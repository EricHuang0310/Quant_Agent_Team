import json
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

from agents.base_agent import TradeSignal, Signal
from core.broker import AlpacaBroker
from core.portfolio import PortfolioManager
from config.settings import TradingConfig
from utils.logger import setup_logger
from utils.prompt_loader import PromptLoader

logger = setup_logger("agent.risk")


@dataclass
class RiskCheck:
    passed: bool
    reason: str
    modified_signal: Optional[TradeSignal] = None


class RiskAgent:
    """Evaluates trade signals against risk constraints.

    Two layers of risk evaluation:
    1. **Quantitative checks** (fast, deterministic) — always run first
       - Circuit breaker, daily loss limit, position concentration, position size
    2. **LLM qualitative review** (optional, slower) — only for edge-case signals
       - Triggered when quantitative checks pass but conditions are borderline
       - Prompt loaded from ``prompts/risk_agent.md``
    """

    _CORRELATED_GROUPS = [
        {"BTC/USD", "WBTC/USD"},
        {"ETH/USD", "MATIC/USD", "ARB/USD", "OP/USD"},
    ]

    def __init__(
        self,
        portfolio: PortfolioManager,
        broker: AlpacaBroker,
        config: TradingConfig,
        anthropic_api_key: str = "",
        llm_model: str = "claude-sonnet-4-5-20250929",
    ):
        self.portfolio = portfolio
        self.broker = broker
        self.config = config
        self._drawdown_high_water: float = config.total_capital
        self._circuit_breaker_active: bool = False
        self._daily_loss: float = 0.0

        # LLM qualitative review (optional — disabled if no API key)
        self._llm_client = (
            anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        )
        self._llm_model = llm_model
        self._prompt_loader = PromptLoader()
        self._recent_trades: List[dict] = []
        self._cycle_count: int = 0
        self._circuit_breaker_reset_cycle: int = -100  # long ago

    # ── Public API ───────────────────────────────────────────

    def evaluate_signal(
        self, signal: TradeSignal, strategy_name: str
    ) -> RiskCheck:
        """Run all risk checks on a proposed trade signal."""
        # Sells / holds always allowed (reduce risk)
        if signal.signal in (Signal.SELL, Signal.STRONG_SELL, Signal.HOLD):
            return RiskCheck(passed=True, reason="sell_or_hold_always_allowed")

        # Layer 1 — quantitative checks
        checks = [
            self._check_circuit_breaker(),
            self._check_daily_loss_limit(),
            self._check_position_concentration(signal),
            self._check_position_size(signal, strategy_name),
        ]
        for check in checks:
            if not check.passed:
                logger.warning(
                    f"Risk BLOCKED {signal.symbol}/{strategy_name}: {check.reason}"
                )
                return check

        # Layer 2 — LLM qualitative review for edge cases
        if self._should_llm_review(signal, strategy_name):
            logger.info(
                f"Edge-case detected for {signal.symbol}/{strategy_name}, "
                f"running LLM risk review..."
            )
            llm_check = self._llm_risk_review(signal, strategy_name)
            if not llm_check.passed:
                logger.warning(
                    f"LLM risk review BLOCKED {signal.symbol}/{strategy_name}: "
                    f"{llm_check.reason}"
                )
            return llm_check

        return RiskCheck(passed=True, reason="all_checks_passed")

    def increment_cycle(self):
        """Call once per trading loop cycle."""
        self._cycle_count += 1

    def record_trade_result(self, trade: dict):
        """Track recent trades for LLM context."""
        self._recent_trades.append(trade)
        if len(self._recent_trades) > 20:
            self._recent_trades = self._recent_trades[-20:]

    # ── Quantitative Checks ─────────────────────────────────

    def _check_circuit_breaker(self) -> RiskCheck:
        """Block all new positions if portfolio drawdown exceeds threshold."""
        try:
            account = self.broker.get_account()
            equity = account["equity"]
        except Exception as e:
            logger.error(f"Cannot fetch account for circuit breaker: {e}")
            return RiskCheck(passed=False, reason="account_fetch_failed")

        if equity > self._drawdown_high_water:
            self._drawdown_high_water = equity

        if self._drawdown_high_water > 0:
            drawdown = (self._drawdown_high_water - equity) / self._drawdown_high_water
        else:
            drawdown = 0.0

        if drawdown > self.config.max_drawdown_pct:
            self._circuit_breaker_active = True
            return RiskCheck(
                passed=False,
                reason=f"circuit_breaker: drawdown={drawdown:.2%} > {self.config.max_drawdown_pct:.2%}",
            )

        self._circuit_breaker_active = False
        return RiskCheck(passed=True, reason="circuit_breaker_ok")

    def _check_daily_loss_limit(self) -> RiskCheck:
        if self._daily_loss > self.config.daily_loss_limit_pct * self.portfolio.total_capital:
            return RiskCheck(
                passed=False,
                reason=f"daily_loss_limit: loss=${self._daily_loss:.2f}",
            )
        return RiskCheck(passed=True, reason="daily_loss_ok")

    def _check_position_concentration(self, signal: TradeSignal) -> RiskCheck:
        try:
            positions = self.broker.get_all_positions()
            total_equity = self.portfolio.total_capital
            for p in positions:
                if p.symbol == signal.symbol.replace("/", ""):
                    current_pct = float(p.market_value) / total_equity if total_equity > 0 else 0
                    if current_pct >= self.config.max_position_pct:
                        return RiskCheck(
                            passed=False,
                            reason=f"concentration: {signal.symbol} already at {current_pct:.1%} "
                                   f"(max {self.config.max_position_pct:.1%})",
                        )
        except Exception as e:
            logger.error(f"Concentration check error: {e}")
        return RiskCheck(passed=True, reason="concentration_ok")

    def _check_position_size(self, signal: TradeSignal, strategy_name: str) -> RiskCheck:
        available = self.portfolio.get_available_capital(strategy_name)
        if available <= 0:
            return RiskCheck(
                passed=False,
                reason=f"no_capital: {strategy_name} has no available capital",
            )
        return RiskCheck(passed=True, reason="position_size_ok")

    # ── LLM Qualitative Review ──────────────────────────────

    def _should_llm_review(self, signal: TradeSignal, strategy_name: str) -> bool:
        """Decide whether this signal warrants LLM qualitative review.

        Returns True for edge-case situations where quantitative checks pass
        but conditions are borderline.
        """
        if not self._llm_client:
            return False

        # Edge 1: Circuit breaker was recently reset (within 3 cycles)
        if self._cycle_count - self._circuit_breaker_reset_cycle < 3:
            return True

        # Edge 2: Approaching concentration limit (>80% of max)
        try:
            positions = self.broker.get_all_positions()
            total_equity = self.portfolio.total_capital
            for p in positions:
                if p.symbol == signal.symbol.replace("/", ""):
                    current_pct = float(p.market_value) / total_equity if total_equity > 0 else 0
                    if current_pct > self.config.max_position_pct * 0.8:
                        return True
        except Exception:
            pass

        # Edge 3: Daily loss above 50% of limit
        limit = self.config.daily_loss_limit_pct * self.portfolio.total_capital
        if limit > 0 and self._daily_loss > limit * 0.5:
            return True

        # Edge 4: Low confidence signal (borderline)
        if 0.4 <= signal.confidence <= 0.55:
            return True

        return False

    def _llm_risk_review(self, signal: TradeSignal, strategy_name: str) -> RiskCheck:
        """Call Claude for qualitative risk review on edge-case signals."""
        try:
            risk_state = self.get_risk_state()
            portfolio_state = {
                "total_capital": self.portfolio.total_capital,
                "strategy": strategy_name,
                "available_capital": self.portfolio.get_available_capital(strategy_name),
            }

            prompt = self._prompt_loader.load("risk_agent.md", {
                "signal_summary": json.dumps({
                    "symbol": signal.symbol,
                    "signal": signal.signal.name,
                    "confidence": signal.confidence,
                    "target_position_pct": signal.target_position_pct,
                    "stop_loss_pct": signal.stop_loss_pct,
                    "metadata": {
                        k: v for k, v in signal.metadata.items()
                        if isinstance(v, (int, float, str, bool))
                    },
                }, indent=2),
                "risk_state": json.dumps(risk_state, indent=2),
                "recent_trades": json.dumps(self._recent_trades[-5:], indent=2),
                "portfolio_state": json.dumps(portfolio_state, indent=2),
            })

            response = self._llm_client.messages.create(
                model=self._llm_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

            data = json.loads(text)
            approved = data.get("approved", True)
            reasoning = data.get("reasoning", "No reasoning")
            risk_score = data.get("risk_score", 0.5)

            logger.info(
                f"LLM risk review: approved={approved}, "
                f"risk_score={risk_score:.2f}, reason={reasoning}"
            )

            if not approved:
                return RiskCheck(passed=False, reason=f"llm_review_rejected: {reasoning}")

            # Check if LLM suggests reducing position size
            mods = data.get("suggested_modifications", {})
            reduce_pct = mods.get("reduce_size_pct")
            if reduce_pct is not None and 0 < reduce_pct <= 1.0:
                reduced_target = signal.target_position_pct * (1 - reduce_pct)
                modified = TradeSignal(
                    symbol=signal.symbol,
                    signal=signal.signal,
                    confidence=signal.confidence,
                    target_position_pct=reduced_target,
                    stop_loss_pct=signal.stop_loss_pct,
                    take_profit_pct=signal.take_profit_pct,
                    metadata={
                        **signal.metadata,
                        "llm_size_reduction": reduce_pct,
                        "llm_reasoning": reasoning,
                    },
                )
                return RiskCheck(
                    passed=True,
                    reason=f"llm_approved_with_reduction({reduce_pct:.0%}): {reasoning}",
                    modified_signal=modified,
                )

            return RiskCheck(passed=True, reason=f"llm_approved: {reasoning}")

        except Exception as e:
            # Fail-open: if LLM review fails, allow the signal through
            logger.warning(f"LLM risk review failed, allowing signal: {e}")
            return RiskCheck(passed=True, reason="llm_review_failed_passthrough")

    # ── State Management ────────────────────────────────────

    def update_daily_loss(self, loss_amount: float):
        self._daily_loss += loss_amount

    def reset_daily_counters(self):
        self._daily_loss = 0.0
        logger.info("Daily risk counters reset")

    def reset_circuit_breaker(self):
        self._circuit_breaker_active = False
        self._circuit_breaker_reset_cycle = self._cycle_count
        account = self.broker.get_account()
        self._drawdown_high_water = account["equity"]
        logger.info(f"Circuit breaker reset. New HWM: {self._drawdown_high_water:.2f}")

    def get_risk_state(self) -> dict:
        return {
            "circuit_breaker_active": self._circuit_breaker_active,
            "high_water_mark": self._drawdown_high_water,
            "daily_loss": self._daily_loss,
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "daily_loss_limit": self.config.daily_loss_limit_pct * self.portfolio.total_capital,
            "cycle_count": self._cycle_count,
            "cycles_since_cb_reset": self._cycle_count - self._circuit_breaker_reset_cycle,
        }
