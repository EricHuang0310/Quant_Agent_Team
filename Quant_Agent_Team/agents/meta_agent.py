import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import anthropic
from alpaca.data.timeframe import TimeFrame

from agents.base_agent import BaseStrategyAgent
from agents.risk_agent import RiskAgent
from config.settings import MetaAgentConfig
from core.indicators import Indicators
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager
from utils.logger import setup_logger
from utils.prompt_loader import PromptLoader

logger = setup_logger("agent.meta")


class MarketRegime(Enum):
    STRONG_TREND_UP = "strong_trend_up"
    TREND_UP = "trend_up"
    RANGING = "ranging"
    TREND_DOWN = "trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class AllocationDecision:
    regime: MarketRegime
    allocations: Dict[str, float]  # strategy_name -> pct (sum to 1.0)
    reasoning: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Default fallback allocation when LLM fails
_FALLBACK_ALLOCATION = {
    "momentum": 0.34,
    "mean_reversion": 0.33,
    "grid": 0.33,
}


class MetaAgent:
    """Orchestrator that uses LLM reasoning to allocate capital across strategies.

    Responsibilities:
    1. Detect market regime (quantitative + LLM reasoning)
    2. Decide capital allocation across strategy agents
    3. Activate/deactivate strategies based on regime
    """

    def __init__(
        self,
        config: MetaAgentConfig,
        strategy_agents: Dict[str, BaseStrategyAgent],
        risk_agent: RiskAgent,
        portfolio: PortfolioManager,
        market_data: MarketDataService,
    ):
        self.config = config
        self.agents = strategy_agents
        self.risk_agent = risk_agent
        self.portfolio = portfolio
        self.market_data = market_data
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._prompt_loader = PromptLoader()
        self._allocation_history: List[AllocationDecision] = []
        self._cycle_count = 0

    def detect_regime(self) -> MarketRegime:
        """Quantitative regime detection based on BTC (market leader)."""
        try:
            df = self.market_data.get_bars("BTC/USD", TimeFrame.Hour, lookback_days=14)
            if len(df) < 50:
                return MarketRegime.RANGING

            adx_df = Indicators.adx(df, 14)
            adx_val = adx_df["ADX_14"].iloc[-1]
            rsi = Indicators.rsi(df, 14).iloc[-1]
            ema_20 = Indicators.ema(df, 20).iloc[-1]
            ema_50 = Indicators.ema(df, 50).iloc[-1]
            close = df.iloc[-1]["close"]

            # Volatility: 20-period std of returns
            returns = df["close"].pct_change().dropna()
            vol = returns.tail(20).std() * (24 ** 0.5)  # annualize hourly

            # Determine regime
            if adx_val != adx_val:
                return MarketRegime.RANGING

            if vol > 0.04:  # very high hourly vol
                return MarketRegime.HIGH_VOLATILITY

            if adx_val > 30:
                if close > ema_20 > ema_50:
                    return MarketRegime.STRONG_TREND_UP if adx_val > 40 else MarketRegime.TREND_UP
                elif close < ema_20 < ema_50:
                    return MarketRegime.STRONG_TREND_DOWN if adx_val > 40 else MarketRegime.TREND_DOWN

            return MarketRegime.RANGING

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime.RANGING

    def _build_llm_context(self) -> str:
        """Assemble structured context for Claude prompt."""
        parts = []

        # 1. Market regime
        regime = self.detect_regime()
        parts.append(f"## Current Market Regime (Quantitative): {regime.value}")

        # 2. BTC key metrics
        try:
            df = self.market_data.get_bars("BTC/USD", TimeFrame.Hour, lookback_days=7)
            if len(df) > 0:
                close = df.iloc[-1]["close"]
                change_24h = (close - df.iloc[-24]["close"]) / df.iloc[-24]["close"] * 100 if len(df) > 24 else 0
                change_7d = (close - df.iloc[0]["close"]) / df.iloc[0]["close"] * 100
                parts.append(
                    f"## BTC/USD\n"
                    f"- Price: ${close:,.2f}\n"
                    f"- 24h Change: {change_24h:+.2f}%\n"
                    f"- 7d Change: {change_7d:+.2f}%"
                )
        except Exception:
            pass

        # 3. Per-strategy state and performance
        for name, agent in self.agents.items():
            state = agent.get_strategy_state()
            alloc = self.portfolio.get_strategy_allocation(name)
            parts.append(
                f"## Strategy: {name}\n"
                f"- Active: {state.get('active', False)}\n"
                f"- Current Allocation: {alloc.current_pct:.1%}\n"
                f"- Target Allocation: {alloc.target_pct:.1%}\n"
                f"- Positions: {len(alloc.positions)}\n"
                f"- Unrealized PnL: ${sum(p.unrealized_pnl for p in alloc.positions):,.2f}"
            )

        # 4. Risk state
        risk_state = self.risk_agent.get_risk_state()
        parts.append(
            f"## Risk State\n"
            f"- Circuit Breaker: {'ACTIVE' if risk_state['circuit_breaker_active'] else 'OK'}\n"
            f"- Daily Loss: ${risk_state['daily_loss']:,.2f} / ${risk_state['daily_loss_limit']:,.2f}\n"
            f"- High Water Mark: ${risk_state['high_water_mark']:,.2f}"
        )

        # 5. Portfolio summary
        perf = self.portfolio.get_performance_summary()
        parts.append(
            f"## Portfolio\n"
            f"- Equity: ${perf['equity']:,.2f}\n"
            f"- Cash: ${perf['cash']:,.2f}\n"
            f"- Total PnL: ${perf['total_pnl']:,.2f}\n"
            f"- Total Trades: {perf['total_trades']}"
        )

        return "\n\n".join(parts)

    def decide_allocation(self) -> AllocationDecision:
        """Use LLM to decide capital allocation across strategies."""
        self._cycle_count += 1
        context = self._build_llm_context()
        regime = self.detect_regime()

        # Load strategy descriptions from .md files for richer LLM context
        strategy_descs = self._prompt_loader.load_all_strategy_descriptions()
        strategy_desc_text = ""
        for name, desc in strategy_descs.items():
            strategy_desc_text += f"\n### {name}\n{desc}\n"

        # Build prompt from .md template with runtime variable injection
        prompt = self._prompt_loader.load("meta_agent.md", {
            "strategy_descriptions": strategy_desc_text,
            "min_allocation_pct": str(self.config.min_allocation_pct),
            "max_allocation_pct": str(self.config.max_allocation_pct),
            "market_context": context,
        })

        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            decision = self._parse_response(response.content[0].text, regime)
            logger.info(
                f"LLM allocation: {decision.allocations} "
                f"regime={decision.regime.value} "
                f"confidence={decision.confidence:.2f}"
            )
        except Exception as e:
            logger.error(f"LLM call failed, using fallback: {e}")
            decision = AllocationDecision(
                regime=regime,
                allocations=_FALLBACK_ALLOCATION.copy(),
                reasoning=f"Fallback: LLM call failed ({e})",
                confidence=0.3,
            )

        self._allocation_history.append(decision)
        return decision

    def _parse_response(self, text: str, fallback_regime: MarketRegime) -> AllocationDecision:
        """Parse and validate LLM JSON response."""
        # Try to extract JSON from response
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {text[:200]}")
            return AllocationDecision(
                regime=fallback_regime,
                allocations=_FALLBACK_ALLOCATION.copy(),
                reasoning="Fallback: JSON parse failed",
                confidence=0.3,
            )

        # Validate allocations
        allocs = data.get("allocations", {})
        expected_keys = {"momentum", "mean_reversion", "grid"}
        for key in expected_keys:
            if key not in allocs:
                allocs[key] = _FALLBACK_ALLOCATION[key]

        # Clamp values
        for key in allocs:
            allocs[key] = max(0.0, min(allocs[key], self.config.max_allocation_pct))

        # Normalize to sum to 1.0
        total = sum(allocs.values())
        if total > 0:
            allocs = {k: v / total for k, v in allocs.items()}
        else:
            allocs = _FALLBACK_ALLOCATION.copy()

        # Parse regime
        regime_str = data.get("regime", fallback_regime.value)
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            regime = fallback_regime

        confidence = max(0.0, min(float(data.get("confidence", 0.5)), 1.0))
        reasoning = data.get("reasoning", "No reasoning provided")

        return AllocationDecision(
            regime=regime,
            allocations=allocs,
            reasoning=reasoning,
            confidence=confidence,
        )

    def get_allocation_history(self) -> List[AllocationDecision]:
        return self._allocation_history

    def get_latest_allocation(self) -> Optional[AllocationDecision]:
        return self._allocation_history[-1] if self._allocation_history else None
