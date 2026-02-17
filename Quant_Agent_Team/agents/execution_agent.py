from typing import List, Tuple

from alpaca.trading.enums import OrderSide

from agents.base_agent import Signal, TradeSignal
from agents.risk_agent import RiskAgent
from core.broker import AlpacaBroker
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager
from utils.logger import setup_logger

logger = setup_logger("agent.execution")

MIN_ORDER_VALUE = 10.0  # minimum $10 order


class ExecutionAgent:
    """Converts approved TradeSignals into actual Alpaca orders."""

    def __init__(
        self,
        broker: AlpacaBroker,
        portfolio: PortfolioManager,
        risk_agent: RiskAgent,
        market_data: MarketDataService,
    ):
        self.broker = broker
        self.portfolio = portfolio
        self.risk_agent = risk_agent
        self.market_data = market_data

    def execute_signals(
        self, signals: List[Tuple[str, TradeSignal]]
    ) -> List[dict]:
        """Process a batch of (strategy_name, signal) pairs."""
        results = []
        for strategy_name, signal in signals:
            result = self._execute_single(strategy_name, signal)
            results.append(result)
            if result["status"] == "filled":
                logger.info(
                    f"Executed: {strategy_name}/{signal.symbol} "
                    f"{signal.signal.name} -> order {result.get('order_id')}"
                )
        return results

    def _execute_single(self, strategy_name: str, signal: TradeSignal) -> dict:
        # 1. Risk check
        risk_check = self.risk_agent.evaluate_signal(signal, strategy_name)
        if not risk_check.passed:
            return {
                "status": "blocked",
                "symbol": signal.symbol,
                "strategy": strategy_name,
                "reason": risk_check.reason,
            }

        # Use modified signal if risk agent adjusted it (e.g. LLM reduced size)
        if risk_check.modified_signal is not None:
            logger.info(
                f"Using modified signal for {signal.symbol}/{strategy_name}: "
                f"target {signal.target_position_pct:.1%} -> "
                f"{risk_check.modified_signal.target_position_pct:.1%}"
            )
            signal = risk_check.modified_signal

        # 2. Get current price
        current_price = self.market_data.get_current_price(signal.symbol)
        if current_price is None or current_price <= 0:
            return {
                "status": "error",
                "symbol": signal.symbol,
                "strategy": strategy_name,
                "reason": "cannot_get_price",
            }

        # 3. Calculate order size
        allocated_capital = self.portfolio.get_available_capital(strategy_name)
        alloc = self.portfolio.get_strategy_allocation(strategy_name)
        total_alloc = alloc.capital_allocated

        if signal.signal in (Signal.BUY, Signal.STRONG_BUY):
            target_value = total_alloc * signal.target_position_pct
            # Check current position in this symbol for this strategy
            current_pos = self.portfolio.get_position(signal.symbol, strategy_name)
            current_value = current_pos.market_value if current_pos else 0.0
            delta = target_value - current_value

            if delta < MIN_ORDER_VALUE:
                return {
                    "status": "skipped",
                    "symbol": signal.symbol,
                    "strategy": strategy_name,
                    "reason": f"delta_too_small (${delta:.2f})",
                }

            # Don't exceed available capital
            delta = min(delta, allocated_capital)
            side = OrderSide.BUY
            qty = delta / current_price

        elif signal.signal in (Signal.SELL, Signal.STRONG_SELL):
            current_pos = self.portfolio.get_position(signal.symbol, strategy_name)
            if current_pos is None or current_pos.qty <= 0:
                return {
                    "status": "skipped",
                    "symbol": signal.symbol,
                    "strategy": strategy_name,
                    "reason": "no_position_to_sell",
                }
            qty = current_pos.qty
            side = OrderSide.SELL

        else:
            return {
                "status": "skipped",
                "symbol": signal.symbol,
                "strategy": strategy_name,
                "reason": "hold_signal",
            }

        # 4. Place order
        try:
            order = self.broker.submit_market_order(signal.symbol, qty, side)
            self.portfolio.record_trade({
                "strategy": strategy_name,
                "symbol": signal.symbol,
                "side": side.value,
                "qty": qty,
                "price": current_price,
                "value": qty * current_price,
                "signal": signal.signal.name,
                "confidence": signal.confidence,
            })
            return {
                "status": "filled",
                "symbol": signal.symbol,
                "strategy": strategy_name,
                "order_id": str(order.id),
                "side": side.value,
                "qty": qty,
                "value": qty * current_price,
            }
        except Exception as e:
            logger.error(f"Order failed for {signal.symbol}: {e}")
            return {
                "status": "error",
                "symbol": signal.symbol,
                "strategy": strategy_name,
                "reason": str(e),
            }
