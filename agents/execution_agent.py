from typing import Dict, List, Optional, Tuple

from alpaca.trading.enums import OrderSide

from agents.base_agent import Signal, TradeSignal
from agents.risk_agent import RiskAgent
from core.broker import AlpacaBroker
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager
from utils.logger import setup_logger

logger = setup_logger("agent.execution")

MIN_ORDER_VALUE = 10.0  # minimum $10 order
STOP_LIMIT_SPREAD = 0.005  # 0.5% below stop price for limit


class ExecutionAgent:
    """Converts approved TradeSignals into actual Alpaca orders.

    Also manages stop-loss orders and grid limit orders.
    """

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
        # Track active stop-loss orders: (symbol, strategy) -> order_id
        self._stop_loss_orders: Dict[Tuple[str, str], str] = {}

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

    def execute_grid_orders(self, grid_agent, symbols: List[str]) -> List[dict]:
        """Place limit orders for pending grid levels."""
        results = []
        for symbol in symbols:
            pending = grid_agent.get_pending_grid_orders(symbol)
            if not pending:
                continue

            current_price = self.market_data.get_current_price(symbol)
            if current_price is None or current_price <= 0:
                continue

            available = self.portfolio.get_available_capital("grid")
            alloc = self.portfolio.get_strategy_allocation("grid")
            per_grid_value = alloc.capital_allocated * grid_agent.config.get("position_per_grid", 0.02)

            for level in pending:
                if level.order_id:
                    continue  # already has an order

                if per_grid_value < MIN_ORDER_VALUE:
                    continue

                qty = per_grid_value / level.price
                try:
                    order = self.broker.submit_limit_order(
                        symbol, qty, level.side, level.price
                    )
                    level.order_id = str(order.id)
                    results.append({
                        "status": "placed",
                        "type": "grid_limit",
                        "symbol": symbol,
                        "strategy": "grid",
                        "order_id": str(order.id),
                        "side": level.side.value,
                        "price": level.price,
                        "qty": qty,
                    })
                    logger.info(
                        f"Grid limit order: {level.side.value} {qty:.6f} {symbol} "
                        f"@ {level.price}"
                    )
                except Exception as e:
                    logger.error(f"Grid order failed for {symbol} @ {level.price}: {e}")
                    results.append({
                        "status": "error",
                        "type": "grid_limit",
                        "symbol": symbol,
                        "strategy": "grid",
                        "reason": str(e),
                    })
        return results

    def check_and_place_stop_losses(self) -> List[dict]:
        """Check all positions and place stop-loss orders where missing."""
        results = []
        for trade in self.portfolio.get_trade_log():
            if trade.get("side") != "buy":
                continue
            symbol = trade.get("symbol", "")
            strategy = trade.get("strategy", "")
            stop_pct = trade.get("stop_loss_pct")
            if not stop_pct or not symbol or not strategy:
                continue

            key = (symbol, strategy)
            if key in self._stop_loss_orders:
                continue  # already has a stop

            pos = self.portfolio.get_position(symbol, strategy)
            if pos is None or pos.qty <= 0:
                continue

            entry_price = pos.avg_entry_price
            stop_price = round(entry_price * (1 - stop_pct), 2)
            limit_price = round(stop_price * (1 - STOP_LIMIT_SPREAD), 2)

            try:
                order = self.broker.submit_stop_limit_order(
                    symbol, pos.qty, OrderSide.SELL, stop_price, limit_price
                )
                self._stop_loss_orders[key] = str(order.id)
                results.append({
                    "status": "placed",
                    "type": "stop_loss",
                    "symbol": symbol,
                    "strategy": strategy,
                    "order_id": str(order.id),
                    "stop_price": stop_price,
                    "limit_price": limit_price,
                })
                logger.info(
                    f"Stop-loss placed: {symbol}/{strategy} "
                    f"stop={stop_price} limit={limit_price}"
                )
            except Exception as e:
                logger.error(f"Stop-loss order failed for {symbol}/{strategy}: {e}")

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
            # Cancel existing stop-loss for this position
            self._cancel_stop_loss(signal.symbol, strategy_name)

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
            trade_record = {
                "strategy": strategy_name,
                "symbol": signal.symbol,
                "side": side.value,
                "qty": qty,
                "price": current_price,
                "value": qty * current_price,
                "signal": signal.signal.name,
                "confidence": signal.confidence,
            }
            # Preserve stop_loss_pct in trade record for check_and_place_stop_losses
            if signal.stop_loss_pct and side == OrderSide.BUY:
                trade_record["stop_loss_pct"] = signal.stop_loss_pct

            self.portfolio.record_trade(trade_record)
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

    def _cancel_stop_loss(self, symbol: str, strategy: str):
        key = (symbol, strategy)
        order_id = self._stop_loss_orders.pop(key, None)
        if order_id:
            try:
                self.broker.cancel_order(order_id)
                logger.info(f"Cancelled stop-loss {order_id} for {symbol}/{strategy}")
            except Exception as e:
                logger.warning(f"Failed to cancel stop-loss {order_id}: {e}")
