from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLimitOrderRequest,
)

from config.settings import AlpacaConfig
from utils.logger import setup_logger

logger = setup_logger("core.broker")


class AlpacaBroker:
    """Unified broker interface wrapping alpaca-py TradingClient for crypto."""

    def __init__(self, config: AlpacaConfig):
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        logger.info(f"Broker initialized (paper={config.paper})")

    # ── Account ──────────────────────────────────────────────

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def get_buying_power(self) -> float:
        return float(self._client.get_account().buying_power)

    # ── Positions ────────────────────────────────────────────

    def get_all_positions(self) -> list:
        return self._client.get_all_positions()

    def get_position(self, symbol: str) -> Optional[object]:
        sym = self._format_symbol(symbol)
        try:
            return self._client.get_open_position(sym)
        except Exception:
            return None

    def close_position(self, symbol: str):
        sym = self._format_symbol(symbol)
        return self._client.close_position(sym)

    def close_all_positions(self):
        self._client.close_all_positions(cancel_orders=True)
        logger.info("All positions closed")

    # ── Orders ───────────────────────────────────────────────

    def submit_market_order(
        self, symbol: str, qty: float, side: OrderSide
    ):
        req = MarketOrderRequest(
            symbol=self._format_symbol(symbol),
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
        )
        order = self._client.submit_order(req)
        logger.info(
            f"Market order: {side.value} {qty:.6f} {symbol} -> {order.id}"
        )
        return order

    def submit_limit_order(
        self, symbol: str, qty: float, side: OrderSide, limit_price: float
    ):
        req = LimitOrderRequest(
            symbol=self._format_symbol(symbol),
            qty=qty,
            side=side,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
        )
        order = self._client.submit_order(req)
        logger.info(
            f"Limit order: {side.value} {qty:.6f} {symbol} @ {limit_price} -> {order.id}"
        )
        return order

    def submit_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        stop_price: float,
        limit_price: float,
    ):
        req = StopLimitOrderRequest(
            symbol=self._format_symbol(symbol),
            qty=qty,
            side=side,
            stop_price=stop_price,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
        )
        order = self._client.submit_order(req)
        logger.info(
            f"StopLimit order: {side.value} {qty:.6f} {symbol} "
            f"stop={stop_price} limit={limit_price} -> {order.id}"
        )
        return order

    def get_open_orders(self) -> list:
        return self._client.get_orders()

    def cancel_order(self, order_id: str):
        self._client.cancel_order_by_id(order_id)
        logger.info(f"Cancelled order {order_id}")

    def cancel_all_orders(self):
        self._client.cancel_orders()
        logger.info("All orders cancelled")

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        """Normalize to Alpaca crypto format: 'BTC/USD'."""
        symbol = symbol.upper().strip()
        if "/" not in symbol:
            if symbol.endswith("USD"):
                symbol = symbol[:-3] + "/USD"
            else:
                symbol = symbol + "/USD"
        return symbol
