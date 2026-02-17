from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide

from agents.base_agent import BaseStrategyAgent, Signal, TradeSignal
from core.indicators import Indicators
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager


@dataclass
class GridLevel:
    price: float
    side: OrderSide
    is_filled: bool = False
    order_id: Optional[str] = None


class GridAgent(BaseStrategyAgent):
    """Grid trading strategy for ranging markets.

    Only activates when ADX < threshold (non-trending).
    Places a grid of buy/sell limit orders within the recent price range.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        portfolio: PortfolioManager,
        config: dict,
    ):
        super().__init__("grid", market_data, portfolio, config)
        self._grids: Dict[str, List[GridLevel]] = {}

    def setup_grid(self, symbol: str, df) -> List[GridLevel]:
        """Calculate grid levels based on recent price range."""
        lookback = self.config.get("range_lookback_days", 7)
        # Use the last N days of data
        recent = df.tail(lookback * 24)  # approximate hours
        if len(recent) < 24:
            recent = df

        high = recent["high"].max()
        low = recent["low"].min()
        mid = (high + low) / 2
        spacing = mid * self.config["grid_spacing_pct"] / 100
        n_levels = self.config["grid_levels"]

        levels = []
        for i in range(1, n_levels + 1):
            buy_price = mid - i * spacing
            sell_price = mid + i * spacing
            if buy_price >= low * 0.98:
                levels.append(GridLevel(price=round(buy_price, 2), side=OrderSide.BUY))
            if sell_price <= high * 1.02:
                levels.append(GridLevel(price=round(sell_price, 2), side=OrderSide.SELL))

        self._grids[symbol] = levels
        self.logger.info(
            f"Grid set up for {symbol}: {len(levels)} levels, "
            f"range=[{low:.2f}, {high:.2f}], mid={mid:.2f}"
        )
        return levels

    def analyze(self, symbol: str) -> TradeSignal:
        df = self.market_data.get_bars(
            symbol, TimeFrame.Hour, lookback_days=14
        )

        if len(df) < 30:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD,
                confidence=0.0, target_position_pct=0.0,
                metadata={"reason": "insufficient_data"},
            )

        adx_df = Indicators.adx(df, self.config.get("adx_period", 14))
        adx_col = f"ADX_{self.config.get('adx_period', 14)}"
        adx_val = adx_df[adx_col].iloc[-1]

        if adx_val != adx_val:  # NaN check
            adx_val = 30  # assume trending if we can't compute

        adx_threshold = self.config.get("adx_threshold", 20)

        if adx_val > adx_threshold:
            # Market is trending — grid not suitable
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                target_position_pct=0.0,
                metadata={"reason": "trending_market", "adx": float(adx_val)},
            )

        # Market is ranging — grid is suitable
        if symbol not in self._grids or not self._grids[symbol]:
            self.setup_grid(symbol, df)

        close = df.iloc[-1]["close"]
        grid = self._grids[symbol]

        # Find nearest unfilled grid levels
        nearest_buy = None
        nearest_sell = None
        for level in grid:
            if level.is_filled:
                continue
            if level.side == OrderSide.BUY and close > level.price:
                if nearest_buy is None or level.price > nearest_buy.price:
                    nearest_buy = level
            elif level.side == OrderSide.SELL and close < level.price:
                if nearest_sell is None or level.price < nearest_sell.price:
                    nearest_sell = level

        # Determine signal based on proximity to grid levels
        signal = Signal.HOLD
        confidence = 0.0
        target_pct = 0.0

        if nearest_buy and (close - nearest_buy.price) / close < 0.01:
            signal = Signal.BUY
            confidence = 0.6
            target_pct = self.config["position_per_grid"]
        elif nearest_sell and (nearest_sell.price - close) / close < 0.01:
            signal = Signal.SELL
            confidence = 0.6
            target_pct = 0.0

        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            target_position_pct=target_pct,
            stop_loss_pct=0.02,
            metadata={
                "adx": float(adx_val),
                "grid_levels": len(grid),
                "unfilled_levels": sum(1 for l in grid if not l.is_filled),
                "mode": "ranging",
            },
        )

    def get_pending_grid_orders(self, symbol: str) -> List[GridLevel]:
        """Returns unfilled grid levels for external order placement."""
        return [l for l in self._grids.get(symbol, []) if not l.is_filled]

    def mark_filled(self, symbol: str, price: float):
        """Mark the closest grid level as filled."""
        for level in self._grids.get(symbol, []):
            if abs(level.price - price) / price < 0.001:
                level.is_filled = True
                break

    def reset_grid(self, symbol: str):
        """Clear grid for a symbol (e.g., when market regime changes)."""
        self._grids.pop(symbol, None)

    def get_strategy_state(self) -> dict:
        grid_info = {}
        for sym, levels in self._grids.items():
            grid_info[sym] = {
                "total_levels": len(levels),
                "filled": sum(1 for l in levels if l.is_filled),
                "unfilled": sum(1 for l in levels if not l.is_filled),
            }
        return {
            "name": self.name,
            "active": self._is_active,
            "type": "grid_trading",
            "params": self.config,
            "grids": grid_info,
        }
