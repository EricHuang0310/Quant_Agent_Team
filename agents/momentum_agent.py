from alpaca.data.timeframe import TimeFrame

from agents.base_agent import BaseStrategyAgent, Signal, TradeSignal
from core.indicators import Indicators
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager


class MomentumAgent(BaseStrategyAgent):
    """Trend-following strategy using MACD + RSI + EMA trend + ADX confirmation.

    Entry logic:
      - BUY  when: ADX > 25 (trending) AND price > EMA(50) AND MACD histogram > 0 AND RSI < 70
      - SELL when: ADX > 25 (trending) AND price < EMA(50) AND MACD histogram < 0 AND RSI > 30
    Stop-loss: 2x ATR from entry price.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        portfolio: PortfolioManager,
        config: dict,
    ):
        super().__init__("momentum", market_data, portfolio, config)

    def analyze(self, symbol: str) -> TradeSignal:
        df = self.market_data.get_bars(
            symbol, TimeFrame.Hour, lookback_days=self.config.get("lookback_days", 30)
        )

        if len(df) < self.config["trend_ema_period"] + 10:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD,
                confidence=0.0, target_position_pct=0.0,
                metadata={"reason": "insufficient_data"},
            )

        macd_df = Indicators.macd(
            df,
            fast=self.config["fast_period"],
            slow=self.config["slow_period"],
            signal=self.config["signal_period"],
        )
        rsi = Indicators.rsi(df, self.config["rsi_period"])
        ema_trend = Indicators.ema(df, self.config["trend_ema_period"])
        adx_df = Indicators.adx(df, self.config["adx_period"])
        atr = Indicators.atr(df, self.config["atr_period"])

        latest = df.iloc[-1]
        close = latest["close"]

        # Extract latest indicator values
        macd_col = f"MACDh_{self.config['fast_period']}_{self.config['slow_period']}_{self.config['signal_period']}"
        macd_hist = macd_df[macd_col].iloc[-1]
        rsi_val = rsi.iloc[-1]
        ema_val = ema_trend.iloc[-1]
        adx_col = f"ADX_{self.config['adx_period']}"
        adx_val = adx_df[adx_col].iloc[-1]
        atr_val = atr.iloc[-1]

        above_trend = close > ema_val
        is_trending = adx_val > 25

        signal = Signal.HOLD
        confidence = 0.0

        if is_trending and above_trend and macd_hist > 0:
            if rsi_val < self.config["rsi_overbought"]:
                signal = Signal.BUY
                confidence = min(0.5 + (adx_val - 25) / 50, 1.0)
                if adx_val > 40 and macd_hist > 0:
                    signal = Signal.STRONG_BUY
                    confidence = min(confidence + 0.1, 1.0)
        elif is_trending and not above_trend and macd_hist < 0:
            if rsi_val > self.config["rsi_oversold"]:
                signal = Signal.SELL
                confidence = min(0.5 + (adx_val - 25) / 50, 1.0)
                if adx_val > 40:
                    signal = Signal.STRONG_SELL
                    confidence = min(confidence + 0.1, 1.0)

        # ATR-based stop loss (2x ATR as % of price)
        stop_loss_pct = (2.0 * atr_val / close) if close > 0 and atr_val == atr_val else 0.05

        target_pct = confidence * 0.8 if signal in (Signal.BUY, Signal.STRONG_BUY) else 0.0

        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            target_position_pct=target_pct,
            stop_loss_pct=stop_loss_pct,
            metadata={
                "macd_hist": float(macd_hist) if macd_hist == macd_hist else 0.0,
                "rsi": float(rsi_val) if rsi_val == rsi_val else 0.0,
                "adx": float(adx_val) if adx_val == adx_val else 0.0,
                "ema_trend": "up" if above_trend else "down",
                "atr": float(atr_val) if atr_val == atr_val else 0.0,
            },
        )

    def get_strategy_state(self) -> dict:
        return {
            "name": self.name,
            "active": self._is_active,
            "type": "momentum_trend_following",
            "params": self.config,
        }
