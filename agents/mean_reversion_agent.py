import pandas as pd
from alpaca.data.timeframe import TimeFrame

from agents.base_agent import BaseStrategyAgent, Signal, TradeSignal
from core.indicators import Indicators
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager


class MeanReversionAgent(BaseStrategyAgent):
    """Mean reversion strategy using Bollinger Bands + Z-Score + RSI.

    Entry logic:
      - BUY  when: Z-score < -entry_threshold AND RSI < oversold
      - SELL when: Z-score > +entry_threshold AND RSI > overbought
      - EXIT when: Z-score returns to within exit_threshold of 0
    Stop-loss: fixed 3%.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        portfolio: PortfolioManager,
        config: dict,
    ):
        super().__init__("mean_reversion", market_data, portfolio, config)

    def analyze(self, symbol: str) -> TradeSignal:
        df = self.market_data.get_bars(
            symbol, TimeFrame.Hour, lookback_days=self.config.get("lookback_days", 30)
        )

        period = self.config["bb_period"]
        if len(df) < period + 10:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD,
                confidence=0.0, target_position_pct=0.0,
                metadata={"reason": "insufficient_data"},
            )

        bb = Indicators.bbands(df, period, self.config["bb_std"])
        rsi = Indicators.rsi(df, self.config["rsi_period"])
        zscore = Indicators.zscore(df["close"], period)

        latest = df.iloc[-1]
        close = latest["close"]
        z = zscore.iloc[-1]
        rsi_val = rsi.iloc[-1]

        # Handle NaN
        if pd.isna(z) or pd.isna(rsi_val):
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD,
                confidence=0.0, target_position_pct=0.0,
                metadata={"reason": "nan_indicators"},
            )

        bb_std = self.config["bb_std"]
        bb_lower_col = f"BBL_{period}_{bb_std}"
        bb_upper_col = f"BBU_{period}_{bb_std}"
        bb_mid_col = f"BBM_{period}_{bb_std}"

        bb_lower = bb[bb_lower_col].iloc[-1]
        bb_upper = bb[bb_upper_col].iloc[-1]
        bb_mid = bb[bb_mid_col].iloc[-1]

        signal = Signal.HOLD
        confidence = 0.0
        target_pct = 0.0

        zscore_entry = self.config["zscore_entry"]
        zscore_exit = self.config["zscore_exit"]

        # Oversold — buy
        if z < -zscore_entry and rsi_val < self.config["rsi_oversold"]:
            signal = Signal.BUY
            confidence = min(abs(z) / 3.0, 1.0)
            target_pct = confidence * 0.6
            if z < -(zscore_entry * 1.5):
                signal = Signal.STRONG_BUY
                confidence = min(confidence + 0.15, 1.0)
                target_pct = confidence * 0.7

        # Overbought — sell
        elif z > zscore_entry and rsi_val > self.config["rsi_overbought"]:
            signal = Signal.SELL
            confidence = min(abs(z) / 3.0, 1.0)
            target_pct = 0.0
            if z > zscore_entry * 1.5:
                signal = Signal.STRONG_SELL

        # Reversion to mean — exit existing positions
        elif abs(z) < zscore_exit:
            signal = Signal.HOLD
            confidence = 0.8
            target_pct = 0.0  # close positions

        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            target_position_pct=target_pct,
            stop_loss_pct=0.03,
            metadata={
                "zscore": float(z),
                "rsi": float(rsi_val),
                "bb_lower": 0.0 if pd.isna(bb_lower) else float(bb_lower),
                "bb_upper": 0.0 if pd.isna(bb_upper) else float(bb_upper),
                "bb_mid": 0.0 if pd.isna(bb_mid) else float(bb_mid),
                "close": float(close),
            },
        )

    def get_strategy_state(self) -> dict:
        return {
            "name": self.name,
            "active": self._is_active,
            "type": "mean_reversion_bollinger",
            "params": self.config,
        }
