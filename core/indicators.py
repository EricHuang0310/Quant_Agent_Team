import numpy as np
import pandas as pd
import pandas_ta as ta


class Indicators:
    """Thin wrapper around pandas-ta for consistent indicator access."""

    @staticmethod
    def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
        return ta.ema(df[col], length=period)

    @staticmethod
    def sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
        return ta.sma(df[col], length=period)

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return ta.rsi(df["close"], length=period)

    @staticmethod
    def macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        return ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

    @staticmethod
    def bbands(
        df: pd.DataFrame, period: int = 20, std: float = 2.0
    ) -> pd.DataFrame:
        return ta.bbands(df["close"], length=period, std=std)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return ta.atr(df["high"], df["low"], df["close"], length=period)

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        return ta.adx(df["high"], df["low"], df["close"], length=period)

    @staticmethod
    def zscore(series: pd.Series, period: int = 20) -> pd.Series:
        mean = series.rolling(period).mean()
        std = series.rolling(period).std()
        return (series - mean) / std

    @staticmethod
    def volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Annualized volatility based on log returns."""
        log_ret = df["close"].pct_change().apply(lambda x: np.log1p(x) if x == x else x)
        return log_ret.rolling(period).std() * (252 ** 0.5)
