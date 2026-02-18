import numpy as np
import pandas as pd
import pytest

from core.indicators import Indicators


def _make_ohlcv(n: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


class TestIndicators:
    def test_ema_length(self):
        df = _make_ohlcv()
        result = Indicators.ema(df, 20)
        assert len(result) == len(df)
        # First 19 values should be NaN
        assert pd.isna(result.iloc[0])
        assert not pd.isna(result.iloc[-1])

    def test_sma_length(self):
        df = _make_ohlcv()
        result = Indicators.sma(df, 20)
        assert len(result) == len(df)

    def test_rsi_range(self):
        df = _make_ohlcv()
        result = Indicators.rsi(df, 14)
        valid = result.dropna()
        assert all(0 <= v <= 100 for v in valid)

    def test_macd_columns(self):
        df = _make_ohlcv()
        result = Indicators.macd(df, fast=12, slow=26, signal=9)
        assert "MACDh_12_26_9" in result.columns
        assert "MACD_12_26_9" in result.columns
        assert "MACDs_12_26_9" in result.columns

    def test_bbands_columns(self):
        df = _make_ohlcv()
        result = Indicators.bbands(df, period=20, std=2.0)
        assert "BBL_20_2.0" in result.columns
        assert "BBU_20_2.0" in result.columns
        assert "BBM_20_2.0" in result.columns

    def test_bbands_upper_above_lower(self):
        df = _make_ohlcv()
        result = Indicators.bbands(df, 20, 2.0)
        valid_idx = result.dropna().index
        assert all(result.loc[valid_idx, "BBU_20_2.0"] > result.loc[valid_idx, "BBL_20_2.0"])

    def test_atr_positive(self):
        df = _make_ohlcv()
        result = Indicators.atr(df, 14)
        valid = result.dropna()
        assert all(v > 0 for v in valid)

    def test_adx_columns(self):
        df = _make_ohlcv()
        result = Indicators.adx(df, 14)
        assert "ADX_14" in result.columns

    def test_adx_range(self):
        df = _make_ohlcv()
        result = Indicators.adx(df, 14)
        valid = result["ADX_14"].dropna()
        assert all(0 <= v <= 100 for v in valid)

    def test_zscore_mean_near_zero(self):
        df = _make_ohlcv(200)
        result = Indicators.zscore(df["close"], 20)
        valid = result.dropna()
        # Zscore should have mean close to 0
        assert abs(valid.mean()) < 0.5

    def test_volatility_positive(self):
        df = _make_ohlcv()
        result = Indicators.volatility(df, 20)
        valid = result.dropna()
        assert all(v >= 0 for v in valid)

    def test_short_data(self):
        """pandas-ta returns None when data is too short for the indicator period."""
        df = _make_ohlcv(5)
        result = Indicators.rsi(df, 14)
        assert result is None
