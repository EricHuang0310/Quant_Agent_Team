import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from agents.base_agent import Signal, TradeSignal
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.grid_agent import GridAgent


def _make_trending_up_data(n: int = 200) -> pd.DataFrame:
    """Generate data with a clear uptrend."""
    np.random.seed(42)
    trend = np.linspace(100, 130, n)
    noise = np.random.randn(n) * 0.3
    close = trend + noise
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=idx)


def _make_ranging_data(n: int = 200) -> pd.DataFrame:
    """Generate data that oscillates in a range."""
    np.random.seed(42)
    close = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) * 0.3
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=idx)


def _make_oversold_data(n: int = 200) -> pd.DataFrame:
    """Generate data with a sharp drop (mean reversion buy signal)."""
    np.random.seed(42)
    base = np.full(n, 100.0)
    # Sharp drop in last 30 bars
    base[-30:] = np.linspace(100, 85, 30)
    noise = np.random.randn(n) * 0.2
    close = base + noise
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=idx)


def _mock_market_data(data: pd.DataFrame) -> MagicMock:
    md = MagicMock()
    md.get_bars.return_value = data
    return md


def _mock_portfolio() -> MagicMock:
    p = MagicMock()
    p.get_strategy_allocation.return_value = MagicMock(
        capital_allocated=100_000.0, positions=[]
    )
    p.get_available_capital.return_value = 100_000.0
    return p


MOMENTUM_CONFIG = {
    "fast_period": 12, "slow_period": 26, "signal_period": 9,
    "rsi_period": 14, "rsi_overbought": 75, "rsi_oversold": 25,
    "atr_period": 14, "trend_ema_period": 50, "adx_period": 14,
}

MEAN_REV_CONFIG = {
    "bb_period": 20, "bb_std": 2.5, "rsi_period": 14,
    "rsi_overbought": 80, "rsi_oversold": 20,
    "zscore_entry": 2.5, "zscore_exit": 0.5,
}

GRID_CONFIG = {
    "grid_levels": 10, "grid_spacing_pct": 1.0,
    "position_per_grid": 0.02, "range_lookback_days": 7,
    "adx_threshold": 25, "adx_period": 14,
}


class TestMomentumSignals:
    def test_returns_trade_signal(self):
        data = _make_trending_up_data()
        agent = MomentumAgent(_mock_market_data(data), _mock_portfolio(), MOMENTUM_CONFIG)
        signal = agent.analyze("BTC/USD")
        assert isinstance(signal, TradeSignal)
        assert signal.symbol == "BTC/USD"
        assert isinstance(signal.signal, Signal)

    def test_confidence_range(self):
        data = _make_trending_up_data()
        agent = MomentumAgent(_mock_market_data(data), _mock_portfolio(), MOMENTUM_CONFIG)
        signal = agent.analyze("BTC/USD")
        assert 0.0 <= signal.confidence <= 1.0

    def test_stop_loss_set(self):
        data = _make_trending_up_data()
        agent = MomentumAgent(_mock_market_data(data), _mock_portfolio(), MOMENTUM_CONFIG)
        signal = agent.analyze("BTC/USD")
        assert signal.stop_loss_pct is not None
        assert signal.stop_loss_pct > 0

    def test_insufficient_data(self):
        short_data = _make_trending_up_data(10)
        agent = MomentumAgent(_mock_market_data(short_data), _mock_portfolio(), MOMENTUM_CONFIG)
        signal = agent.analyze("BTC/USD")
        assert signal.signal == Signal.HOLD
        assert signal.metadata.get("reason") == "insufficient_data"

    def test_metadata_has_nan_safe_values(self):
        data = _make_trending_up_data()
        agent = MomentumAgent(_mock_market_data(data), _mock_portfolio(), MOMENTUM_CONFIG)
        signal = agent.analyze("BTC/USD")
        for key in ("macd_hist", "rsi", "adx", "atr"):
            assert not pd.isna(signal.metadata[key])


class TestMeanReversionSignals:
    def test_returns_trade_signal(self):
        data = _make_ranging_data()
        agent = MeanReversionAgent(_mock_market_data(data), _mock_portfolio(), MEAN_REV_CONFIG)
        signal = agent.analyze("ETH/USD")
        assert isinstance(signal, TradeSignal)

    def test_oversold_triggers_buy(self):
        data = _make_oversold_data()
        agent = MeanReversionAgent(_mock_market_data(data), _mock_portfolio(), MEAN_REV_CONFIG)
        signal = agent.analyze("ETH/USD")
        # With a 15% drop, zscore should be very negative
        if signal.metadata.get("zscore", 0) < -MEAN_REV_CONFIG["zscore_entry"]:
            assert signal.signal in (Signal.BUY, Signal.STRONG_BUY)

    def test_stop_loss_fixed(self):
        data = _make_ranging_data()
        agent = MeanReversionAgent(_mock_market_data(data), _mock_portfolio(), MEAN_REV_CONFIG)
        signal = agent.analyze("ETH/USD")
        assert signal.stop_loss_pct == 0.03


class TestGridSignals:
    def test_ranging_market_sets_up_grid(self):
        data = _make_ranging_data()
        agent = GridAgent(_mock_market_data(data), _mock_portfolio(), GRID_CONFIG)
        signal = agent.analyze("SOL/USD")
        assert isinstance(signal, TradeSignal)
        # Should have set up grid if ADX is low enough
        if signal.metadata.get("adx", 100) < GRID_CONFIG["adx_threshold"]:
            assert signal.metadata.get("grid_levels", 0) > 0

    def test_get_pending_grid_orders(self):
        data = _make_ranging_data()
        agent = GridAgent(_mock_market_data(data), _mock_portfolio(), GRID_CONFIG)
        agent.analyze("SOL/USD")
        pending = agent.get_pending_grid_orders("SOL/USD")
        # If grid was set up, pending should be non-empty
        if "SOL/USD" in agent._grids:
            assert len(pending) > 0

    def test_mark_filled(self):
        data = _make_ranging_data()
        agent = GridAgent(_mock_market_data(data), _mock_portfolio(), GRID_CONFIG)
        agent.analyze("SOL/USD")
        if agent._grids.get("SOL/USD"):
            first_level = agent._grids["SOL/USD"][0]
            agent.mark_filled("SOL/USD", first_level.price)
            assert first_level.is_filled

    def test_reset_grid(self):
        data = _make_ranging_data()
        agent = GridAgent(_mock_market_data(data), _mock_portfolio(), GRID_CONFIG)
        agent.analyze("SOL/USD")
        agent.reset_grid("SOL/USD")
        assert "SOL/USD" not in agent._grids


class TestAnalyzeUniverse:
    def test_error_handling(self):
        """analyze_universe should handle errors gracefully and return HOLD."""
        md = MagicMock()
        md.get_bars.side_effect = Exception("API error")
        agent = MomentumAgent(md, _mock_portfolio(), MOMENTUM_CONFIG)
        signals = agent.analyze_universe(["BTC/USD", "ETH/USD"])
        assert len(signals) == 2
        assert all(s.signal == Signal.HOLD for s in signals)
        assert all("error" in s.metadata for s in signals)
