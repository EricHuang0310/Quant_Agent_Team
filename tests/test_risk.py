import pytest
from unittest.mock import MagicMock, patch

from agents.base_agent import Signal, TradeSignal
from agents.risk_agent import RiskAgent, RiskCheck
from config.settings import TradingConfig


def _make_risk_agent(
    equity: float = 100_000.0,
    cash: float = 50_000.0,
    positions: list = None,
    daily_loss: float = 0.0,
) -> RiskAgent:
    """Create a RiskAgent with mocked dependencies."""
    broker = MagicMock()
    broker.get_account.return_value = {
        "equity": equity, "cash": cash,
        "buying_power": cash, "portfolio_value": equity,
    }
    broker.get_all_positions.return_value = positions or []

    portfolio = MagicMock()
    portfolio.total_capital = equity
    portfolio.get_available_capital.return_value = cash
    portfolio.get_strategy_allocation.return_value = MagicMock(
        capital_allocated=equity * 0.34, positions=[]
    )

    config = TradingConfig(
        total_capital=equity,
        max_position_pct=0.15,
        max_drawdown_pct=0.10,
        daily_loss_limit_pct=0.03,
    )

    agent = RiskAgent(portfolio, broker, config)
    agent._daily_loss = daily_loss
    return agent


def _make_buy_signal(symbol="BTC/USD", confidence=0.7, target_pct=0.1):
    return TradeSignal(
        symbol=symbol,
        signal=Signal.BUY,
        confidence=confidence,
        target_position_pct=target_pct,
    )


class TestCircuitBreaker:
    def test_passes_when_no_drawdown(self):
        agent = _make_risk_agent()
        signal = _make_buy_signal()
        result = agent.evaluate_signal(signal, "momentum")
        assert result.passed

    def test_blocks_when_drawdown_exceeded(self):
        agent = _make_risk_agent(equity=88_000.0)
        agent._drawdown_high_water = 100_000.0
        signal = _make_buy_signal()
        result = agent.evaluate_signal(signal, "momentum")
        assert not result.passed
        assert "circuit_breaker" in result.reason

    def test_sells_always_allowed(self):
        agent = _make_risk_agent(equity=88_000.0)
        agent._drawdown_high_water = 100_000.0
        sell_signal = TradeSignal(
            symbol="BTC/USD", signal=Signal.SELL,
            confidence=0.8, target_position_pct=0.0,
        )
        result = agent.evaluate_signal(sell_signal, "momentum")
        assert result.passed


class TestDailyLossLimit:
    def test_passes_below_limit(self):
        agent = _make_risk_agent(daily_loss=1_000.0)
        signal = _make_buy_signal()
        result = agent.evaluate_signal(signal, "momentum")
        assert result.passed

    def test_blocks_above_limit(self):
        # daily_loss_limit_pct=0.03 on 100k = $3000
        agent = _make_risk_agent(daily_loss=3_500.0)
        signal = _make_buy_signal()
        result = agent.evaluate_signal(signal, "momentum")
        assert not result.passed
        assert "daily_loss_limit" in result.reason

    def test_reset_clears_daily_loss(self):
        agent = _make_risk_agent(daily_loss=5_000.0)
        agent.reset_daily_counters()
        assert agent._daily_loss == 0.0


class TestPositionConcentration:
    def test_blocks_concentrated_position(self):
        pos = MagicMock()
        pos.symbol = "BTCUSD"
        pos.market_value = "20000.0"
        agent = _make_risk_agent(positions=[pos])
        # Override to make float conversion work
        type(pos).market_value = 20_000.0
        signal = _make_buy_signal("BTC/USD")
        result = agent._check_position_concentration(signal)
        assert not result.passed

    def test_passes_small_position(self):
        pos = MagicMock()
        pos.symbol = "BTCUSD"
        type(pos).market_value = 5_000.0
        agent = _make_risk_agent(positions=[pos])
        signal = _make_buy_signal("BTC/USD")
        result = agent._check_position_concentration(signal)
        assert result.passed


class TestPositionSize:
    def test_blocks_no_capital(self):
        agent = _make_risk_agent()
        agent.portfolio.get_available_capital.return_value = 0.0
        signal = _make_buy_signal()
        result = agent._check_position_size(signal, "momentum")
        assert not result.passed

    def test_passes_with_capital(self):
        agent = _make_risk_agent()
        signal = _make_buy_signal()
        result = agent._check_position_size(signal, "momentum")
        assert result.passed


class TestResetCircuitBreaker:
    def test_resets_state(self):
        agent = _make_risk_agent()
        agent._circuit_breaker_active = True
        agent.reset_circuit_breaker()
        assert not agent._circuit_breaker_active

    def test_updates_high_water_mark(self):
        agent = _make_risk_agent(equity=95_000.0)
        agent._drawdown_high_water = 100_000.0
        agent.reset_circuit_breaker()
        assert agent._drawdown_high_water == 95_000.0


class TestHoldSignal:
    def test_hold_always_passes(self):
        agent = _make_risk_agent(daily_loss=5_000.0)
        agent._circuit_breaker_active = True
        hold = TradeSignal(
            symbol="BTC/USD", signal=Signal.HOLD,
            confidence=0.5, target_position_pct=0.0,
        )
        result = agent.evaluate_signal(hold, "momentum")
        assert result.passed


class TestRiskState:
    def test_get_risk_state_returns_dict(self):
        agent = _make_risk_agent()
        state = agent.get_risk_state()
        assert "circuit_breaker_active" in state
        assert "high_water_mark" in state
        assert "daily_loss" in state
        assert "daily_loss_limit" in state

    def test_increment_cycle(self):
        agent = _make_risk_agent()
        initial = agent._cycle_count
        agent.increment_cycle()
        assert agent._cycle_count == initial + 1
