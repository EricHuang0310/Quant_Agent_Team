import pytest
from unittest.mock import MagicMock

from core.portfolio import PortfolioManager, PositionInfo, StrategyAllocation


def _make_portfolio(equity=100_000.0, cash=50_000.0) -> PortfolioManager:
    broker = MagicMock()
    broker.get_account.return_value = {
        "equity": equity, "cash": cash,
        "buying_power": cash, "portfolio_value": equity,
    }
    broker.get_all_positions.return_value = []
    return PortfolioManager(broker, equity)


class TestPositionStrategyMap:
    def test_multiple_strategies_same_symbol(self):
        pm = _make_portfolio()
        pm.assign_position_to_strategy("BTC/USD", "momentum")
        pm.assign_position_to_strategy("BTC/USD", "mean_reversion")
        assert "momentum" in pm._position_strategy_map["BTC/USD"]
        assert "mean_reversion" in pm._position_strategy_map["BTC/USD"]

    def test_remove_position(self):
        pm = _make_portfolio()
        pm.assign_position_to_strategy("BTC/USD", "momentum")
        pm.assign_position_to_strategy("BTC/USD", "mean_reversion")
        pm.remove_position_from_strategy("BTC/USD", "momentum")
        assert "momentum" not in pm._position_strategy_map["BTC/USD"]
        assert "mean_reversion" in pm._position_strategy_map["BTC/USD"]

    def test_remove_last_strategy_cleans_up(self):
        pm = _make_portfolio()
        pm.assign_position_to_strategy("BTC/USD", "momentum")
        pm.remove_position_from_strategy("BTC/USD", "momentum")
        assert "BTC/USD" not in pm._position_strategy_map

    def test_remove_nonexistent_is_safe(self):
        pm = _make_portfolio()
        pm.remove_position_from_strategy("BTC/USD", "momentum")  # no-op


class TestRecordTrade:
    def test_buy_assigns_strategy(self):
        pm = _make_portfolio()
        pm.record_trade({
            "symbol": "BTC/USD", "strategy": "momentum",
            "side": "buy", "qty": 0.1, "price": 50000,
        })
        assert "momentum" in pm._position_strategy_map.get("BTC/USD", set())

    def test_sell_removes_strategy(self):
        pm = _make_portfolio()
        pm.assign_position_to_strategy("BTC/USD", "momentum")
        pm.record_trade({
            "symbol": "BTC/USD", "strategy": "momentum",
            "side": "sell", "qty": 0.1, "price": 50000,
        })
        assert "BTC/USD" not in pm._position_strategy_map

    def test_trade_log_appended(self):
        pm = _make_portfolio()
        pm.record_trade({
            "symbol": "ETH/USD", "strategy": "grid",
            "side": "buy", "qty": 1.0, "price": 3000,
        })
        assert len(pm.get_trade_log()) == 1
        assert pm.get_trade_log()[0]["symbol"] == "ETH/USD"
        assert "timestamp" in pm.get_trade_log()[0]


class TestAllocations:
    def test_set_and_get_allocation(self):
        pm = _make_portfolio()
        pm.set_target_allocation("momentum", 0.4)
        alloc = pm.get_strategy_allocation("momentum")
        assert alloc.target_pct == 0.4
        assert alloc.capital_allocated == 40_000.0

    def test_available_capital_no_positions(self):
        pm = _make_portfolio()
        pm.set_target_allocation("momentum", 0.4)
        available = pm.get_available_capital("momentum")
        assert available == 40_000.0

    def test_get_position_not_found(self):
        pm = _make_portfolio()
        assert pm.get_position("BTC/USD", "momentum") is None


class TestSyncFromBroker:
    def test_split_position_across_strategies(self):
        broker = MagicMock()
        broker.get_account.return_value = {
            "equity": 100_000.0, "cash": 50_000.0,
            "buying_power": 50_000.0, "portfolio_value": 100_000.0,
        }
        pos = MagicMock()
        pos.symbol = "BTCUSD"
        pos.qty = "1.0"
        pos.avg_entry_price = "50000.0"
        pos.current_price = "55000.0"
        pos.market_value = "55000.0"
        pos.unrealized_pl = "5000.0"
        pos.unrealized_plpc = "0.1"
        # Make float() work on these
        type(pos).qty = property(lambda s: "1.0")
        type(pos).avg_entry_price = property(lambda s: "50000.0")
        type(pos).current_price = property(lambda s: "55000.0")
        type(pos).market_value = property(lambda s: "55000.0")
        type(pos).unrealized_pl = property(lambda s: "5000.0")
        type(pos).unrealized_plpc = property(lambda s: "0.1")

        broker.get_all_positions.return_value = [pos]

        pm = PortfolioManager(broker, 100_000.0)
        pm.set_target_allocation("momentum", 0.5)
        pm.set_target_allocation("mean_reversion", 0.5)
        pm.assign_position_to_strategy("BTCUSD", "momentum")
        pm.assign_position_to_strategy("BTCUSD", "mean_reversion")

        pm.sync_from_broker()

        mom_alloc = pm.get_strategy_allocation("momentum")
        mr_alloc = pm.get_strategy_allocation("mean_reversion")

        # Position should be split 50/50
        assert len(mom_alloc.positions) == 1
        assert len(mr_alloc.positions) == 1
        assert mom_alloc.positions[0].market_value == 27_500.0
        assert mr_alloc.positions[0].market_value == 27_500.0
