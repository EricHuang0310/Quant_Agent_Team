from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from core.broker import AlpacaBroker
from utils.logger import setup_logger

logger = setup_logger("core.portfolio")


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    strategy: str


@dataclass
class StrategyAllocation:
    strategy_name: str
    target_pct: float = 0.0
    current_pct: float = 0.0
    capital_allocated: float = 0.0
    positions: List[PositionInfo] = field(default_factory=list)


class PortfolioManager:
    """Central portfolio state, synced from Alpaca account.

    Tracks per-strategy allocations and positions for capital isolation.
    Uses (symbol, strategy) composite tracking to allow multiple strategies
    to hold positions in the same symbol simultaneously.
    """

    def __init__(self, broker: AlpacaBroker, total_capital: float = 100_000.0):
        self._broker = broker
        self._total_capital = total_capital
        self._allocations: Dict[str, StrategyAllocation] = {}
        # symbol -> set of strategies that own positions in it
        self._position_strategy_map: Dict[str, Set[str]] = {}
        self._trade_log: List[dict] = []
        self._equity_history: List[dict] = []

    def sync_from_broker(self) -> dict:
        """Pull latest positions and account data from Alpaca."""
        account = self._broker.get_account()
        self._total_capital = account["equity"]

        positions = self._broker.get_all_positions()

        # Build per-strategy position infos.
        # Alpaca only tracks one aggregate position per symbol, so we
        # split the position proportionally across strategies that own it.
        strategy_positions: Dict[str, List[PositionInfo]] = {
            name: [] for name in self._allocations
        }

        for p in positions:
            symbol = p.symbol
            strategies = self._position_strategy_map.get(symbol, set())
            if not strategies:
                strategies = {"unassigned"}

            # Split position equally across owning strategies
            n = len(strategies)
            for strat in strategies:
                info = PositionInfo(
                    symbol=symbol,
                    qty=float(p.qty) / n,
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value) / n,
                    unrealized_pnl=float(p.unrealized_pl) / n,
                    unrealized_pnl_pct=float(p.unrealized_plpc),
                    strategy=strat,
                )
                if strat in strategy_positions:
                    strategy_positions[strat].append(info)

        # Update allocations with actual positions
        for name, alloc in self._allocations.items():
            alloc.positions = strategy_positions.get(name, [])
            total_mv = sum(p.market_value for p in alloc.positions)
            alloc.current_pct = total_mv / self._total_capital if self._total_capital > 0 else 0.0
            alloc.capital_allocated = self._total_capital * alloc.target_pct

        self._equity_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": account["equity"],
        })

        return account

    def set_target_allocation(self, strategy: str, pct: float):
        if strategy not in self._allocations:
            self._allocations[strategy] = StrategyAllocation(strategy_name=strategy)
        self._allocations[strategy].target_pct = pct
        self._allocations[strategy].capital_allocated = self._total_capital * pct
        logger.info(f"Allocation set: {strategy} = {pct:.1%}")

    def get_strategy_allocation(self, strategy: str) -> StrategyAllocation:
        if strategy not in self._allocations:
            self._allocations[strategy] = StrategyAllocation(strategy_name=strategy)
        return self._allocations[strategy]

    def get_available_capital(self, strategy: str) -> float:
        alloc = self.get_strategy_allocation(strategy)
        used = sum(p.market_value for p in alloc.positions)
        return max(0.0, alloc.capital_allocated - used)

    def assign_position_to_strategy(self, symbol: str, strategy: str):
        if symbol not in self._position_strategy_map:
            self._position_strategy_map[symbol] = set()
        self._position_strategy_map[symbol].add(strategy)

    def remove_position_from_strategy(self, symbol: str, strategy: str):
        if symbol in self._position_strategy_map:
            self._position_strategy_map[symbol].discard(strategy)
            if not self._position_strategy_map[symbol]:
                del self._position_strategy_map[symbol]

    def get_position(self, symbol: str, strategy: Optional[str] = None) -> Optional[PositionInfo]:
        if strategy:
            alloc = self.get_strategy_allocation(strategy)
            for p in alloc.positions:
                if p.symbol == symbol:
                    return p
            return None
        return self._broker.get_position(symbol)

    def record_trade(self, trade: dict):
        trade["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._trade_log.append(trade)
        # Track which strategy owns this symbol
        if "symbol" in trade and "strategy" in trade:
            side = trade.get("side", "")
            if side == "sell":
                self.remove_position_from_strategy(trade["symbol"], trade["strategy"])
            else:
                self.assign_position_to_strategy(trade["symbol"], trade["strategy"])

    def get_performance_summary(self) -> dict:
        account = self._broker.get_account()
        total_pnl = account["equity"] - 100_000.0  # vs initial capital
        per_strategy = {}
        for name, alloc in self._allocations.items():
            strat_pnl = sum(p.unrealized_pnl for p in alloc.positions)
            per_strategy[name] = {
                "target_pct": alloc.target_pct,
                "current_pct": alloc.current_pct,
                "unrealized_pnl": strat_pnl,
                "position_count": len(alloc.positions),
            }
        return {
            "equity": account["equity"],
            "cash": account["cash"],
            "total_pnl": total_pnl,
            "strategies": per_strategy,
            "total_trades": len(self._trade_log),
        }

    def get_trade_log(self) -> List[dict]:
        return self._trade_log

    @property
    def total_capital(self) -> float:
        return self._total_capital
