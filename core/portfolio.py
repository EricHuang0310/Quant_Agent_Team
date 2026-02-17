from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
    """

    def __init__(self, broker: AlpacaBroker, total_capital: float = 100_000.0):
        self._broker = broker
        self._total_capital = total_capital
        self._allocations: Dict[str, StrategyAllocation] = {}
        self._position_strategy_map: Dict[str, str] = {}  # symbol -> strategy
        self._trade_log: List[dict] = []
        self._equity_history: List[dict] = []

    def sync_from_broker(self) -> dict:
        """Pull latest positions and account data from Alpaca."""
        account = self._broker.get_account()
        self._total_capital = account["equity"]

        positions = self._broker.get_all_positions()
        all_position_infos: Dict[str, PositionInfo] = {}
        for p in positions:
            symbol = p.symbol
            strategy = self._position_strategy_map.get(symbol, "unassigned")
            info = PositionInfo(
                symbol=symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc),
                strategy=strategy,
            )
            all_position_infos[symbol] = info

        # Update allocations with actual positions
        for name, alloc in self._allocations.items():
            alloc.positions = [
                p for p in all_position_infos.values() if p.strategy == name
            ]
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
        self._position_strategy_map[symbol] = strategy

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
