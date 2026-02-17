from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from core.market_data import MarketDataService
from core.portfolio import PortfolioManager
from utils.logger import setup_logger


class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TradeSignal:
    symbol: str
    signal: Signal
    confidence: float  # 0.0 – 1.0
    target_position_pct: float  # desired % of strategy's allocated capital
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class BaseStrategyAgent(ABC):
    """Abstract base class for all strategy agents."""

    def __init__(
        self,
        name: str,
        market_data: MarketDataService,
        portfolio: PortfolioManager,
        config: dict,
    ):
        self.name = name
        self.market_data = market_data
        self.portfolio = portfolio
        self.config = config
        self.logger = setup_logger(f"agent.{name}")
        self._is_active = True

    @abstractmethod
    def analyze(self, symbol: str) -> TradeSignal:
        """Analyze a single symbol and return a trading signal."""

    def analyze_universe(self, symbols: List[str]) -> List[TradeSignal]:
        """Analyze all symbols. Calls analyze() per symbol with error handling."""
        signals = []
        for sym in symbols:
            try:
                sig = self.analyze(sym)
                signals.append(sig)
            except Exception as e:
                self.logger.error(f"Error analyzing {sym}: {e}")
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        signal=Signal.HOLD,
                        confidence=0.0,
                        target_position_pct=0.0,
                        metadata={"error": str(e)},
                    )
                )
        return signals

    @abstractmethod
    def get_strategy_state(self) -> dict:
        """Return current strategy state for meta-agent inspection."""

    def activate(self):
        self._is_active = True
        self.logger.info(f"{self.name} activated")

    def deactivate(self):
        self._is_active = False
        self.logger.info(f"{self.name} deactivated")

    @property
    def is_active(self) -> bool:
        return self._is_active
