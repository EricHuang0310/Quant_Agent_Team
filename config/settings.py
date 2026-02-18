import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env from project root or config/ directory
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(_project_root / "config" / ".env")


@dataclass
class AlpacaConfig:
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "") or os.getenv("ALPACA_API_SECRET", "")
    paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
        "LINK/USD", "AVAX/USD", "DOT/USD", "UNI/USD",
    ])
    data_interval_seconds: int = 60
    lookback_days: int = 30
    max_position_pct: float = 0.15
    max_drawdown_pct: float = 0.10
    daily_loss_limit_pct: float = 0.03
    total_capital: float = 100_000.0


@dataclass
class MetaAgentConfig:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-5-20250929"
    rebalance_interval_minutes: int = 15
    min_allocation_pct: float = 0.10
    max_allocation_pct: float = 0.60


@dataclass
class Settings:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    meta_agent: MetaAgentConfig = field(default_factory=MetaAgentConfig)


settings = Settings()
