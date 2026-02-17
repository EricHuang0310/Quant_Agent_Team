import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from utils.logger import setup_logger

logger = setup_logger("core.market_data")

# Cache TTL in seconds per timeframe
_CACHE_TTL = {
    TimeFrame.Minute: 30,
    TimeFrame.Hour: 300,
    TimeFrame.Day: 3600,
}


class MarketDataService:
    """Fetches and caches OHLCV data from Alpaca crypto data API."""

    def __init__(self):
        # CryptoHistoricalDataClient requires no API keys
        self._client = CryptoHistoricalDataClient()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, float] = {}

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Hour,
        lookback_days: int = 30,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for a single symbol.

        Columns: open, high, low, close, volume, vwap, trade_count
        Index: timestamp (UTC)
        """
        if start is None:
            start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        if end is None:
            end = datetime.now(timezone.utc)

        cache_key = f"{symbol}|{timeframe}|{start.date()}|{end.date()}"
        ttl = _CACHE_TTL.get(timeframe, 60)
        if cache_key in self._cache:
            if time.time() - self._cache_ts[cache_key] < ttl:
                return self._cache[cache_key]

        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        bars = self._client.get_crypto_bars(req)
        df = bars.df

        # If multi-index (symbol, timestamp), drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        self._cache[cache_key] = df
        self._cache_ts[cache_key] = time.time()
        logger.debug(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df

    def get_multi_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame = TimeFrame.Hour,
        lookback_days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols. Returns dict of symbol -> DataFrame."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_bars(sym, timeframe, lookback_days)
            except Exception as e:
                logger.error(f"Failed to fetch bars for {sym}: {e}")
        return result

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        try:
            bar = self._client.get_crypto_latest_bar(
                CryptoBarsRequest(symbol_or_symbols=symbol)
            )
            if symbol in bar:
                b = bar[symbol]
                return {
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "timestamp": b.timestamp,
                }
        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Quick helper to get the latest close price."""
        bar = self.get_latest_bar(symbol)
        return bar["close"] if bar else None

    def clear_cache(self):
        self._cache.clear()
        self._cache_ts.clear()
