from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame

from agents.base_agent import BaseStrategyAgent, Signal
from backtesting.results import BacktestResult, compute_metrics
from core.indicators import Indicators
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager
from utils.logger import setup_logger

logger = setup_logger("backtesting")

CRYPTO_FEE = 0.001  # 0.1% per trade (Alpaca crypto fee)
SLIPPAGE = 0.0005   # 0.05% slippage model


class BacktestMarketDataService(MarketDataService):
    """A MarketDataService subclass that returns windowed data for backtesting.

    Instead of monkey-patching get_bars, this cleanly overrides it to return
    only the data visible at the current backtest timestamp.
    """

    def __init__(self, real_service: MarketDataService):
        # Skip parent __init__ (avoids creating a second CryptoHistoricalDataClient)
        self._real_service = real_service
        self._windows: Dict[str, pd.DataFrame] = {}

    def set_window(self, symbol: str, window: pd.DataFrame):
        self._windows[symbol] = window

    def clear_windows(self):
        self._windows.clear()

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Hour,
        lookback_days: int = 30,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if symbol in self._windows:
            return self._windows[symbol]
        return self._real_service.get_bars(symbol, timeframe, lookback_days, start=start, end=end)

    def get_current_price(self, symbol: str) -> Optional[float]:
        if symbol in self._windows and len(self._windows[symbol]) > 0:
            return float(self._windows[symbol].iloc[-1]["close"])
        return self._real_service.get_current_price(symbol)

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        return self._real_service.get_latest_bar(symbol)

    def get_multi_bars(self, symbols, timeframe=TimeFrame.Hour, lookback_days=30):
        return self._real_service.get_multi_bars(symbols, timeframe, lookback_days)

    def clear_cache(self):
        self._real_service.clear_cache()


class BacktestEngine:
    """Walk-forward backtesting engine for strategy agents.

    No lookahead bias: at time T, only data up to T is available.
    """

    def __init__(self, market_data: MarketDataService):
        self.market_data = market_data

    def backtest_strategy(
        self,
        strategy_agent: BaseStrategyAgent,
        symbols: List[str],
        start: datetime,
        end: datetime,
        initial_capital: float = 100_000.0,
        warmup_bars: int = 60,
        fee: float = CRYPTO_FEE,
        slippage: float = SLIPPAGE,
    ) -> BacktestResult:
        """Run a single strategy backtest across multiple symbols.

        Walk-forward approach: for each bar, the agent sees only data up to that bar.
        """
        # Fetch all data upfront
        all_bars: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.market_data.get_bars(sym, TimeFrame.Hour, start=start, end=end)
                if len(df) > warmup_bars:
                    all_bars[sym] = df
                else:
                    logger.warning(f"Skipping {sym}: only {len(df)} bars (need {warmup_bars}+)")
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")

        if not all_bars:
            logger.error("No valid data for any symbol")
            return compute_metrics(
                pd.Series([initial_capital]),
                pd.DataFrame(),
            )

        # Use the longest common index
        ref_sym = max(all_bars, key=lambda s: len(all_bars[s]))
        ref_index = all_bars[ref_sym].index

        # Create a BacktestMarketDataService and inject it into the agent
        bt_market_data = BacktestMarketDataService(self.market_data)
        original_market_data = strategy_agent.market_data
        strategy_agent.market_data = bt_market_data

        # Simulate
        cash = initial_capital
        positions: Dict[str, float] = {}  # symbol -> qty
        entry_prices: Dict[str, float] = {}  # symbol -> avg entry price
        equity_history = []
        trades = []

        try:
            for i in range(warmup_bars, len(ref_index)):
                ts = ref_index[i]

                # Get current prices
                prices = {}
                for sym, df in all_bars.items():
                    if ts in df.index:
                        prices[sym] = df.loc[ts, "close"]
                    elif i < len(df):
                        prices[sym] = df.iloc[min(i, len(df) - 1)]["close"]

                # Generate signals using only data up to current bar
                for sym, df in all_bars.items():
                    if sym not in prices:
                        continue

                    window = df.iloc[: i + 1] if ts in df.index else df.iloc[:i]
                    if len(window) < warmup_bars:
                        continue

                    try:
                        bt_market_data.set_window(sym, window)
                        signal = strategy_agent.analyze(sym)
                    except Exception:
                        continue

                    price = prices[sym]
                    current_qty = positions.get(sym, 0.0)
                    current_value = current_qty * price

                    # Target capital for this strategy on this symbol
                    target_pct = signal.target_position_pct
                    per_symbol_capital = initial_capital / len(all_bars)
                    target_value = per_symbol_capital * target_pct

                    if signal.signal in (Signal.BUY, Signal.STRONG_BUY) and target_value > current_value:
                        buy_value = min(target_value - current_value, cash)
                        if buy_value > 10:
                            cost = buy_value * (1 + fee + slippage)
                            qty = buy_value / price
                            cash -= cost
                            old_qty = positions.get(sym, 0.0)
                            old_entry = entry_prices.get(sym, price)
                            new_qty = old_qty + qty
                            if new_qty > 0:
                                entry_prices[sym] = (old_entry * old_qty + price * qty) / new_qty
                            positions[sym] = new_qty
                            trades.append({
                                "timestamp": ts, "symbol": sym, "side": "buy",
                                "qty": qty, "price": price, "value": buy_value,
                                "fee": buy_value * fee,
                            })

                    elif signal.signal in (Signal.SELL, Signal.STRONG_SELL) and current_qty > 0:
                        sell_value = current_qty * price * (1 - fee - slippage)
                        entry_cost = current_qty * entry_prices.get(sym, price)
                        pnl = sell_value - entry_cost
                        cash += sell_value
                        trades.append({
                            "timestamp": ts, "symbol": sym, "side": "sell",
                            "qty": current_qty, "price": price, "value": sell_value,
                            "fee": sell_value * fee, "pnl": pnl,
                        })
                        positions[sym] = 0.0
                        entry_prices.pop(sym, None)

                    # Stop-loss check: use tracked entry prices
                    if current_qty > 0 and signal.stop_loss_pct and sym in entry_prices:
                        unrealized_loss = (entry_prices[sym] - price) / entry_prices[sym]
                        if unrealized_loss >= signal.stop_loss_pct:
                            sell_value = current_qty * price * (1 - fee - slippage)
                            entry_cost = current_qty * entry_prices[sym]
                            pnl = sell_value - entry_cost
                            cash += sell_value
                            trades.append({
                                "timestamp": ts, "symbol": sym, "side": "sell",
                                "qty": current_qty, "price": price, "value": sell_value,
                                "fee": sell_value * fee, "pnl": pnl,
                                "reason": "stop_loss",
                            })
                            positions[sym] = 0.0
                            entry_prices.pop(sym, None)

                # Record equity
                pos_value = sum(
                    positions.get(sym, 0) * prices.get(sym, 0) for sym in all_bars
                )
                equity_history.append({"timestamp": ts, "equity": cash + pos_value})
        finally:
            # Always restore original market_data
            strategy_agent.market_data = original_market_data
            bt_market_data.clear_windows()

        # Build results
        equity_df = pd.DataFrame(equity_history)
        if len(equity_df) > 0:
            equity_series = equity_df.set_index("timestamp")["equity"]
        else:
            equity_series = pd.Series([initial_capital])

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        result = compute_metrics(equity_series, trades_df)
        logger.info(
            f"Backtest complete: {strategy_agent.name} | "
            f"Return={result.total_return:.2%} | "
            f"Sharpe={result.sharpe_ratio:.2f} | "
            f"MaxDD={result.max_drawdown:.2%} | "
            f"Trades={result.total_trades}"
        )
        return result

    def backtest_blended(
        self,
        agents: Dict[str, BaseStrategyAgent],
        allocations: Dict[str, float],
        symbols: List[str],
        start: datetime,
        end: datetime,
        initial_capital: float = 100_000.0,
    ) -> Dict[str, BacktestResult]:
        """Backtest multiple strategies with fixed allocations.

        Returns per-strategy results.
        """
        results = {}
        for name, agent in agents.items():
            alloc_capital = initial_capital * allocations.get(name, 0.0)
            if alloc_capital <= 0:
                continue
            result = self.backtest_strategy(
                agent, symbols, start, end, initial_capital=alloc_capital
            )
            results[name] = result
            logger.info(f"  {name}: return={result.total_return:.2%}, sharpe={result.sharpe_ratio:.2f}")

        return results
