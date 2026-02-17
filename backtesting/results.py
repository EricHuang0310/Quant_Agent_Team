from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_log: pd.DataFrame

    def summary(self) -> str:
        return (
            f"=== Backtest Results ===\n"
            f"Total Return:   {self.total_return:>10.2%}\n"
            f"Annual Return:  {self.annual_return:>10.2%}\n"
            f"Sharpe Ratio:   {self.sharpe_ratio:>10.2f}\n"
            f"Sortino Ratio:  {self.sortino_ratio:>10.2f}\n"
            f"Max Drawdown:   {self.max_drawdown:>10.2%}\n"
            f"Calmar Ratio:   {self.calmar_ratio:>10.2f}\n"
            f"Win Rate:       {self.win_rate:>10.2%}\n"
            f"Profit Factor:  {self.profit_factor:>10.2f}\n"
            f"Total Trades:   {self.total_trades:>10d}\n"
        )

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
        }


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Compute backtest performance metrics from an equity curve and trade log."""
    returns = equity_curve.pct_change().dropna()

    # Total and annualized return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_hours = len(equity_curve)
    n_years = n_hours / (365.25 * 24) if n_hours > 0 else 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Sharpe (annualized for hourly data)
    excess = returns - risk_free_rate / (365.25 * 24)
    sharpe = (excess.mean() / excess.std() * np.sqrt(365.25 * 24)) if excess.std() > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (returns.mean() / downside_std * np.sqrt(365.25 * 24)) if downside_std > 0 else 0.0

    # Drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_drawdown = abs(dd.min()) if len(dd) > 0 else 0.0

    # Calmar
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    # Trade-level metrics
    total_trades = len(trades) if trades is not None and len(trades) > 0 else 0
    if total_trades > 0 and "pnl" in trades.columns:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / total_trades
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        equity_curve=equity_curve,
        drawdown_curve=dd,
        trade_log=trades if trades is not None else pd.DataFrame(),
    )
