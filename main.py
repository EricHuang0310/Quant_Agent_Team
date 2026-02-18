"""
Quant Agent Team — Multi-Agent Crypto Trading System

Usage:
    python main.py              # Run live paper trading loop
    python main.py --backtest   # Run backtests on historical data
"""

import argparse
import signal as sig
import sys
import time
from datetime import datetime, timedelta, timezone

import yaml

from agents.base_agent import Signal
from agents.execution_agent import ExecutionAgent
from agents.grid_agent import GridAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.meta_agent import MetaAgent
from agents.momentum_agent import MomentumAgent
from agents.risk_agent import RiskAgent
from backtesting.backtest_engine import BacktestEngine
from config.settings import settings
from core.broker import AlpacaBroker
from core.market_data import MarketDataService
from core.portfolio import PortfolioManager
from utils.logger import setup_logger
from utils.notifier import Notifier

logger = setup_logger("main")

# Circuit breaker auto-recovery after N cycles of drawdown being below threshold
CIRCUIT_BREAKER_COOLDOWN_CYCLES = 30


def load_strategy_configs() -> dict:
    with open("config/strategies.yaml") as f:
        return yaml.safe_load(f)


def build_system() -> dict:
    """Wire up all components."""
    broker = AlpacaBroker(settings.alpaca)
    market_data = MarketDataService()
    portfolio = PortfolioManager(broker, settings.trading.total_capital)

    strat_configs = load_strategy_configs()

    strategies = {
        "momentum": MomentumAgent(market_data, portfolio, strat_configs["momentum"]),
        "mean_reversion": MeanReversionAgent(market_data, portfolio, strat_configs["mean_reversion"]),
        "grid": GridAgent(market_data, portfolio, strat_configs["grid"]),
    }

    risk_agent = RiskAgent(
        portfolio, broker, settings.trading,
        anthropic_api_key=settings.meta_agent.anthropic_api_key,
        llm_model=settings.meta_agent.model,
    )
    exec_agent = ExecutionAgent(broker, portfolio, risk_agent, market_data)
    meta_agent = MetaAgent(
        settings.meta_agent, strategies, risk_agent, portfolio, market_data
    )
    notifier = Notifier()

    return {
        "broker": broker,
        "market_data": market_data,
        "portfolio": portfolio,
        "strategies": strategies,
        "risk_agent": risk_agent,
        "execution_agent": exec_agent,
        "meta_agent": meta_agent,
        "notifier": notifier,
    }


def run_trading_loop(system: dict):
    """Main trading loop for live/paper trading."""
    broker = system["broker"]
    portfolio = system["portfolio"]
    strategies = system["strategies"]
    exec_agent = system["execution_agent"]
    meta_agent = system["meta_agent"]
    risk_agent = system["risk_agent"]
    notifier = system["notifier"]
    grid_agent = strategies.get("grid")

    # Graceful shutdown
    running = True

    def shutdown(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received, cleaning up...")
        running = False

    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    # Initial sync
    account = portfolio.sync_from_broker()
    logger.info(
        f"System ready. Account equity: ${account['equity']:,.2f}, "
        f"cash: ${account['cash']:,.2f}"
    )
    logger.info(f"Trading symbols: {settings.trading.symbols}")

    cycle_count = 0
    rebalance_interval = settings.meta_agent.rebalance_interval_minutes
    last_daily_reset_date = datetime.now(timezone.utc).date()
    circuit_breaker_triggered_cycle = None

    while running:
        try:
            cycle_count += 1
            risk_agent.increment_cycle()
            logger.info(f"{'='*60}")
            logger.info(f"Cycle {cycle_count}")

            # 0. Daily loss counter reset on date change
            today = datetime.now(timezone.utc).date()
            if today != last_daily_reset_date:
                risk_agent.reset_daily_counters()
                last_daily_reset_date = today
                logger.info("New day detected — daily risk counters reset")

            # 1. Sync portfolio state from broker
            portfolio.sync_from_broker()

            # 1b. Circuit breaker auto-recovery check
            risk_state = risk_agent.get_risk_state()
            if risk_state["circuit_breaker_active"]:
                if circuit_breaker_triggered_cycle is None:
                    circuit_breaker_triggered_cycle = cycle_count

                cycles_since = cycle_count - circuit_breaker_triggered_cycle
                if cycles_since >= CIRCUIT_BREAKER_COOLDOWN_CYCLES:
                    # Check if drawdown has recovered below threshold
                    account = broker.get_account()
                    equity = account["equity"]
                    hwm = risk_state["high_water_mark"]
                    current_dd = (hwm - equity) / hwm if hwm > 0 else 0
                    if current_dd < risk_state["max_drawdown_pct"] * 0.5:
                        risk_agent.reset_circuit_breaker()
                        circuit_breaker_triggered_cycle = None
                        logger.info(
                            f"Circuit breaker auto-recovered after {cycles_since} cycles "
                            f"(drawdown={current_dd:.2%})"
                        )
                        notifier.notify(
                            f"Circuit breaker auto-recovered (drawdown={current_dd:.2%})"
                        )
            else:
                circuit_breaker_triggered_cycle = None

            # 2. Meta-agent rebalance (every N cycles)
            if cycle_count == 1 or cycle_count % rebalance_interval == 0:
                logger.info("Running meta-agent allocation...")
                decision = meta_agent.decide_allocation()
                logger.info(
                    f"  Regime: {decision.regime.value} | "
                    f"Confidence: {decision.confidence:.2f}"
                )
                logger.info(f"  Allocations: {decision.allocations}")
                logger.info(f"  Reasoning: {decision.reasoning}")

                for strat_name, pct in decision.allocations.items():
                    portfolio.set_target_allocation(strat_name, pct)
                    if pct > 0:
                        strategies[strat_name].activate()
                    else:
                        strategies[strat_name].deactivate()

            # 3. Collect signals from active strategy agents
            all_signals = []
            for name, agent in strategies.items():
                alloc = portfolio.get_strategy_allocation(name)
                if alloc.target_pct <= 0 or not agent.is_active:
                    continue

                signals = agent.analyze_universe(settings.trading.symbols)
                actionable = [
                    (name, s) for s in signals if s.signal != Signal.HOLD
                ]
                if actionable:
                    logger.info(
                        f"  {name}: {len(actionable)} actionable signals "
                        f"(of {len(signals)} total)"
                    )
                    for _, s in actionable:
                        logger.info(
                            f"    {s.symbol}: {s.signal.name} "
                            f"conf={s.confidence:.2f} "
                            f"target={s.target_position_pct:.1%}"
                        )
                all_signals.extend(actionable)

            # 4. Execute signals
            if all_signals:
                results = exec_agent.execute_signals(all_signals)
                filled = [r for r in results if r["status"] == "filled"]
                blocked = [r for r in results if r["status"] == "blocked"]
                if filled:
                    logger.info(f"  Executed: {len(filled)} orders")
                    for f_result in filled:
                        notifier.notify(
                            f"Order filled: {f_result['side']} {f_result['qty']:.6f} "
                            f"{f_result['symbol']} (${f_result['value']:,.2f}) "
                            f"[{f_result['strategy']}]"
                        )
                if blocked:
                    logger.info(f"  Blocked: {len(blocked)} orders")
            else:
                logger.info("  No actionable signals this cycle")

            # 5. Place stop-loss orders for new positions
            stop_results = exec_agent.check_and_place_stop_losses()
            if stop_results:
                logger.info(f"  Stop-losses placed: {len(stop_results)}")

            # 6. Execute grid limit orders
            if grid_agent and grid_agent.is_active:
                grid_results = exec_agent.execute_grid_orders(
                    grid_agent, settings.trading.symbols
                )
                if grid_results:
                    placed = [r for r in grid_results if r["status"] == "placed"]
                    if placed:
                        logger.info(f"  Grid orders placed: {len(placed)}")

            # 7. Log performance summary
            perf = portfolio.get_performance_summary()
            logger.info(
                f"  Portfolio: equity=${perf['equity']:,.2f} | "
                f"PnL=${perf['total_pnl']:,.2f} | "
                f"trades={perf['total_trades']}"
            )

            # Sleep until next cycle
            logger.info(
                f"  Sleeping {settings.trading.data_interval_seconds}s..."
            )
            time.sleep(settings.trading.data_interval_seconds)

        except KeyboardInterrupt:
            shutdown(None, None)
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
            time.sleep(30)

    # Cleanup
    logger.info("Cancelling all open orders...")
    try:
        broker.cancel_all_orders()
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
    logger.info("System shutdown complete.")


def run_backtest():
    """Run backtests on historical data."""
    market_data = MarketDataService()
    portfolio = PortfolioManager.__new__(PortfolioManager)
    portfolio._broker = None
    portfolio._total_capital = 100_000.0
    portfolio._allocations = {}
    portfolio._position_strategy_map = {}
    portfolio._trade_log = []
    portfolio._equity_history = []

    strat_configs = load_strategy_configs()
    strategies = {
        "momentum": MomentumAgent(market_data, portfolio, strat_configs["momentum"]),
        "mean_reversion": MeanReversionAgent(market_data, portfolio, strat_configs["mean_reversion"]),
        "grid": GridAgent(market_data, portfolio, strat_configs["grid"]),
    }

    engine = BacktestEngine(market_data)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=90)

    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

    logger.info(f"Running backtests from {start.date()} to {end.date()}")
    logger.info(f"Symbols: {symbols}")
    logger.info("")

    for name, agent in strategies.items():
        logger.info(f"--- Backtesting: {name} ---")
        result = engine.backtest_strategy(
            agent, symbols, start, end, initial_capital=100_000.0
        )
        print(result.summary())

    # Blended backtest
    logger.info("--- Backtesting: blended (equal weight) ---")
    blended = engine.backtest_blended(
        strategies,
        {"momentum": 0.4, "mean_reversion": 0.4, "grid": 0.2},
        symbols, start, end,
    )
    for name, result in blended.items():
        print(f"\n[{name}]")
        print(result.summary())


def main():
    parser = argparse.ArgumentParser(description="Quant Agent Team")
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtests instead of live trading"
    )
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        system = build_system()
        run_trading_loop(system)


if __name__ == "__main__":
    main()
