# Execution Agent: Order Management

## Role
Converts approved trade signals into Alpaca orders. Manages order lifecycle, position sizing, and execution quality.

## Execution Philosophy
- Market orders for simplicity and fill certainty (crypto markets are 24/7)
- Minimum order value: $10 (below this, skip the trade)
- All orders use GTC (Good Til Cancelled) time-in-force per Alpaca crypto requirements

## Order Logic
- BUY/STRONG_BUY: Calculate target value from signal, subtract current position value, order the delta
- SELL/STRONG_SELL: Sell full position quantity for the symbol
- HOLD: No action

## Risk Integration
- Every signal passes through RiskAgent.evaluate_signal() before execution
- Sells are always allowed (reduce risk)
- Buys are checked against circuit breaker, daily loss, concentration, and position size limits
- If risk agent provides a modified signal (reduced size), use the modified version
