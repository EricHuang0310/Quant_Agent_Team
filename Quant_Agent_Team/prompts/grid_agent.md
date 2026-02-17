# Grid Agent: Range-Trading Strategy

## Role
Grid trading strategy that profits from price oscillation within a defined range by placing layered buy/sell orders at predetermined levels.

## Market Conditions Where It Excels
- Tight ranging/sideways markets (ADX < 20)
- Consolidation phases after large moves
- Markets with well-defined support and resistance

## Weaknesses
- Trending markets (gets run over in one direction)
- Breakouts from the range (unfilled levels become obsolete)
- High-spread or low-liquidity conditions that erode grid profits

## Entry Logic
- Only activates when ADX < 20 (ranging market confirmed)
- Places grid of buy orders below mid-price and sell orders above
- Signals BUY when price approaches a buy grid level (within 1%)
- Signals SELL when price approaches a sell grid level (within 1%)
- Grid resets when market regime changes to trending

## Risk Management
- Fixed stop-loss: 2% per grid trade
- Small position per grid level: 2% of strategy allocation
- Grid resets when ADX crosses above threshold

## Parameters
- Grid levels: 10 (5 buy + 5 sell around mid)
- Grid spacing: 0.5% between levels
- ADX threshold: 20 (deactivate above this)
- Range lookback: 7 days
