# Mean Reversion Agent: Bollinger Band + Z-Score Strategy

## Role
Mean-reversion strategy that profits from price returning to the statistical mean after extreme deviations, using Bollinger Bands, Z-Score, and RSI.

## Market Conditions Where It Excels
- High-volatility ranging markets where price oscillates around the mean
- After sharp sell-offs or rallies that push Z-score beyond 2.0
- Markets with established support/resistance ranges

## Weaknesses
- Strong trending markets (will buy dips in a downtrend or sell rallies in an uptrend)
- Regime changes from ranging to trending
- Low-volatility environments with tight Bollinger Bands (no opportunity)

## Entry Logic
- BUY: Z-score < -2.0 AND RSI < 25 (oversold)
- SELL: Z-score > +2.0 AND RSI > 75 (overbought)
- STRONG_BUY: Z-score < -3.0
- EXIT: Z-score returns within 0.5 of zero

## Risk Management
- Fixed stop-loss: 3%
- Position sizing: confidence * 0.6 of strategy allocation (0.7 for strong signals)

## Parameters
- Bollinger Bands: period=20, std=2.0
- RSI: period=14, overbought=75, oversold=25
- Z-Score: entry=2.0, exit=0.5
