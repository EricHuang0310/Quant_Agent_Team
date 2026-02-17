# Momentum Agent: Trend-Following Strategy

## Role
Trend-following strategy that captures directional moves in crypto markets using MACD, RSI, EMA trend filter, and ADX confirmation.

## Market Conditions Where It Excels
- Strong directional trends (ADX > 25, ideally > 40)
- Sustained breakouts from consolidation ranges
- Markets with clear EMA alignment (price > EMA20 > EMA50 for longs)

## Weaknesses
- Whipsaws in ranging/choppy markets (ADX < 20)
- Late entries near trend exhaustion (RSI approaching overbought during entry)
- Sudden reversals after extended trends

## Entry Logic
- BUY: ADX > 25 AND price > EMA(50) AND MACD histogram > 0 AND RSI < 70
- SELL: ADX > 25 AND price < EMA(50) AND MACD histogram < 0 AND RSI > 30
- STRONG_BUY/SELL: Same conditions but ADX > 40

## Risk Management
- Stop-loss: 2x ATR from entry price (dynamic)
- Position sizing: confidence * 0.8 of strategy allocation

## Parameters
- MACD: fast=12, slow=26, signal=9
- RSI: period=14, overbought=70, oversold=30
- Trend EMA: period=50
- ADX: period=14
- ATR: period=14
