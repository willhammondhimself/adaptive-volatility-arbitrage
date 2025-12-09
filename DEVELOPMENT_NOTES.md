# Development Notes

## Heston FFT Bug Saga (October-November 2025)

Initial implementation based on online examples had severe pricing errors:
- ATM options: ~1-2% error (acceptable)
- ITM/OTM: 30-200% error (completely wrong!)

### Investigation Timeline
1. **Oct 20**: Noticed errors when testing K=80 vs reference prices
2. **Oct 24-27**: Tried different grid spacing - no improvement
3. **Nov 19**: Added implied vol solver to cross-check
4. **Nov 24**: Finally found it - THREE separate bugs:
   - Grid construction: b = λ/2 should be b = π/η
   - Damping factor: wrong sign (negative → positive)
   - Normalization: /2η should be /π

Validated against BrownianNotion implementation and verified errors <0.03%.

## Performance Notes
- MacBook Pro M1, 16GB RAM
- Full-year SPY backtest: ~200k option prices in ~9 minutes
- Heston FFT: ~180ms per 50x50 surface (cold), <5ms (cached)

## Key Dependencies
- yfinance: Sometimes rate-limits, added caching
- scikit-learn: GARCH fitting occasionally fails on low-vol periods
