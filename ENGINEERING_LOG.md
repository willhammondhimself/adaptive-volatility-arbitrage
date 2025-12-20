# Engineering Log

## Dec 19, 2024: Phase 2 - Uncertainty-Aware Position Sizing

Implemented Bayesian LSTM for volatility forecasting with Monte Carlo Dropout.

**New modules:**

- `models/bayesian_lstm.py`: LSTM with MC Dropout for epistemic uncertainty estimation
- `execution/costs.py`: Square-root market impact model (Almgren & Chriss)
- `risk/uncertainty_sizing.py`: Kelly sizing scaled by model confidence

**Key findings:**

- Epistemic uncertainty reveals model confidence, particularly elevated in high-vol regimes
- Square-root impact model penalizes large orders proportionally to sqrt(participation rate)
- Position sizing now scales inversely with model uncertainty via modified Kelly criterion

**Demo:** `examples/demo_uncertainty_sizing.py` generates `docs/bayesian_uncertainty_plot.png`
