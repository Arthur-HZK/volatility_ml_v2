# Research notes

## Key design choices

### Real options rather than fabricated historical chains

The project uses current listed option-chain observations where the data source exposes them. Historical ML work uses underlying prices and VIX because those series are freely accessible and reproducible.

### Walk-forward validation

The model is repeatedly trained on past observations and evaluated on a later block. This better reflects the chronology of a trading research problem than random cross-validation.

### Probabilistic signal

The Random Forest's class probabilities are retained. A threshold on `P(high volatility)` controls the defensive research signal.

### What should be improved next?

- Add a proper economic benchmark.
- Add transaction costs and turnover.
- Use purged/embargoed validation if labels overlap.
- Calibrate probabilities.
- Add multiple assets.
- Use a licensed historical option-chain dataset for genuine historical IV-surface modelling.
- Test whether option-implied features add information beyond VIX and realized volatility.
