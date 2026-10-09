# CrashLab — Mathematical Market Risk Engine

Runnable Python engine and interactive Streamlit interface. All displayed forecasts are computed from input data and fitted or explicitly user-defined parameters. No pretrained AI forecasts, invented crash scores, or demo prices are used.

## Student view (default)

The default interface is designed for ISC Class 12 students: choose a market, load real prices, choose a forecast period and size of fall, then click **Run my simulation**. It requests five years of history and uses up to the most recent 1,260 returns. Each available model simulates 10,000 futures using fixed settings. Results explain the event frequency, label graph axes, and connect percentages, probability and derivatives to Class 12 maths. Technical assumptions are explained without presenting these research models as ISC syllabus requirements.

Changing the market requires reloading its prices. Changing the data, horizon or fall threshold hides old results until the simulation is rerun. CSV uploads take priority over market downloads. The sidebar's **Advanced view** retains the original dashboard, controls, diagnostic models, stress testing and historical replay. Forecast probabilities remain conditional estimates, not claims of reliable crash prediction.

## Run on your MacBook / Windows / Linux

Use Python **3.12** (3.11–3.13 should work). Unzip this folder, open a terminal inside it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`. Open **http://localhost:8501**. For exact versions used in verification, install `requirements-verified.txt` instead of `requirements.txt`.

1. Select an index or enter a Yahoo ticker for an individual stock.
2. Load real historical data; US indices can use the explicit FRED alternative if Yahoo is unavailable. NIFTY/stocks need Yahoo or a real CSV. Provider errors are displayed, never replaced with generated prices.
3. Choose estimation window, horizon, threshold, event definition and models.
4. Click **Run Simulation / Compare Models**. The base simulation ignores stress multipliers. **Stress-Test Market** applies them and labels the result conditional.
5. Inspect parameters, tails, derivatives, paths, risk and diagnostics. Export JSON or CSV.
6. Run historical replay after loading sufficient preceding history and complete subsequent outcomes. Use broad walk-forward validation for ensemble eligibility.

CSV format:

```csv
Date,Close
YYYY-MM-DD,positive_observed_close
```

At least 252 rows; unique dates; no invalid or nonpositive prices. Use consistently adjusted stock prices to avoid treating splits as crashes. The example above specifies a format, not fabricated market observations. Uploaded data provenance is user supplied. Trading-day steps mean one observed close per step; gaps and holidays are not fabricated.

## Implemented computations

- **GBM**: log-return mean/variance calibration; exact daily lognormal increments; optional antithetic normal sampling; analytical terminal and continuously monitored barrier benchmarks.
- **GARCH / GJR-GARCH**: SciPy maximum likelihood; Gaussian or unit-variance Student-t innovations; positive intercept; transformed coefficients enforce alpha + beta + gamma/2 < 0.999. Conditional variance recursion, multi-day forecast and residual diagnostics. Daily log returns are modelled directly, not reinterpreted as simple returns.
- **Merton jumps**: maximum likelihood of daily compound-Poisson Gaussian mixture, counts 0–8 in fitting; exact untruncated Poisson counts in simulation. Daily arrival intensity bounded at 0.5 makes truncated calibration mass negligible. Drift is compensated to preserve the fitted arithmetic expectation. Two fitting starts. Jump mean/intensity/variance are weakly identified from closes; output labels this and flags intensity boundaries.
- **Three-state regimes**: Gaussian hidden Markov model; scaled forward/backward Baum–Welch likelihood optimisation with three initialisations; stochastic transition rows; filtered endpoint probabilities; expected durations; simulated state entry. Unconverged models are disabled. Means/volatilities are fitted freely: data need not produce exactly one growth, uncertainty, and bear state. The lowest mean/volatility state is labelled crisis-like, not proof of a crisis.
- **EVT**: loss-tail peaks over threshold with SciPy GPD maximum likelihood; daily log-loss VaR/ES, normal comparison, tail exceedance probabilities, threshold sensitivity and optional circular block-bootstrap intervals. At least 40 exceedances. Infinite ES when shape ≥1. No fabricated multi-day crash probability from a daily tail fit.
- **Calculus**: Savitzky–Golay cubic smoothing and first/second derivatives in price units per observation. Centred smoothing is descriptive and excluded from prediction/backtesting to avoid future-neighbour leakage.
- **Heston**: physical-measure full-truncation Euler with correlated shocks and nonnegative variance used in price increments. Internal auxiliary variance can go negative; positive part is used consistently. Parameters are illustrative/user supplied, except estimated drift/initial variance. Not option calibrated, not eligible for ensemble stacking.
- **Circular historical block bootstrap**: empirical return sequences with selectable block length; unconditional IID-return baseline uses block length 1.

The default event is a **20% decline from the starting close within 90 future daily observations**. Running-peak maximum drawdown is a separate definition. Threshold crossing is evaluated at every simulated close; between-close crashes are not observed. Terminal VaR/ES are simple-return losses; EVT uses daily log losses. Neither is silently substituted for barrier probability.

Every stochastic model emits threshold probabilities (10/15/20/30/40%), Wilson sampling intervals (IID draws), standard errors, future percentiles, terminal expected value, terminal VaR/ES, maximum drawdowns and sample-size convergence summaries. GBM antithetic errors use independent pair means instead of an IID Bernoulli formula. Zero observed crashes do not prove zero risk: Wilson upper bounds remain positive.

## Replay and ensemble

Each forecast origin calibrates using trailing prices **through that date only**. Actual labels use subsequent complete horizons. No future crisis labels enter fitting. Dates unavailable in the data are excluded. Replay presets cover 1987, 2000–02, 2008 and 2020, but are enabled by actual data availability rather than supplied synthetic history.

Metrics: Brier score, log loss, calibration bins, precision, recall, false-positive rate, missed-crash rate and successive forecast changes. Classification cutoff is 0.5; undefined rates are reported as missing. Baselines: complete past-window empirical event frequency (Jeffreys-smoothed) and unconditional IID-return simulations.

Convex log-loss stacking fits weights jointly on earlier out-of-sample forecasts. Labels crossing into the later test period are **purged**. Later forecasts form an untouched evaluation set, also scored for individual models/baselines. A minimum of 12 complete origins, 6 purged fitting origins and both fitting outcomes is required. Otherwise there is **no ensemble**. Correlated models receive joint weights rather than independent evidence multiplication. Weight uncertainty and structural errors are not covered by Monte Carlo intervals. Small samples and overlapping horizons invalidate strong accuracy claims. No automatic claim of improvement is made if test scores are worse than baselines.

Defaults: 10,000 current paths, 1,000 paths per replay origin, at most 20 chronological origins. For broad validation increase origin count and provide long history. Crisis presets are case studies, not representative validation.

## Verification

```bash
python -m pytest crashlab/tests -q
python benchmark.py your_real_prices.csv --paths 100000 --replay --output experiment.json
```

Tests use explicitly synthetic test fixtures to check algorithms, never as user-facing market history. Tests cover analytical GBM mean/variance/tails, GARCH coefficient recovery and variance forecasts, Poisson counts, transition validity, EVT formulas, derivative correctness, Heston stability, bootstrap, event definitions, reproducibility, convergence, leakage prevention, purged stacking and dashboard interaction.

`verification_real_data.json` and `.replay.csv` record computations from real FRED S&P 500 data retrieved during this build. They are an execution record, not a forecasting accuracy certification. They contain conditional estimates and selected-case replay metrics; consult observation dates and configurations. Raw provider data is not redistributed in this archive.

## Limits and provenance

- Historical fits are physical-measure approximations. Rare-event forecasts have substantial parameter/model uncertainty. Reported path intervals and probability intervals condition on fitted parameters; only the EVT bootstrap adds an explicit tail-fit uncertainty calculation.
- No reliable jump-parameter confidence interval or Heston option calibration is claimed. Regime/GARCH fits can be local optima. Diagnostics use approximate p-values; no multiple-testing correction or formal model-selection claim.
- Calibration likelihood is not out-of-sample performance. Passing numerical tests is not proof of economic predictability.
- Yahoo is a free unofficial-access provider and may rate-limit. FRED index history availability varies; S&P/Dow generally expose a rolling ten-year window. Long historical replay therefore often requires a separately sourced CSV. FRED index closes are price indices, not dividend total-return series.
- Adjusted stock histories can be revised and may contain survivorship/corporate-action information. True point-in-time stock validation requires vintage datasets. No survivorship-free universe or intraday barrier observation is claimed.
- Computation is local CPU work. 100,000 × 90 paths uses about 36 MB for stored float32 prices per model plus temporary diagnostics. Long 756-step API horizons need much more memory. The dashboard processes models sequentially and retains plot samples and terminal/drawdown vectors rather than all full path matrices.
- Core calculations use float64 except stored plot/risk paths (float32). Discrete daily monitoring and numerical discretisation introduce additional errors distinct from Monte Carlo sampling.
- Ensemble weights are event/configuration-specific. Controls that change forecasts hide stale results. Base and stressed forecasts are not combined.

Primary references used to check conventions:

- ARCH volatility/GARCH/Student-t documentation: https://arch.readthedocs.io/en/stable/univariate/univariate_volatility_modeling.html
- Statsmodels regime-switching likelihood/filter conventions: https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html
- SciPy GPD distribution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html
- FRED S&P 500 provenance and availability: https://fred.stlouisfed.org/series/SP500

## Source layout

`crashlab/data.py` providers/validation; `estimation.py` likelihood and descriptive fitting; `simulation.py` vectorised numerical paths; `risk.py` event calculations and analytic GBM checks; `backtest.py` walk-forward evaluation/stacking; `diagnostics.py` tail bootstrap and residual/stability checks; `app.py` interactive dashboard; `benchmark.py` CLI; `crashlab/tests/` verification.
