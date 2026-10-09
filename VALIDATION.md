# Execution verification

Verification date: 9 October 2026.

Real-data integration: FRED SP500 daily closes, 2 January 2018 through 8 October 2026; 2,204 non-missing observations. Source: https://fred.stlouisfed.org/series/SP500 . No raw provider prices are redistributed.

100,000 paths per fitted model; 90 future trading observations; 20% starting-price threshold; calibration uses the latest 1,260 daily log returns. Seed recorded in JSON; models use deterministic seed offsets. All five eligible fitted simulators completed.

| Model | Paths | Simulation and risk calculation seconds |
| --- | ---: | ---: |
| GBM | 100,000 | 0.238 |
| Bootstrap | 100,000 | 0.271 |
| GARCH | 100,000 | 0.579 |
| Jump | 100,000 | 0.663 |
| Regime | 100,000 | 0.837 |

Timing excludes parameter estimation and depends on this machine. It is not a performance guarantee.

The COVID-period replay produced three complete forecast origins using 252-return training windows and 1,000 simulated paths per model/origin. Forecast dates, outcomes, predictions, baselines and forward volatility evaluation are recorded in verification_real_data.replay.csv. This is a small selected-case execution check, not a meaningful predictive accuracy study. It does not qualify for ensemble weights.

Mathematical tests and Streamlit interaction tests are recorded in test-results.txt. The HTTP server health endpoint returned 200 / ok. Test fixtures are synthetic and confined to automated testing. Passing tests checks implementation correctness, not financial forecasting accuracy.

Execution report: verification_real_data.json. Exact top-level package versions: requirements-verified.txt.
