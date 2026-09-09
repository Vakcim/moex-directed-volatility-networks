# Conference abstract

**Predictive Networks Are Not Correlation Networks: Directed Nonlinear
Dependence and Realized-Volatility Forecasting**

We study whether cross-asset predictive networks contain information for
one-day-ahead realized-variance forecasts beyond a strong heterogeneous
autoregressive benchmark. Using a point-in-time universe of Moscow Exchange
stocks, we construct leakage-aware rolling graphs with temporally separated
training, importance and forecasting windows. Pearson networks measure
contemporaneous co-movement, whereas directed CatBoost networks use
out-of-sample source-group permutation importance conditional on the target's
own volatility history and a market factor. We compare current graph state,
graph changes, linear predictive networks and placebo networks, and repeat the
analysis after causal market-factor residualization. The nonlinear predictive
network is structurally distinct from the correlation network: it is strongly
asymmetric, has low edge overlap with Pearson, and changes substantially after
removing the market component. This structural novelty, however, yields little
forecasting gain. CatBoost-current only marginally improves development-sample
QLIKE relative to HAR-RV, without statistically stable superiority, while
dynamic graph features fail to improve forecasts consistently across network
definitions. Residualization changes topology but does not produce a better
forecast than residual HAR-RV. The results separate co-movement from conditional
predictability and show that an economically interesting network can remain too
weak and estimation-sensitive to improve short-horizon volatility forecasts.

Status note: all numerical findings currently refer to the development period
ending 2026-08-10; the prospective holdout remains sealed.

