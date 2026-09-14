# ── ARIMA trend that internalises autocorrelation ───────────────────────────

# ── correct imports ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf        # keep these
from statsmodels.stats.diagnostic import acorr_ljungbox    # << new path
from scipy.stats import t


# 1 ─────────────────────── Load & tidy
df = (
    pd.read_csv("output/annual_valence_indices.csv")
      .query("(avg_valence_index > 0) & (year >= 1980)") 
      .sort_values("year")
)
years = df["year"].values
y_raw = df["avg_valence_index"].values

# 2 ─────────────────────── Stationarity check (ADF)
adf_stat, adf_p, *_ = adfuller(y_raw, autolag="AIC")
print(f"ADF p-value (level) = {adf_p:.3f}")

if adf_p < 0.05:
    d = 0                       # already stationary
    y_series = y_raw
else:
    d = 1                       # first difference to remove drift
    y_series = np.diff(y_raw)
    years_d  = years[1:]        # align years for differenced series
    print(f"ADF p after 1st diff = {adfuller(y_series, autolag='AIC')[1]:.3f}")

# 3 ─────────────────────── Fit ARIMA(1,d,0) with drift
arima = ARIMA(y_series, order=(1, d, 0), trend="t")  # 't' = intercept + drift
res   = arima.fit()
print(res.summary())

# 4 ─────────────────────── Reconstruct fitted line on the original scale
fitted = res.fittedvalues
if d == 1:
    # cumulative sum to get the level back; anchor at first observed value
    fitted_level = np.r_[y_raw[0], y_raw[0] + np.cumsum(fitted)]
else:
    fitted_level = fitted

# 5 ─────────────────────── 95 % forecast ribbon for in-sample fit
se   = res.get_prediction().se_mean
t95  = t.ppf(0.975, df=len(y_series)-2)
if d == 1:
    se_cum = np.r_[0, np.cumsum(se)]          # roughly accumulate s.e.
    lo = fitted_level - t95 * se_cum
    hi = fitted_level + t95 * se_cum
else:
    lo = fitted_level - t95 * se
    hi = fitted_level + t95 * se

# 6 ─────────────────────── Plot
plt.figure(figsize=(14, 5))
plt.scatter(years, y_raw, s=30, c="darkgreen", label="yearly mean")
plt.fill_between(years, lo, hi, color="lightgreen", alpha=.25,
                 label="95 % CI (ARIMA)")
plt.plot(years, fitted_level, c="black", lw=2,
         label=f"ARIMA(1,{d},0) trend")

plt.title("Annual valence index for 'ADHD' (1980–2024)")
plt.ylabel("Valence index (1 = low, 9 = high)")
plt.xlabel("Year")
plt.grid(True, ls="--", alpha=.6)
plt.legend()
plt.tight_layout()
plt.show()

# 7 ─────────────────────── Residual autocorr diagnostic
# Ljung–Box Q-test (lags 6 and 12)
lb = acorr_ljungbox(res.resid, lags=[6, 12], return_df=True)
print(lb[["lb_stat", "lb_pvalue"]])

