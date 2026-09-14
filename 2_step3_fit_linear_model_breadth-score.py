# ── Data-driven ARIMA trend for semantic dispersion ────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t


# 0 ─────────── parameters ─────────────────────────────────────────────────
terms_to_plot = ["adhd"]          # add more if desired
csv_path      = "output/breadth_scores.csv"
y_col         = "cosine_dissim_mean"       # outcome
se_col        = "cosine_dissim_se"         # ±1 SE for scatter ribbon
first_year    = 1980                       # filter

# 1 ─────────── load & tidy ────────────────────────────────────────────────
df = (pd.read_csv(csv_path)
        .astype({"year": int})
        .loc[lambda d: d["year"] >= first_year])

# 2 ─────────── iterate over terms ─────────────────────────────────────────
for term in terms_to_plot:
    sub  = df.query("term == @term").sort_values("year")
    years, y_raw, se_raw = sub["year"].values, sub[y_col].values, sub[se_col].values

    # 2.1  ADF test for stationarity --------------------------------------
    adf_p = adfuller(y_raw, autolag="AIC")[1]
    d = 0 if adf_p < .05 else 1
    if d == 1:
        y_series = np.diff(y_raw)
        years_d  = years[1:]
        print(f"{term}: ADF p (level) = {adf_p:.3f} → differenced once.")
    else:
        y_series = y_raw
        years_d  = years
        print(f"{term}: ADF p (level) = {adf_p:.3f} → already stationary.")

    # 2.2  candidate ARIMA orders (p = 0–2, q = 0–1) -----------------------
    candidates, aic_vals = [], []
    for p in (0, 1, 2):
        for q in (0, 1):
            try:
                res = ARIMA(y_series, order=(p, d, q), trend="t").fit()
                candidates.append(res);  aic_vals.append(res.aic)
            except Exception:
                continue
    best = candidates[np.argmin(aic_vals)]
    order = best.model_orders
    p, q  = order["ar"], order["ma"]
    print(f"{term}: selected ARIMA({p},{d},{q})  (AIC = {best.aic:.2f})")

    # 2.3  residual white-noise check --------------------------------------
    lb = acorr_ljungbox(best.resid, lags=[6, 12], return_df=True)["lb_pvalue"]
    print(f" Ljung–Box p (lag 6, 12) = {lb.iloc[0]:.3f}, {lb.iloc[1]:.3f}")

    # 2.4  reconstruct fitted level if differenced -------------------------
    fitted = best.fittedvalues
    if d == 1:
        fitted_level = np.r_[y_raw[0], y_raw[0] + np.cumsum(fitted)]
    else:
        fitted_level = fitted

    se_fit = best.get_prediction().se_mean
    if d == 1:
        se_cum = np.r_[0, np.cumsum(se_fit)]
        se_use = se_cum
    else:
        se_use = se_fit
    t95 = t.ppf(.975, df=len(y_series)-2)
    lo, hi = fitted_level - t95*se_use, fitted_level + t95*se_use

    # 3 ─────────── plot ---------------------------------------------------
    plt.figure(figsize=(14,5))
    plt.fill_between(years, y_raw-se_raw, y_raw+se_raw,
                     color="lightgrey", alpha=.3, label="± 1 SE (means)")
    plt.scatter(years, y_raw, color="black", s=28, alpha=.6, label="Yearly mean")
    plt.fill_between(years, lo, hi, color="lightgreen", alpha=.25,
                     label="95 % CI (ARIMA)")
    plt.plot(years, fitted_level, color="royalblue", lw=2,
             label=f"ARIMA({p},{d},{q}) trend")
    plt.title(f"Annual semantic dispersion for “{term}” ({years.min()}–{years.max()})")
    plt.ylabel("Cosine distance (1 – cosine similarity)")
    plt.xlabel("Year"); plt.grid(True, ls="--", alpha=.6); plt.legend(); plt.tight_layout()
    plt.show()

# 4 ─────────── console summary ----------------------------------------
names   = best.model.param_names          # list of coefficient names
values  = best.params                     # ndarray of estimates
stderr  = best.bse                        # standard errors

# try drift first (differenced series), then const (level-stationary);
# if neither exists, report that the deterministic trend is zero
if "drift" in names:                      # ARIMA with differencing
    idx_tag = "drift"
elif "const" in names:                    # stationary level model
    idx_tag = "const"
else:                                     # model dropped the deterministic term
    print(f"{term}: model contains no drift/constant term "
          "(trend forced to zero).\n")
    idx_tag = None

if idx_tag is not None:
    idx       = names.index(idx_tag)
    drift     = values[idx]
    drift_se  = stderr[idx]
    ci_low    = drift - 1.96 * drift_se
    ci_high   = drift + 1.96 * drift_se
    print(f"{term}: {idx_tag} = {drift:+.4f} ± {drift_se:.4f}  "
          f"(95 % CI {ci_low:+.3f} … {ci_high:+.3f})\n")


