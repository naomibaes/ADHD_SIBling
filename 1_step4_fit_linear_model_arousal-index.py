import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import t

# ── 1. load & filter ───────────────────────────────────────────────
df = (
    pd.read_csv("output/annual_arousal_indices.csv")
      .query("(avg_arousal_index > 0) & (year >= 1990)") #Nick: changed year from 1980 to 1990
      .sort_values("year")
)

x = df["year"].values
y = df["avg_arousal_index"].values
X = sm.add_constant(x)                      # intercept + year

# ── 2. OLS fit + Durbin–Watson ─────────────────────────────────────
ols_res = sm.OLS(y, X).fit()
dw       = durbin_watson(ols_res.resid)
print(f"Durbin–Watson = {dw:.3f}")

# ── 3. Decide: OLS vs GLS-AR(1) ───────────────────────────────────
if 1.5 < dw < 2.5:                          # little autocorr
    results     = ols_res
    model_label = "OLS"
    print("Using OLS; no substantial autocorrelation detected.")
    rho_label   = ""
else:                                       # strong autocorr → GLS-AR(1)
    from statsmodels.regression.linear_model import GLSAR
    glsar       = GLSAR(y, X, rho=1)
    results     = glsar.iterative_fit(maxiter=10)
    rho         = glsar.rho[0]
    model_label = "GLS-AR(1)"
    rho_label   = f" (ρ = {rho:.2f})"
    print(f"Using GLS-AR(1); ρ = {rho:.2f}")

# ── 4. predicted line & 95 % CI ────────────────────────────────────
y_hat    = results.predict(X)
se_obs   = results.get_prediction(X).se_obs        # pointwise SE
t_crit   = t.ppf(0.975, df=len(x) - 2)             # two-sided 95 %
ci_low   = y_hat - t_crit * se_obs
ci_high  = y_hat + t_crit * se_obs

# ── 5. plot ────────────────────────────────────────────────────────
plt.figure(figsize=(14, 5))

# yearly points
plt.scatter(x, y, color="darkred", s=28, zorder=3, label="yearly mean")

# CI ribbon
plt.fill_between(x, ci_low, ci_high,
                 color="darkred", alpha=.25,
                 label=f"95 % CI ({model_label})")

# trend line
plt.plot(x, y_hat, color="black", lw=2,
         label=f"{model_label} trend{rho_label}")

# cosmetics
plt.title("Annual arousal index for 'ADHD' (1980 – 2025)")
plt.xlabel("Year")
plt.ylabel("Arousal index (Warriner 1 = low, 9 = high)")
plt.grid(True, ls="--", alpha=.6)
plt.legend()
plt.tight_layout()
plt.show()

# ── 6. regression summary ─────────────────────────────────────────
print(results.summary())
