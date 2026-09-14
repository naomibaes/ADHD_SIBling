# ── Plot yearly valence with trend that adapts to autocorrelation ────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import t

# 1 ──────────────────────────── Load & filter
df = (
    pd.read_csv("output/annual_valence_indices.csv")
      .query("(avg_valence_index > 0) & (year >= 1990)")
      .sort_values("year")
)

x = df["year"].values
y = df["avg_valence_index"].values
X = sm.add_constant(x)                       # intercept + year

# 2 ──────────────────────────── OLS fit + DW test
ols_res = sm.OLS(y, X).fit()
dw = durbin_watson(ols_res.resid)
print(f"Durbin–Watson (OLS residuals) = {dw:.3f}")

if 1.5 < dw < 2.5:                           # little autocorr  → keep OLS
    res         = ols_res
    model_label = "OLS"
    rho_label   = ""
    print("Using OLS (autocorrelation negligible).")
else:                                        # strong autocorr → GLS-AR(1)
    from statsmodels.regression.linear_model import GLSAR
    glsar       = GLSAR(y, X, rho=1)
    res         = glsar.iterative_fit(maxiter=10)
    rho         = glsar.rho[0]
    model_label = "GLS-AR(1)"
    rho_label   = f" (ρ = {rho:.2f})"
    print(f"Using GLS-AR(1); ρ = {rho:.2f}")

# 3 ──────────────────────────── Predictions & 95 % CI
pred     = res.get_prediction(X)
y_hat    = pred.predicted_mean
se_obs   = pred.se_obs
t_crit   = t.ppf(0.975, df=len(x) - 2)       # two-sided 95 %
ci_low   = y_hat - t_crit * se_obs
ci_high  = y_hat + t_crit * se_obs

# 4 ──────────────────────────── Figure
plt.figure(figsize=(14, 5))

# yearly means
plt.scatter(x, y, color="darkgreen", s=28, zorder=3, label="yearly mean")

# confidence ribbon
plt.fill_between(x, ci_low, ci_high, color="lightgreen", alpha=.25,
                 label=f"95 % CI ({model_label})")

# trend line
plt.plot(x, y_hat, color="black", lw=2,
         label=f"{model_label} trend{rho_label}")

# cosmetics
plt.xlabel("Year")
plt.ylabel("Valence index (Warriner 1 = low, 9 = high)")
plt.title("Annual valence index for 'ADHD' (1980 – 2024)")
plt.grid(True, ls="--", alpha=.6)
plt.legend()
plt.tight_layout()
plt.show()

# 5 ──────────────────────────── Text summary
print(res.summary())
