# ░░ Plot yearly arousal trajectories with OLS trend (+95 % CI) ░░
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, t

# 1 ─── load & filter ────────────────────────────────────────────────
df  = pd.read_csv("output/annual_arousal_indices.csv")
df  = df[(df["avg_arousal_index"] > 0) & (df["year"] >= 1990)].sort_values("year") #Nick: changed year from 1980 to 1990

x   = df["year"].values           # numeric years for regression
y   = df["avg_arousal_index"].values
se_obs = df["se_arousal_index"].values   # optional: ±1 SE ribbon for yearly means

# 2 ─── ordinary-least-squares fit ─────────────────────────────────
slope, intercept, r, p, stderr = linregress(x, y)
y_hat = intercept + slope * x

# residual mean-square → for CI of the line
resid  = y - y_hat
mse    = np.sum(resid**2) / (len(x) - 2)
x_mean = x.mean()
sxx    = np.sum((x - x_mean)**2)

t_crit = t.ppf(0.975, df=len(x)-2)          # two-sided 95 %
se_line = np.sqrt(mse * (1/len(x) + (x - x_mean)**2 / sxx))
ci_half = t_crit * se_line                  # half-width of CI at each x

# 3 ─── figure ─────────────────────────────────────────────────────
plt.figure(figsize=(14, 5))

# yearly means (+grey ribbon for the original ±SE, optional)
plt.fill_between(x, y-se_obs, y+se_obs, color="lightgrey", alpha=0, label="± 1 SE (means)")
plt.scatter(x, y, color="darkred", s=28, zorder=3)

# OLS fit + 95 % confidence band
plt.plot(x, y_hat, color="black", lw=2, label=f"OLS fit (r = {r:.2f}, p = {p:.4g})")
plt.fill_between(x, y_hat-ci_half, y_hat+ci_half, color="darkred", alpha=.25, label="95 % CI (fit)")

# 4 ─── ticks & labels ─────────────────────────────────────────────
tick_every = 1                           # show every 5th year
ticks       = x[::tick_every]
tick_labels = ticks.astype(str)
plt.xticks(ticks, tick_labels, rotation=45, ha="right")

plt.xlabel("Year")
plt.ylabel("Arousal index  (Warriner 1 = low, 9 = high)")
plt.title("Annual arousal index for “ADHD” (1990 – 2025) with trend ± 95 % CI") #Nick: renamed from schizophrenia to ADHD, 1980 to 1990
plt.grid(True, lw=.8, ls="--", alpha=.6)
plt.legend()
plt.tight_layout()
plt.show()

# … (the rest of the code: load df, fit line, plot) …

# ── stats read-out ───────────────────────────────────────────────
r2 = r**2
print(f"── — OLS summary ─────────────────────")
print(f"n (years)        : {len(x)}")
print(f"Slope            : {slope:+.4f}   (± {stderr:.4f})")
print(f"Intercept        : {intercept:+.4f}")
print(f"R-squared        : {r2:.4f}")
print(f"Pearson r        : {r:.4f}")
print(f"p-value (slope)  : {p:.4g}")
print("────────────────────────────────────────────────────\n")
