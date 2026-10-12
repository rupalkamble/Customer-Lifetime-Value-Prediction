import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
os.makedirs("figures", exist_ok=True)

def savefig(name):
    plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=130); plt.show()

# 1. LOAD (real UCI file if present, else synthetic)
real = [p for p in ["data/online_retail_II.xlsx", "data/online_retail_II.csv"] if os.path.exists(p)]
if real:
    df = pd.read_excel(real[0]) if real[0].endswith("xlsx") else pd.read_csv(real[0], parse_dates=["InvoiceDate"])
    SOURCE = "UCI Online Retail II"
else:
    df = pd.read_csv("data/online_retail_synthetic.csv", parse_dates=["InvoiceDate"])
    SOURCE = "synthetic (same schema as Online Retail II)"
print("Data source:", SOURCE, "| raw rows:", len(df))

# 2. CLEAN
n0 = len(df)
df = df.dropna(subset=["Customer ID"])
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[(df.Quantity > 0) & (df.Price > 0)].drop_duplicates()
df["Revenue"] = df.Quantity * df.Price
print(f"Rows kept after cleaning: {len(df):,} of {n0:,}")

# 3. TIME SPLIT (feature window -> target window)
d_min, d_max = df.InvoiceDate.min(), df.InvoiceDate.max()
cutoff = d_max - pd.Timedelta(days=92)
feat, targ = df[df.InvoiceDate < cutoff], df[df.InvoiceDate >= cutoff]
print("Feature window:", d_min.date(), "->", cutoff.date(), "| Target window:", cutoff.date(), "->", d_max.date())

# 4. FEATURES
g = feat.groupby("Customer ID")
X = pd.DataFrame({
    "Recency": (cutoff - g.InvoiceDate.max()).dt.days,
    "Frequency": g.Invoice.nunique(),
    "Monetary": g.Revenue.sum(),
    "AvgOrderValue": g.Revenue.sum() / g.Invoice.nunique(),
    "Tenure": (cutoff - g.InvoiceDate.min()).dt.days,
    "UniqueProducts": g.StockCode.nunique(),
    "AvgItemsPerOrder": g.Quantity.sum() / g.Invoice.nunique(),
    "IsUK": (g.Country.first() == "United Kingdom").astype(int),
})
X["PurchaseRate"] = X.Frequency / (X.Tenure + 1)
X["Last90dOrders"] = feat[feat.InvoiceDate >= cutoff - pd.Timedelta(days=90)].groupby("Customer ID").Invoice.nunique().reindex(X.index).fillna(0)
y = targ.groupby("Customer ID").Revenue.sum().reindex(X.index).fillna(0).rename("FutureLTV")
print("Customers:", len(X), "| share with zero future spend: %.1f%%" % (100 * (y == 0).mean()))
print(X.isna().sum().sum(), "missing feature values")

# outlier capping (99th percentile)
caps = {c: X[c].quantile(0.99) for c in ["Monetary", "AvgOrderValue", "AvgItemsPerOrder", "UniqueProducts"]}
for c, v in caps.items():
    X[c] = X[c].clip(upper=v)
cap_y = y.quantile(0.99); y = y.clip(upper=cap_y)

# 5. EDA PLOTS
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(y, bins=50, ax=ax[0]); ax[0].set_title("Future 3-month LTV (capped)")
sns.histplot(np.log1p(y), bins=50, ax=ax[1], color="C1"); ax[1].set_title("log(1 + LTV)")
sns.histplot(X.Recency, bins=40, ax=ax[2], color="C2"); ax[2].set_title("Recency (days)")
savefig("01_distributions")

plt.figure(figsize=(9, 7))
sns.heatmap(pd.concat([X, y], axis=1).corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation heatmap"); savefig("02_correlation")

# 6. SPLIT, SEGMENT, SCALE
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rfm = ["Recency", "Frequency", "Monetary"]
seg_sc = StandardScaler().fit(np.log1p(Xtr[rfm]))
km = KMeans(4, random_state=42, n_init=10).fit(seg_sc.transform(np.log1p(Xtr[rfm])))
Xtr = Xtr.assign(Segment=km.labels_)
Xte = Xte.assign(Segment=km.predict(seg_sc.transform(np.log1p(Xte[rfm]))))
Xtr_d = pd.get_dummies(Xtr, columns=["Segment"], prefix="Seg", dtype=int)
Xte_d = pd.get_dummies(Xte, columns=["Segment"], prefix="Seg", dtype=int).reindex(columns=Xtr_d.columns, fill_value=0)

plt.figure(figsize=(8, 4.5))
sns.boxplot(x=Xtr.Segment, y=ytr.values); plt.title("Future LTV by customer segment (train)"); plt.xlabel("Segment"); plt.ylabel("Future LTV")
savefig("03_segments")
print(Xtr.groupby("Segment")[rfm].mean().round(1).assign(Count=Xtr.Segment.value_counts().sort_index(), AvgFutureLTV=ytr.groupby(Xtr.Segment).mean().round(1)))

sc = StandardScaler().fit(Xtr_d)

# 7. MODELS
lr = LinearRegression().fit(sc.transform(Xtr_d), ytr)
gs = GridSearchCV(DecisionTreeRegressor(random_state=42),
                  {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [5, 10, 20, 50]},
                  cv=5, scoring="r2").fit(Xtr_d, ytr)
dt = gs.best_estimator_
print("Best tree params:", gs.best_params_, "| CV R2: %.3f" % gs.best_score_)

res = {}
for name, m, Xt_tr, Xt_te in [("Linear Regression", lr, sc.transform(Xtr_d), sc.transform(Xte_d)),
                              ("Decision Tree", dt, Xtr_d, Xte_d)]:
    p = m.predict(Xt_te)
    res[name] = dict(pred=p, R2=r2_score(yte, p), MAE=mean_absolute_error(yte, p),
                     RMSE=np.sqrt(mean_squared_error(yte, p)),
                     Train_R2=r2_score(ytr, m.predict(Xt_tr)))
results = pd.DataFrame({k: {m: v[m] for m in ["R2", "MAE", "RMSE", "Train_R2"]} for k, v in res.items()}).T.round(3)
print(results)
print("Baseline (predict train mean): MAE = %.1f" % mean_absolute_error(yte, np.full(len(yte), ytr.mean())))

# 8. EVALUATION PLOTS
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for a, (n, v) in zip(ax, res.items()):
    a.scatter(yte, v["pred"], alpha=.35, s=14); lim = [0, max(yte.max(), v["pred"].max())]
    a.plot(lim, lim, "r--"); a.set_title(f"{n}: R²={v['R2']:.2f}"); a.set_xlabel("Actual LTV"); a.set_ylabel("Predicted LTV")
savefig("04_pred_vs_actual")

best = max(res, key=lambda k: res[k]["R2"]); print("Best model on test R²:", best)
resid = yte - res[best]["pred"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(res[best]["pred"], resid, alpha=.35, s=14); ax[0].axhline(0, color="r", ls="--"); ax[0].set_title(f"Residuals vs predicted ({best})"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Residual")
sns.histplot(resid, bins=40, ax=ax[1]); ax[1].set_title("Residual distribution")
savefig("05_residuals")

# 9. INTERPRETATION
coef = pd.Series(lr.coef_, index=Xtr_d.columns).sort_values()
imp = pd.Series(dt.feature_importances_, index=Xtr_d.columns).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
coef.plot.barh(ax=ax[0], color=np.where(coef < 0, "C3", "C0")); ax[0].set_title("Linear Regression coefficients (standardised)")
imp.plot.barh(ax=ax[1], color="C2"); ax[1].set_title("Decision Tree feature importance")
savefig("06_coef_importance")
print("\nTop coefficients:\n", coef.sort_values(key=abs, ascending=False).head(5).round(2))
print("\nTop tree features:\n", imp.sort_values(ascending=False).head(5).round(3))

plt.figure(figsize=(22, 9))
plot_tree(dt, feature_names=Xtr_d.columns, filled=True, max_depth=3, fontsize=9, rounded=True, precision=1)
plt.title("Decision tree (top 3 levels)"); savefig("07_tree")
