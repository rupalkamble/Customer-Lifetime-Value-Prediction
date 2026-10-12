"""Generates a synthetic transaction log with the same schema as the UCI
'Online Retail II' dataset (Invoice, StockCode, Description, Quantity,
InvoiceDate, Price, Customer ID, Country). Used only when the real dataset
is not available. Customer behaviour follows a BG/NBD-style process:
each customer has their own purchase rate, a per-order chance of going
inactive, and a typical basket value."""
import numpy as np, pandas as pd

def generate(n_customers=4500, seed=42):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-09")
    span = (end - start).days
    n_products = 600
    prices = np.round(rng.lognormal(1.0, 0.8, n_products) + 0.5, 2)
    countries = ["United Kingdom"] * 90 + ["Germany", "France", "EIRE", "Spain",
                 "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia", "Norway"]
    rows, inv = [], 500000
    for cid in range(12347, 12347 + n_customers):
        first = int(rng.beta(1.2, 2.5) * span)               # most customers join early
        rate = rng.gamma(1.5, 0.012)                          # orders per day
        p_drop = rng.beta(1.0, 6.0)                           # chance of going inactive after an order
        aov_scale = rng.lognormal(0.0, 0.55)                  # customer's spending level
        qty_pref = rng.choice([1, 2, 4, 6, 12], p=[.25, .25, .2, .2, .1])
        country = rng.choice(countries)
        day = first
        while day < span:
            inv += 1
            n_lines = 1 + rng.poisson(4 * min(aov_scale, 3))
            ts = start + pd.Timedelta(days=int(day), hours=int(rng.integers(8, 18)),
                                      minutes=int(rng.integers(0, 60)))
            for _ in range(n_lines):
                sc = int(rng.integers(0, n_products))
                q = max(1, int(rng.poisson(qty_pref)))
                rows.append((str(inv), f"P{sc:04d}", f"PRODUCT {sc}", q, ts,
                             prices[sc], cid, country))
            if rng.random() < 0.03:                           # occasional return
                inv += 1
                rows.append((f"C{inv}", f"P{sc:04d}", f"PRODUCT {sc}", -q, ts,
                             prices[sc], cid, country))
            if rng.random() < p_drop:
                break
            day += max(1, rng.exponential(1.0 / max(rate, 1e-3)))
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    # guest checkouts with no Customer ID, to exercise the cleaning step
    guest = df.sample(frac=0.04, random_state=1).copy()
    guest["Customer ID"] = np.nan
    guest["Invoice"] = guest["Invoice"].astype(str) + "G"
    df = pd.concat([df, guest]).sort_values("InvoiceDate").reset_index(drop=True)
    return df

if __name__ == "__main__":
    d = generate()
    d.to_csv("data/online_retail_synthetic.csv", index=False)
    print(d.shape, d["Customer ID"].nunique(), d.InvoiceDate.min(), d.InvoiceDate.max())
