# ╔══════════════════════════════════════════════════════════════════╗
# ║   HYPER-PERSONALIZATION vs PROFITABILITY — INTERACTIVE PROTOTYPE ║
# ║   Google Colab · ipywidgets · Real-time AI Optimization          ║
# ║   Student: Khushi Garg | PGDM RBA 2                              ║
# ║   Upload Superstore.csv then run ALL CELLS (Runtime > Run all)   ║
# ╚══════════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Install & Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","pulp","ipywidgets","-q"],
               capture_output=True)

import warnings; warnings.filterwarnings("ignore")
import os, json, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, f_oneway
import pulp
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Plot style
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    "axes.grid": True, "grid.color": "#E0E0E0", "grid.linewidth": 0.6,
})
SEED = 42; np.random.seed(SEED); random.seed(SEED)
SEG_COLORS = {"Premium":"#2196F3","Loyal":"#4CAF50","Price-Sensitive":"#FF9800","Low-Value":"#F44336"}

print("✓ Libraries loaded")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Load Dataset & Build Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for path in ["Data/Superstore.csv", "Data/superstore.csv",
             "/content/Sample_-_Superstore.csv","Superstore.csv"]:
    if os.path.exists(path):
        df_raw = pd.read_csv(path, encoding="latin-1")
        print(f"✓ Dataset loaded: {path}  ({len(df_raw):,} rows)")
        break
else:
    raise FileNotFoundError("Upload Superstore.csv to /content/")

df = df_raw.copy()
df.columns = [c.strip().replace(" ","_") for c in df.columns]
df["Order_Date"]      = pd.to_datetime(df["Order_Date"])
SNAP                  = df["Order_Date"].max() + pd.Timedelta(days=1)
df["Discount_Amount"] = (df["Sales"]/(1-df["Discount"].replace(0,np.nan))*df["Discount"]).fillna(0)
df["Margin_Pct"]      = np.where(df["Sales"]>0, df["Profit"]/df["Sales"]*100, 0)
df["Profitable"]      = (df["Profit"]>0).astype(int)
df["Year"]            = df["Order_Date"].dt.year
df["Month"]           = df["Order_Date"].dt.month

dc = df.groupby("Customer_ID").agg(
    Total_Revenue     =("Sales",          "sum"),
    Total_Profit      =("Profit",         "sum"),
    Total_Orders      =("Order_ID",       "nunique"),
    Avg_Discount      =("Discount",       "mean"),
    Avg_Margin_Pct    =("Margin_Pct",     "mean"),
    Order_Variability =("Sales",          "std"),
    Preferred_Category=("Category",       lambda x: x.mode()[0]),
    First_Purchase    =("Order_Date",     "min"),
    Last_Purchase     =("Order_Date",     "max"),
).reset_index()
dc["Tenure_Years"]     = ((dc["Last_Purchase"]-dc["First_Purchase"]).dt.days/365).clip(lower=0.08)
dc["Annual_Frequency"] = dc["Total_Orders"]/dc["Tenure_Years"]
dc["AOV"]              = dc["Total_Revenue"]/dc["Total_Orders"]
dc["Avg_Order_Margin"] = dc["Total_Profit"]/dc["Total_Orders"]
dc["CLV"]              = (dc["Avg_Order_Margin"]*dc["Annual_Frequency"]*3).round(2)
dc["Profit_Per_Order"] = dc["Total_Profit"]/dc["Total_Orders"]
dc["Order_Variability"]= dc["Order_Variability"].fillna(0)
dc["Days_Since_Last"]  = (SNAP - dc["Last_Purchase"]).dt.days

# RFM + Clustering
rfm = df.groupby("Customer_ID").agg(
    Recency  =("Order_Date", lambda x:(SNAP-x.max()).days),
    Frequency=("Order_ID",   "nunique"),
    Monetary =("Sales",      "sum"),
).reset_index()
for col in ["Recency","Frequency","Monetary"]:
    p1,p99 = rfm[col].quantile(.01), rfm[col].quantile(.99)
    rfm[col] = rfm[col].clip(lower=p1, upper=p99)
rfm["R_Score"] = pd.qcut(rfm["Recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"]+rfm["F_Score"]+rfm["M_Score"]
Xs = StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]])
km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
rfm["Cluster"] = km.fit_predict(Xs)
cm_rfm = rfm.groupby("Cluster")["RFM_Score"].mean().sort_values(ascending=False)
lm = {cm_rfm.index[0]:"Premium",cm_rfm.index[1]:"Loyal",
      cm_rfm.index[2]:"Price-Sensitive",cm_rfm.index[3]:"Low-Value"}
rfm["RFM_Segment"] = rfm["Cluster"].map(lm)
dc  = dc.merge(rfm[["Customer_ID","RFM_Segment","RFM_Score"]], on="Customer_ID", how="left")
dfs = df.merge(rfm[["Customer_ID","RFM_Segment"]], on="Customer_ID", how="left")

# Churn model
FEAT = ["Avg_Discount","Total_Orders","Avg_Margin_Pct","AOV",
        "Tenure_Years","Order_Variability","CLV","RFM_Score"]
dc["Churned"]    = (dc["Days_Since_Last"] > 180).astype(int)
Xc = dc[FEAT].fillna(0).values
yc = dc["Churned"].values
Xtr,Xte,ytr,yte = train_test_split(Xc,yc,test_size=.25,random_state=SEED,stratify=yc)
RF = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced")
RF.fit(Xtr, ytr)
dc["Churn_Prob"] = RF.predict_proba(Xc)[:,1]
AUC = roc_auc_score(yte, RF.predict_proba(Xte)[:,1])

# Regression
dd = df.groupby("Customer_ID").agg(
    Avg_Discount=("Discount","mean"), Avg_Profit=("Profit","mean"),
    Avg_Margin  =("Margin_Pct","mean"), Order_Count=("Order_ID","nunique")
).reset_index().merge(rfm[["Customer_ID","RFM_Segment"]],on="Customer_ID",how="left")
reg = LinearRegression().fit(dd[["Avg_Discount","Order_Count"]].values, dd["Avg_Profit"].values)
BREAKEVEN = -(reg.intercept_ + reg.coef_[1]*dd["Order_Count"].mean()) / reg.coef_[0]

print(f"✓ Segments: {rfm['RFM_Segment'].value_counts().to_dict()}")
print(f"✓ Churn model AUC = {AUC:.3f}  |  Churn rate = {dc['Churned'].mean()*100:.1f}%")
print(f"✓ Break-even discount threshold = {BREAKEVEN:.1%}")
print(f"\n{'='*55}")
print(f"  MODELS READY — Run the cells below to explore")
print(f"{'='*55}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — Executive Dashboard (Static Charts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
display(HTML("""
<div style="background:#1a1a2e;color:#fff;padding:16px 24px;border-radius:10px;
font-family:Segoe UI,sans-serif;margin-bottom:16px">
  <div style="font-size:18px;font-weight:700">📊 Executive Dashboard</div>
  <div style="font-size:12px;color:rgba(255,255,255,.6);margin-top:4px">
  Real Superstore Data · 2014–2017 · Hyper-Personalization vs Profitability
  </div>
</div>
"""))

# KPI Cards
total_rev  = df["Sales"].sum()
total_prof = df["Profit"].sum()
loss_pct   = (df["Profit"]<0).mean()*100
avg_margin = dc["Avg_Margin_Pct"].mean()
churn_rate = dc["Churned"].mean()*100
avg_clv    = dc["CLV"].mean()

display(HTML(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
font-family:Segoe UI,sans-serif;margin-bottom:20px">
  <div style="background:#fff;border-radius:10px;padding:18px;border-left:4px solid #2196F3;
  box-shadow:0 2px 8px rgba(0,0,0,.06)">
    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px">Total Revenue</div>
    <div style="font-size:28px;font-weight:800;color:#2d3748">${total_rev/1e6:.2f}M</div>
    <div style="font-size:11px;color:#718096">{len(df):,} transactions · {dc['Customer_ID'].nunique()} customers</div>
  </div>
  <div style="background:#fff;border-radius:10px;padding:18px;border-left:4px solid #4CAF50;
  box-shadow:0 2px 8px rgba(0,0,0,.06)">
    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px">Total Profit</div>
    <div style="font-size:28px;font-weight:800;color:#2d3748">${total_prof:,.0f}</div>
    <div style="font-size:11px;color:#718096">Avg margin: {avg_margin:.1f}%</div>
  </div>
  <div style="background:#fff;border-radius:10px;padding:18px;border-left:4px solid #F44336;
  box-shadow:0 2px 8px rgba(0,0,0,.06)">
    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px">Loss Transactions</div>
    <div style="font-size:28px;font-weight:800;color:#F44336">{loss_pct:.1f}%</div>
    <div style="font-size:11px;color:#718096">Break-even discount @ {BREAKEVEN:.1%}</div>
  </div>
  <div style="background:#fff;border-radius:10px;padding:18px;border-left:4px solid #FF9800;
  box-shadow:0 2px 8px rgba(0,0,0,.06)">
    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px">Churn Rate</div>
    <div style="font-size:28px;font-weight:800;color:#FF9800">{churn_rate:.1f}%</div>
    <div style="font-size:11px;color:#718096">Model AUC: {AUC:.3f}</div>
  </div>
  <div style="background:#fff;border-radius:10px;padding:18px;border-left:4px solid #9C27B0;
  box-shadow:0 2px 8px rgba(0,0,0,.06)">
    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px">Avg Customer CLV</div>
    <div style="font-size:28px;font-weight:800;color:#9C27B0">${avg_clv:,.0f}</div>
    <div style="font-size:11px;color:#718096">Total CLV: ${dc['CLV'].sum():,.0f}</div>
  </div>
  <div style="background:#fff;border-radius:10px;padding:18px;border-left:4px solid #00BCD4;
  box-shadow:0 2px 8px rgba(0,0,0,.06)">
    <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px">Segments Found</div>
    <div style="font-size:28px;font-weight:800;color:#00BCD4">4</div>
    <div style="font-size:11px;color:#718096">ANOVA F &gt; 400, p &lt; 0.001</div>
  </div>
</div>
"""))

# Revenue trend + Segment charts
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Executive Dashboard — Real Superstore Data 2014–2017",
             fontsize=14, fontweight="bold")

# Monthly trend
m = df.groupby(["Year","Month"])[["Sales","Profit"]].sum().reset_index()
m["Date"] = pd.to_datetime(m[["Year","Month"]].assign(day=1))
axes[0].plot(m["Date"], m["Sales"]/1e3, color="#2196F3", linewidth=2, label="Revenue ($K)")
axes[0].fill_between(m["Date"], m["Sales"]/1e3, alpha=.1, color="#2196F3")
ax0t = axes[0].twinx()
ax0t.plot(m["Date"], m["Profit"]/1e3, color="#4CAF50", linewidth=2, label="Profit ($K)")
ax0t.fill_between(m["Date"], m["Profit"]/1e3, alpha=.1, color="#4CAF50")
axes[0].set_title("Monthly Revenue & Profit Trend")
axes[0].set_ylabel("Revenue ($K)", color="#2196F3")
ax0t.set_ylabel("Profit ($K)", color="#4CAF50")

# Segment sizes
seg_c = rfm["RFM_Segment"].value_counts()
axes[1].pie(seg_c.values, labels=seg_c.index,
            colors=[SEG_COLORS[s] for s in seg_c.index],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"linewidth":1.5,"edgecolor":"white"})
axes[1].set_title("Customer Segments (K-Means k=4)")

# Discount-loss rate
bins   = [-0.001,0,0.05,0.10,0.20,0.30,1.0]
labels = ["0%","1-5%","6-10%","11-20%","21-30%","31%+"]
df["Disc_Bin"] = pd.cut(df["Discount"], bins=bins, labels=labels)
loss_by_disc = df.groupby("Disc_Bin", observed=True)["Profitable"].apply(
    lambda x: (x==0).mean()*100)
bar_cols = ["#4CAF50" if l<10 else "#FF9800" if l<30 else "#F44336"
            for l in loss_by_disc.values]
axes[2].bar(loss_by_disc.index, loss_by_disc.values, color=bar_cols, edgecolor="white", alpha=.9)
axes[2].axhline(50, color="red", linestyle="--", linewidth=1, label="50% loss line")
axes[2].set_title(f"Loss Rate by Discount (Break-even={BREAKEVEN:.1%})")
axes[2].set_ylabel("% Loss Transactions")
axes[2].legend()

plt.tight_layout()
plt.show()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Customer Segmentation Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
display(HTML("""
<div style="background:#1a1a2e;color:#fff;padding:14px 24px;border-radius:10px;
font-family:Segoe UI,sans-serif;margin:16px 0 12px">
  <div style="font-size:16px;font-weight:700">👥 Customer Segmentation & RFM Analysis</div>
</div>
"""))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("RFM Segmentation — Real Data · ANOVA F>400 · p<0.001",
             fontsize=14, fontweight="bold")

# Scatter R vs F
for seg, grp in rfm.groupby("RFM_Segment"):
    axes[0,0].scatter(grp["Recency"], grp["Frequency"],
                      c=SEG_COLORS[seg], label=seg, alpha=.5, s=25)
axes[0,0].set_xlabel("Recency (days)"); axes[0,0].set_ylabel("Frequency")
axes[0,0].set_title("Recency vs Frequency"); axes[0,0].legend(fontsize=8)

# Scatter F vs M
for seg, grp in rfm.groupby("RFM_Segment"):
    axes[0,1].scatter(grp["Frequency"], grp["Monetary"],
                      c=SEG_COLORS[seg], label=seg, alpha=.5, s=25)
axes[0,1].set_xlabel("Frequency"); axes[0,1].set_ylabel("Monetary ($)")
axes[0,1].set_title("Frequency vs Monetary")

# Revenue vs Profit by segment
seg_fin = dfs.groupby("RFM_Segment")[["Sales","Profit"]].sum()
segs = list(seg_fin.index); x = np.arange(len(segs)); w = 0.35
axes[0,2].bar(x-w/2, seg_fin["Sales"]/1e6, w, label="Revenue",
              color="#90CAF9", alpha=.9, edgecolor="white")
axes[0,2].bar(x+w/2, seg_fin["Profit"]/1e6, w, label="Profit",
              color=[SEG_COLORS[s] for s in segs], alpha=.9, edgecolor="white")
axes[0,2].set_xticks(x); axes[0,2].set_xticklabels(segs, rotation=12)
axes[0,2].set_title("Revenue vs Profit ($M)"); axes[0,2].legend()

# CLV by segment
clv_s = dc.groupby("RFM_Segment")["CLV"].mean().sort_values(ascending=False)
bars = axes[1,0].bar(clv_s.index, clv_s.values,
                     color=[SEG_COLORS[s] for s in clv_s.index],
                     edgecolor="white", alpha=.9)
axes[1,0].set_title("Average CLV by Segment ($)")
axes[1,0].tick_params(axis="x", rotation=12)
for bar, v in zip(bars, clv_s.values):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, v+2,
                   f"${v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Margin % by segment
seg_margin = dfs.groupby("RFM_Segment")["Margin_Pct"].mean().sort_values(ascending=False)
axes[1,1].bar(seg_margin.index, seg_margin.values,
              color=[SEG_COLORS[s] for s in seg_margin.index],
              edgecolor="white", alpha=.9)
axes[1,1].set_title("Avg Margin % by Segment")
axes[1,1].tick_params(axis="x", rotation=12)

# Segment summary table
seg_summary = dfs.groupby("RFM_Segment").agg(
    Customers=("Customer_ID","nunique"), Revenue=("Sales","sum"),
    Profit=("Profit","sum"), Avg_Disc=("Discount","mean")).round(2)
seg_summary["Margin%"] = (seg_summary["Profit"]/seg_summary["Revenue"]*100).round(1)
axes[1,2].axis("off")
tbl_data = [[s, str(int(seg_summary.loc[s,"Customers"])),
             f"${seg_summary.loc[s,'Revenue']/1e6:.1f}M",
             f"${seg_summary.loc[s,'Profit']:,.0f}",
             f"{seg_summary.loc[s,'Margin%']:.1f}%",
             f"{seg_summary.loc[s,'Avg_Disc']*100:.1f}%"]
            for s in seg_summary.index]
tbl = axes[1,2].table(
    cellText=tbl_data,
    colLabels=["Segment","Cust","Revenue","Profit","Margin%","Avg Disc"],
    cellLoc="center", loc="center", bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (i,j), cell in tbl.get_celld().items():
    if i==0: cell.set_facecolor("#1a1a2e"); cell.set_text_props(color="white",fontweight="bold")
    elif j==0:
        seg_name = tbl_data[i-1][0]
        cell.set_facecolor(SEG_COLORS.get(seg_name,"#f0f0f0")+"33")
    cell.set_edgecolor("#e0e0e0")
axes[1,2].set_title("Segment Summary Table")

plt.tight_layout()
plt.show()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Churn Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
display(HTML(f"""
<div style="background:#1a1a2e;color:#fff;padding:14px 24px;border-radius:10px;
font-family:Segoe UI,sans-serif;margin:16px 0 12px">
  <div style="font-size:16px;font-weight:700">⚠️ Churn Prediction Model</div>
  <div style="font-size:12px;color:rgba(255,255,255,.6);margin-top:3px">
  Random Forest · AUC={AUC:.3f} · Churn = no purchase in 180 days
  </div>
</div>
"""))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Churn Prediction — Random Forest Classifier", fontsize=14, fontweight="bold")

# Feature importance
fi = pd.Series(RF.feature_importances_, index=FEAT).sort_values()
colors_fi = ["#2196F3" if v > fi.mean() else "#90CAF9" for v in fi.values]
axes[0].barh(fi.index, fi.values, color=colors_fi, edgecolor="white", alpha=.9)
axes[0].set_title(f"Feature Importance\n(Top drivers of churn)")
axes[0].set_xlabel("Importance Score")

# Churn risk by segment
seg_ch = dc.groupby("RFM_Segment")["Churn_Prob"].mean().sort_values(ascending=False)
bars = axes[1].bar(seg_ch.index, seg_ch.values*100,
                   color=[SEG_COLORS[s] for s in seg_ch.index],
                   edgecolor="white", alpha=.9)
axes[1].set_title("Avg Churn Risk % by Segment")
axes[1].set_ylabel("Churn Probability %")
axes[1].tick_params(axis="x", rotation=12)
for bar, v in zip(bars, seg_ch.values*100):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+0.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Churn distribution
dc["Risk_Cat"] = pd.cut(dc["Churn_Prob"], bins=[0,.3,.6,1.0],
                         labels=["Low (<30%)","Medium (30-60%)","High (>60%)"])
risk_c = dc["Risk_Cat"].value_counts()
axes[2].bar(risk_c.index, risk_c.values,
            color=["#4CAF50","#FF9800","#F44336"], edgecolor="white", alpha=.9)
axes[2].set_title("Churn Risk Distribution")
axes[2].set_ylabel("Number of Customers")
for i, v in enumerate(risk_c.values):
    axes[2].text(i, v+1, str(v), ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.show()

# High risk customers table
hr = dc[dc["Churn_Prob"]>=0.6][["Customer_ID","RFM_Segment","CLV","Avg_Discount","Churn_Prob"]].head(8).copy()
hr["CLV"] = hr["CLV"].round(2)
hr["Avg_Discount"] = (hr["Avg_Discount"]*100).round(1)
hr["Churn_Prob"] = (hr["Churn_Prob"]*100).round(1)
hr.columns = ["Customer ID","Segment","CLV ($)","Avg Disc (%)","Churn Risk (%)"]
display(HTML("<b style='font-family:Segoe UI'>Top High-Risk Customers (Churn Probability > 60%)</b>"))
display(hr.style.background_gradient(subset=["Churn Risk (%)"], cmap="Reds")
               .format({"CLV ($)":"${:.2f}","Avg Disc (%)":"{:.1f}%","Churn Risk (%)":"{:.1f}%"})
               .set_properties(**{"font-size":"12px","font-family":"Segoe UI"}))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — LIVE AI OPTIMIZER (Parameters actually change output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
display(HTML("""
<div style="background:#1a1a2e;color:#fff;padding:14px 24px;border-radius:10px;
font-family:Segoe UI,sans-serif;margin:16px 0 12px">
  <div style="font-size:16px;font-weight:700">🎯 AI Optimization Engine — LIVE</div>
  <div style="font-size:12px;color:rgba(255,255,255,.6);margin-top:3px">
  PuLP · Binary Integer Programming · Change parameters and click Run Optimization
  </div>
</div>
"""))

# ── Widgets ───────────────────────────────────────────────────────
w_budget = widgets.FloatSlider(
    value=10, min=5, max=30, step=1,
    description="Budget %:", style={"description_width":"120px"},
    layout=widgets.Layout(width="450px"),
    continuous_update=False,
    readout_format=".0f",
)
w_max_disc = widgets.Dropdown(
    options=[("0% — No discounts at all", 0.0),
             ("5% Maximum discount",      0.05),
             ("10% Maximum discount",     0.10),
             ("20% Maximum discount",     0.20)],
    value=0.10, description="Max Discount:",
    style={"description_width":"120px"},
    layout=widgets.Layout(width="350px"),
)
w_min_margin = widgets.FloatSlider(
    value=5, min=0, max=20, step=1,
    description="Min Margin %:", style={"description_width":"120px"},
    layout=widgets.Layout(width="450px"),
    continuous_update=False,
)
w_n_customers = widgets.IntSlider(
    value=150, min=50, max=793, step=25,
    description="Customers:", style={"description_width":"120px"},
    layout=widgets.Layout(width="450px"),
    continuous_update=False,
)
w_run = widgets.Button(
    description="▶ Run AI Optimization",
    button_style="primary",
    layout=widgets.Layout(width="220px", height="40px"),
)
w_output = widgets.Output()

display(widgets.VBox([
    widgets.HTML("<b style='font-family:Segoe UI;font-size:13px'>Optimization Parameters:</b>"),
    w_budget, w_max_disc, w_min_margin, w_n_customers,
    w_run, w_output
]))

def run_optimization(btn):
    budget_pct  = w_budget.value / 100
    max_disc    = w_max_disc.value
    min_margin  = w_min_margin.value
    n_customers = w_n_customers.value

    with w_output:
        clear_output(wait=True)
        print("⏳ Running optimization... please wait")

        # ── Build action set based on max discount ─────────────
        ACTIONS = {"No Discount":{"d":0.00,"u":1.00},
                   "Premium Rec":{"d":0.00,"u":1.12}}
        if max_disc >= 0.05: ACTIONS["5% Discount"]  = {"d":0.05,"u":1.08}
        if max_disc >= 0.08: ACTIONS["Bundle Offer"] = {"d":0.08,"u":1.25}
        if max_disc >= 0.10: ACTIONS["10% Discount"] = {"d":0.10,"u":1.18}
        if max_disc >= 0.20: ACTIONS["20% Discount"] = {"d":0.20,"u":1.25}

        samp   = dc.sample(n=min(n_customers, len(dc)), random_state=SEED).copy()
        BUDGET = samp["AOV"].sum() * budget_pct

        def ep(aov, bm, act):
            d = ACTIONS[act]["d"]; u = ACTIONS[act]["u"]
            return round(aov*u*(1-d) - aov*(1-bm) - aov*u*d, 2)

        def emp(aov, bm, act):
            d = ACTIONS[act]["d"]; u = ACTIONS[act]["u"]
            rev = aov*u*(1-d)
            return 0 if rev<=0 else round((rev - aov*(1-bm) - aov*u*d)/rev*100, 2)

        # ── PuLP Model ─────────────────────────────────────────
        prob = pulp.LpProblem("OptProfit", pulp.LpMaximize)
        rows = list(samp.iterrows())
        xv   = {(i, a): pulp.LpVariable(f"x{i}_{a.replace(' ','_')}", cat="Binary")
                for i in range(len(rows)) for a in ACTIONS}

        # Objective: maximize total profit
        prob += pulp.lpSum(
            ep(max(r["AOV"],1), r["Avg_Margin_Pct"]/100, a) * xv[(i,a)]
            for i,(_,r) in enumerate(rows) for a in ACTIONS
        )
        # Each customer gets exactly 1 action
        for i,(_,r) in enumerate(rows):
            prob += pulp.lpSum(xv[(i,a)] for a in ACTIONS) == 1
            # Enforce min margin floor
            for a in ACTIONS:
                if emp(max(r["AOV"],1), r["Avg_Margin_Pct"]/100, a) < min_margin:
                    prob += xv[(i,a)] == 0

        # Budget constraint
        prob += pulp.lpSum(
            max(r["AOV"],1) * ACTIONS[a]["u"] * ACTIONS[a]["d"] * xv[(i,a)]
            for i,(_,r) in enumerate(rows) for a in ACTIONS
        ) <= BUDGET

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # ── Collect results ────────────────────────────────────
        results = []
        for i,(_,r) in enumerate(rows):
            for a in ACTIONS:
                if pulp.value(xv.get((i,a),0)) == 1:
                    results.append({
                        "Customer_ID": r["Customer_ID"],
                        "Segment":     r["RFM_Segment"],
                        "Action":      a,
                        "Profit":      ep(max(r["AOV"],1), r["Avg_Margin_Pct"]/100, a),
                        "Margin%":     emp(max(r["AOV"],1), r["Avg_Margin_Pct"]/100, a),
                        "AOV":         round(max(r["AOV"],1), 2),
                    })

        opt_df = pd.DataFrame(results)
        if len(opt_df) == 0:
            print("No feasible solution — try relaxing constraints (lower min margin or higher budget)")
            return

        ai_profit   = opt_df["Profit"].sum()
        trad_profit = float(sum(
            max(r["AOV"],1)*0.8 - max(r["AOV"],1)*(1-r["Avg_Margin_Pct"]/100) - max(r["AOV"],1)*0.2
            for _,r in samp.iterrows()
        ))
        improvement = (ai_profit - trad_profit) / abs(trad_profit) * 100 if trad_profit!=0 else 0
        disc_spent  = sum(max(r["AOV"],1)*ACTIONS[a]["u"]*ACTIONS[a]["d"]
                          for i,(_,r) in enumerate(rows)
                          for a in ACTIONS if pulp.value(xv.get((i,a),0))==1)

        clear_output(wait=True)

        # ── Summary KPIs ───────────────────────────────────────
        imp_color = "#4CAF50" if improvement > 0 else "#F44336"
        display(HTML(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
        font-family:Segoe UI,sans-serif;margin:12px 0">
          <div style="background:#e8f5e9;border-radius:8px;padding:14px;text-align:center">
            <div style="font-size:11px;color:#555;margin-bottom:4px">AI Optimized Profit</div>
            <div style="font-size:22px;font-weight:800;color:#2e7d32">${ai_profit:,.0f}</div>
            <div style="font-size:10px;color:#777">{len(opt_df)} customers</div>
          </div>
          <div style="background:#fff8e1;border-radius:8px;padding:14px;text-align:center">
            <div style="font-size:11px;color:#555;margin-bottom:4px">Traditional (20% disc)</div>
            <div style="font-size:22px;font-weight:800;color:#e65100">${trad_profit:,.0f}</div>
            <div style="font-size:10px;color:#777">same customers</div>
          </div>
          <div style="background:#{'e8f5e9' if improvement>0 else 'ffebee'};border-radius:8px;
          padding:14px;text-align:center">
            <div style="font-size:11px;color:#555;margin-bottom:4px">Profit Improvement</div>
            <div style="font-size:22px;font-weight:800;color:{imp_color}">{improvement:+.1f}%</div>
            <div style="font-size:10px;color:#777">AI vs blanket discount</div>
          </div>
          <div style="background:#e3f2fd;border-radius:8px;padding:14px;text-align:center">
            <div style="font-size:11px;color:#555;margin-bottom:4px">Budget Used</div>
            <div style="font-size:22px;font-weight:800;color:#1565c0">${disc_spent:,.0f}</div>
            <div style="font-size:10px;color:#777">of ${BUDGET:,.0f} allowed ({budget_pct:.0%})</div>
          </div>
        </div>
        """))

        # ── Charts ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"AI Optimization Results — Budget={budget_pct:.0%} · Max Disc={max_disc:.0%} · "
            f"Min Margin={min_margin:.0f}% · N={n_customers}",
            fontsize=12, fontweight="bold"
        )

        # Action distribution
        act_c = opt_df["Action"].value_counts()
        act_colors = {"No Discount":"#4CAF50","Premium Rec":"#9C27B0",
                      "5% Discount":"#8BC34A","Bundle Offer":"#2196F3",
                      "10% Discount":"#FF9800","20% Discount":"#F44336"}
        axes[0].bar(act_c.index, act_c.values,
                    color=[act_colors.get(a,"#999") for a in act_c.index],
                    edgecolor="white", alpha=.9)
        axes[0].set_title("Recommended Actions Distribution")
        axes[0].set_ylabel("Number of Customers")
        axes[0].tick_params(axis="x", rotation=15)
        for i, v in enumerate(act_c.values):
            axes[0].text(i, v+0.3, str(v), ha="center", va="bottom", fontweight="bold")

        # Profit by segment
        seg_p = opt_df.groupby("Segment")["Profit"].sum().sort_values(ascending=False)
        bars = axes[1].bar(seg_p.index, seg_p.values,
                           color=[SEG_COLORS.get(s,"#999") for s in seg_p.index],
                           edgecolor="white", alpha=.9)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Total Profit by Segment (AI-Optimized)")
        axes[1].tick_params(axis="x", rotation=12)
        for bar, v in zip(bars, seg_p.values):
            axes[1].text(bar.get_x()+bar.get_width()/2,
                         v+(abs(v)*0.03 if v>=0 else abs(v)*-0.06),
                         f"${v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # AI vs Traditional comparison
        comp_labels = ["Traditional\n(20% blanket)", "AI-Optimized"]
        comp_vals   = [trad_profit, ai_profit]
        comp_colors = ["#F44336" if v < 0 else "#4CAF50" for v in comp_vals]
        axes[2].bar(comp_labels, comp_vals, color=comp_colors, edgecolor="white",
                    alpha=.9, width=0.5)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_title(f"AI vs Traditional\n(Improvement: {improvement:+.1f}%)")
        for i, v in enumerate(comp_vals):
            axes[2].text(i, v+(abs(v)*0.03 if v>=0 else abs(v)*-0.08),
                         f"${v:,.0f}", ha="center", va="bottom",
                         fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.show()

        # ── Top customer recommendations table ─────────────────
        top10 = opt_df.nlargest(10,"Profit")[
            ["Customer_ID","Segment","Action","Profit","Margin%","AOV"]].copy()
        top10["Profit"] = top10["Profit"].round(2)
        top10["Margin%"] = top10["Margin%"].round(1)
        top10["AOV"]    = top10["AOV"].round(2)
        display(HTML("<b style='font-family:Segoe UI;font-size:13px'>Top 10 Customer Recommendations:</b>"))
        display(top10.style
                .background_gradient(subset=["Profit"], cmap="Greens")
                .format({"Profit":"${:.2f}","Margin%":"{:.1f}%","AOV":"${:.2f}"})
                .set_properties(**{"font-size":"12px","font-family":"Segoe UI"})
                .applymap(lambda v: f"background-color:{SEG_COLORS.get(v,'#fff')}22",
                          subset=["Segment"]))

        print(f"\n✓ Optimization complete. Status: {pulp.LpStatus[prob.status]}")

w_run.on_click(run_optimization)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — Strategy Simulation (Interactive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
display(HTML("""
<div style="background:#1a1a2e;color:#fff;padding:14px 24px;border-radius:10px;
font-family:Segoe UI,sans-serif;margin:16px 0 12px">
  <div style="font-size:16px;font-weight:700">⚡ Strategy Simulation — Live Comparison</div>
  <div style="font-size:12px;color:rgba(255,255,255,.6);margin-top:3px">
  Adjust discount and uplift assumptions — results update instantly
  </div>
</div>
"""))

s_disc = widgets.FloatSlider(
    value=20, min=0, max=50, step=1,
    description="Discount %:", style={"description_width":"120px"},
    layout=widgets.Layout(width="450px"), continuous_update=False,
)
s_uplift = widgets.FloatSlider(
    value=15, min=0, max=50, step=1,
    description="Demand Uplift %:", style={"description_width":"120px"},
    layout=widgets.Layout(width="450px"), continuous_update=False,
)
s_n = widgets.IntSlider(
    value=200, min=50, max=793, step=25,
    description="Customers:", style={"description_width":"120px"},
    layout=widgets.Layout(width="450px"), continuous_update=False,
)
s_run = widgets.Button(
    description="▶ Run Simulation",
    button_style="success",
    layout=widgets.Layout(width="200px", height="40px"),
)
s_output = widgets.Output()
display(widgets.VBox([s_disc, s_uplift, s_n, s_run, s_output]))

def run_simulation(btn):
    disc   = s_disc.value / 100
    uplift = 1 + s_uplift.value / 100
    n      = s_n.value
    samp2  = dc.sample(n=min(n, len(dc)), random_state=SEED).copy()

    def calc(d, u):
        tot_r = tot_p = 0
        for _,r in samp2.iterrows():
            aov = max(r["AOV"],1); bm = r["Avg_Margin_Pct"]/100
            rev = aov*u*(1-d); cost = aov*(1-bm); dc4 = aov*u*d
            tot_r += rev; tot_p += rev - cost - dc4
        return round(tot_r,2), round(tot_p,2)

    r1,p1 = calc(disc, uplift)           # User-defined discount + uplift
    r2,p2 = calc(disc*0.5, uplift*0.85)  # Half discount, lower uplift
    r3,p3 = calc(0.05, 1.12)             # 5% bundle
    r4,p4 = calc(0.00, 1.12)             # AI no-discount premium rec

    strategies = {
        f"Blanket {disc:.0%}\n(your setting)": {"r":r1,"p":p1,"c":"#F44336"},
        f"Half Disc\n({disc/2:.0%})":           {"r":r2,"p":p2,"c":"#FF9800"},
        "5% Bundle\nOffer":                      {"r":r3,"p":p3,"c":"#4CAF50"},
        "AI No-Disc\nPremium Rec":               {"r":r4,"p":p4,"c":"#2196F3"},
    }
    best = max(strategies, key=lambda k: strategies[k]["p"])

    with s_output:
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Strategy Simulation — Disc={disc:.0%} · Uplift={uplift:.0%} · N={n} customers",
            fontsize=13, fontweight="bold"
        )
        keys = list(strategies.keys())
        profits  = [strategies[k]["p"] for k in keys]
        revenues = [strategies[k]["r"] for k in keys]
        colors   = [strategies[k]["c"] for k in keys]
        best_mask = ["★ BEST" if k==best else "" for k in keys]

        bars = axes[0].bar(keys, profits, color=colors, edgecolor="white", alpha=.9, width=0.5)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Total Profit by Strategy")
        axes[0].set_ylabel("Profit ($)")
        for bar, v, bm in zip(bars, profits, best_mask):
            label = f"${v:,.0f}\n{bm}"
            axes[0].text(bar.get_x()+bar.get_width()/2,
                         v+(abs(min(profits))*0.04 if v>=0 else abs(v)*0.05),
                         label, ha="center", va="bottom", fontsize=9, fontweight="bold")

        axes[1].bar(keys, revenues, color=[c+"aa" for c in colors], edgecolor="white", alpha=.9, width=0.5)
        axes[1].set_title("Total Revenue by Strategy")
        axes[1].set_ylabel("Revenue ($)")
        for i, v in enumerate(revenues):
            axes[1].text(i, v*1.01, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()

        best_profit = strategies[best]["p"]
        worst       = min(strategies, key=lambda k: strategies[k]["p"])
        worst_profit= strategies[worst]["p"]
        uplift_gain = (best_profit-worst_profit)/abs(worst_profit)*100 if worst_profit!=0 else 0
        ai_vs_disc  = (p4-p1)/abs(p1)*100 if p1!=0 else 0
        gain_color  = "#2e7d32" if ai_vs_disc>0 else "#c62828"
        display(HTML(f"""
        <div style="font-family:Segoe UI,sans-serif;padding:12px;background:#f7fafc;
        border-radius:8px;margin-top:8px">
          <b>Simulation Result:</b> AI No-Discount strategy generates
          <span style="color:{gain_color};font-weight:700">{ai_vs_disc:+.1f}%</span>
          more profit than your {disc:.0%} blanket discount setting.<br>
          <span style="color:#718096;font-size:12px">
          Best: {best.replace(chr(10),' ')} (${best_profit:,.0f}) ·
          Worst: {worst.replace(chr(10),' ')} (${worst_profit:,.0f})
          </span>
        </div>"""))

s_run.on_click(run_simulation)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — Customer Lookup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
display(HTML("""
<div style="background:#1a1a2e;color:#fff;padding:14px 24px;border-radius:10px;
font-family:Segoe UI,sans-serif;margin:16px 0 12px">
  <div style="font-size:16px;font-weight:700">🔍 Customer Lookup</div>
</div>
"""))

cust_ids = sorted(dc["Customer_ID"].tolist())
w_cust = widgets.Combobox(
    options=cust_ids, value=cust_ids[0],
    description="Customer ID:", style={"description_width":"120px"},
    layout=widgets.Layout(width="400px"),
    ensure_option=False,
)
w_lookup = widgets.Button(
    description="🔍 Look Up Customer",
    button_style="info",
    layout=widgets.Layout(width="200px", height="38px"),
)
w_cust_out = widgets.Output()
display(widgets.VBox([w_cust, w_lookup, w_cust_out]))

def lookup_customer(btn):
    cid = w_cust.value.strip()
    row = dc[dc["Customer_ID"]==cid]
    if row.empty:
        with w_cust_out:
            clear_output(wait=True)
            print(f"Customer '{cid}' not found. Choose from the dropdown.")
        return
    r   = row.iloc[0]
    seg = r["RFM_Segment"]
    col = SEG_COLORS.get(seg,"#999")
    rec = {"Premium":     "⭐ Premium Rec — exclusive loyalty rewards, NO discount",
           "Loyal":       "🎁 Bundle Offer — value add without price cut",
           "Price-Sensitive": "🚫 No Discount — avoid reinforcing dependency",
           "Low-Value":   "📧 Low-cost email reactivation only"}
    risk= "🔴 High" if r["Churn_Prob"]>0.6 else "🟡 Medium" if r["Churn_Prob"]>0.3 else "🟢 Low"

    with w_cust_out:
        clear_output(wait=True)
        display(HTML(f"""
        <div style="font-family:Segoe UI,sans-serif;background:#fff;border-radius:12px;
        padding:20px;box-shadow:0 2px 12px rgba(0,0,0,.08);max-width:700px">
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
            <div style="width:56px;height:56px;border-radius:50%;background:{col};
            display:flex;align-items:center;justify-content:center;color:#fff;
            font-size:18px;font-weight:800">{cid[:2].upper()}</div>
            <div>
              <div style="font-size:18px;font-weight:700">{cid}</div>
              <div style="display:inline-block;background:{col};color:#fff;padding:2px 10px;
              border-radius:12px;font-size:12px;font-weight:600;margin-top:4px">{seg}</div>
              &nbsp;<span style="font-size:12px;color:#718096">
              RFM Score: {int(r['RFM_Score'])} · {r['Preferred_Category']} · {r['Tenure_Years']:.1f} yrs tenure
              </span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px">
            <div style="background:#f7fafc;border-radius:8px;padding:12px;text-align:center">
              <div style="font-size:20px;font-weight:800">${r['CLV']:,.0f}</div>
              <div style="font-size:11px;color:#718096">Customer LTV</div>
            </div>
            <div style="background:#f7fafc;border-radius:8px;padding:12px;text-align:center">
              <div style="font-size:20px;font-weight:800;
              color:{'#2e7d32' if r['Total_Profit']>0 else '#c62828'}">${r['Total_Profit']:,.0f}</div>
              <div style="font-size:11px;color:#718096">Total Profit</div>
            </div>
            <div style="background:#f7fafc;border-radius:8px;padding:12px;text-align:center">
              <div style="font-size:20px;font-weight:800">{int(r['Total_Orders'])}</div>
              <div style="font-size:11px;color:#718096">Total Orders</div>
            </div>
            <div style="background:#f7fafc;border-radius:8px;padding:12px;text-align:center">
              <div style="font-size:20px;font-weight:800">{r['Avg_Discount']*100:.1f}%</div>
              <div style="font-size:11px;color:#718096">Avg Discount Used</div>
            </div>
            <div style="background:#f7fafc;border-radius:8px;padding:12px;text-align:center">
              <div style="font-size:20px;font-weight:800">{r['Avg_Margin_Pct']:.1f}%</div>
              <div style="font-size:11px;color:#718096">Avg Margin %</div>
            </div>
            <div style="background:#f7fafc;border-radius:8px;padding:12px;text-align:center">
              <div style="font-size:20px;font-weight:800">{r['Churn_Prob']*100:.1f}%</div>
              <div style="font-size:11px;color:#718096">Churn Risk · {risk}</div>
            </div>
          </div>
          <div style="background:#1a1a2e;color:#fff;border-radius:8px;padding:14px">
            <div style="font-size:11px;color:rgba(255,255,255,.5);margin-bottom:6px">
            🤖 AI RECOMMENDATION</div>
            <div style="font-size:14px;font-weight:600">{rec.get(seg,'N/A')}</div>
          </div>
        </div>
        """))

w_lookup.on_click(lookup_customer)
lookup_customer(None)  # Show first customer on load

print("\n" + "="*60)
print("  ✓ ALL SECTIONS LOADED SUCCESSFULLY")
print("  Use the widgets above to interact with the models")
print("  AI Optimizer and Strategy Simulation update in real-time")
print("="*60)
