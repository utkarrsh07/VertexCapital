"""VertexCapital Terminal v1 — local research dashboard. Run with streamlit."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "recommendations"
PORT = BASE / "strategy_v2" / "portfolio"
ANALYTICS = BASE / "strategy_v2" / "analytics"

st.set_page_config(page_title="VertexCapital | Research Terminal", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
:root{color-scheme:dark}
.stApp{background:#080c13;color:#e8edf5}
[data-testid="stSidebar"]{background:#101722}
[data-testid="stMetric"]{background:#111a26;border:1px solid #233042;border-radius:12px;padding:18px}
h1,h2,h3{letter-spacing:-.035em}
div[data-testid="stDataFrame"]{border:1px solid #253244;border-radius:10px}
.stTabs [data-baseweb="tab"]{font-weight:650}
</style>""", unsafe_allow_html=True)

def read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return {}

def fmt_pct(x):
    try: return f"{float(x):+.2%}"
    except (ValueError,TypeError): return "—"

def fmt_money(x):
    try: return f"${float(x):,.2f}"
    except (ValueError,TypeError): return "—"

def line_plot(df, x, series, title, percent=False):
    fig=go.Figure()
    palette=["#F3A84D","#69A9EF","#6FD3AD","#BBA1F2"]
    for i,(column,label) in enumerate(series):
        if column in df:
            fig.add_trace(go.Scatter(x=df[x],y=df[column],name=label,mode="lines",
                                     line=dict(width=2.5,color=palette[i%len(palette)])))
    fig.update_layout(title=title,template="plotly_dark",paper_bgcolor="#101722",
                      plot_bgcolor="#101722",font=dict(color="#cdd7e6"),
                      margin=dict(l=25,r=25,t=55,b=25),height=385,
                      legend=dict(orientation="h",y=1.12),hovermode="x unified")
    if percent: fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig,use_container_width=True)

st.sidebar.markdown("## ◆ VERTEXCAPITAL")
st.sidebar.caption("QUANTITATIVE RESEARCH TERMINAL")
section=st.sidebar.radio("Workspace",["Overview","Trade Recommendations","Strategy Analytics","Trade History","Data & Disclosures"])
st.sidebar.divider()
st.sidebar.info("RESEARCH ONLY · Not investment advice · No brokerage connectivity")
if st.sidebar.button("Refresh local data"):
    st.rerun()

summary=read_json(PORT/"summary.json")
equity=read_csv(PORT/"equity_curve.csv")
executed=read_csv(PORT/"executed_trades.csv")
rejected=read_csv(PORT/"rejected_trades.csv")
audit=read_csv(BASE/"strategy_v2"/"trade_audit.csv")
recommendations=[]
for path in [BASE/"trade_plans.csv",BASE/"current_trade_plans.csv",BASE/"recommendations.csv"]:
    if path.exists(): recommendations.append(path)
if not recommendations and BASE.exists():
    recommendations=[f for f in BASE.glob("*.csv") if "plan" in f.name.lower() or "recommend" in f.name.lower()]
rec=read_csv(recommendations[0]) if recommendations else pd.DataFrame()

st.markdown("# ◆ VertexCapital")
st.caption("Institutional-style research workspace  /  Local data snapshot  /  Not live market data")
st.warning("Exploratory simulations only. Historical prediction provenance, survivorship bias, trade execution, currency treatment and costs remain unverified. Do not use this dashboard as a live trading signal.",icon="⚠️")

if section=="Overview":
    cols=st.columns(4)
    vals=[("Portfolio value",fmt_money(summary.get("final_value"))),
          ("Simulated return",fmt_pct(summary.get("total_return"))),
          ("SPY return",fmt_pct(summary.get("benchmark_return"))),
          ("Maximum drawdown",fmt_pct(summary.get("maximum_drawdown")))]
    for c,(name,value) in zip(cols,vals): c.metric(name,value)
    if not equity.empty and {"date","portfolio_value","benchmark_value"}.issubset(equity.columns):
        equity["date"]=pd.to_datetime(equity["date"],errors="coerce")
        line_plot(equity,"date",[("portfolio_value","VertexCapital"),("benchmark_value","SPY buy & hold")],"Simulated equity vs benchmark")
    else:
        st.info("Portfolio curve missing. Run: python backend/portfolio_simulation_v2.py")
    left,right=st.columns([1,1])
    with left:
        st.subheader("Execution summary")
        st.metric("Executed trades",summary.get("executed_trades","—"))
        st.metric("Rejected trades",summary.get("rejected_trades","—"))
    with right:
        st.subheader("Research period")
        st.write(f"**Start:** {summary.get('start','—')}")
        st.write(f"**End:** {summary.get('end','—')}")
        st.write(f"**Peak concurrent positions:** {summary.get('max_concurrent_positions','—')}")
elif section=="Trade Recommendations":
    st.subheader("Current research scenarios")
    st.caption("These are proposed levels, not calibrated price forecasts. No trade may be appropriate.")
    if rec.empty: st.info("No current recommendation CSV detected. Run: python backend/recommendations.py")
    else:
        cols=rec.columns.tolist()
        ticker_col="ticker" if "ticker" in cols else None
        if ticker_col:
            search=st.text_input("Filter ticker",placeholder="e.g. AAPL")
            if search: rec=rec[rec[ticker_col].astype(str).str.contains(search,case=False,regex=False)]
        st.dataframe(rec,use_container_width=True,hide_index=True)
        st.download_button("Export scenarios CSV",rec.to_csv(index=False).encode(),"vertex_recommendations.csv","text/csv")
elif section=="Strategy Analytics":
    st.subheader("Strategy-level diagnostics")
    analytics_files=list(ANALYTICS.glob("*.json")) if ANALYTICS.exists() else []
    for f in analytics_files:
        obj=read_json(f)
        if isinstance(obj,dict) and obj:
            st.caption(f.name)
            st.json(obj,expanded=False)
    if not equity.empty and "drawdown" in equity:
        equity["date"]=pd.to_datetime(equity["date"],errors="coerce")
        line_plot(equity,"date",[("drawdown","Drawdown")],"Portfolio drawdown",True)
    if not executed.empty and "pnl" in executed:
        st.subheader("Simulated realized P&L by ticker")
        agg=executed.groupby("ticker",as_index=False)["pnl"].sum().sort_values("pnl",ascending=False)
        fig=go.Figure(go.Bar(x=agg["ticker"],y=agg["pnl"],marker_color="#F3A84D"))
        fig.update_layout(template="plotly_dark",paper_bgcolor="#101722",plot_bgcolor="#101722",height=350)
        st.plotly_chart(fig,use_container_width=True)
elif section=="Trade History":
    st.subheader("Executed simulated trades")
    if executed.empty: st.info("No portfolio execution records found.")
    else:
        st.dataframe(executed,use_container_width=True,hide_index=True)
        st.download_button("Export trade history",executed.to_csv(index=False).encode(),"vertex_executed_trades.csv","text/csv")
    st.subheader("Rejected simulated trades")
    if not rejected.empty: st.dataframe(rejected,use_container_width=True,hide_index=True)
    st.subheader("Historical trade audit")
    if not audit.empty: st.dataframe(audit,use_container_width=True,hide_index=True)
elif section=="Data & Disclosures":
    st.subheader("Local data sources")
    for label,path in [("Portfolio summary",PORT/"summary.json"),
                       ("Equity curve",PORT/"equity_curve.csv"),
                       ("Executed trades",PORT/"executed_trades.csv"),
                       ("Historical audit",BASE/"strategy_v2"/"trade_audit.csv")]:
        st.write(("✓" if path.exists() else "✗"),label,"—",str(path.relative_to(ROOT)))
    st.subheader("Research limitations")
    st.write("This terminal reads local CSV/JSON files and does not execute trades. It is not a live portfolio tracker. Reconstructed performance can differ materially from executable returns because of fills, corporate actions, FX, costs, data quality, overlapping signals, and sample selection.")
    if summary: st.json(summary)
