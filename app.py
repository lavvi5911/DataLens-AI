import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from datetime import datetime

from utils.data_loader import load_dataset, validate_dataset
from utils.analyzer import DataAnalyzer
from utils.detector import IssueDetector
from utils.recommender import RecommendationEngine
from utils.explainer import InsightExplainer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens AI · Dataset Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def _plotly_layout():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="DM Sans, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
    )


def _build_report(df, issues, recs, health_score):
    lines = [
        "=" * 60,
        "  DataLens AI — Dataset Diagnostic Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"DATASET HEALTH SCORE: {health_score}/100",
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB",
        "",
        "── ISSUES DETECTED ──────────────────────────────────────",
    ]
    if issues:
        sorder = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for iss in sorted(issues, key=lambda x: sorder.get(x["severity"], 9)):
            lines.append(f"\n[{iss['severity'].upper()}] {iss['title']}")
            lines.append(f"  {iss['description']}")
            lines.append(f"  Fix: {iss['fix']}")
    else:
        lines.append("No significant issues detected.")
    lines += ["", "── RECOMMENDATIONS ──────────────────────────────────────"]
    for section, items in recs.items():
        if items:
            lines.append(f"\n{section.replace('_', ' ').upper()}")
            for item in items:
                lines.append(f"  * {item['title']}")
                lines.append(f"    {item['detail']}")
    lines += ["", "=" * 60, "End of Report", "=" * 60]
    return "\n".join(lines)


# ── Session state ─────────────────────────────────────────────────────────────
for _k in ["df", "analysis", "issues", "recommendations", "insights", "health_score"]:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🔬</span>
        <span class="logo-text">DataLens<span class="logo-accent">AI</span></span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    nav = st.radio(
        "",
        ["📤 Upload Dataset", "📊 Data Overview", "🚨 Issues Detected",
         "🧠 AI Insights", "💡 Recommendations"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if st.session_state.health_score is not None:
        score = st.session_state.health_score
        color = "#22c55e" if score >= 75 else "#f59e0b" if score >= 50 else "#ef4444"
        st.markdown(f"""
        <div class="health-widget">
            <div class="health-label">Dataset Health Score</div>
            <div class="health-score" style="color:{color}">{score}</div>
            <div class="health-bar-bg">
                <div class="health-bar-fill" style="width:{score}%;background:{color}"></div>
            </div>
            <div class="health-note">{"Excellent" if score >= 75 else "Needs Work" if score >= 50 else "Critical Issues"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-footer">v1.0 · Built with ❤️</p>', unsafe_allow_html=True)


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-bg"></div>
    <h1 class="hero-title">AI Dataset Debugger</h1>
    <p class="hero-sub">Self-Diagnosing Data Intelligence · Uncover Hidden Issues Instantly</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "📤 Upload Dataset":
    st.markdown('<div class="section-title">Upload Your Dataset</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            '<div class="upload-zone-label">Drop a CSV file to begin deep analysis</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

        if uploaded:
            with st.spinner("Analyzing your dataset…"):
                df, error = load_dataset(uploaded)
                if error:
                    st.markdown(f'<div class="error-box">⚠️ {error}</div>', unsafe_allow_html=True)
                else:
                    valid, msg = validate_dataset(df)
                    if not valid:
                        st.markdown(f'<div class="error-box">⚠️ {msg}</div>', unsafe_allow_html=True)
                    else:
                        st.session_state.df = df
                        analyzer    = DataAnalyzer(df)
                        detector    = IssueDetector(df)
                        recommender = RecommendationEngine(df, detector)
                        explainer   = InsightExplainer(df, detector, analyzer)

                        st.session_state.analysis        = analyzer.run()
                        st.session_state.issues          = detector.detect_all()
                        st.session_state.recommendations = recommender.generate()
                        st.session_state.insights        = explainer.generate()
                        st.session_state.health_score    = detector.health_score()

                        st.markdown(f"""
                        <div class="success-box">
                            ✅ Dataset loaded —
                            <strong>{df.shape[0]:,}</strong> rows ×
                            <strong>{df.shape[1]}</strong> columns.
                            Navigate using the sidebar.
                        </div>
                        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">What DataLens Detects</div>
            <ul class="feature-list">
                <li>🔴 Missing values &amp; patterns</li>
                <li>🟠 Outliers (IQR method)</li>
                <li>🟡 Class imbalance</li>
                <li>🔵 Feature redundancy</li>
                <li>🟣 Multicollinearity</li>
                <li>⚡ Data leakage risk</li>
                <li>📉 Skewness &amp; distribution</li>
                <li>🔁 Duplicate rows</li>
                <li>🎯 High-cardinality columns</li>
                <li>📊 Low-variance features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <div class="empty-text">No dataset loaded yet</div>
            <div class="empty-sub">Upload a CSV file above to begin</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Data Overview":
    if st.session_state.df is None:
        st.markdown('<div class="warn-box">📤 Please upload a dataset first.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.df
        st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)

        n_rows, n_cols = df.shape
        n_numeric  = df.select_dtypes(include=np.number).shape[1]
        n_missing  = int(df.isnull().sum().sum())
        miss_pct   = round(n_missing / (n_rows * n_cols) * 100, 1)

        c1, c2, c3, c4 = st.columns(4)
        for col_ui, label, value, icon in [
            (c1, "Rows",             f"{n_rows:,}",                 "📋"),
            (c2, "Columns",          str(n_cols),                   "🗂️"),
            (c3, "Numeric Features", str(n_numeric),                "🔢"),
            (c4, "Missing Cells",    f"{n_missing:,} ({miss_pct}%)", "🕳️"),
        ]:
            with col_ui:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["📋 Sample Data", "📈 Statistics", "🌡️ Correlation", "📊 Distributions"])

        with t1:
            st.markdown('<div class="tab-subtitle">First 50 rows of your dataset</div>', unsafe_allow_html=True)
            st.dataframe(df.head(50), use_container_width=True)

        with t2:
            st.markdown('<div class="tab-subtitle">Descriptive statistics for all features</div>', unsafe_allow_html=True)
            desc = df.describe(include="all").T.reset_index().rename(columns={"index": "Feature"})
            st.dataframe(desc, use_container_width=True)

        with t3:
            num_df = df.select_dtypes(include=np.number)
            if num_df.shape[1] < 2:
                st.info("Need at least 2 numeric columns for a correlation matrix.")
            else:
                corr = num_df.corr()
                fig  = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Feature Correlation Matrix",
                )
                fig.update_layout(**_plotly_layout())
                st.plotly_chart(fig, use_container_width=True)

        with t4:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available.")
            else:
                chosen   = st.selectbox("Select feature", num_cols)
                col_data = df[chosen].dropna()
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Box Plot"])
                fig.add_trace(go.Histogram(x=col_data, name="Histogram",
                                           marker_color="#6366f1", opacity=0.8), row=1, col=1)
                fig.add_trace(go.Box(y=col_data, name="Boxplot",
                                     marker_color="#8b5cf6", line_color="#a78bfa"), row=1, col=2)
                fig.update_layout(title=f"Distribution of {chosen}", **_plotly_layout())
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ISSUES DETECTED
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🚨 Issues Detected":
    if st.session_state.issues is None:
        st.markdown('<div class="warn-box">📤 Please upload a dataset first.</div>', unsafe_allow_html=True)
    else:
        issues = st.session_state.issues
        df     = st.session_state.df

        st.markdown('<div class="section-title">Issues Detected</div>', unsafe_allow_html=True)

        sorder  = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        scolors = {"critical": "#ef4444", "high": "#f97316", "medium": "#f59e0b", "low": "#22c55e"}
        sicons  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

        if not issues:
            st.markdown(
                '<div class="success-box">🎉 No significant issues detected! Your dataset looks clean.</div>',
                unsafe_allow_html=True,
            )
        else:
            counts = {s: sum(1 for i in issues if i["severity"] == s)
                      for s in ["critical", "high", "medium", "low"]}
            badge_html = "".join(
                f'<span class="badge" style="background:{scolors[s]}">'
                f'{sicons[s]} {counts[s]} {s.capitalize()}</span>'
                for s in ["critical", "high", "medium", "low"] if counts[s]
            )
            st.markdown(f'<div class="badge-row">{badge_html}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            for iss in sorted(issues, key=lambda x: sorder.get(x["severity"], 9)):
                sev = iss["severity"]
                c   = scolors[sev]
                ico = sicons[sev]
                st.markdown(f"""
                <div class="issue-card" style="border-left:4px solid {c}">
                    <div class="issue-header">
                        <span class="issue-icon">{ico}</span>
                        <span class="issue-title">{iss['title']}</span>
                        <span class="issue-badge" style="background:{c}22;color:{c}">{sev.upper()}</span>
                    </div>
                    <div class="issue-desc">{iss['description']}</div>
                    <div class="issue-fix">💡 <strong>Fix:</strong> {iss['fix']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Missing values chart
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=True)
            if len(missing):
                st.markdown('<div class="chart-title">Missing Values by Feature</div>', unsafe_allow_html=True)
                bcolors = [
                    "#ef4444" if v / len(df) > 0.3 else "#f59e0b" if v / len(df) > 0.1 else "#6366f1"
                    for v in missing.values
                ]
                fig = go.Figure(go.Bar(
                    x=missing.values, y=missing.index, orientation="h",
                    marker_color=bcolors,
                    text=[f"{v / len(df) * 100:.1f}%" for v in missing.values],
                    textposition="outside",
                ))
                fig.update_layout(title="Missing Values per Column",
                                  xaxis_title="Missing Count", yaxis_title="",
                                  **_plotly_layout())
                st.plotly_chart(fig, use_container_width=True)

            # Outlier boxplots
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                st.markdown('<div class="chart-title">Outlier Analysis</div>', unsafe_allow_html=True)
                fig = go.Figure()
                for c in num_cols[:8]:
                    fig.add_trace(go.Box(y=df[c].dropna(), name=c,
                                         boxpoints="outliers",
                                         marker_color="#8b5cf6", line_color="#a78bfa"))
                fig.update_layout(title="Boxplots — Outlier Detection", **_plotly_layout())
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🧠 AI Insights":
    if st.session_state.insights is None:
        st.markdown('<div class="warn-box">📤 Please upload a dataset first.</div>', unsafe_allow_html=True)
    else:
        insights = st.session_state.insights
        df       = st.session_state.df

        st.markdown('<div class="section-title">AI Insights</div>', unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-sub\">Plain-English explanations of what's happening in your data</div>",
            unsafe_allow_html=True,
        )

        tcolors = {"info": "#6366f1", "warning": "#f59e0b", "success": "#22c55e", "error": "#ef4444"}
        ticons  = {"info": "💡", "warning": "⚠️", "success": "✅", "error": "🔴"}

        for ins in insights:
            t   = ins.get("type", "info")
            c   = tcolors.get(t, "#6366f1")
            ico = ticons.get(t, "💡")
            act = f'<div class="insight-action">→ {ins["action"]}</div>' if ins.get("action") else ""
            st.markdown(f"""
            <div class="insight-card" style="border-top:3px solid {c}">
                <div class="insight-header">{ico} {ins['title']}</div>
                <div class="insight-body">{ins['body']}</div>
                {act}
            </div>
            """, unsafe_allow_html=True)

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.markdown('<div class="chart-title">Categorical Feature Distribution</div>', unsafe_allow_html=True)
            chosen_cat = st.selectbox("Select column", cat_cols, key="cat_sel")
            vc = df[chosen_cat].value_counts().head(20)
            fig = px.bar(x=vc.index, y=vc.values,
                         labels={"x": chosen_cat, "y": "Count"},
                         color=vc.values.tolist(),
                         color_continuous_scale="Purpor")
            fig.update_layout(title=f"Value Counts: {chosen_cat}", **_plotly_layout())
            st.plotly_chart(fig, use_container_width=True)

        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            skew = num_df.skew().sort_values(key=abs, ascending=False)
            bcolors = [
                "#ef4444" if abs(v) > 2 else "#f59e0b" if abs(v) > 1 else "#22c55e"
                for v in skew.values
            ]
            fig = go.Figure(go.Bar(
                x=skew.index, y=skew.values,
                marker_color=bcolors,
                text=[f"{v:.2f}" for v in skew.values],
                textposition="outside",
            ))
            for thresh, color, label in [(1, "#f59e0b", "Moderate"), (2, "#ef4444", "Severe")]:
                fig.add_hline(y=thresh,  line_dash="dot", line_color=color,
                              annotation_text=f"{label} (+)")
                fig.add_hline(y=-thresh, line_dash="dot", line_color=color,
                              annotation_text=f"{label} (-)")
            fig.update_layout(title="Feature Skewness",
                              xaxis_title="Feature", yaxis_title="Skewness",
                              **_plotly_layout())
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "💡 Recommendations":
    if st.session_state.recommendations is None:
        st.markdown('<div class="warn-box">📤 Please upload a dataset first.</div>', unsafe_allow_html=True)
    else:
        recs = st.session_state.recommendations
        df   = st.session_state.df

        st.markdown('<div class="section-title">Recommendations & Next Steps</div>', unsafe_allow_html=True)

        report_txt = _build_report(df, st.session_state.issues, recs, st.session_state.health_score)
        b64 = base64.b64encode(report_txt.encode()).decode()
        st.markdown(
            f'<a href="data:text/plain;base64,{b64}" download="datalens_report.txt" '
            f'class="download-btn">⬇️ Download Full Report (.txt)</a>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        pcols = ["#6366f1", "#6366f1", "#8b5cf6", "#8b5cf6", "#a78bfa", "#a78bfa"]
        for section_key, section_title, icon in [
            ("preprocessing",       "Preprocessing Steps",     "🧹"),
            ("feature_engineering", "Feature Engineering",     "⚙️"),
            ("encoding",            "Encoding Recommendations","🔤"),
            ("modeling",            "Model Suggestions",       "🤖"),
        ]:
            items = recs.get(section_key, [])
            if not items:
                continue
            st.markdown(f'<div class="rec-section-title">{icon} {section_title}</div>', unsafe_allow_html=True)
            for idx, item in enumerate(items, 1):
                pc = pcols[min(idx - 1, len(pcols) - 1)]
                code_html = f'<code class="rec-code">{item["code"]}</code>' if item.get("code") else ""
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-num" style="background:{pc}">{idx}</div>
                    <div class="rec-content">
                        <div class="rec-title">{item['title']}</div>
                        <div class="rec-detail">{item['detail']}</div>
                        {code_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
