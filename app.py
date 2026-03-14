import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Partner Readiness Dashboard", page_icon="📊", layout="wide")
DATA_FILE = Path("partner_training_dashboard_data.csv")

THEME = {
    "bg": "#07111f",
    "panel": "#0f1c2f",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "gold": "#d4a843",
    "blue": "#5aa0ff",
    "green": "#2ac38b",
    "rose": "#ef6b8a",
    "purple": "#9f7aea",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=THEME["text"], family="Inter, sans-serif", size=12),
    margin=dict(l=18, r=18, t=48, b=18),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(212,168,67,0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(90,160,255,0.10), transparent 24%),
                linear-gradient(180deg, #07101c 0%, #0b1627 52%, #0f172a 100%);
            color: {THEME['text']};
        }}
        .block-container {{max-width: 1380px; padding-top: 1rem; padding-bottom: 2rem;}}
        section[data-testid="stSidebar"] {{background: linear-gradient(180deg, #0b1422 0%, #101b2d 100%);}}
        .hero {{background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.08); border-radius: 24px; padding: 24px; margin-bottom: 16px;}}
        .eyebrow {{color: {THEME['gold']}; text-transform: uppercase; font-size: 12px; letter-spacing: 0.16em; font-weight: 700; margin-bottom: 10px;}}
        .hero-title {{font-size: 2.2rem; font-weight: 700; line-height: 1.05; margin-bottom: 8px;}}
        .hero-sub {{color: {THEME['muted']}; font-size: 1rem; max-width: 900px;}}
        .section-card {{background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025)); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 16px 16px 8px 16px; margin-bottom: 14px;}}
        .insight-card {{background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 14px 16px; min-height: 92px;}}
        .insight-title {{color: {THEME['gold']}; text-transform: uppercase; font-size: 12px; letter-spacing: 0.12em; margin-bottom: 6px;}}
        .insight-body {{font-size: 0.95rem; color: {THEME['text']}; line-height: 1.45;}}
        div[data-testid="stMetric"] {{background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 10px 14px;}}
        div[data-testid="stMetric"] label {{color: {THEME['muted']} !important;}}
        .stTabs [data-baseweb="tab-list"] {{gap: 10px;}}
        .stTabs [data-baseweb="tab"] {{background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 999px; height: 42px; padding: 0 16px; color: {THEME['muted']};}}
        .stTabs [aria-selected="true"] {{background: linear-gradient(180deg, rgba(212,168,67,0.18), rgba(212,168,67,0.08)); color: white !important; border-color: rgba(212,168,67,0.35);}}
        .small-note {{color: {THEME['muted']}; font-size: 0.88rem; margin-top: -2px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_style(fig, title=None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.12)")
    return fig


def card_start(title, subtitle=None):
    note = f'<div class="small-note">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div class="section-card"><h3 style="margin:0 0 6px 0;">{title}</h3>{note}', unsafe_allow_html=True)


def card_end():
    st.markdown('</div>', unsafe_allow_html=True)


def insight_box(title, body):
    st.markdown(f'<div class="insight-card"><div class="insight-title">{title}</div><div class="insight-body">{body}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_FILE.exists():
        st.error("partner_training_dashboard_data.csv was not found in the project folder.")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    for c in ["signup_date", "activation_date", "certification_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["active_30d"] = (df["days_since_last_login"] <= 30).astype(int)
    df["completion_gap"] = (df["modules_assigned"] - df["modules_completed"]).clip(lower=0)
    df["readiness_score"] = (
        df["training_completion_pct"] * 0.32
        + df["quiz_avg_score"] * 0.18
        + df["assessment_score"] * 0.22
        + df["compliance_score"] * 0.16
        + (40 - df["days_since_last_login"].clip(0, 40)) * 0.3
    ).clip(0, 100).round(1)
    df["readiness_status_model"] = pd.cut(
        df["readiness_score"],
        bins=[-1, 55, 72, 100],
        labels=["Not Ready", "Almost Ready", "Ready"],
    ).astype(str)
    df["customer_ready_model"] = (df["readiness_score"] >= 72).astype(int)
    df["certification_status_model"] = np.where(
        df["readiness_score"] >= 76,
        "Certified",
        np.where(df["readiness_score"] >= 60, "In Progress", "Not Certified"),
    )
    df["risk_band"] = pd.cut(
        df["performance_score"],
        bins=[-1, 55, 70, 100],
        labels=["High Risk", "Watchlist", "Stable"],
    ).astype(str)
    return df


@st.cache_data(show_spinner=False)
def build_models(df):
    feature_cols = [
        "partner_type", "role", "region", "city_tier", "training_path",
        "months_active", "onboarding_days", "modules_assigned", "modules_completed",
        "training_completion_pct", "learning_hours", "live_sessions_attended",
        "mentor_sessions", "portal_logins_30d", "days_since_last_login", "reminders_sent",
        "quiz_attempts", "quiz_avg_score", "assessment_score", "compliance_score",
        "field_visits", "leads_assigned", "tickets_handled", "avg_resolution_hours",
        "qa_audit_score", "complaints_count", "repeat_issue_rate", "partner_nps", "customer_csat"
    ]
    X = df[feature_cols].copy()
    y_class = df["at_risk_flag"].astype(int)
    y_reg = df["performance_score"].astype(float)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.25, random_state=42, stratify=y_class)
    clf = Pipeline([("prep", pre), ("model", RandomForestClassifier(n_estimators=220, random_state=42, max_depth=8))])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    class_metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "auc": round(roc_auc_score(y_test, prob), 3),
    }

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.25, random_state=42)
    reg = Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=240, random_state=42, max_depth=10))])
    reg.fit(Xr_train, yr_train)
    pred_reg = reg.predict(Xr_test)
    reg_metrics = {
        "mae": round(mean_absolute_error(yr_test, pred_reg), 2),
        "r2": round(r2_score(yr_test, pred_reg), 3),
    }

    feature_names = clf.named_steps["prep"].get_feature_names_out()
    importances = clf.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(12)

    scored = df.copy()
    scored["risk_probability"] = clf.predict_proba(X)[:, 1]
    scored["predicted_performance_score"] = reg.predict(X)
    return class_metrics, reg_metrics, imp_df, scored


@st.cache_data(show_spinner=False)
def run_segmentation(df):
    seg_cols = ["training_completion_pct", "portal_logins_30d", "assessment_score", "customer_csat", "performance_score"]
    z = StandardScaler().fit_transform(df[seg_cols])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(z)
    out = df.copy()
    out["segment"] = labels
    stats = out.groupby("segment").agg(
        partners=("partner_id", "count"),
        completion=("training_completion_pct", "mean"),
        logins=("portal_logins_30d", "mean"),
        assess=("assessment_score", "mean"),
        csat=("customer_csat", "mean"),
        performance=("performance_score", "mean"),
        risk=("at_risk_flag", "mean"),
    ).round(2)
    names = []
    actions = []
    for _, row in stats.iterrows():
        if row["performance"] >= stats["performance"].quantile(0.75):
            names.append("Elite Performers")
            actions.append("Reward, certify fast, and use as benchmark cohort")
        elif row["completion"] >= stats["completion"].median() and row["logins"] >= stats["logins"].median():
            names.append("Engaged Builders")
            actions.append("Push to final certification and deployment")
        elif row["risk"] >= stats["risk"].quantile(0.75):
            names.append("At-Risk Operators")
            actions.append("Trigger coaching, reminders, and manager follow-up")
        else:
            names.append("Low-Activity Learners")
            actions.append("Improve engagement with nudges and mentor sessions")
    stats["segment_name"] = names
    stats["recommended_action"] = actions
    out = out.merge(stats[["segment_name"]], left_on="segment", right_index=True, how="left")
    return out, stats.reset_index()


df = load_data()
class_metrics, reg_metrics, imp_df, scored_df = build_models(df)
seg_df, seg_stats = run_segmentation(df)

with st.sidebar:
    st.title("📊 Partner Dashboard")
    st.caption("Readiness, training, performance, and risk intelligence")
    st.markdown("---")
    regions = sorted(df["region"].dropna().unique())
    roles = sorted(df["role"].dropna().unique())
    partner_types = sorted(df["partner_type"].dropna().unique())
    statuses = sorted(df["readiness_status_model"].dropna().unique())

    selected_regions = st.multiselect("Region", regions, default=regions)
    selected_roles = st.multiselect("Role", roles, default=roles)
    selected_types = st.multiselect("Partner type", partner_types, default=partner_types)
    selected_status = st.multiselect("Readiness band", statuses, default=statuses)
    date_min = df["signup_date"].min().date()
    date_max = df["signup_date"].max().date()
    selected_dates = st.date_input("Signup date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

start_date, end_date = selected_dates if isinstance(selected_dates, tuple) and len(selected_dates) == 2 else (date_min, date_max)
filtered = scored_df[
    scored_df["region"].isin(selected_regions)
    & scored_df["role"].isin(selected_roles)
    & scored_df["partner_type"].isin(selected_types)
    & scored_df["readiness_status_model"].isin(selected_status)
    & scored_df["signup_date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))
].copy()

seg_filtered = seg_df[seg_df["partner_id"].isin(filtered["partner_id"])].copy()
if filtered.empty:
    st.warning("No partners match the selected filters.")
    st.stop()

hero = f"""
<div class=\"hero\">
  <div class=\"eyebrow\">Partner Readiness & Training Intelligence</div>
  <div class=\"hero-title\">Partner Management Dashboard</div>
  <div class=\"hero-sub\">Track onboarding, training, certification readiness, field performance, and intervention risk in one admin view. Current selection: {len(filtered):,} partners across {filtered['region'].nunique()} regions and {filtered['role'].nunique()} roles.</div>
</div>
"""
st.markdown(hero, unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Partners", f"{len(filtered):,}")
k2.metric("Active 30D", f"{filtered['active_30d'].mean() * 100:.1f}%")
k3.metric("Training completion", f"{filtered['training_completion_pct'].mean():.1f}%")
k4.metric("Ready for deployment", f"{filtered['customer_ready_model'].mean() * 100:.1f}%")
k5.metric("At-risk partners", f"{filtered['at_risk_flag'].mean() * 100:.1f}%")
k6.metric("Avg CSAT", f"{filtered['customer_csat'].mean():.2f}")

ins1, ins2, ins3 = st.columns(3)
with ins1:
    top_region = filtered.groupby("region")["performance_score"].mean().sort_values(ascending=False).index[0]
    insight_box("Best region", f"{top_region} currently leads on average partner performance in the filtered view.")
with ins2:
    low_role = filtered.groupby("role")["training_completion_pct"].mean().sort_values().index[0]
    insight_box("Training gap", f"{low_role} has the weakest average training completion and likely needs stronger follow-up.")
with ins3:
    top_action = filtered["next_best_action"].value_counts().index[0]
    insight_box("Priority action", f"The most common recommended intervention is: {top_action}.")

overview_tab, training_tab, performance_tab, predictive_tab, actions_tab = st.tabs(["Overview", "Training", "Performance", "Predictive", "Actions"])

with overview_tab:
    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        card_start("Onboarding and readiness funnel", "Shows how the partner base moves from signup into onboarding, training, and readiness bands.")
        funnel_df = pd.DataFrame({
            "stage": ["Signed up", "Activated", "In training", "Model certified", "Ready"],
            "count": [
                len(filtered),
                filtered["activation_date"].notna().sum(),
                (filtered["training_completion_pct"] >= 40).sum(),
                (filtered["certification_status_model"] == "Certified").sum(),
                filtered["customer_ready_model"].sum(),
            ]
        })
        fig = go.Figure(go.Funnel(y=funnel_df["stage"], x=funnel_df["count"], textinfo="value+percent initial"))
        chart_style(fig, "Partner funnel")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Cohort trend", "Monthly signup cohorts plotted against average completion rate.")
        cohort = filtered.groupby("cohort_month", as_index=False).agg(partners=("partner_id", "count"), completion=("training_completion_pct", "mean"))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=cohort["cohort_month"], y=cohort["partners"], name="Partners"), secondary_y=False)
        fig.add_trace(go.Scatter(x=cohort["cohort_month"], y=cohort["completion"], mode="lines+markers", name="Avg completion"), secondary_y=True)
        chart_style(fig, "Cohort growth")
        fig.update_yaxes(title_text="Partners", secondary_y=False)
        fig.update_yaxes(title_text="Completion %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    c3, c4 = st.columns(2)
    with c3:
        card_start("Readiness by region", "Compares modeled readiness mix across regions.")
        region_ready = filtered.groupby(["region", "readiness_status_model"]).size().reset_index(name="partners")
        fig = px.bar(region_ready, x="region", y="partners", color="readiness_status_model", barmode="stack")
        chart_style(fig, "Readiness distribution")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c4:
        card_start("Regional leaderboard", "Average leaderboard score by region for quick benchmarking.")
        region_perf = filtered.groupby("region", as_index=False)["leaderboard_score"].mean().sort_values("leaderboard_score", ascending=False)
        fig = px.bar(region_perf, x="leaderboard_score", y="region", orientation="h", color="leaderboard_score", color_continuous_scale="Blues")
        chart_style(fig, "Leaderboard by region")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

with training_tab:
    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        card_start("Completion vs assessment", "Shows whether training progress is translating into stronger assessment scores.")
        fig = px.scatter(filtered, x="training_completion_pct", y="assessment_score", color="readiness_status_model", size="portal_logins_30d", hover_data=["role", "region", "partner_type"])
        chart_style(fig, "Training quality")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Role-region heatmap", "Highlights uneven completion performance across operating teams.")
        heat = pd.pivot_table(filtered, index="role", columns="region", values="training_completion_pct", aggfunc="mean").fillna(0)
        fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Tealgrn")
        chart_style(fig, "Completion heatmap")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    c3, c4 = st.columns(2)
    with c3:
        card_start("Quiz scores by path", "Distribution view for content effectiveness across learning paths.")
        fig = px.box(filtered, x="training_path", y="quiz_avg_score", color="training_path")
        chart_style(fig, "Quiz outcomes")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c4:
        card_start("Follow-up pressure", "Reminder volume versus completion gap by role.")
        reminder = filtered.groupby("role", as_index=False).agg(reminders=("reminders_sent", "mean"), gap=("completion_gap", "mean"))
        fig = px.scatter(reminder, x="reminders", y="gap", size="gap", color="role", hover_name="role")
        chart_style(fig, "Reminder efficiency")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

with performance_tab:
    c1, c2 = st.columns(2)
    with c1:
        card_start("CSAT by certification band", "Compares customer satisfaction against modeled certification status.")
        cert_csat = filtered.groupby("certification_status_model", as_index=False)["customer_csat"].mean().sort_values("customer_csat", ascending=False)
        fig = px.bar(cert_csat, x="certification_status_model", y="customer_csat", color="customer_csat", color_continuous_scale="Sunset")
        chart_style(fig, "CSAT impact")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Performance vs completion", "Tests whether stronger learning progress translates into stronger field performance.")
        fig = px.scatter(filtered, x="training_completion_pct", y="performance_score", color="region", size="customer_csat", hover_data=["role", "partner_type"])
        chart_style(fig, "Learning to performance")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    c3, c4 = st.columns(2)
    with c3:
        card_start("Commercial productivity", "Conversion rate and sales value across partner roles.")
        sales = filtered.groupby("role", as_index=False).agg(conversion=("conversion_rate", "mean"), sales_value=("sales_value", "mean"))
        fig = px.bar(sales, x="role", y="sales_value", color="conversion", color_continuous_scale="Blues")
        chart_style(fig, "Sales by role")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c4:
        card_start("Service quality", "Resolution time and QA quality across roles.")
        service = filtered.groupby("role", as_index=False).agg(resolution=("avg_resolution_hours", "mean"), qa=("qa_audit_score", "mean"))
        fig = px.scatter(service, x="resolution", y="qa", size="qa", color="role", hover_name="role")
        chart_style(fig, "Resolution vs QA")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

with predictive_tab:
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        card_start("Risk model metrics", "Random forest classification predicts whether a partner is at risk.")
        m1, m2 = st.columns(2)
        m1.metric("Risk model accuracy", class_metrics["accuracy"])
        m2.metric("Risk model AUC", class_metrics["auc"])
        m3, m4 = st.columns(2)
        m3.metric("Performance model MAE", reg_metrics["mae"])
        m4.metric("Performance model R²", reg_metrics["r2"])
        card_end()
    with c2:
        card_start("Feature importance", "Top variables driving at-risk classification.")
        fig = px.bar(imp_df.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Sunset")
        chart_style(fig, "Risk drivers")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    c3, c4 = st.columns(2)
    with c3:
        card_start("Predicted risk matrix", "Partners with the highest predicted risk should be prioritized for intervention.")
        risk_view = filtered[["partner_id", "partner_name", "region", "role", "training_completion_pct", "customer_csat", "risk_probability", "next_best_action"]].sort_values("risk_probability", ascending=False).head(15)
        st.dataframe(risk_view, use_container_width=True, hide_index=True, height=360)
        card_end()
    with c4:
        card_start("Predicted vs actual performance", "Regression output helps quantify how well training and engagement explain field outcomes.")
        sample = filtered[["performance_score", "predicted_performance_score", "region", "role"]].sample(min(700, len(filtered)), random_state=42)
        fig = px.scatter(sample, x="performance_score", y="predicted_performance_score", color="region", hover_data=["role"])
        fig.add_shape(type="line", x0=20, y0=20, x1=100, y1=100, line=dict(dash="dash"))
        chart_style(fig, "Regression fit")
        st.plotly_chart(fig, use_container_width=True)
        card_end()

with actions_tab:
    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        card_start("Partner segments", "K-means groups partners by learning, engagement, CSAT, and performance behavior.")
        fig = px.scatter(seg_filtered, x="training_completion_pct", y="performance_score", color="segment_name", size="portal_logins_30d", hover_data=["region", "role", "customer_csat"])
        chart_style(fig, "Segment map")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Segment actions", "Each segment maps to a practical intervention or deployment action.")
        seg_table = seg_stats.rename(columns={"segment_name": "Segment", "completion": "Completion %", "logins": "Logins", "assess": "Assess", "csat": "CSAT", "performance": "Performance", "risk": "Risk rate", "recommended_action": "Recommended action"})
        st.dataframe(seg_table, use_container_width=True, hide_index=True, height=360)
        card_end()

    c3, c4 = st.columns(2)
    with c3:
        card_start("Action demand", "Shows which admin interventions are most needed across the selected partner base.")
        act = filtered["next_best_action"].value_counts().reset_index()
        act.columns = ["action", "partners"]
        fig = px.bar(act, x="partners", y="action", orientation="h", color="partners", color_continuous_scale="Blues")
        chart_style(fig, "Next-best actions")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c4:
        card_start("Top partners", "Highest leaderboard partners can be used as role models, coaches, or pilot cohort members.")
        leaders = filtered[["partner_id", "partner_name", "region", "role", "leaderboard_score", "customer_csat", "performance_score"]].sort_values("leaderboard_score", ascending=False).head(15)
        st.dataframe(leaders, use_container_width=True, hide_index=True, height=320)
        card_end()

st.download_button(
    "Download filtered data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="partner_dashboard_filtered.csv",
    mime="text/csv",
)
