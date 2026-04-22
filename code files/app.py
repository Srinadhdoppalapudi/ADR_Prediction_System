import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from app_inference_utils import predict_adverse_reactions
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="ADR Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None


def find_existing_image(possible_names):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in possible_names:
        full_path = os.path.join(base_dir, name)
        if os.path.exists(full_path):
            return full_path
    return None


def build_background_css(image_names, overlay_1="rgba(4, 8, 30, 0.62)", overlay_2="rgba(4, 8, 30, 0.78)"):
    bg_path = find_existing_image(image_names)

    if bg_path:
        ext = "png" if bg_path.lower().endswith(".png") else "jpg"
        bg_base64 = get_base64_image(bg_path)
        if bg_base64:
            return f"""
            background:
                linear-gradient({overlay_1}, {overlay_2}),
                url("data:image/{ext};base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            """

    return """
    background:
        radial-gradient(circle at top left, #132047 0%, #0a1430 40%, #040918 100%);
    """


def get_welcome_background_css():
    # Use the BLUE futuristic medical image for welcome page
    return build_background_css(
        [
            "background.png",
            "background.jpg",
            "ChatGPT Image Apr 20, 2026, 05_15_05 PM.png",
            "ChatGPT Image Apr 20, 2026, 05_15_09 PM.png"
        ],
        overlay_1="rgba(2, 8, 28, 0.40)",
        overlay_2="rgba(2, 8, 28, 0.58)"
    )


def get_dashboard_background_css():
    # Use the NEW medicine/lab image for prediction page
    return build_background_css(
        [
            "dashboard_bg.png",
            "dashboard_bg.jpg",
            "ChatGPT Image Apr 20, 2026, 05_49_20 PM.png"
        ],
        overlay_1="rgba(18, 5, 22, 0.58)",
        overlay_2="rgba(8, 6, 18, 0.78)"
    )


# -------------------------------------------------
# GLOBAL UI
# -------------------------------------------------
# -------------------------------------------------
# GLOBAL UI
# -------------------------------------------------
def apply_global_ui(page_name):
    background_css = get_welcome_background_css() if page_name == "welcome" else get_dashboard_background_css()

    st.markdown(
        f"""
        <style>
        .stApp {{
            {background_css}
            color: white;
        }}

        [data-testid="stAppViewContainer"] {{
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
            height: 0rem !important;
        }}

        header {{
            visibility: hidden !important;
            height: 0rem !important;
        }}

        .block-container {{
            padding-top: 0.3rem !important;
            padding-bottom: 2rem !important;
            max-width: 1420px;
        }}

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: rgba(4, 10, 28, 0.88);
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(14px);
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* Welcome page */
        .welcome-shell {{
            min-height: 86vh;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem 1rem 1rem 1rem;
        }}

        .welcome-content {{
            max-width: 980px;
            margin: 0 auto;
        }}

        .welcome-small {{
            color: #d8b4fe;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .welcome-main {{
            font-size: 4.3rem;
            line-height: 1.05;
            font-weight: 900;
            background: linear-gradient(90deg, #8fd3ff, #c084fc, #f5f3ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 22px;
            text-transform: uppercase;
        }}

        .welcome-desc {{
            max-width: 940px;
            margin: 0 auto 24px auto;
            color: #edf4ff;
            font-size: 1.16rem;
            line-height: 1.9;
        }}

        .welcome-badges {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 10px;
            margin-bottom: 28px;
        }}

        .welcome-badge {{
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            padding: 10px 18px;
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            backdrop-filter: blur(8px);
        }}

        .explore-button-wrap {{
            max-width: 260px;
            margin: 0 auto;
        }}

        /* Dashboard cards */
        .hero-card {{
            background: linear-gradient(135deg, rgba(24, 14, 42, 0.62), rgba(68, 40, 120, 0.32));
            border: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(14px);
            border-radius: 26px;
            padding: 22px 28px;
            margin-bottom: 14px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        }}

        .hero-title {{
            font-size: 2.1rem;
            font-weight: 900;
            color: #ffffff;
            margin-bottom: 10px;
            line-height: 1.15;
        }}

        .hero-subtitle {{
            font-size: 1rem;
            color: #f3e8ff;
            line-height: 1.75;
            max-width: 1080px;
        }}

        .demo-box {{
            background: rgba(82, 54, 129, 0.42);
            border-left: 4px solid #a78bfa;
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 18px;
            color: #faf5ff;
            font-size: 0.92rem;
            line-height: 1.5;
            width: 100%;
            backdrop-filter: blur(10px);
        }}

        .selected-card-green {{
            background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(59,130,246,0.10));
            border: 1px solid rgba(52,211,153,0.20);
            border-radius: 18px;
            padding: 14px 18px;
            color: white;
            font-size: 1.04rem;
            backdrop-filter: blur(12px);
        }}

        .selected-card-blue {{
            background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(168,85,247,0.12));
            border: 1px solid rgba(129,140,248,0.20);
            border-radius: 18px;
            padding: 14px 18px;
            color: white;
            font-size: 1.04rem;
            backdrop-filter: blur(12px);
        }}

        .section-title {{
            font-size: 2rem;
            font-weight: 900;
            color: white;
            margin-top: 12px;
            margin-bottom: 12px;
        }}

        .result-card {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 12px;
            backdrop-filter: blur(10px);
        }}

        .result-title {{
            font-size: 1.08rem;
            font-weight: 800;
            color: white;
            margin-bottom: 10px;
        }}

        .plot-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 10px 12px;
            backdrop-filter: blur(10px);
            margin-bottom: 14px;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white !important;
            border-radius: 16px;
            border: none;
            font-weight: 800;
            padding: 0.86rem 1.2rem;
            width: 100%;
            box-shadow: 0 10px 24px rgba(59,130,246,0.24);
        }}

        .stDownloadButton > button {{
            background: linear-gradient(90deg, #db2777, #7c3aed);
            color: white !important;
            border-radius: 16px;
            border: none;
            font-weight: 800;
            padding: 0.86rem 1.2rem;
            width: 100%;
            box-shadow: 0 10px 24px rgba(124,58,237,0.24);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 10px 18px;
            color: white !important;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.10);
            padding: 14px;
            border-radius: 18px;
            backdrop-filter: blur(10px);
        }}

        /* -------- STRONG SELECTBOX / DROPDOWN FIX -------- */

        /* closed select box */
        div[data-baseweb="select"] {{
            color: #f8fafc !important;
        }}

        div[data-baseweb="select"] > div {{
            background: rgba(23, 19, 43, 0.92) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            min-height: 54px !important;
        }}

        div[data-baseweb="select"] > div:hover {{
            background: rgba(32, 24, 58, 0.96) !important;
        }}

        div[data-baseweb="select"] * {{
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }}

        div[data-baseweb="select"] input {{
            color: #f8fafc !important;
            caret-color: #f8fafc !important;
        }}

        div[data-baseweb="select"] div[role="combobox"] {{
            background: transparent !important;
            color: #f8fafc !important;
        }}

        div[data-baseweb="select"] div[role="combobox"] > div {{
            background: transparent !important;
            color: #f8fafc !important;
        }}

        div[data-baseweb="select"] span {{
            color: #f8fafc !important;
            font-weight: 600 !important;
        }}

        div[data-baseweb="select"] svg {{
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }}

        /* dropdown popup container */
        div[role="listbox"] {{
            background: rgba(18, 18, 30, 0.98) !important;
            color: #ffffff !important;
        }}

        /* baseweb menu container */
        ul[role="listbox"] {{
            background: rgba(18, 18, 30, 0.98) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 12px !important;
            padding: 6px !important;
        }}

        /* dropdown options */
        ul[role="listbox"] li,
        div[role="option"] {{
            background: rgba(18, 18, 30, 0.98) !important;
            color: #ffffff !important;
            font-weight: 500 !important;
        }}

        /* all text inside options */
        ul[role="listbox"] li *,
        div[role="option"] * {{
            color: #ffffff !important;
            fill: #ffffff !important;
        }}

        /* hovered option */
        ul[role="listbox"] li:hover,
        div[role="option"]:hover {{
            background: rgba(99,102,241,0.30) !important;
            color: #ffffff !important;
        }}

        /* selected / highlighted option */
        ul[role="listbox"] li[aria-selected="true"],
        div[role="option"][aria-selected="true"] {{
            background: rgba(124,58,237,0.35) !important;
            color: #ffffff !important;
        }}

        /* fallback for portal-rendered dropdown */
        [data-baseweb="popover"] ul,
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] div[role="option"] {{
            background: rgba(18, 18, 30, 0.98) !important;
            color: #ffffff !important;
        }}

        [data-baseweb="popover"] li *,
        [data-baseweb="popover"] div[role="option"] * {{
            color: #ffffff !important;
        }}

        body *[role="option"] {{
            color: #ffffff !important;
        }}

        /* Inputs and table */
        .stTextInput input {{
            background: rgba(23, 19, 43, 0.88) !important;
            color: white !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }}

        .stDataFrame {{
            background: rgba(255,255,255,0.04);
            border-radius: 16px;
        }}

        /* General text */
        h1, h2, h3, h4, h5, h6, p, label, div {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "sample_data.csv")  # change if needed

        chunk_size = 10000
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        unique_drugs = sorted(df["drug_name"].dropna().unique())
        drug_indication_map = {}
        drug_indication_side_effects_map = {}

        for drug in unique_drugs:
            drug_data = df[df["drug_name"] == drug]

            drug_indications = drug_data["indication_name"].dropna().unique()
            drug_indication_map[drug] = sorted(drug_indications)

            drug_indication_side_effects_map[drug] = {}
            for indication in drug_indications:
                indication_data = drug_data[drug_data["indication_name"] == indication]
                side_effects = indication_data["side_effect_name"].dropna().unique()
                drug_indication_side_effects_map[drug][indication] = sorted(side_effects)

        return df, drug_indication_map, unique_drugs, drug_indication_side_effects_map

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None


# -------------------------------------------------
# WELCOME PAGE
# -------------------------------------------------
def show_welcome_page():
    st.markdown(
        """
        <div class="welcome-shell">
            <div class="welcome-content">
                <div class="welcome-small">Welcome to</div>
                <div class="welcome-main">Adverse Drug Reaction<br>Prediction System</div>
                <div class="welcome-desc">
                    An intelligent AI-powered platform that analyzes <b>drug</b>, <b>indication</b>,
                    and <b>side-effect</b> patterns to predict potential adverse drug reactions with
                    probability and confidence insights. This project demonstrates how machine learning
                    can support safer decision-making and risk-aware healthcare analytics.
                </div>
                <div class="welcome-badges">
                    <div class="welcome-badge">🧠 AI & Machine Learning</div>
                    <div class="welcome-badge">💊 Drug Safety Analysis</div>
                    <div class="welcome-badge">📊 Interactive Dashboard</div>
                    <div class="welcome-badge">🔬 Data-Driven Insights</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c2:
        st.markdown('<div class="explore-button-wrap">', unsafe_allow_html=True)
        if st.button("🚀 Explore Dashboard", key="go_to_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------------------
# DASHBOARD HEADER
# -------------------------------------------------
def show_dashboard_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🏥 Adverse Drug Reaction Prediction Dashboard</div>
            <div class="hero-subtitle">
                This dashboard presents AI-driven predictions of potential adverse drug reactions
                based on selected drug and medical indication combinations. It provides probability
                scores, confidence insights, and interactive visual analytics to help interpret
                predicted side effects in a clear, modern, and data-driven way.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="demo-box">
            <b>🧬 Project Overview:</b> This interface is designed to demonstrate how machine learning
            can support drug safety analysis by identifying possible adverse reactions from drug–indication
            relationships. Explore prediction rankings, confidence levels, and visual patterns through the dashboard below.
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
def show_sidebar(unique_drugs, drug_indication_map, drug_indication_side_effects_map):
    st.sidebar.markdown("## 🔍 Input Parameters")
    st.sidebar.markdown(
        "<p style='font-size:13px; color:#e9d5ff; margin-top:-8px;'>Choose a drug and indication to generate ADR predictions</p>",
        unsafe_allow_html=True
    )

    selected_drug = st.sidebar.selectbox(
        "Select Drug",
        options=unique_drugs,
        help="Choose the drug you want to analyze",
        key="drug_select"
    )

    if selected_drug:
        drug_indications = drug_indication_map.get(selected_drug, [])
        st.sidebar.info(f"{selected_drug} has {len(drug_indications)} available indications")

        if drug_indications:
            selected_indication = st.sidebar.selectbox(
                "Select Medical Indication",
                options=drug_indications,
                help="Choose the medical condition the drug is used for",
                key="indication_select"
            )
        else:
            selected_indication = None
            st.sidebar.warning(f"No indications found for {selected_drug}")
    else:
        selected_indication = None

    if selected_drug and selected_indication:
        available_side_effects = drug_indication_side_effects_map.get(selected_drug, {}).get(selected_indication, [])
        st.sidebar.info(f"{len(available_side_effects)} side effects available for prediction")

    predict_button = st.sidebar.button("🔮 Predict Adverse Reactions")
    back_button = st.sidebar.button("🏠 Back to Welcome Page")

    if back_button:
        st.session_state.page = "welcome"
        st.session_state.run_prediction = False
        st.rerun()

    return selected_drug, selected_indication, predict_button


# -------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------
def render_visualizations(results_df):
    top_df = results_df.head(10).copy()
    top_df["Rank"] = range(1, len(top_df) + 1)

    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    st.markdown("### Top Side Effects by Probability")

    fig_bar = px.bar(
        top_df.sort_values(by="Probability", ascending=True),
        x="Probability",
        y="Side Effect",
        orientation="h",
        text="Probability",
        color="Confidence",
        color_continuous_scale="Magma"
    )

    fig_bar.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.4f}<br>Confidence: %{marker.color:.4f}<extra></extra>"
    )

    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=560,
        xaxis_title="Probability",
        yaxis_title="Side Effect",
        margin=dict(l=40, r=40, t=20, b=30)
    )

    fig_bar.update_yaxes(tickfont=dict(size=12))
    fig_bar.update_xaxes(range=[0, 1])

    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    v1, v2 = st.columns([1.45, 1])

    with v1:
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        st.markdown("### Probability vs Confidence")

        fig_scatter = px.scatter(
            top_df,
            x="Probability",
            y="Confidence",
            size="Confidence",
            color="Probability",
            hover_name="Side Effect",
            hover_data={
                "Probability": ":.4f",
                "Confidence": ":.4f",
                "Rank": True
            },
            color_continuous_scale="Plasma"
        )

        fig_scatter.update_traces(
            marker=dict(
                line=dict(width=1, color="white"),
                sizemode="diameter",
                opacity=0.90
            )
        )

        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=500,
            xaxis_title="Probability",
            yaxis_title="Confidence",
            margin=dict(l=30, r=20, t=20, b=20)
        )

        fig_scatter.update_xaxes(range=[0, 1], gridcolor="rgba(255,255,255,0.18)")
        fig_scatter.update_yaxes(range=[0, 1], gridcolor="rgba(255,255,255,0.18)")

        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        st.markdown("### Confidence Overview")

        avg_conf = float(results_df["Confidence"].mean())
        avg_prob = float(results_df["Probability"].mean())

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Average Confidence", "Remaining"],
                    values=[avg_conf, max(0.0001, 1 - avg_conf)],
                    hole=0.70,
                    textinfo="none",
                    marker=dict(colors=["#a855f7", "rgba(255,255,255,0.08)"])
                )
            ]
        )

        fig_donut.update_layout(
            annotations=[
                dict(
                    text=f"{avg_conf:.0%}",
                    x=0.5,
                    y=0.5,
                    font_size=28,
                    font_color="white",
                    showarrow=False
                )
            ],
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False
        )

        st.plotly_chart(fig_donut, use_container_width=True)
        st.metric("Average Confidence", f"{avg_conf:.4f}")
        st.metric("Average Probability", f"{avg_prob:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    st.markdown("### Detailed Interactive Results")

    display_df = results_df.copy()
    display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.4f}")
    display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------
# -------------------------------------------------
# DASHBOARD PAGE
# -------------------------------------------------
def show_dashboard():
    with st.spinner("Loading data..."):
        df, drug_indication_map, unique_drugs, drug_indication_side_effects_map = load_data()

    if df is None:
        st.error("Failed to load data. Please check your data file.")
        return

    show_dashboard_header()

    left_col, right_col = st.columns([1, 3], gap="large")

    with left_col:
        st.markdown("## 🔍 Input Parameters")
        st.markdown(
            "<p style='font-size:13px; color:#e9d5ff; margin-top:-8px;'>Choose a drug and indication to generate ADR predictions</p>",
            unsafe_allow_html=True
        )

        selected_drug = st.selectbox(
            "Select Drug",
            options=unique_drugs,
            help="Choose the drug you want to analyze",
            key="drug_select_main"
        )

        if selected_drug:
            drug_indications = drug_indication_map.get(selected_drug, [])
            st.info(f"{selected_drug} has {len(drug_indications)} available indications")

            if drug_indications:
                selected_indication = st.selectbox(
                    "Select Medical Indication",
                    options=drug_indications,
                    help="Choose the medical condition the drug is used for",
                    key="indication_select_main"
                )
            else:
                selected_indication = None
                st.warning(f"No indications found for {selected_drug}")
        else:
            selected_indication = None

        if selected_drug and selected_indication:
            available_side_effects = drug_indication_side_effects_map.get(
                selected_drug, {}
            ).get(selected_indication, [])
            st.info(f"{len(available_side_effects)} side effects available for prediction")

        predict_button = st.button("🔮 Predict Adverse Reactions", key="predict_main")
        back_button = st.button("🏠 Back to Welcome Page", key="back_main")

        if back_button:
            st.session_state.page = "welcome"
            st.session_state.run_prediction = False
            st.rerun()

    with right_col:
        # Always store current selections
        st.session_state.selected_drug = selected_drug
        st.session_state.selected_indication = selected_indication

        if predict_button:
            if not selected_drug or not selected_indication:
                st.warning("Please select both a drug and an indication.")
                st.session_state.run_prediction = False
            else:
                st.session_state.run_prediction = True

        if st.session_state.get("run_prediction", False):
            selected_drug = st.session_state.get("selected_drug")
            selected_indication = st.session_state.get("selected_indication")

            if not selected_drug or not selected_indication:
                st.warning("Please select both a drug and an indication.")
                return

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="selected-card-green"><b>Selected Drug:</b> {selected_drug}</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f'<div class="selected-card-blue"><b>Selected Indication:</b> {selected_indication}</div>',
                    unsafe_allow_html=True
                )

            with st.spinner("Running prediction... This may take a moment."):
                try:
                    side_effects_list = drug_indication_side_effects_map.get(
                        selected_drug, {}
                    ).get(selected_indication, [])

                    if not side_effects_list:
                        st.error("No side effects found for this drug–indication pair.")
                        return

                    predictions = predict_adverse_reactions(
                        selected_drug,
                        selected_indication,
                        side_effects_list
                    )

                    if not predictions:
                        st.error("No predictions generated.")
                        return

                    results_df = pd.DataFrame(
                        predictions,
                        columns=["Side Effect", "Probability", "Confidence"]
                    )

                    st.success(f"✅ Generated {len(predictions)} predictions successfully!")

                    items_per_page = 10
                    total_pages = (len(results_df) + items_per_page - 1) // items_per_page

                    page = 1
                    if total_pages > 1:
                        page = st.selectbox(
                            f"Select Page (1-{total_pages})",
                            range(1, total_pages + 1),
                            index=0,
                            key="page_select"
                        )

                    start_idx = (page - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, len(results_df))
                    page_df = results_df.iloc[start_idx:end_idx].copy()

                    st.markdown('<div class="section-title">📈 Summary Statistics</div>', unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Total Predictions", len(results_df))
                    with m2:
                        st.metric("Highest Probability", f"{results_df['Probability'].max():.4f}")
                    with m3:
                        st.metric("Average Probability", f"{results_df['Probability'].mean():.4f}")

                    tab1, tab2, tab3 = st.tabs(["📋 Predictions", "📊 Visualization", "📥 Export"])

                    with tab1:
                        st.markdown("### Predicted Adverse Reactions")

                        for idx, row in page_df.iterrows():
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="result-title">#{idx + 1} {row['Side Effect']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            rc1, rc2 = st.columns([4, 1])
                            with rc1:
                                st.progress(float(min(max(row["Probability"], 0), 1)))
                            with rc2:
                                st.write(f"**{row['Probability']:.4f}**")

                            rc3, rc4 = st.columns(2)
                            with rc3:
                                st.metric("Probability", f"{row['Probability']:.4f}")
                            with rc4:
                                st.metric("Confidence", f"{row['Confidence']:.4f}")

                            st.markdown("---")

                        st.markdown("### Search / Explore Results")
                        query = st.text_input("Search side effect name", key="search_side_effect")
                        if query:
                            filtered = results_df[results_df["Side Effect"].str.contains(query, case=False, na=False)]
                            st.dataframe(filtered, use_container_width=True)
                        else:
                            st.dataframe(results_df, use_container_width=True)

                    with tab2:
                        render_visualizations(results_df)

                    with tab3:
                        st.markdown("### Download Full Prediction Results")
                        csv = results_df.to_csv(index=False)

                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"adr_predictions_{selected_drug}_{selected_indication}.csv",
                            mime="text/csv"
                        )

                        st.dataframe(results_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

        else:
            st.info("Use the left-side input panel to select a drug and medical indication, then click Predict Adverse Reactions.")

            st.markdown("""
            ### 🚀 How to Use
            1. Select a **Drug**
            2. Select a **Medical Indication**
            3. Click **Predict Adverse Reactions**
            4. Explore predictions, visualizations, confidence insights, and export results
            """)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Drugs", len(unique_drugs))
            with c2:
                total_indications = sum(len(v) for v in drug_indication_map.values())
                st.metric("Drug-Indication Pairs", total_indications)
            with c3:
                st.metric("Total Side Effects", len(df["side_effect_name"].dropna().unique()))


# -------------------------------------------------
# MAIN ROUTER
# -------------------------------------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if "run_prediction" not in st.session_state:
        st.session_state.run_prediction = False

    apply_global_ui(st.session_state.page)

    if st.session_state.page == "welcome":
        show_welcome_page()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()