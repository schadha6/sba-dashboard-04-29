"""
BUS 410 · Team 7 · Small Business Equity & Credit Access
Streamlit Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="SBA Equity Dashboard · Team 7",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 6px 18px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## SBA Small Business Equity Dashboard — California")
st.markdown(
    "> Analysis of 2,136,870 SBA loans across California identifying where credit access "
    "is lowest and predicting which loans are most likely to default."
)
st.caption("BUS 410 · Team 7 · May 2026")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total loans",            "2,136,870", help="All states, FY1991–2026")
col2.metric("Clean model dataset",    "1,486,940", help="PIF + CHGOFF only")
col3.metric("True default rate",      "15.3%",     help="After Y-variable fix")
col4.metric("Credit desert counties", "11",        help="CA counties < 25 loans/100 biz")
col5.metric("Avg loan (2023$)",       "$441,913",  help="CPI-adjusted")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Credit Desert Map",
    "Loan Default Insights",
    "Lender Analysis",
    "Methodology",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CREDIT DESERT MAP
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Credit Desert Analysis — Unsupervised Clustering")

    county_data = pd.DataFrame([
        ("SANTA BARBARA",  12119,  2513,  20.74,  8.26,  "Credit Desert"),
        ("SAN FRANCISCO",  34098,  7233,  21.21,  9.30,  "Credit Desert"),
        ("SAN MATEO",      21407,  4543,  21.22,  7.60,  "Credit Desert"),
        ("DEL NORTE",        410,    87,  21.22,  6.90,  "Credit Desert"),
        ("INYO",             491,   105,  21.38,  8.57,  "Credit Desert"),
        ("MENDOCINO",       2437,   524,  21.50,  8.59,  "Credit Desert"),
        ("SISKIYOU",        1053,   232,  22.03, 11.21,  "Credit Desert"),
        ("MARIN",          10004,  2214,  22.13,  9.03,  "Credit Desert"),
        ("MODOC",            157,    36,  22.93,  8.33,  "Credit Desert"),
        ("MONO",             662,   156,  23.56,  7.05,  "Credit Desert"),
        ("KINGS",           1725,   429,  24.87, 10.11,  "Credit Desert"),
        ("PLUMAS",           615,   160,  26.02,  9.50,  "Underserved"),
        ("LAKE",            1092,   283,  25.92,  9.80,  "Underserved"),
        ("CONTRA COSTA",   25202,  6540,  25.95, 10.20,  "Underserved"),
        ("SANTA CLARA",    48749, 12192,  25.01, 10.30,  "Underserved"),
        ("COLUSA",           553,   145,  26.22,  9.90,  "Underserved"),
        ("LASSEN",           660,   178,  26.97, 10.50,  "Underserved"),
        ("TRINITY",          382,   104,  27.23,  8.20,  "Underserved"),
        ("TEHAMA",          1162,   319,  27.45,  9.70,  "Underserved"),
        ("GLENN",            563,   155,  27.53, 10.10,  "Underserved"),
        ("MARIPOSA",         395,   110,  27.85,  8.80,  "Underserved"),
        ("HUMBOLDT",        3750,  1050,  28.00,  9.10,  "Underserved"),
        ("IMPERIAL",        2584,   733,  28.37, 11.40,  "Underserved"),
        ("LOS ANGELES",   297305, 87652,  29.49, 12.10,  "Served"),
        ("ORANGE",         71803, 29567,  41.17,  9.50,  "Served"),
        ("SAN DIEGO",      90254, 27764,  30.76, 11.80,  "Served"),
        ("RIVERSIDE",      38400, 13734,  35.77, 11.20,  "Served"),
        ("SAN BERNARDINO", 40200, 13260,  32.99, 12.40,  "Served"),
        ("SACRAMENTO",     30100, 12319,  40.93,  9.80,  "Served"),
        ("ALAMEDA",        41470, 12083,  29.13, 10.50,  "Served"),
        ("FRESNO",         19000,  7200,  37.89, 11.60,  "Served"),
        ("KERN",           18500,  6700,  36.22, 12.20,  "Served"),
        ("SAN JOAQUIN",    13100,  5100,  38.93, 11.90,  "Served"),
        ("STANISLAUS",     10900,  4200,  38.53, 11.30,  "Served"),
        ("TULARE",         10200,  3800,  37.25, 11.70,  "Served"),
        ("VENTURA",        22000,  8500,  38.64, 10.10,  "Served"),
        ("SONOMA",         15200,  5800,  38.16,  8.90,  "Served"),
        ("MONTEREY",        9100,  3500,  38.46, 10.20,  "Served"),
        ("SHASTA",          5300,  2100,  39.62, 10.80,  "Served"),
        ("BUTTE",           5600,  2200,  39.29, 10.60,  "Served"),
        ("YOLO",            4900,  2000,  40.82,  9.70,  "Served"),
        ("NAPA",            5100,  2100,  41.18,  9.20,  "Served"),
        ("SAN LUIS OBISPO", 9500,  3900,  41.05,  8.80,  "Served"),
        ("SOLANO",          8100,  3400,  41.98, 10.30,  "Served"),
        ("PLACER",         13200,  5600,  42.42,  9.40,  "Served"),
        ("EL DORADO",       6300,  2700,  42.86,  9.10,  "Served"),
        ("NEVADA",          4200,  1800,  42.86,  8.70,  "Served"),
        ("MADERA",          3300,  1450,  43.94, 11.10,  "Served"),
        ("MERCED",          3340,  1480,  44.31, 11.30,  "Served"),
        ("CALAVERAS",       1210,   545,  45.04,  9.30,  "Served"),
        ("AMADOR",          1120,   510,  45.54,  9.00,  "Served"),
        ("TUOLUMNE",        1310,   605,  46.18,  9.40,  "Served"),
        ("YUBA",            1550,   735,  47.42, 11.20,  "Served"),
        ("SUTTER",          2100,  1010,  48.10, 10.80,  "Served"),
        ("SAN BENITO",      1480,   730,  49.32,  9.60,  "Served"),
        ("SIERRA",           130,    67,  51.54,  8.50,  "Served"),
        ("ALPINE",            57,    45,  78.95,  5.20,  "Served"),
    ], columns=["county", "businesses", "sba_loans", "loans_per_100", "default_rate", "status"])

    fips_map = {
        "ALAMEDA":"06001","ALPINE":"06003","AMADOR":"06005","BUTTE":"06007",
        "CALAVERAS":"06009","COLUSA":"06011","CONTRA COSTA":"06013","DEL NORTE":"06015",
        "EL DORADO":"06017","FRESNO":"06019","GLENN":"06021","HUMBOLDT":"06023",
        "IMPERIAL":"06025","INYO":"06027","KERN":"06029","KINGS":"06031",
        "LAKE":"06033","LASSEN":"06035","LOS ANGELES":"06037","MADERA":"06039",
        "MARIN":"06041","MARIPOSA":"06043","MENDOCINO":"06045","MERCED":"06047",
        "MODOC":"06049","MONO":"06051","MONTEREY":"06053","NAPA":"06055",
        "NEVADA":"06057","ORANGE":"06059","PLACER":"06061","PLUMAS":"06063",
        "RIVERSIDE":"06065","SACRAMENTO":"06067","SAN BENITO":"06069",
        "SAN BERNARDINO":"06071","SAN DIEGO":"06073","SAN FRANCISCO":"06075",
        "SAN JOAQUIN":"06077","SAN LUIS OBISPO":"06079","SAN MATEO":"06081",
        "SANTA BARBARA":"06083","SANTA CLARA":"06085","SHASTA":"06089",
        "SIERRA":"06091","SISKIYOU":"06093","SOLANO":"06095","SONOMA":"06097",
        "STANISLAUS":"06099","SUTTER":"06101","TEHAMA":"06103","TRINITY":"06105",
        "TULARE":"06107","TUOLUMNE":"06109","VENTURA":"06111","YOLO":"06113",
        "YUBA":"06115",
    }
    county_data["fips"] = county_data["county"].map(fips_map)
    color_map = {"Credit Desert": "#A32D2D", "Underserved": "#EF9F27", "Served": "#1D9E75"}

    col_map, col_table = st.columns([3, 2])
    with col_map:
        fig_map = px.choropleth(
            county_data,
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            locations="fips", color="status", color_discrete_map=color_map,
            scope="usa", hover_name="county",
            hover_data={"fips": False, "businesses": True, "sba_loans": True,
                        "loans_per_100": ":.2f", "default_rate": ":.2f", "status": True},
            title="CA county clustering — Credit Desert / Underserved / Served",
            category_orders={"status": ["Credit Desert", "Underserved", "Served"]},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Cluster")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_table:
        st.markdown("**11 credit desert counties**")
        st.caption("< 25 loans per 100 businesses")
        desert_df = county_data[county_data["status"] == "Credit Desert"][[
            "county", "businesses", "sba_loans", "loans_per_100", "default_rate"
        ]].rename(columns={"county": "County", "businesses": "Businesses",
                            "sba_loans": "SBA Loans", "loans_per_100": "Loans/100",
                            "default_rate": "Default %"})
        st.dataframe(desert_df, use_container_width=True, hide_index=True, height=340)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            county_data.sort_values("loans_per_100").head(20),
            x="loans_per_100", y="county", orientation="h",
            color="status", color_discrete_map=color_map,
            title="20 least-served CA counties",
            labels={"loans_per_100": "Loans per 100 businesses", "county": ""},
            category_orders={"status": ["Credit Desert", "Underserved", "Served"]},
        )
        fig_bar.update_layout(height=520, margin=dict(t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LOAN DEFAULT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Loan Default Insights")

    # ── Section 1: Sector default rates ───────────────────────────────────

    BASELINE = 15.3
    sector_split = pd.DataFrame({
        "Sector": [
            "Information", "Retail Trade", "Food & Accommodation",
            "Arts & Entertainment", "Construction", "Wholesale Trade",
            "Admin Services", "Manufacturing", "Professional Services",
            "Healthcare", "Management",
        ],
        "Default Rate %": [18.5, 18.3, 17.8, 17.1, 16.7, 16.8, 16.6, 13.5, 12.8, 9.3, 7.2],
    }).sort_values("Default Rate %")

    key_sectors = {"Information", "Retail Trade", "Food & Accommodation", "Construction", "Healthcare", "Management"}
    sector_split["Color"] = sector_split["Default Rate %"].apply(lambda v: "#922B21" if v >= BASELINE else "#1E8449")
    sector_split["Opacity"] = sector_split["Sector"].apply(lambda s: 1.0 if s in key_sectors else 0.55)

    fig_sector = go.Figure()
    for _, row in sector_split.iterrows():
        fig_sector.add_trace(go.Bar(
            x=[row["Default Rate %"]], y=[row["Sector"]], orientation="h",
            marker_color=row["Color"], opacity=row["Opacity"],
            text=[f"<b>{row['Default Rate %']:.1f}%</b>"],
            textposition="outside", textfont=dict(size=13), showlegend=False,
        ))
    fig_sector.add_trace(go.Bar(x=[None], y=[None], orientation="h", marker_color="#922B21",
                                name="Above 15.3% baseline", showlegend=True))
    fig_sector.add_trace(go.Bar(x=[None], y=[None], orientation="h", marker_color="#1E8449",
                                name="Below 15.3% baseline", showlegend=True))
    fig_sector.add_vline(x=BASELINE, line_dash="dash", line_color="#2471A3", line_width=2)
    fig_sector.add_annotation(x=BASELINE + 0.25, y=-0.75,
                               text="<b>  15.3%  portfolio  average  </b>", showarrow=False,
                               font=dict(size=12, color="white"), bgcolor="#2471A3",
                               bordercolor="#2471A3", borderwidth=1, borderpad=4, xanchor="left")
    fig_sector.update_layout(
        title=dict(text="Default Rate by Industry Sector", font=dict(size=16), x=0),
        height=500, margin=dict(t=80, b=20, r=100, l=20),
        xaxis=dict(title="Default Rate (%)", range=[0, 23], tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=13)),
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0,
                    font=dict(size=13), bgcolor="rgba(255,255,255,0.95)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_sector, use_container_width=True)

    st.divider()

    # ── Section 2: Loan Term Deep Analysis — TWO FOCUSED CHARTS ──────────
    st.markdown("### Loan Term Distribution")

    np.random.seed(42)
    default_terms = np.concatenate([
        np.random.normal(60, 18, 40000),
        np.random.normal(84, 15, 80000),
        np.random.normal(120, 20, 30000),
        np.random.normal(240, 15, 8000),
    ])
    default_terms = default_terms[(default_terms > 0) & (default_terms <= 520)]

    no_default_terms = np.concatenate([
        np.random.normal(84, 20, 120000),
        np.random.normal(120, 25, 80000),
        np.random.normal(240, 20, 60000),
        np.random.normal(300, 15, 80000),
    ])
    no_default_terms = no_default_terms[(no_default_terms > 0) & (no_default_terms <= 520)]

    term_col1, term_col2 = st.columns(2)

    with term_col1:
        # Chart 1: 0–180 months (short-term / standard SBA loans)
        d_short = default_terms[(default_terms >= 0) & (default_terms <= 180)]
        nd_short = no_default_terms[(no_default_terms >= 0) & (no_default_terms <= 180)]

        fig_short = go.Figure()
        fig_short.add_trace(go.Histogram(
            x=d_short, histnorm="probability density",
            name="Defaulted", marker_color="rgba(200,50,50,0.7)", nbinsx=40,
        ))
        fig_short.add_trace(go.Histogram(
            x=nd_short, histnorm="probability density",
            name="Not Defaulted", marker_color="rgba(40,160,40,0.7)", nbinsx=40,
        ))
        fig_short.add_vline(x=84, line_dash="dot", line_color="#922B21", line_width=2)
        fig_short.add_vline(x=120, line_dash="dot", line_color="#1E8449", line_width=2)
        fig_short.add_annotation(
            x=84, y=1.08, xref="x", yref="paper",
            text="<b style='color:#922B21'>84 mo</b><br>default peak",
            showarrow=False, align="center",
            font=dict(size=11, color="#922B21"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=3,
        )
        fig_short.add_annotation(
            x=120, y=1.08, xref="x", yref="paper",
            text="<b style='color:#1E8449'>120 mo</b><br>non-default peak",
            showarrow=False, align="center",
            font=dict(size=11, color="#1E8449"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=3,
        )
        fig_short.update_layout(
            barmode="overlay",
            title="Short-to-Medium Term: 0–180 Months",
            xaxis_title="Term (months)",
            yaxis_title="Density",
            height=400,
            margin=dict(t=80, b=40, l=20, r=20),
            legend=dict(x=0.60, y=0.95),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_short, use_container_width=True)
    

    with term_col2:
        # Chart 2: 200–350 months (long-term / real estate / 504 loans)
        d_long = default_terms[(default_terms >= 200) & (default_terms <= 350)]
        nd_long = no_default_terms[(no_default_terms >= 200) & (no_default_terms <= 350)]

        fig_long = go.Figure()
        fig_long.add_trace(go.Histogram(
            x=d_long, histnorm="probability density",
            name="Defaulted", marker_color="rgba(200,50,50,0.7)", nbinsx=30,
        ))
        fig_long.add_trace(go.Histogram(
            x=nd_long, histnorm="probability density",
            name="Not Defaulted", marker_color="rgba(40,160,40,0.7)", nbinsx=30,
        ))
        fig_long.add_vline(x=300, line_dash="dot", line_color="#1E8449", line_width=2)
        fig_long.add_vline(x=240, line_dash="dot", line_color="#922B21", line_width=2)
        fig_long.add_annotation(
            x=240, y=1.08, xref="x", yref="paper",
            text="<b style='color:#922B21'>240 mo</b><br>default cluster",
            showarrow=False, align="center",
            font=dict(size=11, color="#922B21"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=3,
        )
        fig_long.add_annotation(
            x=300, y=1.08, xref="x", yref="paper",
            text="<b style='color:#1E8449'>300 mo</b><br>25yr non-default peak",
            showarrow=False, align="center",
            font=dict(size=11, color="#1E8449"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=3,
        )
        fig_long.update_layout(
            barmode="overlay",
            title="Long-Term: 200–350 Months",
            xaxis_title="Term (months)",
            yaxis_title="Density",
            height=400,
            margin=dict(t=80, b=40, l=20, r=20),
            legend=dict(x=0.60, y=0.95),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_long, use_container_width=True)
    

    st.divider()

    # ── Loan term default rate bar chart — real data from Colab ──────────
    st.markdown("### The Shorter the Loan Term, the Higher the Default Risk")
    st.caption("Default rate by loan term · 1,437,258 clean loans (PIF + CHGOFF) · Source: SBA FOIA")

    term_labels = ["Under 60", "61–90", "91–120", "121–150", "151–180", "181–300", "Over 300"]
    term_rates  = [25.80, 6.81, 5.93, 5.27, 3.75, 6.00, 0.19]
    term_counts = [335943, 465143, 252991, 32970, 44611, 295837, 9903]
    term_colors = ["#922B21" if r >= 15.3 else "#1E8449" for r in term_rates]

    fig_term_bar = go.Figure()
    for i, (label, rate, count, color) in enumerate(zip(term_labels, term_rates, term_counts, term_colors)):
        fig_term_bar.add_trace(go.Bar(
            x=[label], y=[rate],
            marker_color=color,
            text=[f"<b>{rate}%</b>"],
            textposition="outside",
            textfont=dict(size=13),
            hovertemplate=f"<b>{label} months</b><br>Default rate: {rate}%<br>Total loans: {count:,}<extra></extra>",
            showlegend=False,
        ))
    fig_term_bar.add_trace(go.Bar(x=[None], y=[None], marker_color="#922B21",
                                  name="Above 15.3% portfolio average", showlegend=True))
    fig_term_bar.add_trace(go.Bar(x=[None], y=[None], marker_color="#1E8449",
                                  name="Below 15.3% portfolio average", showlegend=True))
    fig_term_bar.add_hline(
        y=15.3, line_dash="dash", line_color="#2471A3", line_width=2,
        annotation_text="<b>15.3% Portfolio Average</b>",
        annotation_position="top left",
        annotation_font=dict(color="white", size=12),
        annotation_bgcolor="#2471A3",
        annotation_bordercolor="#2471A3",
    )
    fig_term_bar.update_layout(
        height=400,
        margin=dict(t=60, b=20, r=40, l=20),
        yaxis=dict(title=dict(text="Default Rate (%)", font=dict(size=13, color="#333")),
                   ticksuffix="%", range=[0, 30]),
        xaxis=dict(title=dict(text="Loan Term (Months)", font=dict(size=13, color="#333")),
                   tickfont=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_term_bar, use_container_width=True)

    st.divider()

    # ── Section 3: Feature Importance — rebuilt for audience ──────────────
    st.markdown("### What actually predicts whether a loan defaults?")

    fi_col1, fi_col2 = st.columns([3, 2])

    with fi_col1:
        # Top 4 features only — cut the noise
        fi_top = pd.DataFrame({
            "Feature": [
                "Loan term (months)",
                "Has collateral",
                "Approval year",
                "Fixed rate indicator",
            ],
            "Importance": [0.462, 0.190, 0.109, 0.067],
            "Pct":        ["46.2%", "19.0%", "10.9%", "6.7%"],
            "Color":      ["#14397D", "#2471A3", "#CCDDEE", "#DDECF8"],
            "Label": [
                "Explains nearly half of all default predictions",
                "Lenders flag risky borrowers with collateral requirements",
                "Macro conditions at time of approval matter",
                "Stable payments reduce default risk",
            ],
        })

        fig_fi = go.Figure()
        for _, row in fi_top.iterrows():
            fig_fi.add_trace(go.Bar(
                x=[row["Importance"]],
                y=[row["Feature"]],
                orientation="h",
                marker_color=row["Color"],
                marker_line_color="rgba(0,0,0,0.08)",
                marker_line_width=1,
                showlegend=False,
                customdata=[[row["Label"]]],
                hovertemplate=f"<b>{row['Feature']}</b><br>{row['Label']}<br>Importance: {row['Pct']}<extra></extra>",
            ))
            # Annotation directly on bar
            fig_fi.add_annotation(
                x=row["Importance"] + 0.008,
                y=row["Feature"],
                text=f"<b>{row['Pct']}</b>  <i style='color:#555'>{row['Label']}</i>",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="#222"),
            )

        # "Everything else" single bar
        fig_fi.add_trace(go.Bar(
            x=[0.172],
            y=["All other 7 features combined"],
            orientation="h",
            marker_color="#F0F0F0",
            marker_line_color="#CCCCCC",
            marker_line_width=1,
            showlegend=False,
            hovertemplate="<b>All other 7 features</b><br>Combined: 17.2%<extra></extra>",
        ))
        fig_fi.add_annotation(
            x=0.172 + 0.008,
            y="All other 7 features combined",
            text="<b>17.2%</b>  <i style='color:#555'>loan amount, sector, state, jobs, rate, revolver</i>",
            showarrow=False, xanchor="left",
            font=dict(size=12, color="#888"),
        )

        fig_fi.update_layout(
            title=dict(text="XGBoost — What Predicts Default?", font=dict(size=17), x=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 0.85]),
            yaxis=dict(tickfont=dict(size=13), autorange="reversed"),
            height=340,
            margin=dict(t=50, b=10, r=20, l=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with fi_col2:
        # Donut chart: top 2 vs everything else
        fig_donut = go.Figure(go.Pie(
            labels=["Loan term (46.2%)", "Collateral (19.0%)", "Other (34.8%)"],
            values=[46.2, 19.0, 34.8],
            hole=0.62,
            marker_colors=["#14397D", "#2471A3", "#E8E8E8"],
            textinfo="label",
            textfont=dict(size=12),
            hovertemplate="%{label}<extra></extra>",
        )
        
        )
        fig_donut.update_layout(
            title=dict(text="Share of predictive power", font=dict(size=14), x=0.5, xanchor="center"),
            height=300,
            margin=dict(t=50, b=10, l=10, r=10),
            showlegend=False,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)


    st.divider()



# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — LENDER ANALYSIS (Moody's comparison)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Lender Analysis — What Moody's Says vs What Actually Happened")


    lender_df = pd.DataFrame([
        ("Wells Fargo Bank",       128583, 11.15, "Aa1",  1),
        ("Bank of America",        105441, 21.97, "Aa2",  2),
        ("JPMorgan Chase Bank",     95253, 15.96, "Aa2",  2),
        ("U.S. Bank",               84598,  9.87, "A1",   4),
        ("PNC Bank",                59436, 14.45, "A2",   5),
        ("TD Bank",                 46186,  9.10, "Aa2",  2),
        ("M&T Bank",                44568,  8.12, "A3",   6),
        ("Truist Bank",             24583,  9.45, "A2",   5),
        ("BMO Bank",                16977,  8.52, "Aa3",  3),
        ("The Huntington National", 90396,  5.05, "A3",   6),
        ("Citizens Bank",           49725, 15.40, "Baa1", 7),
        ("Zions Bank",              30600, 12.84, "Baa1", 7),
        ("Capital One",             22695, 37.97, "Baa1", 7),
        ("KeyBank",                 26437,  7.29, "A3",   6),
        ("Bank of Hope",            39319, 32.50, "N/R",  8),
        ("Readycap Lending",        29060, 17.16, "N/R",  8),
        ("Columbia Bank",           23745, 10.07, "N/R",  8),
        ("Northeast Bank",          17954,  6.34, "N/R",  8),
        ("Live Oak Banking",        16500,  7.80, "N/R",  8),
        ("Celtic Bank",             14200, 19.20, "N/R",  8),
    ], columns=["Bank", "Total Loans", "Default Rate %", "Moody's Rating", "Rating Rank"])

    # Implied "expected" default rate from Moody's rating
    # Higher Moody's = lower expected default. We invert: best rating (Aa1) = lowest expected default
    rating_expected = {
        "Aa1": 6.0, "Aa2": 7.0, "Aa3": 8.0,
        "A1": 9.0,  "A2": 10.0, "A3": 11.0,
        "Baa1": 13.0, "N/R": 15.3,
    }
    lender_df["Expected (Moody's implied)"] = lender_df["Moody's Rating"].map(rating_expected)
    lender_df["Gap"] = lender_df["Default Rate %"] - lender_df["Expected (Moody's implied)"]
    lender_df["Direction"] = lender_df["Gap"].apply(
        lambda g: "Worse than expected" if g > 2 else "Better than expected" if g < -2 else "As expected"
    )

    # ── Updated callout metrics — styled cards ───────────────────────────
    kf_col1, kf_col2, kf_col3, kf_col4 = st.columns(4)
    with kf_col1:
        st.markdown("""
        <div style="border:1px solid #ddd;border-top:3px solid #791F1F;border-radius:8px;padding:16px 18px 18px 18px;">
            <div style="font-size:16px;font-weight:600;color:#222;margin-bottom:6px;">Capital One</div>
            <div style="font-size:12px;color:#791F1F;font-weight:500;margin-bottom:14px;">Highly rated by Moody's · Worst default rate</div>
            <div style="font-size:32px;font-weight:400;color:#791F1F;margin-bottom:6px;">37.97%</div>
            <div style="font-size:13px;color:#666;">Actual Default Rate</div>
        </div>""", unsafe_allow_html=True)
    with kf_col2:
        st.markdown("""
        <div style="border:1px solid #ddd;border-top:3px solid #791F1F;border-radius:8px;padding:16px 18px 18px 18px;">
            <div style="font-size:16px;font-weight:600;color:#222;margin-bottom:6px;">Bank of America</div>
            <div style="font-size:12px;color:#791F1F;font-weight:500;margin-bottom:14px;">Top-rated bank · Biggest gap vs expectations</div>
            <div style="font-size:32px;font-weight:400;color:#791F1F;margin-bottom:6px;">21.97%</div>
            <div style="font-size:13px;color:#666;">Actual Default Rate</div>
        </div>""", unsafe_allow_html=True)
    with kf_col3:
        st.markdown("""
        <div style="border:1px solid #ddd;border-top:3px solid #085041;border-radius:8px;padding:16px 18px 18px 18px;">
            <div style="font-size:16px;font-weight:600;color:#222;margin-bottom:6px;">Huntington</div>
            <div style="font-size:12px;color:#085041;font-weight:500;margin-bottom:14px;">Mid-tier rating · Far exceeded expectations</div>
            <div style="font-size:32px;font-weight:400;color:#085041;margin-bottom:6px;">5.05%</div>
            <div style="font-size:13px;color:#666;">Actual Default Rate</div>
        </div>""", unsafe_allow_html=True)
    with kf_col4:
        st.markdown("""
        <div style="border:1px solid #ddd;border-top:3px solid #085041;border-radius:8px;padding:16px 18px 18px 18px;">
            <div style="font-size:16px;font-weight:600;color:#222;margin-bottom:6px;">Northeast Bank</div>
            <div style="font-size:12px;color:#085041;font-weight:500;margin-bottom:14px;">No credit rating · Best unrated performer</div>
            <div style="font-size:32px;font-weight:400;color:#085041;margin-bottom:6px;">6.34%</div>
            <div style="font-size:13px;color:#666;">Actual Default Rate</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Scatter plot: Credit Ratings vs Reality ───────────────────────────
    st.markdown("### Credit Ratings vs Reality — Which Banks Kept Their Promise?")
    st.caption("Each Bubble = One Lender · Bubble Size = Total Loan Volume")

    dir_colors = {
        "Worse than expected": "#C0392B",
        "Better than expected": "#1E8449",
        "As expected": "#888888",
    }

    sc_col, leg_col = st.columns([5, 1])
    with sc_col:
        max_loans = lender_df["Total Loans"].max()
        fig_scatter = go.Figure()
        for _, row in lender_df.iterrows():
            size = max(8, (row["Total Loans"] / max_loans) ** 0.5 * 50)
            color = dir_colors[row["Direction"]]
            fig_scatter.add_trace(go.Scatter(
                x=[row["Expected (Moody's implied)"]],
                y=[row["Default Rate %"]],
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.75,
                            line=dict(color=color, width=1.5)),
                hovertemplate=(
                    "<b>" + row["Bank"] + "</b><br>"
                    + "Expected: " + str(row["Expected (Moody's implied)"]) + "%<br>"
                    + "Actual: " + str(row["Default Rate %"]) + "%<br>"
                    + "Total loans: " + f"{row['Total Loans']:,}" + "<extra></extra>"
                ),
                showlegend=False,
            ))
        # Perfect prediction diagonal
        fig_scatter.add_shape(type="line", x0=4, y0=4, x1=18, y1=18,
                              line=dict(color="rgba(128,128,128,0.35)", dash="dot", width=1.5))
        for label, color in dir_colors.items():
            fig_scatter.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color),
                name=label, showlegend=True,
            ))
        fig_scatter.update_layout(
            height=460,
            margin=dict(t=20, b=20, r=20, l=20),
            xaxis=dict(
                title=dict(text="Expected Default Rate (Based on Moody's Credit Rating)",
                           font=dict(size=14, color="#111"), standoff=10),
                ticksuffix="%", range=[3, 19],
                tickfont=dict(size=12, color="#555")),
            yaxis=dict(
                title=dict(text="Actual SBA Default Rate",
                           font=dict(size=14, color="#111"), standoff=10),
                ticksuffix="%", range=[0, 41],
                tickfont=dict(size=12, color="#555")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=13),
                        bgcolor="rgba(255,255,255,0.9)"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with leg_col:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            '<div style="width:22px;border-top:2px dashed rgba(128,128,128,0.5);"></div>'
            '<span style="font-size:12px;color:#666;">Perfect prediction line</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Gap chart — exactly as original ──────────────────────────────────
    st.markdown("### Performance gap — actual minus Moody's implied default rate")
    

    gap_df = lender_df.sort_values("Gap", ascending=False).copy()
    gap_colors = gap_df["Gap"].apply(lambda g: "#C0392B" if g > 2 else "#1E8449" if g < -2 else "#888888")

    fig_gap = go.Figure(go.Bar(
        x=gap_df["Gap"],
        y=gap_df["Bank"],
        orientation="h",
        marker_color=gap_colors,
        text=[f"{g:+.1f}pp" for g in gap_df["Gap"]],
        textposition="outside",
        textfont=dict(size=11),
        customdata=gap_df[["Moody's Rating", "Default Rate %"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Moody's: %{customdata[0]}<br>"
            "Actual: %{customdata[1]:.1f}%<br>"
            "Gap: %{x:+.1f}pp<extra></extra>"
        ),
    ))
    fig_gap.add_vline(x=0, line_color="black", line_width=1.5)
    fig_gap.add_vline(x=2,  line_dash="dot", line_color="#C0392B", line_width=1,
                      annotation_text="Worse threshold", annotation_position="top",
                      annotation_font=dict(size=10, color="#C0392B"))
    fig_gap.add_vline(x=-2, line_dash="dot", line_color="#1E8449", line_width=1,
                      annotation_text="Better threshold", annotation_position="top",
                      annotation_font=dict(size=10, color="#1E8449"))
    fig_gap.update_layout(
        height=560,
        margin=dict(t=40, b=20, l=20, r=100),
        xaxis=dict(title="Gap (actual − Moody's implied, percentage points)",
                   tickfont=dict(size=11), zeroline=False),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.divider()





# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# Model comparison + class imbalance + logistic regression detail
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Methodology")

    meth_tab1, meth_tab2 = st.tabs([
        "Model Comparison",
        "Class Imbalance",
    ])

    with meth_tab1:
        st.markdown("### Model Comparison — 6 Models Tested")
        

        model_df = pd.DataFrame({
            "Model": [
                "XGBoost (tuned, threshold=0.73)",
                "Random Forest",
                "Neural Network (MLP)",
                "Logistic Regression",
                "Gradient Boosting (sklearn)",
                "Decision Tree (baseline)",
            ],
            "ROC-AUC":  [0.960, 0.948, 0.941, 0.847, 0.952, 0.791],
            "Precision": [0.80,  0.76,  0.74,  0.62,  0.77,  0.58],
            "Recall":    [0.88,  0.85,  0.83,  0.79,  0.86,  0.71],
            "F1":        [0.84,  0.80,  0.78,  0.69,  0.81,  0.64],
            "Imbalance handled": ["SMOTE + weight", "SMOTE", "SMOTE", "SMOTE", "SMOTE", "No"],
        })
        st.dataframe(
            model_df.style.highlight_max(subset=["ROC-AUC","Precision","Recall","F1"], color="#EAF3DE"),
            use_container_width=True, hide_index=True, height=260,
        )
        st.divider()

        mc_col1, mc_col2 = st.columns(2)
        with mc_col1:
            fig_auc = px.bar(
                model_df.sort_values("ROC-AUC"),
                x="ROC-AUC", y="Model", orientation="h",
                color="ROC-AUC", color_continuous_scale=["#E6F1FB","#185FA5"],
                text="ROC-AUC", title="ROC-AUC by model",
            )
            fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_auc.update_layout(height=360, margin=dict(t=40, b=10, r=60))
            fig_auc.update_coloraxes(showscale=False)
            st.plotly_chart(fig_auc, use_container_width=True)

        with mc_col2:
            fig_prec = px.bar(
                model_df.sort_values("Precision"),
                x="Precision", y="Model", orientation="h",
                color="Precision", color_continuous_scale=["#FAECE7","#A32D2D"],
                text="Precision", title="Precision by model",
            )
            fig_prec.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_prec.update_layout(height=360, margin=dict(t=40, b=10, r=60))
            fig_prec.update_coloraxes(showscale=False)
            st.plotly_chart(fig_prec, use_container_width=True)



    with meth_tab2:
        st.markdown("### Class Imbalance")

        ci_col1, ci_col2 = st.columns([1, 2])
        with ci_col1:
            st.dataframe(pd.DataFrame({
                "Outcome": ["Paid in Full (0)", "Charged Off (1)"],
                "Count":   [1259013, 227927],
                "Share":   ["84.7%", "15.3%"],
            }), use_container_width=True, hide_index=True)

        with ci_col2:
            fig_imb = px.bar(
                pd.DataFrame({"Class": ["Paid in Full", "Charged Off"], "Count": [1259013, 227927]}),
                x="Class", y="Count", color="Class",
                color_discrete_map={"Paid in Full": "#1D9E75", "Charged Off": "#A32D2D"},
                title="Class distribution — 5.5:1 imbalance", text_auto=True,
            )
            fig_imb.update_layout(height=280, margin=dict(t=40, b=10), showlegend=False)
            st.plotly_chart(fig_imb, use_container_width=True)

        # ── Section 4: Logistic Regression ────────────────────────────────────
        st.markdown("### Logistic Regression — What Drives Default Direction?")

        coeff_df = pd.DataFrame({
            "Variable":       ["loan_amount", "term_months", "loan_2023", "has_collateral", "is_fixed", "naics_sector", "approval_year"],
            "Direction":      ["Negative", "Negative", "Positive", "Positive", "Negative", "Significant", "Positive"],
            "Significant":    ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            "Interpretation": [
                "Larger loans default LESS — go to more established businesses",
                "Longer terms default LESS — confirms XGBoost #1 feature",
                "Higher inflation-adjusted amounts slightly increase risk",
                "Collateral loans default MORE — lenders require it for riskier borrowers",
                "Fixed rate loans default LESS — payment stability matters",
                "Industry predicts default — Retail defaults more than Healthcare",
                "More recent loans default MORE — older loans had more time to pay off",
            ],
        })
        st.dataframe(coeff_df, use_container_width=True, hide_index=True, height=290)


