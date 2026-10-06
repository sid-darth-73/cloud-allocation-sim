"""
Cloud Resource Allocation Simulator - Streamlit Dashboard.

An interactive dashboard comparing traditional and resource-aware allocation algorithms
(Round Robin, First Fit, Best Fit Euclidean, Resource-Aware Cosine Alignment)
under multi-dimensional constraints (CPU, RAM, Bandwidth).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from algorithms import (
    BestFitAllocator,
    FirstFitAllocator,
    ResourceAwareAllocator,
    RoundRobinAllocator,
)
from models import Cluster, PhysicalMachine, Request
from metrics import (
    calculate_cluster_metrics,
    calculate_pm_metrics,
    generate_comparison_table,
    pm_metrics_to_dataframe,
)
from simulator import (
    generate_synthetic_workload,
    get_static_baseline_requests,
    run_benchmark,
    run_dynamic_timeline_simulation,
)

# ---------------------------------------------------------
# Page Configuration & CSS Theming
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cloud Resource Allocation Simulator",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888888;
        margin-bottom: 4px;
    }
    .metric-card-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-active { background-color: #2E7D32; color: #E8F5E9; }
    .badge-idle { background-color: #757575; color: #FAFAFA; }
    .badge-cpu { background-color: #E53935; color: #FFEBEE; }
    .badge-ram { background-color: #1E88E5; color: #E3F2FD; }
    .badge-bw { background-color: #8E24AA; color: #F3E5F5; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar Controls & Cluster Setup
# ---------------------------------------------------------
st.sidebar.title("⚙️ Simulation Settings")

workload_mode = st.sidebar.radio(
    "Workload Source",
    [
        "Static Lab Baseline (R1-R4)",
        "Dynamic Synthetic Workload (20-50 Reqs)",
        "Custom Workload Builder",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ PM Cluster Configuration")
cluster_size = st.sidebar.slider("Number of PMs", min_value=2, max_value=8, value=4, step=1)
pm_cpu_cap = st.sidebar.number_input("PM CPU Capacity (Cores)", min_value=2.0, max_value=64.0, value=8.0, step=1.0)
pm_ram_cap = st.sidebar.number_input("PM RAM Capacity (GB)", min_value=4.0, max_value=128.0, value=16.0, step=2.0)
pm_bw_cap = st.sidebar.number_input("PM Bandwidth Capacity (Mbps)", min_value=10.0, max_value=500.0, value=50.0, step=10.0)

# Build Cluster template
cluster_template = Cluster.create_default(
    num_pms=cluster_size,
    cpu_cap=pm_cpu_cap,
    ram_cap=pm_ram_cap,
    bw_cap=pm_bw_cap,
)

# Workload Generation Logic
requests: list[Request] = []

if workload_mode == "Static Lab Baseline (R1-R4)":
    requests = get_static_baseline_requests()
elif workload_mode == "Dynamic Synthetic Workload (20-50 Reqs)":
    st.sidebar.subheader("🎲 Synthetic Workload Parameters")
    num_reqs = st.sidebar.slider("Number of Requests", min_value=10, max_value=50, value=30, step=5)
    seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
    min_cpu, max_cpu = st.sidebar.slider("CPU Request Range (Cores)", 1.0, 8.0, (1.0, 6.0))
    min_ram, max_ram = st.sidebar.slider("RAM Request Range (GB)", 1.0, 16.0, (2.0, 12.0))
    min_bw, max_bw = st.sidebar.slider("Bandwidth Request Range (Mbps)", 5.0, 50.0, (5.0, 35.0))
    requests = generate_synthetic_workload(
        num_requests=num_reqs,
        seed=seed,
        cpu_range=(min_cpu, max_cpu),
        ram_range=(min_ram, max_ram),
        bw_range=(min_bw, max_bw),
    )
else:
    st.sidebar.subheader("✍️ Add Custom Request")
    with st.sidebar.form("custom_req_form", clear_on_submit=False):
        c_id = st.text_input("Request ID", value=f"R{len(st.session_state.get('custom_reqs', [])) + 1}")
        c_cpu = st.number_input("CPU Cores", min_value=0.5, max_value=32.0, value=2.0, step=0.5)
        c_ram = st.number_input("RAM GB", min_value=0.5, max_value=64.0, value=4.0, step=0.5)
        c_bw = st.number_input("Bandwidth Mbps", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
        c_add = st.form_submit_button("Add Request")
        if c_add:
            if "custom_reqs" not in st.session_state:
                st.session_state.custom_reqs = get_static_baseline_requests()
            st.session_state.custom_reqs.append(
                Request(id=c_id, cpu=c_cpu, ram=c_ram, bw=c_bw, arrival_time=len(st.session_state.custom_reqs), lifetime=10.0)
            )

    if "custom_reqs" not in st.session_state:
        st.session_state.custom_reqs = get_static_baseline_requests()

    if st.sidebar.button("Reset to Default Baseline"):
        st.session_state.custom_reqs = get_static_baseline_requests()

    requests = st.session_state.custom_reqs


# ---------------------------------------------------------
# Run Benchmark
# ---------------------------------------------------------
benchmark_results = run_benchmark(requests, cluster_template=cluster_template)
comparison_df = generate_comparison_table(benchmark_results)


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("☁️ Cloud Resource Allocation Lab Simulator")
st.markdown(
    """
    **Comparative Evaluation of Multi-Dimensional Resource Allocation Algorithms**  
    Analyzing **Round Robin (RR)**, **First Fit (FF)**, **Best Fit (BF - Euclidean)**, and **Resource-Aware (RAA - Cosine Alignment)**
    under multi-resource constraints (CPU, RAM, Bandwidth) with server consolidation and resource skew metrics.
    """
)

# Tabs
tab_summary, tab_hosts, tab_steps, tab_timeline, tab_math = st.tabs(
    [
        "📊 Benchmark & Comparison",
        "🖥️ PM-Level Utilization",
        "🔍 Step-by-Step Visualizer",
        "⏱️ Dynamic Lifecycle Timeline",
        "📐 Formulations & Lab Theory",
    ]
)


# =========================================================
# TAB 1: Benchmark & Comparison Cards
# =========================================================
with tab_summary:
    st.subheader("📋 Request Queue Inspector")
    col_q1, col_q2 = st.columns([3, 1])

    req_data = [r.to_dict() for r in requests]
    req_df = pd.DataFrame(req_data)

    with col_q1:
        st.dataframe(
            req_df[["id", "cpu", "ram", "bw", "arrival_time", "lifetime"]].rename(
                columns={
                    "id": "Request ID",
                    "cpu": "CPU (Cores)",
                    "ram": "RAM (GB)",
                    "bw": "Bandwidth (Mbps)",
                    "arrival_time": "Arrival (s)",
                    "lifetime": "Lifetime (s)",
                }
            ),
            use_container_width=True,
            height=200 if len(requests) <= 5 else 260,
        )

    with col_q2:
        tot_cpu_req = sum(r.cpu for r in requests)
        tot_ram_req = sum(r.ram for r in requests)
        tot_bw_req = sum(r.bw for r in requests)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Demanded Resources</div>
                <div style="font-size: 1.1rem; font-weight:600; margin-top:6px;">
                    <span class="badge badge-cpu">CPU</span> <b>{tot_cpu_req:.1f}</b> Cores<br>
                    <span class="badge badge-ram">RAM</span> <b>{tot_ram_req:.1f}</b> GB<br>
                    <span class="badge badge-bw">BW</span> <b>{tot_bw_req:.1f}</b> Mbps
                </div>
                <div style="margin-top:8px; font-size:0.8rem; color:#888;">
                    Cluster Total: {cluster_template.total_cpu_capacity:.0f} Cores | {cluster_template.total_ram_capacity:.0f} GB | {cluster_template.total_bw_capacity:.0f} Mbps
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🏆 Side-by-Side Algorithm Comparison Cards")

    algo_cols = st.columns(4)
    card_colors = {
        "Round Robin (RR)": "#FFA726",
        "First Fit (FF)": "#42A5F5",
        "Best Fit (BF)": "#AB47BC",
        "Resource-Aware (RAA)": "#26A69A",
    }

    for idx, (algo_name, data) in enumerate(benchmark_results.items()):
        metrics = data["cluster_metrics"]
        alloc_res = data["allocation_result"]
        color = card_colors.get(algo_name, "#42A5F5")

        with algo_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div class="metric-card-title" style="color: {color}; font-size: 0.95rem;">{algo_name}</div>
                    <div style="margin: 10px 0;">
                        <span style="font-size:0.85rem; color:#888;">Active PMs:</span>
                        <span style="font-size:1.4rem; font-weight:700; color:{color};"> {metrics['active_pms']}</span> / {metrics['total_pms']}
                    </div>
                    <div style="font-size:0.85rem; line-height: 1.6;">
                        • <b>Avg Cluster Util:</b> {metrics['average_cluster_util_pct']}%<br>
                        • <b>Resource Imbalance:</b> <code>{metrics['average_imbalance_index']:.6f}</code><br>
                        • <b>Consolidation:</b> {metrics['server_consolidation_pct']}%<br>
                        • <b>Accepted:</b> {alloc_res.allocated_count}/{alloc_res.total_requests}<br>
                        • <b>Runtime:</b> {metrics['execution_time_us']} μs
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Detailed Benchmark Matrix")
    st.dataframe(comparison_df, use_container_width=True)

    # Visualizations: Grouped Bar Chart & Radar Chart
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Cluster Resource Utilization by Algorithm")
        fig_util = go.Figure()
        algos = list(benchmark_results.keys())
        cpu_vals = [benchmark_results[a]["cluster_metrics"]["cluster_cpu_util_pct"] for a in algos]
        ram_vals = [benchmark_results[a]["cluster_metrics"]["cluster_ram_util_pct"] for a in algos]
        bw_vals = [benchmark_results[a]["cluster_metrics"]["cluster_bw_util_pct"] for a in algos]

        fig_util.add_trace(go.Bar(name="CPU Util %", x=algos, y=cpu_vals, marker_color="#EF5350"))
        fig_util.add_trace(go.Bar(name="RAM Util %", x=algos, y=ram_vals, marker_color="#42A5F5"))
        fig_util.add_trace(go.Bar(name="Bandwidth Util %", x=algos, y=bw_vals, marker_color="#AB47BC"))

        fig_util.update_layout(
            barmode="group",
            yaxis_title="Utilization (%)",
            xaxis_title="Algorithm",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=30, b=40),
            height=380,
        )
        st.plotly_chart(fig_util, use_container_width=True)

    with col_chart2:
        st.subheader("🕸️ Multi-Criteria Radar Comparison")
        # Normalize metrics for radar comparison
        categories = ["Consolidation %", "Cluster Util %", "Resource Balance", "Placement Rate %", "Speed Score"]
        fig_radar = go.Figure()

        # Find max runtime to invert speed
        max_time = max(benchmark_results[a]["cluster_metrics"]["execution_time_us"] for a in algos) or 1.0

        for a in algos:
            m = benchmark_results[a]["cluster_metrics"]
            res = benchmark_results[a]["allocation_result"]
            # Balance score: 100 * (1 / (1 + 50 * imbalance))
            balance_score = round(100.0 / (1.0 + 50.0 * m["average_imbalance_index"]), 1)
            # Speed score: 100 * (min_time / time)
            min_time = min(benchmark_results[x]["cluster_metrics"]["execution_time_us"] for x in algos)
            speed_score = round(100.0 * (min_time / max(m["execution_time_us"], 1e-3)), 1)

            values = [
                m["server_consolidation_pct"],
                m["average_cluster_util_pct"],
                balance_score,
                res.acceptance_ratio_pct,
                speed_score,
            ]
            # close polygon
            values.append(values[0])

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    name=a,
                    fill="toself",
                    fillcolor=card_colors.get(a, "#42A5F5") + "22",
                    line=dict(color=card_colors.get(a, "#42A5F5"), width=2),
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=30, b=50),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# =========================================================
# TAB 2: Host (PM) Level Deep Dive
# =========================================================
with tab_hosts:
    st.subheader("🖥️ Host-Level Resource Allocation & Imbalance")

    selected_algo = st.selectbox(
        "Select Algorithm to Inspect Host Breakdown",
        list(benchmark_results.keys()),
        index=0,
    )

    algo_cluster: Cluster = benchmark_results[selected_algo]["cluster"]
    pm_df = benchmark_results[selected_algo]["pm_metrics_df"]

    # PM Host Cards
    pm_cols = st.columns(len(algo_cluster.machines))
    for idx, pm in enumerate(algo_cluster.machines):
        with pm_cols[idx]:
            state_badge = '<span class="badge badge-active">ACTIVE</span>' if pm.is_active else '<span class="badge badge-idle">IDLE</span>'
            req_list = ", ".join([r.id for r in pm.allocated_requests]) if pm.allocated_requests else "None"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b>{pm.id}</b>
                        {state_badge}
                    </div>
                    <div style="margin: 8px 0; font-size:0.8rem; color:#888;">
                        Requests: <b>{req_list}</b>
                    </div>
                    <div style="font-size:0.85rem; margin-top:6px;">
                        <div>CPU: <b>{pm.allocated_cpu:.1f}</b> / {pm.cpu_capacity:.0f} Cores ({pm.cpu_utilization_pct:.1f}%)</div>
                        <div>RAM: <b>{pm.allocated_ram:.1f}</b> / {pm.ram_capacity:.0f} GB ({pm.ram_utilization_pct:.1f}%)</div>
                        <div>BW: <b>{pm.allocated_bw:.1f}</b> / {pm.bw_capacity:.0f} Mbps ({pm.bw_utilization_pct:.1f}%)</div>
                    </div>
                    <div style="margin-top:8px; font-size:0.78rem; color:#E0E0E0;">
                        Skew Index: <code>{pm.imbalance_index:.6f}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Visual Grouped Bar Chart of PM-Level Resource Utilizations
    st.markdown("---")
    col_pm_chart1, col_pm_chart2 = st.columns(2)

    with col_pm_chart1:
        st.markdown(f"#### PM Utilization Breakdown - {selected_algo}")
        fig_pm_bars = go.Figure()
        pms = [pm.id for pm in algo_cluster.machines]
        cpu_pms = [pm.cpu_utilization_pct for pm in algo_cluster.machines]
        ram_pms = [pm.ram_utilization_pct for pm in algo_cluster.machines]
        bw_pms = [pm.bw_utilization_pct for pm in algo_cluster.machines]

        fig_pm_bars.add_trace(go.Bar(name="CPU %", x=pms, y=cpu_pms, marker_color="#EF5350"))
        fig_pm_bars.add_trace(go.Bar(name="RAM %", x=pms, y=ram_pms, marker_color="#42A5F5"))
        fig_pm_bars.add_trace(go.Bar(name="BW %", x=pms, y=bw_pms, marker_color="#AB47BC"))

        fig_pm_bars.update_layout(
            barmode="group",
            yaxis_title="Utilization (%)",
            xaxis_title="Physical Machine",
            yaxis=dict(range=[0, 105]),
            height=360,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_pm_bars, use_container_width=True)

    with col_pm_chart2:
        st.markdown("#### Cross-Algorithm PM Imbalance / Skew Comparison")
        # Compare Imbalance across all 4 algorithms for each PM
        fig_imb = go.Figure()
        for a_name in benchmark_results.keys():
            c: Cluster = benchmark_results[a_name]["cluster"]
            imbs = [calculate_pm_metrics(p)["imbalance_index"] for p in c.machines]
            fig_imb.add_trace(go.Bar(name=a_name, x=pms, y=imbs, marker_color=card_colors.get(a_name)))

        fig_imb.update_layout(
            barmode="group",
            yaxis_title="Imbalance / Skewness Index",
            xaxis_title="Physical Machine",
            height=360,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_imb, use_container_width=True)

    st.markdown("#### Full Host Telemetry Table")
    st.dataframe(
        pm_df[[
            "pm_id", "is_active", "num_requests", "requests",
            "allocated_cpu", "allocated_ram", "allocated_bw",
            "cpu_util_pct", "ram_util_pct", "bw_util_pct",
            "imbalance_index"
        ]],
        use_container_width=True,
    )


# =========================================================
# TAB 3: Step-by-Step Allocation Visualizer
# =========================================================
with tab_steps:
    st.subheader("🔍 Step-by-Step Request Placement Visualizer")
    st.markdown(
        "Observe the sequential decision-making process for each incoming request, "
        "including host candidate scores and mathematical rationales."
    )

    step_algo = st.selectbox(
        "Select Allocation Algorithm for Visualizer",
        list(benchmark_results.keys()),
        index=3,  # Resource Aware
        key="step_algo_select",
    )

    steps = benchmark_results[step_algo]["allocation_result"].steps
    if steps:
        step_idx = st.slider(
            "Select Request Step",
            min_value=1,
            max_value=len(steps),
            value=1,
            step=1,
            format="Step %d",
        )

        curr_step = steps[step_idx - 1]
        col_s1, col_s2 = st.columns([2, 3])

        with col_s1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Incoming Request: {curr_step.request_id}</div>
                    <div style="font-size: 1.1rem; margin-top:8px;">
                        • CPU Required: <b>{curr_step.request_cpu:.1f}</b> Cores<br>
                        • RAM Required: <b>{curr_step.request_ram:.1f}</b> GB<br>
                        • Bandwidth Required: <b>{curr_step.request_bw:.1f}</b> Mbps
                    </div>
                    <hr style="margin: 12px 0;">
                    <div style="font-size: 0.95rem;">
                        <b>Placement Status:</b> {'✅ SUCCESS' if curr_step.success else '❌ REJECTED'}<br>
                        <b>Assigned Host:</b> <span style="color:#1E88E5; font-weight:700;">{curr_step.allocated_pm_id}</span><br>
                        <b>Decision Metric:</b> <code>{curr_step.decision_metric_name}</code> = <code>{curr_step.decision_metric_value:.4f}</code>
                    </div>
                    <div style="margin-top:10px; font-size:0.85rem; color:#BBB; font-style: italic;">
                        "{curr_step.rationale}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_s2:
            st.markdown("#### Candidate Host Evaluation Scores")
            if curr_step.pm_candidate_scores:
                score_df = pd.DataFrame(
                    [
                        {"Host": pm_id, f"Score ({curr_step.decision_metric_name})": score}
                        for pm_id, score in curr_step.pm_candidate_scores.items()
                    ]
                )
                fig_scores = px.bar(
                    score_df,
                    x="Host",
                    y=f"Score ({curr_step.decision_metric_name})",
                    color="Host",
                    text_auto=True,
                )
                fig_scores.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_scores, use_container_width=True)

        # Simulation playback up to this step
        st.markdown("---")
        st.markdown(f"#### Cluster State at Step {step_idx}")

        # Replay allocations up to step_idx
        replay_cluster = cluster_template.clone()
        algo_class = type(benchmark_results[step_algo]["allocator"])
        replay_alloc = algo_class()
        replay_alloc.reset_state()

        for s in range(step_idx):
            req_s = requests[s]
            replay_alloc.allocate_single(req_s, replay_cluster, step_num=s + 1)

        replay_cols = st.columns(len(replay_cluster.machines))
        for idx, pm in enumerate(replay_cluster.machines):
            with replay_cols[idx]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <b>{pm.id}</b> ({'Active' if pm.is_active else 'Idle'})
                        <div style="font-size:0.8rem; margin:6px 0;">
                            Hosts: {[r.id for r in pm.allocated_requests]}
                        </div>
                        <div style="font-size:0.85rem;">
                            CPU: {pm.allocated_cpu:.1f} / {pm.cpu_capacity:.0f} C ({pm.cpu_utilization_pct:.1f}%)<br>
                            RAM: {pm.allocated_ram:.1f} / {pm.ram_capacity:.0f} GB ({pm.ram_utilization_pct:.1f}%)<br>
                            BW: {pm.allocated_bw:.1f} / {pm.bw_capacity:.0f} M ({pm.bw_utilization_pct:.1f}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =========================================================
# TAB 4: Dynamic Lifecycle Timeline
# =========================================================
with tab_timeline:
    st.subheader("⏱️ Dynamic Discrete-Event Lifecycle Timeline")
    st.markdown(
        "Simulates arrival and departure lifecycles over continuous time. "
        "As requests complete their execution duration, resources are released and hosts may transition back to idle."
    )

    if st.button("▶️ Run Dynamic Timeline Simulation", key="btn_run_timeline"):
        with st.spinner("Executing event-driven simulation for all algorithms..."):
            timeline_dict = run_dynamic_timeline_simulation(requests, cluster_template=cluster_template)
            st.session_state.timeline_results = timeline_dict

    if "timeline_results" in st.session_state:
        t_results = st.session_state.timeline_results

        # Active PMs over time
        fig_time_pms = go.Figure()
        for a_name, df_t in t_results.items():
            fig_time_pms.add_trace(
                go.Scatter(
                    x=df_t["time"],
                    y=df_t["active_pms"],
                    mode="lines+markers",
                    name=a_name,
                    line=dict(color=card_colors.get(a_name), width=2),
                )
            )

        fig_time_pms.update_layout(
            title="Active Servers (PMs) Over Time",
            xaxis_title="Simulation Time (seconds)",
            yaxis_title="Active PM Count",
            height=360,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_time_pms, use_container_width=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            # Average Utilization over time
            fig_time_util = go.Figure()
            for a_name, df_t in t_results.items():
                fig_time_util.add_trace(
                    go.Scatter(
                        x=df_t["time"],
                        y=df_t["avg_cluster_util_pct"],
                        mode="lines",
                        name=a_name,
                        line=dict(color=card_colors.get(a_name), width=2),
                    )
                )
            fig_time_util.update_layout(
                title="Cluster Resource Utilization (%) Over Time",
                xaxis_title="Time (s)",
                yaxis_title="Average Utilization (%)",
                height=340,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_time_util, use_container_width=True)

        with col_t2:
            # Resource Skew / Imbalance over time
            fig_time_skew = go.Figure()
            for a_name, df_t in t_results.items():
                fig_time_skew.add_trace(
                    go.Scatter(
                        x=df_t["time"],
                        y=df_t["imbalance_index"],
                        mode="lines",
                        name=a_name,
                        line=dict(color=card_colors.get(a_name), width=2),
                    )
                )
            fig_time_skew.update_layout(
                title="Resource Skew / Imbalance Over Time",
                xaxis_title="Time (s)",
                yaxis_title="Imbalance Index",
                height=340,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_time_skew, use_container_width=True)
    else:
        st.info("Click the 'Run Dynamic Timeline Simulation' button above to simulate arrival/departure lifecycles.")


# =========================================================
# TAB 5: Formulations & Lab Theory
# =========================================================
with tab_math:
    st.subheader("📐 Mathematical Formulations & Cloud Computing Lab Theory")

    st.markdown(
        r"""
        ### 1. Multi-Dimensional Resource Request & Host Model
        Each cloud request $R_i$ demands resources across $D=3$ dimensions:
        $$R_i = \left( r_i^{\text{cpu}}, r_i^{\text{ram}}, r_i^{\text{bw}} \right)$$

        A Physical Machine (PM) $m$ has capacities $C_m = (C^{\text{cpu}}, C^{\text{ram}}, C^{\text{bw}})$.
        At any point, host utilization fractions $u_{m, d} \in [0, 1]$ are:
        $$u_{m, \text{cpu}} = \frac{\sum_{i \in m} r_i^{\text{cpu}}}{C^{\text{cpu}}}, \quad
          u_{m, \text{ram}} = \frac{\sum_{i \in m} r_i^{\text{ram}}}{C^{\text{ram}}}, \quad
          u_{m, \text{bw}} = \frac{\sum_{i \in m} r_i^{\text{bw}}}{C^{\text{bw}}}$$

        ---

        ### 2. Multi-Dimensional Resource Imbalance / Skewness Index
        When resources are consumed unevenly (e.g. CPU exhausted while RAM remains idle), 
        the server suffers from **resource stranding**. To quantify this, the resource skewness index is computed as:
        $$u_{m, \text{avg}} = \frac{u_{m, \text{cpu}} + u_{m, \text{ram}} + u_{m, \text{bw}}}{3}$$

        $$\text{Imbalance}(m) = \frac{(u_{m, \text{cpu}} - u_{m, \text{avg}})^2 + 
                                       (u_{m, \text{ram}} - u_{m, \text{avg}})^2 + 
                                       (u_{m, \text{bw}} - u_{m, \text{avg}})^2}{3}$$
        *A lower imbalance index indicates balanced consumption across all dimensions.*

        ---

        ### 3. Allocation Algorithm Formulations

        #### A. Round Robin (RR)
        Assigns incoming requests sequentially in circular order using pointer $k$:
        $$k \leftarrow (k + 1) \pmod N$$
        - **Strength:** Zero computational overhead, evenly spreads requests.
        - **Weakness:** Activates all PMs regardless of load (terrible consolidation), leading to high energy consumption and higher imbalance.

        #### B. First Fit (FF)
        Scans PMs linearly starting from index $0$:
        $$\text{PM}^* = \arg\min_m \{ m \mid m \text{ can allocate } R_i \}$$
        - **Strength:** Packs lower-index PMs first, leaving trailing PMs idle (high server consolidation).
        - **Weakness:** Oblivious to multi-dimensional skew.

        #### C. Best Fit (BF) — Normalized Euclidean Residual Space
        Evaluates post-allocation remaining residual capacity vector $\mathbf{v}^{\text{rem}}_m$:
        $$d_{\text{residual}}(m) = \sqrt{\left(\frac{C^{\text{cpu}} - \sum r^{\text{cpu}}}{C^{\text{cpu}}}\right)^2 + 
                                        \left(\frac{C^{\text{ram}} - \sum r^{\text{ram}}}{C^{\text{ram}}}\right)^2 + 
                                        \left(\frac{C^{\text{bw}} - \sum r^{\text{bw}}}{C^{\text{bw}}}\right)^2}$$
        $$\text{PM}^* = \arg\min_m d_{\text{residual}}(m)$$
        - **Strength:** Minimizes leftover slack space on hosts, producing tight bin packing.

        #### D. Resource-Aware Allocation (RAA) — Vector Alignment via Cosine Similarity
        Normalizes the request demand vector $\mathbf{R}$ and the host residual capacity vector $\mathbf{A}_m$:
        $$\mathbf{R} = \left[ \frac{r^{\text{cpu}}}{C^{\text{cpu}}}, \frac{r^{\text{ram}}}{C^{\text{ram}}}, \frac{r^{\text{bw}}}{C^{\text{bw}}} \right], \quad
          \mathbf{A}_m = \left[ \frac{\text{rem}_{m}^{\text{cpu}}}{C^{\text{cpu}}}, \frac{\text{rem}_{m}^{\text{ram}}}{C^{\text{ram}}}, \frac{\text{rem}_{m}^{\text{bw}}}{C^{\text{bw}}} \right]$$

        $$\text{CosineSimilarity}(\mathbf{R}, \mathbf{A}_m) = \frac{\mathbf{R} \cdot \mathbf{A}_m}{\|\mathbf{R}\|_2 \|\mathbf{A}_m\|_2}$$
        $$\text{Score}(m) = \text{CosineSimilarity}(\mathbf{R}, \mathbf{A}_m) - \lambda \cdot \text{Imbalance}_{\text{post}}(m)$$
        - **Strength:** Balances multi-dimensional consumption by pairing complementary requests (e.g. CPU-heavy with RAM-heavy), preventing stranded capacity and minimizing resource skew.
        """
    )
