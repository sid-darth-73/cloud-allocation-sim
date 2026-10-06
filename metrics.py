"""
Evaluation Metrics for Multi-Dimensional Cloud Resource Allocation.

Computes:
1. Individual PM Resource Utilization: CPU (%), RAM (%), Bandwidth (%)
2. Average Cluster Resource Utilization (%)
3. Total Active PMs Used (Server Consolidation Efficiency)
4. Multi-dimensional Resource Imbalance / Skewness Index:
   Imbalance = ((u_cpu - u_avg)^2 + (u_ram - u_avg)^2 + (u_bw - u_avg)^2) / 3
5. Algorithm Execution Time in microseconds (μs)
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from models import Cluster, PhysicalMachine


def calculate_pm_imbalance(pm: PhysicalMachine) -> float:
    """Calculate the multi-dimensional resource imbalance index for a single PM.
    
    Formula:
        u_avg = (u_cpu + u_ram + u_bw) / 3.0
        Imbalance = ((u_cpu - u_avg)^2 + (u_ram - u_avg)^2 + (u_bw - u_avg)^2) / 3.0
        
    Returns 0.0 if the PM is inactive (idle).
    """
    if not pm.is_active:
        return 0.0
    u_cpu = pm.cpu_utilization
    u_ram = pm.ram_utilization
    u_bw = pm.bw_utilization
    u_avg = (u_cpu + u_ram + u_bw) / 3.0
    return ((u_cpu - u_avg) ** 2 + (u_ram - u_avg) ** 2 + (u_bw - u_avg) ** 2) / 3.0


def calculate_pm_metrics(pm: PhysicalMachine) -> Dict[str, Any]:
    """Extract complete metrics for an individual Physical Machine."""
    return {
        "pm_id": pm.id,
        "is_active": pm.is_active,
        "num_requests": len(pm.allocated_requests),
        "requests": [r.id for r in pm.allocated_requests],
        "allocated_cpu": round(pm.allocated_cpu, 2),
        "allocated_ram": round(pm.allocated_ram, 2),
        "allocated_bw": round(pm.allocated_bw, 2),
        "cpu_capacity": pm.cpu_capacity,
        "ram_capacity": pm.ram_capacity,
        "bw_capacity": pm.bw_capacity,
        "cpu_util_pct": round(pm.cpu_utilization_pct, 2),
        "ram_util_pct": round(pm.ram_utilization_pct, 2),
        "bw_util_pct": round(pm.bw_utilization_pct, 2),
        "avg_util_pct": round(pm.average_utilization * 100.0, 2),
        "imbalance_index": round(calculate_pm_imbalance(pm), 6),
    }


def calculate_cluster_metrics(cluster: Cluster, execution_time_us: float = 0.0) -> Dict[str, Any]:
    """Calculate aggregated cluster-level metrics.
    
    Args:
        cluster: Evaluated cluster state.
        execution_time_us: Algorithm runtime in microseconds.
        
    Returns:
        Dictionary of consolidated cluster metrics.
    """
    active_pms = cluster.active_machines
    active_count = len(active_pms)
    total_pms = len(cluster.machines)

    # Imbalance: mean across active PMs (if any), otherwise 0.0
    imbalances = [calculate_pm_imbalance(pm) for pm in active_pms]
    avg_imbalance = (sum(imbalances) / active_count) if active_count > 0 else 0.0
    max_imbalance = max(imbalances) if imbalances else 0.0

    # Total allocated requests
    total_allocated_reqs = sum(len(pm.allocated_requests) for pm in cluster.machines)

    return {
        "total_pms": total_pms,
        "active_pms": active_count,
        "idle_pms": total_pms - active_count,
        "server_consolidation_pct": round(((total_pms - active_count) / total_pms * 100.0), 2) if total_pms > 0 else 0.0,
        "total_allocated_requests": total_allocated_reqs,
        "cluster_cpu_util_pct": round(cluster.cluster_cpu_utilization_pct, 2),
        "cluster_ram_util_pct": round(cluster.cluster_ram_utilization_pct, 2),
        "cluster_bw_util_pct": round(cluster.cluster_bw_utilization_pct, 2),
        "average_cluster_util_pct": round(cluster.average_cluster_utilization_pct, 2),
        "average_imbalance_index": round(avg_imbalance, 6),
        "max_imbalance_index": round(max_imbalance, 6),
        "execution_time_us": round(execution_time_us, 2),
    }


def pm_metrics_to_dataframe(cluster: Cluster) -> pd.DataFrame:
    """Format PM-level metrics as a Pandas DataFrame for visualization."""
    data = [calculate_pm_metrics(pm) for pm in cluster.machines]
    df = pd.DataFrame(data)
    return df


def generate_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a side-by-side comparison DataFrame of all algorithms."""
    rows = []
    for algo_name, data in results_dict.items():
        metrics = data.get("cluster_metrics", {})
        alloc_res = data.get("allocation_result")
        rows.append({
            "Algorithm": algo_name,
            "Active PMs": metrics.get("active_pms", 0),
            "Idle PMs": metrics.get("idle_pms", 0),
            "Cluster CPU Util (%)": metrics.get("cluster_cpu_util_pct", 0.0),
            "Cluster RAM Util (%)": metrics.get("cluster_ram_util_pct", 0.0),
            "Cluster BW Util (%)": metrics.get("cluster_bw_util_pct", 0.0),
            "Avg Cluster Util (%)": metrics.get("average_cluster_util_pct", 0.0),
            "Avg Resource Imbalance": metrics.get("average_imbalance_index", 0.0),
            "Accepted / Total": f"{alloc_res.allocated_count if alloc_res else 0}/{alloc_res.total_requests if alloc_res else 0}",
            "Execution Time (μs)": metrics.get("execution_time_us", 0.0),
        })
    return pd.DataFrame(rows)
