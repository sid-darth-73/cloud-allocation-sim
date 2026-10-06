"""
Simulation Engine for Cloud Multi-Dimensional Resource Allocation.

Supports:
1. Static Lab Baseline: The exact 4 requests (R1-R4) on 4 PMs.
2. Synthetic Workload Generator: Parameterized dynamic requests (20-50+ reqs).
3. Batch Allocation Benchmark: Evaluates RR, FF, BF, and RAA side-by-side.
4. Dynamic Discrete-Event Timeline Simulation: Tracks arrival/departure lifecycles
   and records time-series metrics (active hosts, multi-resource utilization, imbalance).
"""

from __future__ import annotations

import heapq
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from algorithms import (
    BaseAllocator,
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


def get_static_baseline_requests() -> List[Request]:
    """Construct the exact 4 lab baseline requests:
    - R1: 2 Cores CPU, 4 GB RAM, 10 Mbps Bandwidth
    - R2: 4 Cores CPU, 2 GB RAM, 20 Mbps Bandwidth
    - R3: 1 Core CPU, 8 GB RAM, 5 Mbps Bandwidth
    - R4: 6 Cores CPU, 4 GB RAM, 30 Mbps Bandwidth
    """
    return [
        Request(id="R1", cpu=2.0, ram=4.0, bw=10.0, arrival_time=0.0, lifetime=10.0),
        Request(id="R2", cpu=4.0, ram=2.0, bw=20.0, arrival_time=1.0, lifetime=12.0),
        Request(id="R3", cpu=1.0, ram=8.0, bw=5.0, arrival_time=2.0, lifetime=15.0),
        Request(id="R4", cpu=6.0, ram=4.0, bw=30.0, arrival_time=3.0, lifetime=10.0),
    ]


def generate_synthetic_workload(
    num_requests: int = 30,
    seed: int = 42,
    cpu_range: Tuple[float, float] = (1.0, 6.0),
    ram_range: Tuple[float, float] = (2.0, 12.0),
    bw_range: Tuple[float, float] = (5.0, 35.0),
    lifetime_range: Tuple[float, float] = (10.0, 40.0),
    arrival_interval_range: Tuple[float, float] = (1.0, 4.0),
) -> List[Request]:
    """Generate dynamic synthetic workload with arrival and departure parameters.
    
    Args:
        num_requests: Number of requests to generate (e.g. 20 to 50).
        seed: Random seed for deterministic reproducibility.
        cpu_range: (min_cpu, max_cpu) in cores.
        ram_range: (min_ram, max_ram) in GB.
        bw_range: (min_bw, max_bw) in Mbps.
        lifetime_range: (min_life, max_life) in seconds.
        arrival_interval_range: Inter-arrival time interval.
        
    Returns:
        List of generated Request objects.
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    current_time = 0.0

    for i in range(1, num_requests + 1):
        # Generate varied resource profiles (CPU-bound, RAM-bound, BW-bound, Balanced)
        profile_type = rng.choice(["cpu_heavy", "ram_heavy", "bw_heavy", "balanced"])
        
        if profile_type == "cpu_heavy":
            cpu = round(rng.uniform(4.0, min(8.0, cpu_range[1])), 1)
            ram = round(rng.uniform(ram_range[0], 4.0), 1)
            bw = round(rng.uniform(bw_range[0], 20.0), 1)
        elif profile_type == "ram_heavy":
            cpu = round(rng.uniform(cpu_range[0], 2.0), 1)
            ram = round(rng.uniform(6.0, min(14.0, ram_range[1])), 1)
            bw = round(rng.uniform(bw_range[0], 15.0), 1)
        elif profile_type == "bw_heavy":
            cpu = round(rng.uniform(cpu_range[0], 3.0), 1)
            ram = round(rng.uniform(ram_range[0], 6.0), 1)
            bw = round(rng.uniform(25.0, min(45.0, bw_range[1])), 1)
        else:
            cpu = round(rng.uniform(cpu_range[0], cpu_range[1]), 1)
            ram = round(rng.uniform(ram_range[0], ram_range[1]), 1)
            bw = round(rng.uniform(bw_range[0], bw_range[1]), 1)

        lifetime = round(rng.uniform(lifetime_range[0], lifetime_range[1]), 1)
        requests.append(
            Request(
                id=f"REQ_{i:03d}",
                cpu=cpu,
                ram=ram,
                bw=bw,
                arrival_time=round(current_time, 1),
                lifetime=lifetime,
            )
        )
        current_time += rng.uniform(arrival_interval_range[0], arrival_interval_range[1])

    return requests


def get_standard_allocators() -> List[BaseAllocator]:
    """Instantiate the 4 evaluated allocators."""
    return [
        RoundRobinAllocator(),
        FirstFitAllocator(),
        BestFitAllocator(),
        ResourceAwareAllocator(),
    ]


def run_benchmark(
    requests: List[Request],
    cluster_template: Optional[Cluster] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute all 4 algorithms on identical cloned clusters.
    
    Args:
        requests: Batch of requests to allocate.
        cluster_template: Base cluster configuration (defaults to 4 PMs).
        
    Returns:
        Dictionary mapping algorithm name to its results:
        {
            algo_name: {
                "allocator": BaseAllocator,
                "allocation_result": AllocationResult,
                "cluster": Cluster,
                "cluster_metrics": Dict,
                "pm_metrics_df": DataFrame,
            }
        }
    """
    if cluster_template is None:
        cluster_template = Cluster.create_default(num_pms=4)

    results: Dict[str, Dict[str, Any]] = {}
    allocators = get_standard_allocators()

    for alloc in allocators:
        # Clone fresh cluster so each algorithm starts from identical clean state
        cluster = cluster_template.clone()
        alloc_res = alloc.allocate_all(requests, cluster)
        c_metrics = calculate_cluster_metrics(cluster, alloc_res.execution_time_us)
        pm_df = pm_metrics_to_dataframe(cluster)

        results[alloc.name] = {
            "allocator": alloc,
            "allocation_result": alloc_res,
            "cluster": cluster,
            "cluster_metrics": c_metrics,
            "pm_metrics_df": pm_df,
        }

    return results


def run_dynamic_timeline_simulation(
    requests: List[Request],
    cluster_template: Optional[Cluster] = None,
) -> Dict[str, pd.DataFrame]:
    """Perform event-driven simulation tracking allocation and departure lifecycles over time.
    
    Events:
    - ARRIVAL: request attempts allocation on cluster.
    - DEPARTURE: active request releases PM resources upon lifetime expiration.
    
    Returns:
        Dict mapping algorithm name to time-series DataFrame of cluster state.
    """
    if cluster_template is None:
        cluster_template = Cluster.create_default(num_pms=4)

    timeline_data_by_algo: Dict[str, pd.DataFrame] = {}
    allocators = get_standard_allocators()

    for alloc in allocators:
        cluster = cluster_template.clone()
        alloc.reset_state()

        # Priority queue for simulation events: (timestamp, event_type, priority, payload)
        # Event types: 'arrival', 'departure'
        event_queue: List[Tuple[float, str, int, Any]] = []
        event_counter = 0

        for req in requests:
            heapq.heappush(event_queue, (req.arrival_time, "arrival", event_counter, req))
            event_counter += 1

        records: List[Dict[str, Any]] = []
        active_allocations: Dict[str, str] = {}  # req_id -> pm_id

        # Record initial state t = 0
        records.append({
            "time": 0.0,
            "event": "START",
            "active_pms": 0,
            "allocated_requests": 0,
            "cluster_cpu_pct": 0.0,
            "cluster_ram_pct": 0.0,
            "cluster_bw_pct": 0.0,
            "avg_cluster_util_pct": 0.0,
            "imbalance_index": 0.0,
        })

        while event_queue:
            t, event_type, _, payload = heapq.heappop(event_queue)

            if event_type == "arrival":
                req: Request = payload
                step = alloc.allocate_single(req, cluster)
                if step.success and step.allocated_pm_id:
                    active_allocations[req.id] = step.allocated_pm_id
                    # Schedule departure event
                    dep_time = t + req.lifetime
                    heapq.heappush(event_queue, (dep_time, "departure", event_counter, req))
                    event_counter += 1

            elif event_type == "departure":
                req: Request = payload
                pm_id = active_allocations.pop(req.id, None)
                if pm_id:
                    pm = cluster.get_machine(pm_id)
                    if pm:
                        pm.release(req.id)

            # Sample cluster state at event timestamp
            c_metrics = calculate_cluster_metrics(cluster)
            records.append({
                "time": round(t, 2),
                "event": f"{event_type.upper()}_{payload.id}",
                "active_pms": c_metrics["active_pms"],
                "allocated_requests": c_metrics["total_allocated_requests"],
                "cluster_cpu_pct": c_metrics["cluster_cpu_util_pct"],
                "cluster_ram_pct": c_metrics["cluster_ram_util_pct"],
                "cluster_bw_pct": c_metrics["cluster_bw_util_pct"],
                "avg_cluster_util_pct": c_metrics["average_cluster_util_pct"],
                "imbalance_index": c_metrics["average_imbalance_index"],
            })

        timeline_data_by_algo[alloc.name] = pd.DataFrame(records)

    return timeline_data_by_algo


def print_cli_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Pretty print lab comparison summary to terminal."""
    comp_df = generate_comparison_table(results)
    print("\n" + "=" * 90)
    print("        CLOUD RESOURCE ALLOCATION LAB EVALUATION BENCHMARK")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90)

    for algo_name, data in results.items():
        print(f"\n--- {algo_name} Host Allocation Details ---")
        cluster: Cluster = data["cluster"]
        pm_df = pm_metrics_to_dataframe(cluster)
        cols_to_print = [
            "pm_id",
            "is_active",
            "num_requests",
            "requests",
            "cpu_util_pct",
            "ram_util_pct",
            "bw_util_pct",
            "imbalance_index",
        ]
        print(pm_df[cols_to_print].to_string(index=False))
    print("\n")


if __name__ == "__main__":
    print("[1/2] Running Static Baseline (R1 - R4 on 4 PMs)...")
    static_reqs = get_static_baseline_requests()
    static_results = run_benchmark(static_reqs)
    print_cli_summary(static_results)

    print("[2/2] Running Extended Synthetic Workload (30 Dynamic Requests)...")
    synth_reqs = generate_synthetic_workload(num_requests=30, seed=42)
    synth_results = run_benchmark(synth_reqs)
    print_cli_summary(synth_results)
