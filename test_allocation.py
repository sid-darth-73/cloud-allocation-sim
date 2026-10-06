"""
Unit and integration tests for Cloud Allocation Simulator.
"""

import math
import pytest

from models import Request, PhysicalMachine, Cluster
from algorithms import (
    RoundRobinAllocator,
    FirstFitAllocator,
    BestFitAllocator,
    ResourceAwareAllocator,
)
from metrics import (
    calculate_pm_imbalance,
    calculate_pm_metrics,
    calculate_cluster_metrics,
    generate_comparison_table,
)
from simulator import (
    get_static_baseline_requests,
    generate_synthetic_workload,
    run_benchmark,
    run_dynamic_timeline_simulation,
)


def test_request_properties():
    req = Request(id="R1", cpu=2.0, ram=4.0, bw=10.0, arrival_time=1.0, lifetime=5.0)
    assert req.departure_time == 6.0
    assert req.as_vector() == [2.0, 4.0, 10.0]
    data = req.to_dict()
    assert data["id"] == "R1"
    assert data["cpu"] == 2.0


def test_physical_machine_allocation_and_release():
    pm = PhysicalMachine(id="PM1", cpu_capacity=8.0, ram_capacity=16.0, bw_capacity=50.0)
    assert pm.remaining_cpu == 8.0
    assert pm.is_active is False
    assert pm.imbalance_index == 0.0

    r1 = Request(id="R1", cpu=2.0, ram=4.0, bw=10.0)
    assert pm.can_allocate(r1) is True
    allocated = pm.allocate(r1)
    assert allocated is True
    assert pm.is_active is True
    assert pm.allocated_cpu == 2.0
    assert pm.remaining_cpu == 6.0
    assert pm.allocated_ram == 4.0
    assert pm.allocated_bw == 10.0
    assert pm.cpu_utilization == 2.0 / 8.0
    assert pm.ram_utilization == 4.0 / 16.0
    assert pm.bw_utilization == 10.0 / 50.0

    # Over-capacity request
    huge = Request(id="Huge", cpu=10.0, ram=1.0, bw=1.0)
    assert pm.can_allocate(huge) is False
    assert pm.allocate(huge) is False

    # Release
    released = pm.release("R1")
    assert released is True
    assert pm.is_active is False
    assert pm.remaining_cpu == 8.0


def test_imbalance_calculation():
    pm = PhysicalMachine(id="PM1", cpu_capacity=8.0, ram_capacity=16.0, bw_capacity=50.0)
    # CPU: 4/8 = 0.5, RAM: 2/16 = 0.125, BW: 20/50 = 0.4
    r = Request(id="R2", cpu=4.0, ram=2.0, bw=20.0)
    pm.allocate(r)

    u_cpu = 0.5
    u_ram = 0.125
    u_bw = 0.4
    u_avg = (u_cpu + u_ram + u_bw) / 3.0
    expected_imb = ((u_cpu - u_avg) ** 2 + (u_ram - u_avg) ** 2 + (u_bw - u_avg) ** 2) / 3.0

    assert math.isclose(pm.imbalance_index, expected_imb, rel_tol=1e-5)
    assert math.isclose(calculate_pm_imbalance(pm), expected_imb, rel_tol=1e-5)


def test_cluster_cloning():
    cluster = Cluster.create_default(num_pms=4)
    r1 = Request(id="R1", cpu=2.0, ram=4.0, bw=10.0)
    cluster.machines[0].allocate(r1)

    cloned = cluster.clone()
    assert len(cloned.machines) == 4
    assert cloned.machines[0].allocated_cpu == 2.0
    # Modifying clone shouldn't affect original
    cloned.machines[1].allocate(r1)
    assert cluster.machines[1].allocated_cpu == 0.0


def test_static_baseline_allocations():
    reqs = get_static_baseline_requests()
    assert len(reqs) == 4
    results = run_benchmark(reqs)

    # 1. Round Robin distributes across all 4 PMs
    rr_data = results["Round Robin (RR)"]
    assert rr_data["cluster_metrics"]["active_pms"] == 4
    assert rr_data["allocation_result"].allocated_count == 4

    # 2. First Fit packs into 2 PMs (PM1 gets R1, R2, R3; PM2 gets R4)
    ff_data = results["First Fit (FF)"]
    assert ff_data["cluster_metrics"]["active_pms"] == 2
    assert ff_data["cluster_metrics"]["idle_pms"] == 2
    assert ff_data["allocation_result"].allocated_count == 4
    pm1_ff = ff_data["cluster"].get_machine("PM1")
    assert [r.id for r in pm1_ff.allocated_requests] == ["R1", "R2", "R3"]

    # 3. Best Fit packs into 2 PMs
    bf_data = results["Best Fit (BF)"]
    assert bf_data["cluster_metrics"]["active_pms"] == 2
    assert bf_data["allocation_result"].allocated_count == 4

    # 4. Resource Aware packs into 2 PMs
    raa_data = results["Resource-Aware (RAA)"]
    assert raa_data["cluster_metrics"]["active_pms"] == 2
    assert raa_data["allocation_result"].allocated_count == 4

    # All share identical cluster resource utilization %
    expected_cpu_util = (13.0 / 32.0) * 100.0
    assert math.isclose(ff_data["cluster_metrics"]["cluster_cpu_util_pct"], expected_cpu_util, rel_tol=1e-2)


def test_synthetic_workload_generation():
    reqs = generate_synthetic_workload(num_requests=25, seed=123)
    assert len(reqs) == 25
    for r in reqs:
        assert r.cpu > 0
        assert r.ram > 0
        assert r.bw > 0
        assert r.lifetime > 0


def test_dynamic_timeline_simulation():
    reqs = get_static_baseline_requests()
    timeline = run_dynamic_timeline_simulation(reqs)
    assert "Round Robin (RR)" in timeline
    assert "First Fit (FF)" in timeline
    assert "Best Fit (BF)" in timeline
    assert "Resource-Aware (RAA)" in timeline

    df_rr = timeline["Round Robin (RR)"]
    assert not df_rr.empty
    assert "active_pms" in df_rr.columns
    assert "avg_cluster_util_pct" in df_rr.columns
