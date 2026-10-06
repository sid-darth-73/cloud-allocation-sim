"""
Cloud Resource Allocation Algorithms.

Implements:
1. Round Robin (RR): Cyclic host assignment.
2. First Fit (FF): First host with sufficient capacity from index 0.
3. Best Fit (BF): Minimizes normalized Euclidean residual capacity space.
4. Resource-Aware Allocation (RAA): Vector alignment via Cosine Similarity between
   request demand and host residual capacity to prevent stranded resources and
   minimize multi-dimensional resource skew.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models import Cluster, PhysicalMachine, Request


@dataclass
class AllocationStep:
    """Record of an individual request placement decision."""
    step: int
    request_id: str
    request_cpu: float
    request_ram: float
    request_bw: float
    allocated_pm_id: Optional[str]
    success: bool
    decision_metric_name: str
    decision_metric_value: float
    pm_candidate_scores: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "request_id": self.request_id,
            "cpu": self.request_cpu,
            "ram": self.request_ram,
            "bw": self.request_bw,
            "allocated_pm": self.allocated_pm_id if self.allocated_pm_id else "REJECTED",
            "success": self.success,
            "metric": self.decision_metric_name,
            "metric_value": round(self.decision_metric_value, 4),
            "rationale": self.rationale,
        }


@dataclass
class AllocationResult:
    """Aggregated outcome of running an allocation algorithm on a cluster."""
    algorithm_name: str
    cluster: Cluster
    steps: List[AllocationStep]
    allocated_count: int
    rejected_count: int
    total_requests: int
    execution_time_us: float

    @property
    def acceptance_ratio_pct(self) -> float:
        """Percentage of requests successfully accommodated."""
        if self.total_requests == 0:
            return 100.0
        return (self.allocated_count / self.total_requests) * 100.0


class BaseAllocator(ABC):
    """Abstract Base Class for multi-dimensional cloud allocation policies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def reset_state(self) -> None:
        """Reset internal algorithm state (e.g., RR pointers). Override if stateful."""
        pass

    @abstractmethod
    def allocate_single(
        self, request: Request, cluster: Cluster, step_num: int = 1
    ) -> AllocationStep:
        """Allocate a single request to the most appropriate PM in the cluster."""
        raise NotImplementedError

    def allocate_all(
        self, requests: List[Request], cluster: Cluster
    ) -> AllocationResult:
        """Execute allocation for a batch of requests with high-resolution timing."""
        self.reset_state()
        steps: List[AllocationStep] = []
        allocated_count = 0
        rejected_count = 0

        # High-resolution benchmark timing
        start_ns = time.perf_counter_ns()

        for idx, req in enumerate(requests, start=1):
            step = self.allocate_single(req, cluster, step_num=idx)
            steps.append(step)
            if step.success:
                allocated_count += 1
            else:
                rejected_count += 1

        end_ns = time.perf_counter_ns()
        execution_time_us = (end_ns - start_ns) / 1000.0

        return AllocationResult(
            algorithm_name=self.name,
            cluster=cluster,
            steps=steps,
            allocated_count=allocated_count,
            rejected_count=rejected_count,
            total_requests=len(requests),
            execution_time_us=round(execution_time_us, 2),
        )


class RoundRobinAllocator(BaseAllocator):
    """Round Robin (RR) Allocator.
    
    Distributes incoming requests cyclically across machines.
    Maintains a circular pointer to balance placement counts across hosts.
    """

    def __init__(self):
        super().__init__(
            name="Round Robin (RR)",
            description="Cyclic round-robin placement advancing through PMs sequentially.",
        )
        self.current_index: int = 0

    def reset_state(self) -> None:
        self.current_index = 0

    def allocate_single(
        self, request: Request, cluster: Cluster, step_num: int = 1
    ) -> AllocationStep:
        num_pms = len(cluster.machines)
        if num_pms == 0:
            return AllocationStep(
                step=step_num,
                request_id=request.id,
                request_cpu=request.cpu,
                request_ram=request.ram,
                request_bw=request.bw,
                allocated_pm_id=None,
                success=False,
                decision_metric_name="RoundRobinIndex",
                decision_metric_value=float(self.current_index),
                rationale="Cluster contains zero PMs.",
            )

        candidate_scores: Dict[str, float] = {}
        # Try PMs cyclically starting from current_index
        for offset in range(num_pms):
            idx = (self.current_index + offset) % num_pms
            pm = cluster.machines[idx]
            candidate_scores[pm.id] = float(offset)
            if pm.can_allocate(request):
                pm.allocate(request)
                chosen_pm_id = pm.id
                self.current_index = (idx + 1) % num_pms
                return AllocationStep(
                    step=step_num,
                    request_id=request.id,
                    request_cpu=request.cpu,
                    request_ram=request.ram,
                    request_bw=request.bw,
                    allocated_pm_id=chosen_pm_id,
                    success=True,
                    decision_metric_name="RoundRobinIndex",
                    decision_metric_value=float(idx),
                    pm_candidate_scores=candidate_scores,
                    rationale=f"Assigned cyclically to {chosen_pm_id} at index {idx}.",
                )

        return AllocationStep(
            step=step_num,
            request_id=request.id,
            request_cpu=request.cpu,
            request_ram=request.ram,
            request_bw=request.bw,
            allocated_pm_id=None,
            success=False,
            decision_metric_name="RoundRobinIndex",
            decision_metric_value=float(self.current_index),
            pm_candidate_scores=candidate_scores,
            rationale="Insufficient capacity across all PMs in the cluster.",
        )


class FirstFitAllocator(BaseAllocator):
    """First Fit (FF) Allocator.
    
    Scans PMs strictly from index 0 to N-1 and assigns the request to the first
    PM that satisfies the multi-dimensional capacity constraints.
    Favors server consolidation onto lower-indexed machines.
    """

    def __init__(self):
        super().__init__(
            name="First Fit (FF)",
            description="First fit placement scanning linearly from PM1 to PM_N to consolidate load.",
        )

    def allocate_single(
        self, request: Request, cluster: Cluster, step_num: int = 1
    ) -> AllocationStep:
        candidate_scores: Dict[str, float] = {}
        for idx, pm in enumerate(cluster.machines):
            feasible = pm.can_allocate(request)
            candidate_scores[pm.id] = 1.0 if feasible else 0.0
            if feasible:
                pm.allocate(request)
                return AllocationStep(
                    step=step_num,
                    request_id=request.id,
                    request_cpu=request.cpu,
                    request_ram=request.ram,
                    request_bw=request.bw,
                    allocated_pm_id=pm.id,
                    success=True,
                    decision_metric_name="FirstFeasibleIndex",
                    decision_metric_value=float(idx),
                    pm_candidate_scores=candidate_scores,
                    rationale=f"Placed on first available host {pm.id} (index {idx}).",
                )

        return AllocationStep(
            step=step_num,
            request_id=request.id,
            request_cpu=request.cpu,
            request_ram=request.ram,
            request_bw=request.bw,
            allocated_pm_id=None,
            success=False,
            decision_metric_name="FirstFeasibleIndex",
            decision_metric_value=-1.0,
            pm_candidate_scores=candidate_scores,
            rationale="No PM has sufficient residual capacity to accommodate request.",
        )


class BestFitAllocator(BaseAllocator):
    """Best Fit (BF) Allocator - Euclidean Residual Space Minimization.
    
    Evaluates all feasible PMs and places the request onto the PM that leaves
    the smallest normalized Euclidean residual capacity space:
        res_norm = sqrt( (rem_cpu_post / cap_cpu)^2 +
                         (rem_ram_post / cap_ram)^2 +
                         (rem_bw_post / cap_bw)^2 )
    
    This tight-fitting heuristic packs hosts tightly and minimizes stranded leftovers.
    """

    def __init__(self):
        super().__init__(
            name="Best Fit (BF)",
            description="Minimizes post-allocation normalized Euclidean residual space.",
        )

    @staticmethod
    def compute_residual_euclidean_distance(
        pm: PhysicalMachine, request: Request
    ) -> float:
        """Calculate the normalized Euclidean norm of residual capacity after allocation."""
        rem_cpu = (pm.remaining_cpu - request.cpu) / pm.cpu_capacity
        rem_ram = (pm.remaining_ram - request.ram) / pm.ram_capacity
        rem_bw = (pm.remaining_bw - request.bw) / pm.bw_capacity
        return math.sqrt(rem_cpu**2 + rem_ram**2 + rem_bw**2)

    def allocate_single(
        self, request: Request, cluster: Cluster, step_num: int = 1
    ) -> AllocationStep:
        best_pm: Optional[PhysicalMachine] = None
        min_residual = float("inf")
        candidate_scores: Dict[str, float] = {}

        for pm in cluster.machines:
            if pm.can_allocate(request):
                res_dist = self.compute_residual_euclidean_distance(pm, request)
                candidate_scores[pm.id] = round(res_dist, 4)
                if res_dist < min_residual:
                    min_residual = res_dist
                    best_pm = pm

        if best_pm is not None:
            best_pm.allocate(request)
            return AllocationStep(
                step=step_num,
                request_id=request.id,
                request_cpu=request.cpu,
                request_ram=request.ram,
                request_bw=request.bw,
                allocated_pm_id=best_pm.id,
                success=True,
                decision_metric_name="MinResidualEuclideanDistance",
                decision_metric_value=min_residual,
                pm_candidate_scores=candidate_scores,
                rationale=f"Selected {best_pm.id} minimizing residual distance to {min_residual:.4f}.",
            )

        return AllocationStep(
            step=step_num,
            request_id=request.id,
            request_cpu=request.cpu,
            request_ram=request.ram,
            request_bw=request.bw,
            allocated_pm_id=None,
            success=False,
            decision_metric_name="MinResidualEuclideanDistance",
            decision_metric_value=float("inf"),
            pm_candidate_scores=candidate_scores,
            rationale="No feasible PM found.",
        )


class ResourceAwareAllocator(BaseAllocator):
    """Resource-Aware Allocation (RAA) using Cosine Similarity / Vector Alignment.
    
    Aligns the multi-dimensional request demand vector R with the PM's remaining
    capacity vector A to prevent resource fragmentation (stranded capacity).
    
    Vector Formulation:
        R = [ r_cpu/C_cpu, r_ram/C_ram, r_bw/C_bw ]
        A = [ a_cpu/C_cpu, a_ram/C_ram, a_bw/C_bw ]
        
        CosineSimilarity(R, A) = (R . A) / ( ||R|| * ||A|| )
    
    A higher cosine similarity means the request consumes resources in the exact
    proportions available on that host, avoiding premature exhaustion of any single
    resource and minimizing overall resource skew.
    
    Tie-breaking: PM with lower post-allocation imbalance index.
    """

    def __init__(self, skew_penalty_weight: float = 0.5):
        super().__init__(
            name="Resource-Aware (RAA)",
            description="Vector alignment via Cosine Similarity between demand and available capacity.",
        )
        self.skew_penalty_weight = skew_penalty_weight

    @staticmethod
    def compute_cosine_similarity(pm: PhysicalMachine, request: Request) -> float:
        """Compute cosine similarity between normalized request and PM residual capacity."""
        # Normalized demand vector
        r_cpu = request.cpu / pm.cpu_capacity
        r_ram = request.ram / pm.ram_capacity
        r_bw = request.bw / pm.bw_capacity

        # Normalized available capacity vector
        a_cpu = pm.remaining_cpu / pm.cpu_capacity
        a_ram = pm.remaining_ram / pm.ram_capacity
        a_bw = pm.remaining_bw / pm.bw_capacity

        dot_product = (r_cpu * a_cpu) + (r_ram * a_ram) + (r_bw * a_bw)
        norm_r = math.sqrt(r_cpu**2 + r_ram**2 + r_bw**2)
        norm_a = math.sqrt(a_cpu**2 + a_ram**2 + a_bw**2)

        if norm_r == 0.0 or norm_a == 0.0:
            return 0.0

        cos_sim = dot_product / (norm_r * norm_a)
        # Numerical clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, cos_sim))

    @staticmethod
    def compute_post_allocation_imbalance(
        pm: PhysicalMachine, request: Request
    ) -> float:
        """Simulate post-allocation imbalance index on candidate PM."""
        post_cpu = (pm.allocated_cpu + request.cpu) / pm.cpu_capacity
        post_ram = (pm.allocated_ram + request.ram) / pm.ram_capacity
        post_bw = (pm.allocated_bw + request.bw) / pm.bw_capacity
        u_avg = (post_cpu + post_ram + post_bw) / 3.0
        return ((post_cpu - u_avg) ** 2 + (post_ram - u_avg) ** 2 + (post_bw - u_avg) ** 2) / 3.0

    def score_pm(self, pm: PhysicalMachine, request: Request) -> Tuple[float, float]:
        """Score candidate PM. Higher cosine similarity is preferred; lower imbalance breaks ties."""
        cos_sim = self.compute_cosine_similarity(pm, request)
        post_imbalance = self.compute_post_allocation_imbalance(pm, request)
        # Combined composite score: Alignment minus skew penalty
        composite_score = cos_sim - (self.skew_penalty_weight * post_imbalance)
        return composite_score, cos_sim

    def allocate_single(
        self, request: Request, cluster: Cluster, step_num: int = 1
    ) -> AllocationStep:
        best_pm: Optional[PhysicalMachine] = None
        best_score = -float("inf")
        best_cos_sim = -float("inf")
        candidate_scores: Dict[str, float] = {}

        for pm in cluster.machines:
            if pm.can_allocate(request):
                composite_score, cos_sim = self.score_pm(pm, request)
                candidate_scores[pm.id] = round(cos_sim, 4)

                # Prioritize host with superior alignment score
                if composite_score > best_score:
                    best_score = composite_score
                    best_cos_sim = cos_sim
                    best_pm = pm

        if best_pm is not None:
            best_pm.allocate(request)
            return AllocationStep(
                step=step_num,
                request_id=request.id,
                request_cpu=request.cpu,
                request_ram=request.ram,
                request_bw=request.bw,
                allocated_pm_id=best_pm.id,
                success=True,
                decision_metric_name="CosineAlignmentScore",
                decision_metric_value=best_cos_sim,
                pm_candidate_scores=candidate_scores,
                rationale=(
                    f"Selected {best_pm.id} with cosine alignment = {best_cos_sim:.4f} "
                    f"(composite score = {best_score:.4f})."
                ),
            )

        return AllocationStep(
            step=step_num,
            request_id=request.id,
            request_cpu=request.cpu,
            request_ram=request.ram,
            request_bw=request.bw,
            allocated_pm_id=None,
            success=False,
            decision_metric_name="CosineAlignmentScore",
            decision_metric_value=-1.0,
            pm_candidate_scores=candidate_scores,
            rationale="No feasible PM found.",
        )
