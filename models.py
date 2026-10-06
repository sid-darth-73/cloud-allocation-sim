"""
Data models for Cloud Resource Allocation Simulation.

Defines:
- Request: Multi-dimensional resource request (CPU, RAM, Bandwidth) with optional dynamic lifetime.
- PhysicalMachine: Homogeneous host representation with capacity tracking, allocation, and rollback.
- Cluster: Collection of physical machines with aggregate capacity and state management.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Request:
    """Represents a multi-dimensional cloud resource request.
    
    Attributes:
        id: Unique identifier for the request (e.g. 'R1').
        cpu: Required CPU cores.
        ram: Required RAM in Gigabytes (GB).
        bw: Required Bandwidth in Megabits per second (Mbps).
        arrival_time: Simulation timestamp when the request arrives.
        lifetime: Duration for which the request holds allocated resources.
    """
    id: str
    cpu: float
    ram: float
    bw: float
    arrival_time: float = 0.0
    lifetime: float = 0.0

    @property
    def departure_time(self) -> float:
        """Timestamp when the request completes and releases resources."""
        return self.arrival_time + self.lifetime

    def as_vector(self) -> List[float]:
        """Return resource requirements as a 3D vector [CPU, RAM, BW]."""
        return [float(self.cpu), float(self.ram), float(self.bw)]

    def to_dict(self) -> Dict[str, float | str]:
        """Convert request to dictionary format."""
        return {
            "id": self.id,
            "cpu": self.cpu,
            "ram": self.ram,
            "bw": self.bw,
            "arrival_time": self.arrival_time,
            "lifetime": self.lifetime,
            "departure_time": self.departure_time,
        }


@dataclass
class PhysicalMachine:
    """Represents a Physical Machine (PM / Host) in the cloud cluster.
    
    Attributes:
        id: Unique host identifier (e.g., 'PM1').
        cpu_capacity: Total CPU core capacity (default: 8 Cores).
        ram_capacity: Total RAM capacity in GB (default: 16 GB).
        bw_capacity: Total Bandwidth capacity in Mbps (default: 50 Mbps).
        allocated_requests: List of currently hosted requests.
    """
    id: str
    cpu_capacity: float = 8.0
    ram_capacity: float = 16.0
    bw_capacity: float = 50.0
    allocated_requests: List[Request] = field(default_factory=list)

    @property
    def allocated_cpu(self) -> float:
        """Sum of CPU cores allocated to running requests."""
        return sum(req.cpu for req in self.allocated_requests)

    @property
    def allocated_ram(self) -> float:
        """Sum of RAM (GB) allocated to running requests."""
        return sum(req.ram for req in self.allocated_requests)

    @property
    def allocated_bw(self) -> float:
        """Sum of Bandwidth (Mbps) allocated to running requests."""
        return sum(req.bw for req in self.allocated_requests)

    @property
    def remaining_cpu(self) -> float:
        """Residual unallocated CPU cores."""
        return max(0.0, self.cpu_capacity - self.allocated_cpu)

    @property
    def remaining_ram(self) -> float:
        """Residual unallocated RAM in GB."""
        return max(0.0, self.ram_capacity - self.allocated_ram)

    @property
    def remaining_bw(self) -> float:
        """Residual unallocated Bandwidth in Mbps."""
        return max(0.0, self.bw_capacity - self.allocated_bw)

    @property
    def cpu_utilization(self) -> float:
        """CPU utilization fraction [0.0, 1.0]."""
        return self.allocated_cpu / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    @property
    def ram_utilization(self) -> float:
        """RAM utilization fraction [0.0, 1.0]."""
        return self.allocated_ram / self.ram_capacity if self.ram_capacity > 0 else 0.0

    @property
    def bw_utilization(self) -> float:
        """Bandwidth utilization fraction [0.0, 1.0]."""
        return self.allocated_bw / self.bw_capacity if self.bw_capacity > 0 else 0.0

    @property
    def cpu_utilization_pct(self) -> float:
        """CPU utilization percentage (0% to 100%)."""
        return self.cpu_utilization * 100.0

    @property
    def ram_utilization_pct(self) -> float:
        """RAM utilization percentage (0% to 100%)."""
        return self.ram_utilization * 100.0

    @property
    def bw_utilization_pct(self) -> float:
        """Bandwidth utilization percentage (0% to 100%)."""
        return self.bw_utilization * 100.0

    @property
    def average_utilization(self) -> float:
        """Mean resource utilization across CPU, RAM, and BW fractions."""
        return (self.cpu_utilization + self.ram_utilization + self.bw_utilization) / 3.0

    @property
    def is_active(self) -> bool:
        """True if the PM is currently hosting at least one request."""
        return len(self.allocated_requests) > 0

    @property
    def imbalance_index(self) -> float:
        """Multi-dimensional Resource Imbalance / Skewness Index for this PM.
        
        Formula:
            Imbalance = ((u_cpu - u_avg)^2 + (u_ram - u_avg)^2 + (u_bw - u_avg)^2) / 3
        where u_* are resource utilization fractions in [0.0, 1.0].
        If PM is inactive (idle), imbalance is defined as 0.0.
        """
        if not self.is_active:
            return 0.0
        u_cpu = self.cpu_utilization
        u_ram = self.ram_utilization
        u_bw = self.bw_utilization
        u_avg = (u_cpu + u_ram + u_bw) / 3.0
        return ((u_cpu - u_avg) ** 2 + (u_ram - u_avg) ** 2 + (u_bw - u_avg) ** 2) / 3.0

    def can_allocate(self, request: Request, tol: float = 1e-6) -> bool:
        """Check if PM has sufficient residual capacity along all 3 dimensions.
        
        Args:
            request: The request under consideration.
            tol: Floating point tolerance.
            
        Returns:
            bool: True if feasible, False otherwise.
        """
        return (
            (self.remaining_cpu + tol >= request.cpu)
            and (self.remaining_ram + tol >= request.ram)
            and (self.remaining_bw + tol >= request.bw)
        )

    def allocate(self, request: Request) -> bool:
        """Allocate a request to this PM if capacity constraints are met.
        
        Args:
            request: The request to allocate.
            
        Returns:
            bool: True if successfully allocated, False if infeasible.
        """
        if self.can_allocate(request):
            self.allocated_requests.append(request)
            return True
        return False

    def release(self, request_id: str) -> bool:
        """Rollback or release a request by its ID.
        
        Args:
            request_id: Identifier of the request to remove.
            
        Returns:
            bool: True if found and removed, False otherwise.
        """
        for idx, req in enumerate(self.allocated_requests):
            if req.id == request_id:
                self.allocated_requests.pop(idx)
                return True
        return False

    def reset(self) -> None:
        """Clear all allocations and restore host to initial idle state."""
        self.allocated_requests.clear()

    def to_dict(self) -> Dict[str, float | str | int | bool]:
        """Export current host state as dictionary."""
        return {
            "pm_id": self.id,
            "cpu_capacity": self.cpu_capacity,
            "ram_capacity": self.ram_capacity,
            "bw_capacity": self.bw_capacity,
            "allocated_cpu": round(self.allocated_cpu, 2),
            "allocated_ram": round(self.allocated_ram, 2),
            "allocated_bw": round(self.allocated_bw, 2),
            "remaining_cpu": round(self.remaining_cpu, 2),
            "remaining_ram": round(self.remaining_ram, 2),
            "remaining_bw": round(self.remaining_bw, 2),
            "cpu_util_pct": round(self.cpu_utilization_pct, 2),
            "ram_util_pct": round(self.ram_utilization_pct, 2),
            "bw_util_pct": round(self.bw_utilization_pct, 2),
            "imbalance_index": round(self.imbalance_index, 6),
            "is_active": self.is_active,
            "num_requests": len(self.allocated_requests),
            "requests": [r.id for r in self.allocated_requests],
        }


@dataclass
class Cluster:
    """Manages the homogeneous/heterogeneous pool of Physical Machines.
    
    Attributes:
        machines: List of PhysicalMachine instances.
    """
    machines: List[PhysicalMachine] = field(default_factory=list)

    @classmethod
    def create_default(
        cls,
        num_pms: int = 4,
        cpu_cap: float = 8.0,
        ram_cap: float = 16.0,
        bw_cap: float = 50.0,
    ) -> Cluster:
        """Factory method to construct standard lab cluster (PM1 to PM4)."""
        pms = [
            PhysicalMachine(
                id=f"PM{i+1}",
                cpu_capacity=cpu_cap,
                ram_capacity=ram_cap,
                bw_capacity=bw_cap,
            )
            for i in range(num_pms)
        ]
        return cls(machines=pms)

    @property
    def active_machines(self) -> List[PhysicalMachine]:
        """Return only active PMs (hosting at least one request)."""
        return [pm for pm in self.machines if pm.is_active]

    @property
    def total_active_pms(self) -> int:
        """Count of active PMs."""
        return len(self.active_machines)

    @property
    def total_cpu_capacity(self) -> float:
        """Total CPU cores available across all PMs in cluster."""
        return sum(pm.cpu_capacity for pm in self.machines)

    @property
    def total_ram_capacity(self) -> float:
        """Total RAM (GB) available across all PMs in cluster."""
        return sum(pm.ram_capacity for pm in self.machines)

    @property
    def total_bw_capacity(self) -> float:
        """Total Bandwidth (Mbps) available across all PMs in cluster."""
        return sum(pm.bw_capacity for pm in self.machines)

    @property
    def total_allocated_cpu(self) -> float:
        """Total CPU cores currently allocated across cluster."""
        return sum(pm.allocated_cpu for pm in self.machines)

    @property
    def total_allocated_ram(self) -> float:
        """Total RAM (GB) currently allocated across cluster."""
        return sum(pm.allocated_ram for pm in self.machines)

    @property
    def total_allocated_bw(self) -> float:
        """Total Bandwidth (Mbps) currently allocated across cluster."""
        return sum(pm.allocated_bw for pm in self.machines)

    @property
    def cluster_cpu_utilization_pct(self) -> float:
        """Cluster-wide CPU utilization percentage."""
        return (
            (self.total_allocated_cpu / self.total_cpu_capacity * 100.0)
            if self.total_cpu_capacity > 0
            else 0.0
        )

    @property
    def cluster_ram_utilization_pct(self) -> float:
        """Cluster-wide RAM utilization percentage."""
        return (
            (self.total_allocated_ram / self.total_ram_capacity * 100.0)
            if self.total_ram_capacity > 0
            else 0.0
        )

    @property
    def cluster_bw_utilization_pct(self) -> float:
        """Cluster-wide Bandwidth utilization percentage."""
        return (
            (self.total_allocated_bw / self.total_bw_capacity * 100.0)
            if self.total_bw_capacity > 0
            else 0.0
        )

    @property
    def average_cluster_utilization_pct(self) -> float:
        """Average Cluster Resource Utilization across CPU, RAM, and BW (%)."""
        return (
            self.cluster_cpu_utilization_pct
            + self.cluster_ram_utilization_pct
            + self.cluster_bw_utilization_pct
        ) / 3.0

    @property
    def average_active_imbalance(self) -> float:
        """Average resource imbalance index across active PMs.
        
        Returns 0.0 if no PMs are active.
        """
        active = self.active_machines
        if not active:
            return 0.0
        return sum(pm.imbalance_index for pm in active) / len(active)

    def get_machine(self, machine_id: str) -> Optional[PhysicalMachine]:
        """Find PM by identifier."""
        for pm in self.machines:
            if pm.id == machine_id:
                return pm
        return None

    def reset(self) -> None:
        """Reset all physical machines to idle state."""
        for pm in self.machines:
            pm.reset()

    def clone(self) -> Cluster:
        """Create a deep copy of cluster for isolated algorithmic experiments."""
        return copy.deepcopy(self)
