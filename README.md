# Multi-Dimensional Cloud Resource Allocation Simulator

A production-grade, modular Python simulation platform and interactive Streamlit laboratory designed for academic **Cloud Computing Lab Evaluations**.

This platform benchmarks traditional vs. resource-aware allocation policies under multi-dimensional constraints (CPU, RAM, and Bandwidth), examining server consolidation efficiency, resource stranding, and multi-dimensional load skewness.

---

## 🚀 Key Features

1. **Four Multi-Dimensional Allocation Algorithms:**
   - **Round Robin (RR):** Cyclic assignment distributing requests evenly across physical machines.
   - **First Fit (FF):** Linear host scan concentrating requests onto lower-index servers for server consolidation.
   - **Best Fit (BF):** Tight-packing heuristic minimizing the normalized Euclidean residual space:
     $$\min_m \sqrt{\left(\frac{\text{rem}_{m}^{\text{cpu}} - r^{\text{cpu}}}{C^{\text{cpu}}}\right)^2 + \left(\frac{\text{rem}_{m}^{\text{ram}} - r^{\text{ram}}}{C^{\text{ram}}}\right)^2 + \left(\frac{\text{rem}_{m}^{\text{bw}} - r^{\text{bw}}}{C^{\text{bw}}}\right)^2}$$
   - **Resource-Aware Allocation (RAA):** Vector alignment via Cosine Similarity between the request demand vector $\mathbf{R}$ and the PM residual capacity vector $\mathbf{A}_m$:
     $$\text{CosineSimilarity}(\mathbf{R}, \mathbf{A}_m) = \frac{\mathbf{R} \cdot \mathbf{A}_m}{\|\mathbf{R}\|_2 \|\mathbf{A}_m\|_2}$$
     Aligns resource consumption to prevent stranded single-dimension bottlenecks (e.g. CPU exhausted while RAM is idle).

2. **Metrics & Evaluation:**
   - **Per-PM Resource Utilization:** CPU (%), RAM (%), Bandwidth (%).
   - **Average Cluster Utilization (%).**
   - **Server Consolidation Efficiency:** Active PM count vs. Idle PMs turned off for energy conservation.
   - **Multi-Dimensional Resource Imbalance / Skewness Index:**
     $$\text{Imbalance}(m) = \frac{(u_{m, \text{cpu}} - u_{m, \text{avg}})^2 + (u_{m, \text{ram}} - u_{m, \text{avg}})^2 + (u_{m, \text{bw}} - u_{m, \text{avg}})^2}{3}$$
   - **Algorithm Execution Time:** Microsecond-level ($\mu s$) execution timing.

3. **Workload Modes Supported:**
   - **Static Lab Baseline:** Exact 4 requests ($R_1$ to $R_4$) evaluated on 4 PMs ($8\text{ Cores}, 16\text{ GB}, 50\text{ Mbps}$).
   - **Dynamic Synthetic Workload:** Parameterized 20–50 requests with arrival times and lifetimes.
   - **Event-Driven Lifecycle Simulation:** Tracks arrival, execution duration, and resource release upon completion.
   - **Custom Workload Builder:** Interactive request injection directly within the dashboard.

4. **Interactive Modern UI (`app.py`):**
   - Metric cards, Radar comparison diagrams, Grouped multi-resource bar charts, Step-by-step playback visualizer, and live time-series lifecycle plots.

---

## 📁 Repository Structure

```
cloud-allocation-sim/
├── .venv/                   # Python virtual environment (pre-configured)
├── models.py                # Request, PhysicalMachine, and Cluster domain dataclasses
├── algorithms.py            # BaseAllocator, RR, FF, BF (Euclidean), and RAA (Cosine Alignment)
├── metrics.py               # Utilization, skewness/imbalance, consolidation, and benchmark metrics
├── simulator.py             # CLI runner, synthetic generator, and event-driven timeline engine
├── app.py                   # Streamlit web dashboard with interactive Plotly visualizers
├── test_allocation.py       # Automated unit and integration test suite
├── requirements.txt         # Pinned project dependencies
└── README.md                # Academic laboratory report and operational documentation
```

---

## 🛠️ Installation & Setup

A virtual environment (`.venv`) is already initialized in this repository.

To activate the virtual environment:
```bash
# On macOS / Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

To install/verify dependencies:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Project

### 1. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the CLI Simulation Benchmark
```bash
python simulator.py
```

### 3. Run Automated Unit Tests
```bash
pytest -v
```

---

## 📊 Static Baseline Lab Results Summary

### Request Set:
- **R1:** $2\text{ Cores CPU}, 4\text{ GB RAM}, 10\text{ Mbps Bandwidth}$
- **R2:** $4\text{ Cores CPU}, 2\text{ GB RAM}, 20\text{ Mbps Bandwidth}$
- **R3:** $1\text{ Core CPU}, 8\text{ GB RAM}, 5\text{ Mbps Bandwidth}$
- **R4:** $6\text{ Cores CPU}, 4\text{ GB RAM}, 30\text{ Mbps Bandwidth}$

### Cluster Configuration:
- $4\text{ PMs (PM1 to PM4)}$, each with $8\text{ Cores CPU}, 16\text{ GB RAM}, 50\text{ Mbps Bandwidth}$.

### Benchmark Results:

| Algorithm | Active PMs | Idle PMs | Cluster CPU Util | Cluster RAM Util | Cluster BW Util | Avg Cluster Util | Resource Imbalance | Server Consolidation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Round Robin (RR)** | 4 | 0 | 40.62% | 28.12% | 32.50% | 33.75% | 0.025764 | 0.0% |
| **First Fit (FF)** | 2 | 2 | 40.62% | 28.12% | 32.50% | 33.75% | 0.025347 | 50.0% |
| **Best Fit (BF)** | 2 | 2 | 40.62% | 28.12% | 32.50% | 33.75% | 0.025347 | 50.0% |
| **Resource-Aware (RAA)** | 2 | 2 | 40.62% | 28.12% | 32.50% | 33.75% | 0.025347 | 50.0% |

### Key Observations:
- **Round Robin** spreads the 4 requests across all 4 PMs ($PM_1 \dots PM_4$), leaving zero idle hosts and yielding the highest imbalance on $PM_2$ and $PM_3$ due to resource skew.
- **First Fit, Best Fit, and Resource-Aware** pack $R_1, R_2, R_3$ onto $PM_1$ and $R_4$ onto $PM_2$, leaving $PM_3$ and $PM_4$ completely idle.
- On $PM_1$, $R_2$ (CPU-heavy: 4 Cores, 2 GB) and $R_3$ (RAM-heavy: 1 Core, 8 GB) complement each other, achieving **87.5% CPU and 87.5% RAM utilization**, keeping skew near zero on the packed host!
