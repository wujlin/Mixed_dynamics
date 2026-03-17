# Mixed Feedback Dynamics in Collective Emotion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A theoretical and empirical framework for understanding phase transitions in collective emotion through competing feedback channels.**

---

## 📄 About

This repository contains the code and data for a research project on phase transitions in collective emotion on social media platforms.

**Preprint**: Coming soon

---

## 🔬 Overview

Collective emotions can shift abruptly on social platforms, producing rapid polarization and surges of activity. This repository provides:

1. **Theoretical framework**: A minimal mixed-feedback model identifying critical channel-balance boundaries beyond which neutral states destabilize into self-sustaining polarization.

2. **Simulation tools**: Agent-based models on network topologies (Erdős–Rényi, Barabási–Albert) to validate mean-field predictions.

3. **Empirical validation**: Analysis of large-scale social-media data (COVID-19 discussions on Weibo) showing predicted signatures:
   - **Activity–jump coupling**: Elevated activity → larger polarization jumps
   - **Media-composition–volatility coupling**: User-generated dominance → higher volatility

---

## 🗂️ Repository Structure

```
emotion_dynamics/
├── src/                      # Core library
│   ├── theory.py            # Mean-field analysis & critical points
│   ├── sde_solver.py        # Stochastic differential equations solver
│   ├── network_sim.py       # Agent-based network simulations
│   ├── plot_style.py        # Unified plotting style
│   ├── utils.py             # Helper functions
│   └── empirical/           # Empirical analysis modules
│       ├── data_loader.py   # Load and preprocess Weibo data
│       ├── proxies.py       # Operational proxies (Q, a, r_proxy)
│       ├── statistics.py    # Segment-level statistics
│       └── validation.py    # Hypothesis testing
│
├── scripts/                  # Data processing scripts
│   ├── 01_scrape_weibo.py   # Weibo data collection
│   ├── 02_clean_data.py     # Data cleaning & deduplication
│   ├── 03_llm_annotation.py # LLM-based arousal annotation
│   ├── 04_build_timeseries.py # Construct time-series proxies
│   ├── repo_hygiene/        # Repository maintenance helpers
│   └── ...
│
├── notebooks/                # Analysis notebooks & figure generation
│   ├── 01_Theory_and_Potential.ipynb
│   ├── 02_Network_Topology.ipynb
│   ├── 03_Critical_Slowing_Down.ipynb
│   ├── 04_Sensitivity_Chi_Landscape.ipynb
│   ├── 05_Annotation_Pipeline.ipynb
│   ├── 07_Empirical_Validation.ipynb
│   ├── make_fig*.py         # Figure generation scripts
│   └── regenerate_all_figures.sh
│
├── tests/                    # Unit tests
│   ├── test_theory.py
│   ├── test_sde_solver.py
│   ├── test_network_sim.py
│   └── test_utils.py
│
├── data/                     # Processed data & configs
│   ├── derived/             # Aggregated time-series (public)
│   └── config/              # Configuration files
│
├── outputs/                  # Simulation results & figures
│   └── data/                # Cached simulation results (.npz)
│
├── docs/                     # Documentation
│   ├── architecture/
│   │   └── CODE_STRUCTURE.md # Detailed code architecture
│   ├── code_data_structure.md
│   ├── dataset_description.md
│   ├── repo_hygiene/        # Repo cleanup and publication hygiene
│   ├── visual_style_guide.md
│   └── vllm_qwen_setup.md
│
├── legacy/                   # Archived drafts and historical materials
│   ├── HSSC/                # Early manuscript versions
│   └── ...
│
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wujlin/Mixed_dynamics.git
cd Mixed_dynamics

# Create virtual environment (Python 3.9+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Theory Simulations

```python
from src.theory import MixedFeedbackModel
from src.sde_solver import solve_sde
import matplotlib.pyplot as plt

# Create model instance
model = MixedFeedbackModel(
    phi=0.7,      # High arousal threshold
    theta=0.3,    # Low arousal threshold
    k=10,         # Information density
    n_m=1.0,      # Mainstream channel strength
    n_w=1.0       # User-generated channel strength
)

# Compute critical point
r_c = model.compute_critical_point()
print(f"Critical channel balance: r_c = {r_c:.3f}")

# Run SDE simulation
t, q, a = solve_sde(model, r=0.8, T=1000, dt=0.01)

# Plot trajectories
plt.plot(t, q, label='Polarization q(t)')
plt.plot(t, a, label='Activity a(t)')
plt.legend()
plt.show()
```

### Run Network Simulations

```python
from src.network_sim import NetworkSimulation
import networkx as nx

# Create Erdős–Rényi network
G = nx.erdos_renyi_graph(n=1000, p=0.01)

# Initialize simulation
sim = NetworkSimulation(
    graph=G,
    model=model,
    r=0.8,
    beta=0.0  # Local coupling strength
)

# Run simulation
sim.run(n_sweeps=100)

# Get macroscopic observables
Q, a = sim.get_polarization(), sim.get_activity()
print(f"Final polarization: Q = {Q:.3f}, activity: a = {a:.3f}")
```

### Reproduce Paper Figures

```bash
# Regenerate all main figures
cd notebooks
bash regenerate_all_figures.sh

# Or regenerate specific figures
python make_fig1a_bifurcation.py
python make_fig3a_csd_scaling.py
```

---

## 📊 Data

### Empirical Data (Weibo Corpus)

The raw Weibo corpus (March 2020 – July 2025) contains user identifiers and **cannot be shared publicly** due to privacy and platform terms-of-service restrictions.

**De-identified derived data** used in the paper:
- Aggregated time-series (4-hour windows): `data/derived/timeseries_4h.csv`
- Segment-level statistics: `data/derived/segments_pooled.csv`
- High-density subset: `data/derived/segments_high_density.csv`

These files contain only:
- Window-level arousal fractions (X_H, X_M, X_L)
- Media-type counts (mainstream, user-generated, government)
- Derived proxies (Q, a, r_proxy, volatility, jump intensity)

**No user IDs, usernames, or post content** are included.

### Simulation Data

Cached simulation results (`.npz` files) are available in `outputs/data/`:
- Mean-field bifurcation diagrams
- Network simulation results (ER, BA topologies)
- Parameter landscape scans
- Critical slowing down signatures

---

## 🧪 Methods

### Theoretical Model

**Macroscopic variables**:
- **Polarization direction**: $q(t) = \rho_H(t) - \rho_L(t)$
- **System activity**: $a(t) = \rho_H(t) + \rho_L(t) = 1 - \rho_M(t)$

**Mixed feedback environment**:
```
p_env(q, a; r) = [(1-r) n_m p_main(q) + r n_w p_we(q, a)] / [(1-r) n_m + r n_w]
```
where:
- $p_{\text{main}}(q) = (1-q)/2$ (stabilizing mainstream channel)
- $p_{\text{we}}(q,a) = (a+q)/2$ (amplifying user-generated channel)
- $r \in [0,1]$ (channel balance control parameter)

**Critical point** (symmetric regime):
```
r_c = [n_m(χ+2)] / [n_m(χ+2) + n_w(χ-2)]
```
where $χ$ is the psychological sensitivity.

### Empirical Proxies

**Operational definitions** (window-level):
- **Polarization**: $Q = X_H - X_L$ (from public user posts)
- **Activity**: $a = X_H + X_L$
- **Media composition**: $r_{\text{proxy}} = n_{\text{wemedia}} / (n_{\text{wemedia}} + n_{\text{mainstream}} + n_{\text{gov}})$

**Segment-level statistics** (weekly aggregation):
- **Volatility**: $\text{std}(Q)$
- **Jump intensity**: $\text{jump}_{q95}$ (95th percentile of $|dQ/dt|$)

### LLM Annotation

Emotional arousal (High/Medium/Low) and risk content (risk/norisk) are annotated using **Qwen3-8B** with structured prompts.

**Validation** (n=5000 manual ratings):
- Arousal: 85% accuracy, Cohen's κ=0.79
- Risk: 87% accuracy, κ=0.82

---

## 📈 Key Results

### 1. Direction–Intensity Bifurcation

- **Symmetric regime**: Pitchfork bifurcation at $r_c \approx 0.753$
- **Activity-coupled regime**: Direction–intensity entanglement → crossover instead of sharp transition
- **Implication**: Elevated activity signals fragility *before* polarization emerges

### 2. Network Robustness

- Transition persists on ER and BA topologies
- Finite-size scaling (Binder cumulant) confirms critical point
- Local coupling $\beta$ shifts boundary but does not suppress instability

### 3. Parameter Landscape

**Most vulnerable systems** (lowest $r_c$):
- High user-generated dominance ($n_w/n_m \uparrow$)
- High psychological sensitivity ($\chi \uparrow$)
- Strong local reinforcement ($\beta \uparrow$)

### 4. Empirical Signatures

**H1 (Activity–jump coupling)**: Pooled dataset, Pearson $r=0.241$, $p=0.00798$

**H2 (Media–volatility coupling)**: High-density subset, Pearson $r=0.429$, $p=6.03 \times 10^{-7}$

Both signatures robust to controls for sampling density.

---

## 🔧 Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

This project uses `black` for code formatting:
```bash
black src/ scripts/ notebooks/*.py
```

### Adding New Features

1. Implement in `src/` with docstrings (NumPy style)
2. Add unit tests in `tests/`
3. Update `docs/architecture/CODE_STRUCTURE.md` if adding new modules
4. Run tests and ensure all pass

---

## 📜 Citation

If you use this code or data in your research, please cite this repository:

```bibtex
@software{mixed_dynamics,
  title={Mixed Feedback Dynamics in Collective Emotion},
  author={{Mixed Dynamics Team}},
  year={2026},
  url={https://github.com/wujlin/Mixed_dynamics}
}
```

A paper describing this work is in preparation.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions about the code or data, please open an issue on GitHub.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We thank the developers of:
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [NetworkX](https://networkx.org/) (core scientific computing)
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) (visualization)
- [Qwen](https://github.com/QwenLM/Qwen) (LLM annotation)

---

## 🗺️ Roadmap

- [ ] Publish de-identified empirical data
- [ ] Add interactive demo (Jupyter widgets / Streamlit)
- [ ] Extend to multi-platform comparisons (Weibo vs Twitter vs Reddit)
- [ ] Develop intervention simulation framework

---

**Last updated**: 2026-01-16
