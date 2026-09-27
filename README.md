# Shariah-CFE-1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22998850.svg)](https://doi.org/10.5281/zenodo.22998850)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference Python implementation and reproducibility materials for the
working paper

> **Shariah Computational Financial Engineering I —
> Reconstructing Retail FX and CFD Market Access:
> A Structural Incompatibility Analysis and First-Principles
> Alternative Architecture**
>
> Muhammad Taha Asif
> School of Mathematics and Statistics, The University of Sydney
> [ORCID: 0009-0003-5277-5908](https://orcid.org/0009-0003-5277-5908)
> DOI: [10.5281/zenodo.22998850](https://doi.org/10.5281/zenodo.22998850)

The paper formalises the standard retail Contract-for-Difference
(CFD) architecture as a coupled stochastic system driven by a Bates
(1996) jump-diffusion with stochastic volatility, proves that the
standard CFD violates five foundational Shariah constraints
(*ribā*, *maysir*, *gharar*, *qabḍ*, *bay' al-kāli' bi'l-kāli'*),
and constructs three alternative market-access architectures. This
repository holds the Python code that reproduces every figure,
table, and numerical claim in Section 6 of the paper.

## Repository layout

```
Shariah-CFE-1/
├── README.md              (this file)
├── LICENSE                (MIT)
├── requirements.txt       (Python dependencies)
├── src/
│   ├── bates.py           Bates model simulator (Assumption 3.1)
│   ├── contracts.py       CFD and Sharikat al-'Inan mechanics
│   ├── metrics.py         VaR, CVaR, Sharpe, Sortino, bootstrap
│   ├── simulate.py        Monte Carlo grid runner
│   ├── figures.py         Publication-quality figure generation
│   └── ml_analysis.py     XGBoost margin-call classifier (§6.4)
├── figures/               Generated PDFs (referenced from paper)
├── tables/                Simulation results as CSV
└── notebooks/             Jupyter notebooks (optional; not required)
```

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/Muhammad-Taha-Asif/Shariah-CFE-1.git
cd Shariah-CFE-1

# 2. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 3. Run the full simulation grid (~2 minutes on a modern laptop)
cd src
python simulate.py

# 4. Generate all figures (~30 seconds)
python figures.py
python ml_analysis.py
```

Output PDFs land in `figures/`; the numerical grid lands in
`tables/results_grid.csv`.

## Reproducibility

All Monte Carlo simulations use fixed seeds derived from a
session-wide base seed (`SEED_BASE = 20260928` in
`src/simulate.py`). Running the code exactly as released reproduces
every number reported in Section 6 of the paper. No proprietary
data are used; the illustrative Bates parameter calibrations in
`src/bates.py` are documented in the paper's Table 3.

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@techreport{Asif2026SCFE1,
  author       = {Asif, Muhammad Taha},
  title        = {Shariah Computational Financial Engineering I —
                  Reconstructing Retail {FX} and {CFD} Market Access:
                  A Structural Incompatibility Analysis and
                  First-Principles Alternative Architecture},
  institution  = {The University of Sydney},
  year         = {2026},
  type         = {Working Paper},
  doi          = {10.5281/zenodo.22998850},
  url          = {https://doi.org/10.5281/zenodo.22998850}
}
```

## License

MIT License. See `LICENSE` for details. Contributions and
scholarly comments are welcome via GitHub Issues.

## Contact

Muhammad Taha Asif — asifmuhammadtaha@gmail.com
