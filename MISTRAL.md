# MISTRAL.md - wmin-model Codebase Index

## Overview

**wmin-model** is a [Colibri](https://github.com/HEP-PBSP/colibri) PDF-model implementing the POD (Proper Orthogonal Decomposition) parametrisation for Bayesian inference on PDFs (Parton Distribution Functions), as presented in [arXiv:2507.16913](https://arxiv.org/pdf/2507.16913).

---

## Repository Structure

```
wmin-model/
├── wmin/
│   ├── __init__.py          # Package init, version = "0.1.0"
│   ├── api.py               # ReportEngine API with colibri + wmin providers
│   ├── app.py               # WminApp class extending colibriApp
│   ├── basis.py             # POD basis construction functions
│   ├── config.py            # WminConfig class for PDF model production
│   ├── model.py             # WMinPDF - weight minimization PDF model
│   ├── utils.py             # Utility functions (arclength, sign_flips, likelihood_time)
│   │
│   ├── runcards/            # Example configuration files
│   │   ├── likelihood_time.yaml     # Likelihood timing benchmark
│   │   ├── pod_basis_example.yaml  # POD basis construction example
│   │   └── shift_lhadf_members.py  # Post-fit member shifting script
│   │
│   └── tests/               # Test suite
│       ├── test_fit.py             # Fit tests
│       ├── test_likelihood.py      # Likelihood tests
│       ├── test_wmin_app.py        # App tests
│       ├── test_wmin_basis.py      # Basis construction tests
│       ├── test_wmin_utils.py      # Utility function tests
│       ├── wmin_conftest.py        # Pytest fixtures
│       ├── regression_results/     # Regression test data
│       └── regression_runcards/     # Regression test configurations
│
├── README.md                # Installation, usage, citing
├── pyproject.toml           # Poetry build config, entry point: wmin = "wmin.app:main"
├── environment.yml           # Conda dev environment (Python >= 3.11)
├── ci_environment.yml        # CI environment (no colibri git dep)
└── LICENSE                   # GNU General Public License v3
```

---

## Key Components

### Core Modules

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `wmin/model.py` | Weight minimization PDF model | `WMinPDF` class with `param_names`, `grid_values_func` |
| `wmin/basis.py` | POD basis construction | `n3fit_pdf_model`, `n3fit_pdf_grid`, `get_X_matrix`, `pod_basis`, `write_pod_basis` |
| `wmin/config.py` | Configuration | `WminConfig.produce_pdf_model()` |
| `wmin/utils.py` | Utilities | `likelihood_time`, `arclength_pdfgrid`, `arclength_outliers`, `sign_flips_counter`, `sign_flip_selection`, `FLAV_INFO` |
| `wmin/api.py` | Programmatic API | `API` instance combining colibri + wmin providers |
| `wmin/app.py` | CLI application | `WminApp` class, `wmin_providers` list |

### Entry Points

- **CLI**: `wmin` command (via `wmin.app:main`)
- **Python API**: `from wmin.api import API`

---

## Workflows

### 1. PDF Fits with POD Parametrisation

Perform PDF fits on [NNPDF data](https://github.com/NNPDF/nnpdf) using Colibri's Bayesian workflow with linear POD parametrisation.

**Key function**: `WMinPDF.grid_values_func()` implements:

```
f_{j,wm} = f_j + sum_i(w_i * (f_i - f_j))
```

This automatically satisfies sum rules. The central replica is always included.

### 2. POD-Basis Construction

Generate a Proper Orthogonal Decomposition basis:

1. Use `n3fit_pdf_model()` to generate a neural network PDF model
2. Sample replicas with `n3fit_pdf_grid()`
3. Filter outliers with `arclength_outliers()` and `sign_flip_selection()`
4. Perform SVD with `pod_basis()` to extract principal components
5. Export with `write_pod_basis()`

See `wmin/runcards/pod_basis_example.yaml` for a complete example.

---

## Configuration Reference

### wmin_providers (app.py:10-14)

```python
wmin_providers = [
    "wmin.model",
    "wmin.utils",
    "wmin.basis",
]
```

### WMinPDF Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `wminpdfset` | str | LHAPDF PDF set name |
| `n_basis` | int | Number of basis functions |

### FLAV_INFO (utils.py:17-52)

Default preprocessing exponents for NNPDF40 parametrisation:

- **sng, g, v, v3, v8, t3, t8, t15** flavour configurations
- Each with `smallx` and `largex` exponent ranges
- All have `trainable: False`

---

## Dependencies

### Runtime

- Python >= 3.11
- colibri (from HEP-PBSP)
- reportengine
- validphys
- jax, jaxlib
- numpy, pandas
- tensorflow
- lhapdf
- n3fit (for model generation)
- dill (for model serialization)
- mpi4py, mpich (for MPI support)
- ultranest (for nested sampling)

### Development

- pytest, hypothesis
- sphinx, recommonmark, sphinx_rtd_theme

---

## Testing

```bash
# Run all tests
pytest wmin/tests/

# Specific test files
pytest wmin/tests/test_wmin_basis.py
pytest wmin/tests/test_likelihood.py
pytest wmin/tests/test_fit.py
```

### Regression Tests

Regression test data is stored in:

- `wmin/tests/regression_results/` - Expected outputs
- `wmin/tests/regression_runcards/` - Test configurations

---

## Important Constants

| Constant | Value | Source |
|----------|-------|--------|
| `LHAPDF_XGRID` | LHAPDF x-grid | `colibri.constants` |
| `EXPORT_LABELS` | Export flavour labels | `colibri.constants` |
| `FLAVOUR_TO_ID_MAPPING` | Flavour to index mapping | `colibri.constants` |
| `FIT_XGRID` | Fit x-grid | Various |
| `Q` (default) | 1.65 GeV | Parametrisation scale |

---

## Data Flow

### Basis Construction

```
n3fit_pdf_model() 
  → n3fit_pdf_grid() 
    → arclength_pdfgrid() → arclength_outliers() → filter 
    → sign_flip_selection() → filter
  → get_X_matrix() → center data
  → pod_basis() → SVD → U, S, Vt
  → write_pod_basis() → export to LHAPDF format
```

### PDF Evaluation

```
WMinPDF(wminpdfset, n_basis)
  → grid_values_func(interpolation_grid)
    → convolution.evolution.grid_values()
    → build wmin_input_grid (centered on central replica)
    → return wmin_param(weights) closure
```

---

## Recent Changes (HEAD: b6078a1)

PR #23: Renamed `mc` and `ns` settings throughout the codebase:

- Updated filter.yml files in regression_results
- Updated mc_result.csv files
- Updated runcard YAML files
- Removed deprecated references from utils.py

---

## Known Issues & TODOs

1. **basis.py:75**: `n3fit_pdf_grid` TODO - write using jax
2. **basis.py:230-232**: Post-fit central member handling requires manual script execution
3. **utils.py:280**: `sign_flips_counter` discards last 25 points for numerical stability
4. **basis.py:245-251**: Manual intervention needed after evolution to fix central member

---

## File Index

### Python Modules (by size)

| File | Lines | Description |
|------|-------|-------------|
| `wmin/basis.py` | 253 | POD basis construction |
| `wmin/utils.py` | 316 | Utility functions |
| `wmin/model.py` | 99 | WMinPDF model |
| `wmin/tests/test_wmin_basis.py` | 446 | Basis tests |
| `wmin/tests/test_likelihood.py` | ~200 | Likelihood tests |
| `wmin/config.py` | 45 | Configuration |
| `wmin/api.py` | 29 | API module |
| `wmin/app.py` | 27 | CLI app |

### YAML Configurations

| File | Purpose |
|------|---------|
| `wmin/runcards/pod_basis_example.yaml` | POD basis example |
| `wmin/runcards/likelihood_time.yaml` | Likelihood timing benchmark |
| `wmin/tests/regression_runcards/*.yaml` | Regression test configs |

---

## External References

- **Paper**: [A linear PDF model for Bayesian inference](https://arxiv.org/pdf/2507.16913) (Costantini et al., 2025)
- **Colibri**: <https://github.com/HEP-PBSP/colibri>
- **NNPDF**: <https://github.com/NNPDF/nnpdf>
- **n3fit**: <https://github.com/NNPDF/n3fit>
- **ReportEngine**: <https://github.com/NNPDF/reportengine>
- **ValidPhys**: <https://github.com/NNPDF/validphys>
- **NNPOD Wiki**: <https://github.com/comane/NNPOD-wiki.git>

---

## Usage Examples

### Using the API

```python
from wmin.api import API
fig = API.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
fig.show()
```

### Running from CLI

```bash
# Run the wmin application
wmin --help

# Run with a specific runcard
wmin wmin/runcards/pod_basis_example.yaml
```

### Creating a POD Basis

```python
from wmin.basis import n3fit_pdf_model, n3fit_pdf_grid, pod_basis, write_pod_basis

# Generate model
model = n3fit_pdf_model(
    flav_info=FLAV_INFO,
    replica_range_settings={"min_replica": 1, "max_replica": 5},
    nodes=[25, 20, 8],
    activations=["tanh", "tanh", "linear"],
)

# Generate grid
pdf_grid = n3fit_pdf_grid(model, filter_arclength_outliers=True)

# Compute POD basis
U, phi0 = pod_basis(pdf_grid, Neig=10)

# Export
write_pod_basis((U, phi0), output_path="./my_basis")
```

---

*Generated for commit b6078a1 (2025-10-03)*
