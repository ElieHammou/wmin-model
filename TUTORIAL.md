# wmin-model Tutorial: POD Basis Construction for NSBI Gluon Analysis

This tutorial guides you through constructing Proper Orthogonal Decomposition (POD) bases for PDF analysis using **wmin-model**, with a focus on the NSBI (Non-Singlet Bias Investigation) gluon studies.

---

## Overview

The **wmin-model** implements a linear PDF parametrisation using POD (Principal Component Analysis) to create compact, efficient bases for Bayesian inference. This tutorial covers the workflow used in the [nsbi_gluon_basis]( /Users/eliehammou/Projects/NSBI_gluon/nsbi_gluon_basis) analysis folder.

### Key Concepts

1. **POD Basis**: A set of orthogonal functions that capture the dominant modes of variation in PDF replicas
2. **Weight Minimisation (wmin)**: Linear parametrisation `f_wmin = f_0 + Σ w_i * (f_i - f_0)` that automatically satisfies sum rules
3. **Gluon-Focused Analysis**: Construct bases where only the gluon PDF varies, with other flavours fixed from external PDF sets

---

## Prerequisites

### Environment Setup

```bash
# From wmin-model directory
conda env create -f environment.yml
conda activate wmin-model-dev
```

Or for the NSBI-specific environment:
```bash
conda create python=3.12 -n pdf4sbi
conda install numpy jupyter lhapdf matplotlib nnpdf
```

### Required Packages
- `lhapdf` - LHAPDF PDF interface
- `nnpdf` - NNPDF framework
- `numpy`, `matplotlib` - Numerical and plotting
- `jupyter` - For notebook-based analysis
- `tensorflow` - For neural network PDF model generation

---

## Step 1: Configure POD Basis Construction

### Basic Run Card

Create a YAML configuration file (e.g., `pod_config.yaml`):

```yaml
meta:
  title: POD basis
  author: Your Name
  keywords: ["POD basis", "wmin"]

# NNPDF Neural Net Architecture settings
replica_range_settings:
  min_replica: 1
  max_replica: 20000

# Use external PDF set for non-gluon flavours
overwrite_non_gluon_from_lhapdf: true
lhapdf_set: NNPDF31_nnlo_as_0118_hessian

impose_sumrule: true
filter_sr_outliers: true

fitbasis: EVOL

nodes: [25, 20, 8]
activations: ["tanh", "tanh", "linear"]
initializer_name: "glorot_normal"
layer_type: "dense"

# Number of POD components to keep
Neig: 30

# Theory ID for evolution after SVD
theoryid: 40_000_000

actions_:
  - write_pod_basis
```

### Available Pre-configured Bases

From the NSBI analysis, several POD bases are available in `generating_POD/`:

| Basis | PDF Set | Replicas | Neig | Description |
|-------|---------|----------|------|-------------|
| `gluon_POD_nongluon_NNPDF31_hessian` | NNPDF31_nnlo_as_0118_hessian | 20000 | 30 | Gluon-only variation, NNPDF3.1 |
| `gluon_POD_nongluon_NNPDF40` | NNPDF40_nnlo_as_01180 | 20000 | 30 | Gluon-only variation, NNPDF4.0 |
| `gluon_POD_nongluon_PDF4LHC21` | PDF4LHC21 | 20000 | 30 | Gluon-only variation, PDF4LHC21 |
| `POD_vanilla_Elie` | Custom | 40000 | 50 | Full PDF variation |

---

## Step 2: Generate Neural Network PDF Model

The POD basis construction starts with generating random PDF replicas using a neural network.

### Python API

```python
from wmin.basis import n3fit_pdf_model, n3fit_pdf_grid
from wmin.utils import FLAV_INFO

# Generate NNPDF-style neural network model
model = n3fit_pdf_model(
    flav_info=FLAV_INFO,
    replica_range_settings={"min_replica": 1, "max_replica": 20000},
    nodes=[25, 20, 8],
    activations=["tanh", "tanh", "linear"],
    initializer_name="glorot_normal",
    layer_type="dense",
    fitbasis="EVOL",
)

# Generate PDF grid on LHAPDF x-grid
pdf_grid = n3fit_pdf_grid(
    model,
    filter_arclength_outliers=True
)
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `nodes` | Neural network hidden layer sizes | [25, 20, 8] |
| `activations` | Activation functions | ["tanh", "tanh", "linear"] |
| `replica_range_settings` | Number of replicas to generate | min: 1, max: 20000 |
| `impose_sumrule` | Apply sum rule constraints | True |
| `filter_arclength_outliers` | Remove pathological replicas | True |

---

## Step 3: Filter and Select Replicas

### Arclength Filtering

Removes replicas with unphysical arclength (measure of PDF smoothness):

```python
from wmin.utils import arclength_pdfgrid, arclength_outliers

# Calculate arclength for each replica
arclengths = arclength_pdfgrid(xgrid, pdf_grid)

# Identify outliers using interquartile range
outliers = arclength_outliers(arclengths)

# Filter out outliers
pdf_grid_filtered = np.delete(pdf_grid, outliers, axis=0)
```

### Sign Flip Selection

Ensures valence flavours (V, V3, V8, T3, T8) have at most one sign flip for numerical stability during evolution:

```python
from wmin.utils import sign_flip_selection

# Filter replicas with excessive sign flips
pdf_grid_clean = sign_flip_selection(pdf_grid_filtered)
```

---

## Step 4: Perform Singular Value Decomposition (SVD)

### Center the Data

```python
from wmin.basis import get_X_matrix

# Convert to 2D matrix and center
X, phi0 = get_X_matrix(pdf_grid_clean)
# X shape: (n_flavours * n_x, n_replicas)
# phi0 shape: (n_flavours * n_x, 1) - mean PDF
```

### Compute POD Basis

```python
from wmin.basis import pod_basis

# Perform SVD and extract first Neig components
pod, phi0 = pod_basis(pdf_grid_clean, Neig=30)
# pod shape: (Neig, n_flavours, n_x)
# phi0 shape: (n_flavours, n_x) - central member
```

### Eigenvalue Analysis

The eigenvalues indicate the variance captured by each POD component:

```python
# From generating_POD/pod_eigenvalues.csv
# mode_index,eigenvalue,eigenvalue_ratio
# 1,1.57554625e+06,1.00000000e+00
# 2,1.33596969e+05,8.47940668e-02
# 3,2.70955781e+04,1.71975773e-02
# ...
```

**Interpretation**: The first component captures ~100% of variance, second ~8.5%, third ~1.7%, etc. Typically 10-30 components capture >99% of the variance.

---

## Step 5: Export POD Basis

### Write to LHAPDF Format

```python
from wmin.basis import write_pod_basis

write_pod_basis(
    pod_basis=(pod, phi0),
    output_path="./my_pod_basis",
    Q=1.65,  # Parametrisation scale in GeV
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS
)
```

### Post-Processing

After evolution, run the member shifting script:

```bash
python shift_lhadf_members.py my_pod_basis/postfit/my_pod_basis
```

This:
1. Removes the post-fit generated central member
2. Shifts all members down by one index
3. Makes replica_1 the new central member

**Important**: Decrement `NumMembers` by 1 in the LHAPDF `.info` file.

---

## Step 6: Distance Calculation and Validation

### Distance Metrics

Use the distance functions from `distance_def.py` to validate your POD basis:

```python
from distance_def import wmin_distance, pdf_grid_allflav

# Calculate distance between target PDF and POD reconstruction
original, reconstructed, weights, distance = wmin_distance(
    pdf_target="NNPDF31_nnlo_as_0118",
    center_pdf_grid=center_grid,
    wmin_basis=basis_replicas,
    flavours=[21, 1, -1, 2, -2, 3, -3, 4, -4],  # g, u, ubar, d, dbar, s, sbar, c, cbar
    x_grid=LHAPDF_XGRID,
    Q=1.65,
    minimisation_type="chi2",
    target_distance_mode="mean_replicas"
)
```

### Minimisation Types

| Type | Description | Use Case |
|------|-------------|----------|
| `chi2` | Chi-squared minimisation | Default, most common |
| `l1` | L1 (absolute) minimisation | Robust to outliers |

### Target Distance Modes

| Mode | Description |
|------|-------------|
| `central` | Distance for central member only |
| `mean_replicas` | Average distance over all replicas |

---

## Step 7: Using the POD Basis for Fits

### Configure wmin-model with POD Basis

```yaml
# fit_config.yaml
wmin_settings:
  wminpdfset: path/to/your/pod_basis
  n_basis: 30

theoryid: 40_000_000

# Other fit settings...
```

### Run the Fit

```bash
wmin fit_config.yaml
```

---

## Gluon-Only POD Construction

For NSBI gluon analysis, you can fix all non-gluon flavours from an external PDF set:

```python
import lhapdf
import numpy as np

# Load external PDF set
pdf_set = lhapdf.getPDFSet("NNPDF31_nnlo_as_0118_hessian")
central_pdf = pdf_set.mkPDF(0)

# Get non-gluon flavours from external set
non_gluon_flavours = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]

# For each replica in your POD basis:
for i, replica_grid in enumerate(pod_basis):
    # Replace non-gluon flavours with external PDF
    for flav in non_gluon_flavours:
        idx = FLAVOUR_TO_ID_MAPPING.get(flav, flav)
        replica_grid[flav] = central_pdf.xfxQ(flav, xgrid, Q)
```

---

## Practical Examples from NSBI Analysis

### Example 1: NNPDF3.1 Hessian Basis

```yaml
# gluon_POD_nongluon_NNPDF31_hessian.yaml
replica_range_settings:
  min_replica: 1
  max_replica: 20000

overwrite_non_gluon_from_lhapdf: true
lhapdf_set: NNPDF31_nnlo_as_0118_hessian

impose_sumrule: true
filter_sr_outliers: true

Neig: 30
theoryid: 40_000_000
```

### Example 2: Vanilla POD (Full Variation)

```yaml
# POD_vanilla_Elie.yaml
replica_range_settings:
  min_replica: 1
  max_replica: 40000

overwrite_non_gluon_from_lhapdf: false

impose_sumrule: true
filter_sr_outliers: true

Neig: 50
theoryid: 40_000_000
```

---

## Visualization

### Plot POD Basis Components

```python
import matplotlib.pyplot as plt
import numpy as np

# Load POD basis
pod, phi0 = ...  # From your basis

# Plot first few components for gluon (flavour 21)
gluon_idx = 8  # Index for gluon in flavour list
xgrid = LHAPDF_XGRID

plt.figure(figsize=(12, 8))
for i in range(5):
    plt.plot(xgrid, pod[i, gluon_idx, :], label=f'Component {i+1}')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('x')
plt.ylabel('POD basis function')
plt.yscale('log')
plt.legend()
plt.title('First 5 POD Basis Components (Gluon)')
plt.savefig('pod_basis_components.png')
```

### Plot Eigenvalue Spectrum

```python
# From pod_eigenvalues.csv
eigenvalues = np.loadtxt('pod_eigenvalues.csv', delimiter=',', skiprows=1, usecols=1)

plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(eigenvalues)+1), eigenvalues / eigenvalues[0], 'o-')
plt.xlabel('Mode Index')
plt.ylabel('Normalized Eigenvalue')
plt.title('POD Eigenvalue Spectrum')
plt.grid(True, alpha=0.3)
plt.savefig('pod_eigenvalues.png')
```

---

## Troubleshooting

### Common Issues

1. **Ill-conditioned matrix**: Reduce `Neig` or increase number of replicas
   ```
   Condition number > 1e12 indicates numerical instability
   ```

2. **Sign flip violations**: Increase replica filtering or adjust neural network architecture

3. **Sum rule violations**: Ensure `impose_sumrule: true` and verify quadrature integration

4. **Memory issues**: Reduce `max_replica` or use smaller neural network

### Debugging Tips

```python
# Check condition number
import numpy as np
X, _ = get_X_matrix(pdf_grid)
cond = np.linalg.cond(X @ X.T)
print(f"Condition number: {cond:.2e}")

# Check sum rules
from colibri.constants import FLAVOUR_TO_ID_MAPPING
for flav in ['V', 'V3', 'V8', 'T3', 'T8', 'T15']:
    idx = FLAVOUR_TO_ID_MAPPING[flav]
    integral = np.trapz(pdf_grid[0, idx, :] * XGRID)
    print(f"{flav} sum rule: {integral:.6f}")
```

---

## Best Practices

1. **Start small**: Use 1000-5000 replicas for initial testing
2. **Validate filtering**: Check that arclength and sign flip filters remove <10% of replicas
3. **Monitor eigenvalues**: Ensure first 10-20 components capture >99% of variance
4. **Test reconstruction**: Verify that POD basis can reconstruct original PDFs with distance < 1%
5. **Document configuration**: Keep track of all parameters in YAML files

---

## References

- **Paper**: [A linear PDF model for Bayesian inference](https://arxiv.org/pdf/2507.16913) (Costantini et al., 2025)
- **Colibri Documentation**: https://hep-pbsp.github.io/colibri
- **NNPDF Framework**: https://github.com/NNPDF/nnpdf
- **NNPOD Wiki**: https://github.com/comane/NNPOD-wiki.git

---

## File Structure for Analysis

Recommended organization for your POD basis analysis:

```
my_analysis/
├── configs/
│   ├── pod_basis.yaml          # POD construction config
│   └── fit_config.yaml         # Fit configuration
├── input/
│   └── external_pdf_sets/      # External PDF sets for non-gluon
├── output/
│   ├── pod_basis/              # Generated POD basis
│   │   ├── replicas/           # Individual basis replicas
│   │   └── postfit/            # After evolution
│   ├── pod_eigenvalues.csv     # Eigenvalue spectrum
│   └── figures/                # Plots and visualizations
├── notebooks/
│   ├── generate_pod.ipynb      # POD generation notebook
│   ├── validate_basis.ipynb    # Validation notebook
│   └── analyze_results.ipynb   # Analysis notebook
└── scripts/
    ├── shift_lhadf_members.py  # Post-processing script
    └── custom_utils.py          # Custom utility functions
```

---

*Tutorial based on NSBI gluon analysis workflow from /Users/eliehammou/Projects/NSBI_gluon/nsbi_gluon_basis*
