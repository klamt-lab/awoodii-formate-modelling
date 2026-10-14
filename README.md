# Code for Formate Modeling and Thermodynamic Analysis of *Acetobacterium woodii*

This repository combines the *Acetobacterium woodii* metabolic model, measured D-stat fluxes, and thermodynamic data to investigate formate metabolism. The workflow fits experimental measurements, characterizes the fitted states, and examines how the NADH/NAD ratio affects the maximum thermodynamic driving force (MDF), metabolite concentrations, and reaction bottlenecks.

The input measurements cover DStat 09 average and DStat 10–18. The current campaign selection includes all ten flux fits and fitted characterizations, followed by a NADH/NAD lower-bound scan and ratio optimization for DStat 17 at 717 h.

## Location of key outputs

**MDF plot:**  `Results/03_NAD_Pool_Scan_with_Bottlenecks_PGCD0/plots/nadh_nad_lower_bound_scan_mdf_only.png`

**Raw data for the plot**: `Results/03_NAD_Pool_Scan_with_Bottlenecks_PGCD0/03_NAD_Pool_Scan_with_Bottlenecks_PGCD0_summary.csv` or the xlsx if you prefer

**Exact Ratio value**: `Results/04_NADH_NAD_Ratio_Exact_Value/04_NADH_NAD_Ratio_Exact_Value_summary.xlsx` under **Objective_Value** (it is log value so you have to calculate: $e^{-1.03992}$  = 0.3535)

---

## 1. Quick Start (Installation)

This project uses **[uv](https://github.com/astral-sh/uv)** for Python environment management. Dependencies are specified in `pyproject.toml` and pinned in `uv.lock`.

Run these commands from the project root directory:

```bash
# 1. Create the virtual environment (Python 3.10)
uv venv .venv -p 3.10

# 2. Activate the environment
source .venv/bin/activate

# 3. Install the locked dependencies
uv sync --locked
```

## 2. Manual Step: CPLEX Installation

The full model requires a licensed IBM CPLEX installation. The CPLEX Python package installed from PyPI has a Community Edition size limit that is too small for this model. The version specified for this project is **22.1.1.0**.

Install the Python package from your licensed CPLEX installation into `.venv`:

```bash
# Adjust the path to your CPLEX installation
uv pip install --python .venv/bin/python --no-deps /path/to/cplex/python/3.10/x86-64_linux
```

On the development machine, this directory is `/home/jasmin/cplex/python/3.10/x86-64_linux`.

Repeat this step if a later `uv sync` replaces the licensed package. Use `uv run --no-sync` for the commands below to retain the installed solver. The first flux fit checks whether CPLEX can solve the full model; an import check alone does not detect the Community Edition size limit.

## 3. Manual Step: Java Runtime

The measurement-fitting workflow uses CNApy [1] and requires a Java runtime. Install Java through your operating system's package manager and check that it is available:

```bash
java -version
```

---

## Repository Layout at a Glance

| Path                                  | Purpose                                                          |
| ------------------------------------- | ---------------------------------------------------------------- |
| `Models/`                           | SBML model used for flux fitting and thermodynamic analyses.     |
| `Data/`                             | Standard Gibbs-energy data and metabolite concentration bounds.  |
| `config/`                           | YAML components defining the model, environments, and analyses.  |
| `mymodelingpackage/`                | Bundled solver, experiment, campaign, export, and plotting code. |
| `run_campaign.py`                   | Campaign selection and sequential execution.                     |
| `plot_nadh_nad_lower_bound_scan.py` | Figures for one scan condition.                                  |
| `Results/`                          | Campaign outputs, summaries, and figures.                        |
| `pyproject.toml` / `uv.lock`      | Python requirements and locked dependency versions.              |

The `mymodelingpackage/` bundle supports `FluxValidation`, `Characterization`, and `VariableScan`, including concentration variability, cofactor ratios, bottleneck analysis, and both plotting modes.

---

## Model Assets

All model files are stored under `Models/`.

| File                        | Notes                                    |
| --------------------------- | ---------------------------------------- |
| `A_woodii_cleaned.xml.gz` | Irreversible*A. woodii* model from [2] |

The `Wildtype` component in `config/models.yml` selects this model, its Gibbs-energy data, the biomass reaction id, and model adjustments.

---

## Data Assets

### Thermodynamic Data

All files are stored under `Data/`.

| File                                          | Notes                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------ |
| `dG0_wildtype.json`                         | Standard reaction Gibbs energies used by the thermodynamic solver.                   |
| `concentrations_awoodii_h2_co2_medium.json` | Base metabolite concentration bounds, supplemented by environment-specific settings. |

### Configuration and Measurement Data

All files are stored under `config/`.

| File                                            | Notes                                                                                                |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `models.yml`                                  | Model selection and reaction-bound adjustments.                                                      |
| `analyses.yml`                                | Flux fitting, fitted characterization, NADH/NAD scanning, and ratio optimization settings.           |
| `environments_d_stat_raw.yml`                 | Measured biomass, formate, acetate, CO₂, and H₂ rates for the ten D-stat conditions.               |
| `environments_d_stat_fitted_generated.yml`    | Fitted environments written directly here after flux validation and loaded for subsequent campaigns. |
| `environments_d_stat_fitted_formate_feed.yml` | Active formate-feed environment settings; contains DStat 17 at 717 h.                               |

After flux fitting, the runner updates the formate-feed substrate, product, and flux-adjustment fields in memory from the new fits. Each template retains its own concentration settings. These updates do not rewrite the formate-feed YAML files.

---

## Running Campaigns/Analyses

### Configuration System

Model, environment, and analysis settings are defined as **components** in the YAML files under `config/`. Each experiment combines three component names:

```python
{"components": ["Wildtype", "MeasurementValidation_DStat_17_t717h", "FluxValidation_ATPM"]}
```

Edit `CAMPAIGNS_TO_RUN` directly in `run_campaign.py` to select experiments. Campaigns run in their listed order. Keep flux validation first when generating fresh fitted environments.

Run the configured campaigns from the project root:

```bash
uv run --no-sync python run_campaign.py
```

### A. Flux Validation

**Purpose:** Fit the raw measurements to the stoichiometric model and maximize ATP maintenance using the fitted fluxes.

1. **Setup:** `FluxValidation_ATPM` in `config/analyses.yml`, combined with the raw D-stat environments.
2. **Campaign:** `01_Flux_Validation`; currently selects all ten conditions.
3. **Output:** Fit results, corrected CNApy scenarios, and fitted environment files under `Results/01_Flux_Validation/`.
4. **Generated configuration:** The combined fitted-environment library is written directly to `config/environments_d_stat_fitted_generated.yml` and loaded for subsequent campaigns.

### B. Fitted Characterization

**Purpose:** Characterize each fitted condition through three optimizations: maximize MDF, maximize ATP maintenance, then maximize MDF at the fixed maximum ATP-maintenance flux.

1. **Setup:** `MDF_Characterization` in `config/analyses.yml`.
2. **Campaign:** `02_Fitted_Characterizations`; currently selects all ten fitted conditions.
3. **Output:** Optimization results, concentration variability, and cofactor-ratio ranges under `Results/02_Fitted_Characterizations/`.

### C. NADH/NAD Lower-Bound Scan

**Purpose:** Determine how increasing the NADH/NAD lower bound affects MDF and its thermodynamic bottlenecks.

1. **Setup:** `MDF_Bottlenecks_NADH_NAD_Lower_Bound_Scan_PGCD0` in `config/analyses.yml`.
2. **Scan range:** Nine evenly spaced physical ratios from **0.10 to 0.50 in steps of 0.05**. The YAML stores their natural logarithms for the constraint on `x_nadh_c - x_nad_c`.
3. **Campaign:** `03_NAD_Pool_Scan_with_Bottlenecks_PGCD0`; currently selects DStat 17 at 717 h.
4. **Output:** MDF, reaction bottlenecks, concentration variability, and cofactor ratios under `Results/03_NAD_Pool_Scan_with_Bottlenecks_PGCD0/`.

The thermodynamic analyses apply the ΔG° overrides specified in `analyses.yml`, including zero values for PGCD and both LDAPAT directions. These are Gibbs-energy adjustments, not reaction knockouts. Bottleneck screening requires repeated optimization at each scan point and can take considerable time.

### D. NADH/NAD Ratio Optimization

**Purpose:** Maximize the NADH/NAD ratio with MDF fixed to zero.

1. **Setup:** `NADH_NAD_Ratio_Characterization` in `config/analyses.yml`. The objective maximizes `x_nadh_c - x_nad_c`, the natural logarithm of the ratio.
2. **Campaign:** `04_NADH_NAD_Ratio_Exact_Value`; currently selects DStat 17 at 717 h.
3. **Output:** Results under `Results/04_NADH_NAD_Ratio_Exact_Value/`.

---

## Visualization

The plotting script uses editable `RESULTS_FOLDER` and `MODE` constants. Select `MODE = "mdf_only"` for the MDF figure or `MODE = "all"` for the MDF, variability, and bottleneck figures.

```bash
uv run --no-sync python plot_nadh_nad_lower_bound_scan.py
```

The default reads `Results/03_NAD_Pool_Scan_with_Bottlenecks_PGCD0/` and requires exactly one matching scan CSV. If that folder contains multiple conditions, set `RESULTS_FOLDER` to the desired condition subfolder relative to `Results/`.

The MDF-only figure uses enlarged fonts with no title or legend.

### Figure Outputs

The script saves **HTML, SVG, and PNG** files under `Results/<RESULTS_FOLDER>/plots/`. The NADH/NAD axis displays physical ratios on a linear scale.

---

## Reference

[1] [CNApy: a CellNetAnalyzer GUI in Python for analyzing and designing metabolic networks](https://doi.org/10.1093/bioinformatics/btab828)

[2] [Thermodynamic metabolic modeling of growth and bioproduction potential of the acetogen Acetobacterium woodii with and without redox cofactor swaps](https://doi.org/10.1016/j.synbio.2026.05.007)
