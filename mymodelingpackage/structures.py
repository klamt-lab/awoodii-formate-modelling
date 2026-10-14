from __future__ import annotations

"""Core data containers for the new campaign framework.

This module is intentionally focused on pure dataclasses (no behavior/methods).
It defines the runtime contract between setup, solvers, experiments, and execution.

Structure overview:
1. Engine configuration containers:
   Static solver defaults and solver-specific runtime configuration.
2. Biological configuration containers:
   Model and environment settings used to build scenarios.
3. Scenario containers:
   Request and resolved scenario payloads (biology only).
4. Execution containers:
   Run/job metadata and optional solver overrides.
5. Experiment and campaign containers:
   The executable unit (`ExperimentDefinition`) and campaign-level grouping (`CampaignDefinition`).
"""

from typing import Annotated, Any, Literal, Optional, Union

import cobra
from cobrak.dataclasses import Model as CobrakModel
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Engine Settings Containers
# =============================================================================

# CPLEX defaults
DEFAULT_CPLEX_PARAMS: dict[str, Any] = {
    "timelimit": 600,
    "mip.tolerances.integrality": 0.0,
    "lpmethod": 1,
    "mip.display": 0,
    "emphasis.numerical": 1
}


class ThermoSolverCPLEXConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    to_solve: Literal["thermo"] = "thermo"                        # Discriminator for the thermo solver path, e.g. "thermo".
    # Parameters for docplex_optmdf():
    M_VALUE: float = Field(default=5_000.0, gt=0)                # Big-M constant used in the MILP formulation, e.g. 5000.0.
    MAX_FLUX: float = Field(default=1_000.0, gt=0)               # Hard cap for reaction bounds, e.g. 1000.0 for +/-1000 flux limits.
    STANDARD_R: float = 8.314e-3                                 # Gas constant in kJ/mol/K, e.g. 0.008314.
    STANDARD_T: float = Field(default=298.15, gt=0)             # Temperature in Kelvin, e.g. 298.15 for 25 C.
    with_loop_constraints: bool = False                          # Prevent simultaneous forward and reverse activity, e.g. True for split-reaction loop blocking.
    # Internal CPLEX solver configurations:
    cplex_params: dict[str, Any] = Field(default_factory=lambda: DEFAULT_CPLEX_PARAMS.copy())  # Raw CPLEX parameter overrides, e.g. {"timelimit": 600, "emphasis.numerical": 1}.


class MCSSolverOPTLANGConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    to_solve: Literal["mcs"] = "mcs"                     # Discriminator for the MCS solver path, e.g. "mcs".
    enum_method: int = Field(default=2, ge=0)            # Enumeration strategy passed to the MCS solver, e.g. 2.
    network_compression: bool = True                     # Compress the network before enumeration when possible, e.g. True.
    include_model_bounds: bool = True                    # Include original model bounds in the MCS problem, e.g. True.
    use_kn_in_dual: bool = False                         # Enable knock-in dual handling when a future workflow supplies knock-in indices.
    exclude_boundary_reactions_as_cuts: bool = False     # Ask optlang-enumerator to exclude boundary reactions from cut candidates, e.g. False.
    fva_tolerance: float = Field(default=1e-9, ge=0.0)   # FVA tolerance used inside network compression, e.g. 1e-9.
    big_m: float = Field(default=0.0, ge=0.0, validation_alias=AliasChoices("big_m", "bigM"))  # Big-M override passed to compute_mcs(); 0 uses optlang-enumerator default behavior.
    mip_opt_tol: float = Field(default=1e-6, ge=0.0)     # MIP optimality tolerance passed to optlang-enumerator, e.g. 1e-6.
    mip_feas_tol: float = Field(default=1e-6, ge=0.0)    # MIP feasibility tolerance passed to optlang-enumerator, e.g. 1e-6.
    mip_int_tol: float = Field(default=1e-6, ge=0.0)     # MIP integrality tolerance passed to optlang-enumerator, e.g. 1e-6.
    results_cache_dir: Optional[str] = None              # Optional cache directory for optlang-enumerator intermediate results.


class MCSSolverSDConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    to_solve: Literal["mcs_sd"] = "mcs_sd"                         # Discriminator for the strain-design MCS solver path, e.g. "mcs_sd".
    solution_approach: Literal["any", "best", "populate"] = "populate"  # StrainDesign solution approach, e.g. "populate" for multiple designs.
    network_compression: bool = True                               # Compress the network before strain-design enumeration when possible, e.g. True.
    big_m: Optional[float] = Field(default=None, ge=0.0, validation_alias=AliasChoices("big_m", "M"))  # Optional Big-M value for strain-design MILP construction; None uses indicator constraints when available.
    random_seed: Optional[int] = Field(default=None, ge=1)          # Optional deterministic StrainDesign seed, e.g. 12345.


class NullSolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    to_solve: Literal["none"] = "none"                # Discriminator for workflows that do not need a solver, e.g. "none".


class StoichiometricSolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    to_solve: Literal["stoichiometric"] = "stoichiometric"  # Discriminator for LP/FBA-style stoichiometric solves, e.g. "stoichiometric".
    interface_name: Literal["optlang"] = "optlang"          # Explicit optlang/cobrapy interface backing the stoichiometric solver, e.g. "optlang".
    solver_name: str                                         # Required cobrapy backend override, e.g. "glpk", "cplex", or "gurobi".


class CobrakSolverConfig(BaseModel):
    """Configure COBRA-k linear and nonlinear optimization workflows.

    COBRA-k uses CPLEX through Pyomo for LP/MILP formulations and IPOPT for
    the reduced kinetic NLP. The configuration is immutable because one solver
    instance may be reused across many scenario and scan-point solves.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    # ------------------------------------------------------------------
    # Backend and formulation selection
    # ------------------------------------------------------------------
    to_solve: Literal["cobrak"] = "cobrak"                              # Campaign solver discriminator; must remain "cobrak".
    formulation: Literal[
        "stoichiometric_lp",
        "enzyme_lp",
        "thermodynamic_lp",
        "enzyme_thermodynamic_lp",
        "kinetic_nlp",
    ] = "enzyme_thermodynamic_lp"                                      # Constraint layer to execute, e.g. "thermodynamic_lp" or "kinetic_nlp".

    # ------------------------------------------------------------------
    # Pyomo solver configuration
    # ------------------------------------------------------------------
    # Pyomo backends are separate because CPLEX cannot solve the nonlinear
    # kinetic formulation. Kinetic runs use CPLEX for their LP/FVA seed and
    # IPOPT for the reduced nonlinear problem.
    linear_solver_name: Literal["cplex_direct"] = "cplex_direct"          # Required Pyomo LP/MILP interface; fixed to CPLEX direct.
    nonlinear_solver_name: Literal["ipopt"] = "ipopt"                   # Required Pyomo NLP backend for kinetic_nlp.
    linear_solver_options: dict[str, float | int | str] = Field(default_factory=lambda: {"threads": 1})  # Persistent CPLEX options, e.g. {"threads": 1}.
    linear_solver_attrs: dict[str, float | int | str] = Field(default_factory=dict)  # Attributes assigned to the Pyomo CPLEX solver object.
    linear_solve_options: dict[str, Any] = Field(default_factory=dict)    # Extra keyword arguments passed to each CPLEX solve call.
    nonlinear_solver_options: dict[str, float | int | str] = Field(default_factory=lambda: {"max_iter": 4000, "halt_on_ampl_error": "yes"})  # Persistent IPOPT options.
    nonlinear_solver_attrs: dict[str, float | int | str] = Field(default_factory=dict)  # Attributes assigned to the Pyomo IPOPT solver object.
    nonlinear_solve_options: dict[str, Any] = Field(default_factory=dict)  # Extra keyword arguments passed to each IPOPT solve call.

    # ------------------------------------------------------------------
    # Shared COBRA-k behavior and validation
    # ------------------------------------------------------------------
    with_variability_analysis: bool = False                             # Run COBRA-k variability before the mandatory LP and pass its bounds to later stages.
    with_loop_constraints: bool = False                                 # Add COBRA-k loop constraints where supported.
    ignore_nonlinear_terms: bool = False                                # Omit nonlinear terms in compatible LP helper calls.
    verbose_solver: bool = False                                        # Forward detailed CPLEX/IPOPT output to the console.
    missing_data_policy: Literal["strict", "warn"] = "strict"          # Raise on missing dG0/enzyme/kinetic data or continue with warnings.
    var_data_abs_epsilon: float = Field(default=1e-5, ge=0.0)           # Absolute epsilon used when applying variability data to COBRA-k models.

    # ------------------------------------------------------------------
    # Kinetic active-set search and final NLP
    # ------------------------------------------------------------------
    kinetic_best_result_file: Optional[str] = None                       # Saved COBRA-k best-result dictionary used as the active-reaction set; without it evolution and postprocessing run first.
    kinetic_with_kappa: bool = True                                     # Include substrate/product saturation terms in the kinetic NLP.
    kinetic_with_gamma: bool = True                                     # Include thermodynamic displacement terms in the kinetic NLP.
    kinetic_with_iota: bool = False                                     # Include inhibition terms when annotated and requested.
    kinetic_with_alpha: bool = False                                    # Include activation terms when annotated and requested.
    kinetic_approximation_value: float = Field(default=1e-4, ge=0.0)    # Numerical approximation used by COBRA-k nonlinear expressions.
    kinetic_strict_mode: bool = False                                   # Require strict kinetic parameter handling in COBRA-k NLP construction.
    kinetic_single_strict_reactions: list[str] = Field(default_factory=list)  # Reaction variants that should use strict kinetics individually.
    kinetic_nlp_verbose: bool = False                                  # Forward IPOPT output for the kinetic NLP stage only.
    kinetic_evolution_feasible_starts: int = Field(default=3, gt=0)     # Number of distinct feasible NLP starts requested during evolutionary initialization.
    kinetic_evolution_max_metarounds: int = Field(default=3, gt=0)      # Maximum initialization metarounds used to find feasible starts.
    kinetic_evolution_rounds_per_metaround: int = Field(default=2, gt=0)  # Sampling rounds run per initialization metaround and worker.
    kinetic_evolution_max_deactivated_reactions: int = Field(default=5, gt=0)  # Maximum randomly deactivated reactions in one initialization sample.
    kinetic_evolution_num_generations: int = Field(default=50, gt=0)     # Number of genetic-algorithm generations.
    kinetic_evolution_population_size: Optional[int] = Field(default=None, gt=0)  # Population size; None lets COBRA-k use the available CPU count.
    kinetic_evolution_random_seed: int = Field(default=12345, ge=0, le=4_294_967_295)  # Reproducible Python and NumPy seed for evolutionary sampling and genetic optimization.

    # ------------------------------------------------------------------
    # Optional initial variability analysis
    # ------------------------------------------------------------------
    variability_input_file: Optional[str] = None                        # Optional saved variability dictionary loaded instead of calculating FVA.
    variability_output_file: Optional[str] = None                       # Optional destination for the effective variability dictionary.
    variability_calculate_reacs: bool = True                            # Calculate reaction-variant ranges.
    variability_calculate_concs: bool = True                            # Calculate concentration-variable ranges.
    variability_calculate_rest: bool = True                             # Calculate remaining COBRA-k variable ranges.
    variability_verbose: bool = False                                   # Forward per-objective CPLEX output from variability analysis.
    variability_active_reactions: list[str] = Field(default_factory=list)  # Explicit reaction variants kept active during variability; empty derives the optimized reaction direction.
    variability_min_active_flux: float = Field(default=1e-5, ge=0.0)       # Minimum flux imposed on explicitly active reactions.
    variability_min_flux_cutoff: float = Field(default=1e-5, ge=0.0)       # Cutoff used to round small reaction lower bounds to zero.
    variability_min_enzyme_cutoff: float = Field(default=1e-5, ge=0.0)     # Cutoff used to round small enzyme lower bounds to zero.
    variability_reaction_upper_floor: Optional[float] = Field(default=None, ge=0.0)  # Optional floor for effective reaction upper bounds.
    variability_parallel_verbosity: int = Field(default=0, ge=0)        # COBRA-k/joblib progress verbosity.
    variability_cache_file: Optional[str] = None                        # Optional JSON cache; supports scenario and component placeholders.
    variability_cache_overwrite: bool = False                           # Recompute and replace an existing variability cache.

# Using a discriminator makes deserializing YAML dictionaries into objects incredibly fast and safe
SolverConfig = Annotated[
    Union[ThermoSolverCPLEXConfig, MCSSolverOPTLANGConfig, MCSSolverSDConfig, NullSolverConfig, StoichiometricSolverConfig, CobrakSolverConfig],
    Field(discriminator="to_solve")
]


# =============================================================================
# Biological Settings Containers
# =============================================================================

class ModelConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # for solve()
    model_file: str                                                    # Model SBML/JSON file to load, e.g. "A_woodii_cleaned.xml.gz" or "e_coli_core.xml.gz".
    cobrak_model_file: Optional[str] = None                             # Optional native JSON or annotated SBML paired with the clean COBRApy model.
    cobrak_preserve_model_state: bool = False                           # Keep a saved COBRA-k JSON model unchanged instead of validating and synchronizing it against the clean reporting model.
    dG0_file: Optional[str] = None                                     # Optional thermodynamic data file, e.g. "dG0_awoodii.json".
    biomass_id: Optional[str] = None                                   # Biomass reaction id for this model, e.g. "biomass" or "BIOMASS_Ecoli_core_w_GAM".
    forward_reaction_suffix: str = "_FWD"                              # Forward split-reaction suffix, e.g. "_FWD".
    reverse_reaction_suffix: str = "_REV"                              # Reverse split-reaction suffix, e.g. "_REV".
    model_adjustments: dict[str, Any] = Field(default_factory=dict)    # Bounds or other model-level overrides, e.g. {"ATPM": (1.0, 1000.0)} or {"EX_ac_e_FWD": [0, 0]}.
    reaction_string_overrides: dict[str, str] = Field(default_factory=dict)  # Optional reaction equation replacements applied after loading, e.g. {"FRUK": "f1p_c <=> fdp_c"}.
    
    # Groups of mutually exclusive reaction families or explicit reaction IDs.
    # Example:
    #   [["HDCR"]] keeps one HDCR-family chemistry active at a time.
    #   [["PFOR", "PFL"]] forbids PFOR and PFL from being active together.
    # Backward-compatible input alias: `cofac_alternatives`.
    exclusive_reaction_groups: list[list[Any]] = Field(                # Mutually exclusive reaction groups, e.g. [["HDCR"]] or [["PFOR", "PFL"]]; accepts legacy `cofac_alternatives` on input.
        default_factory=list,
        validation_alias=AliasChoices("exclusive_reaction_groups", "cofac_alternatives"),
    )                                                                  
    extra: dict[str, Any] = Field(default_factory=dict)                # Free-form future model settings, e.g. {"notes": "use alt biomass"}.

    @property
    def cofac_alternatives(self) -> list[list[Any]]:
        """Backward-compatible alias for legacy code paths."""
        return self.exclusive_reaction_groups


class EnvironmentConfig(BaseModel):
    concentrations_file: Optional[str] = None                                                    # Optional concentrations file for the environment, e.g. "concentrations_awoodii_h2_co2_medium.json".
    substrates: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)                          # Substrate condition definitions keyed by name, e.g. {"co2_h2": [{"id": "EX_co2_e_REV", "lb": 0, "ub": 60}]}.
    products: dict[str, dict[str, Any]] = Field(default_factory=dict)                                  # Product definitions keyed by name, e.g. {"acetate": {"id": "EX_ac_e_FWD", "lb": 0.0, "ub": 1000.0}}.
    substrate_product_adjustments: dict[str, dict[str, list[dict[str, Any]]]] = Field(default_factory=dict)  # Pair-specific bound adjustments, e.g. {"glucose_anaerobic": {"acetate": [{"id": "ATPM", "lb": 2.0, "ub": 1000.0}]}}.
    concentration_adjustments: dict[str, dict[str, dict[str, dict[str, float]]]] = Field(default_factory=dict)  # Pair-specific concentration overrides, e.g. {"co2_h2": {"acetate": {"nad_c": {"min": 1e-6, "max": 1e-3}}}}.
    extra: dict[str, Any] = Field(default_factory=dict)                                                # Free-form future environment settings, e.g. {"notes": "serum-free"}.


# =============================================================================
# Scenario Containers (Biology Only)
# =============================================================================

class BiologicalScenario(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_id: str                              # Human-readable identifier composed from model, environment, substrate, and product labels, e.g. "a_woodii__a_woodii_env__co2_h2__acetate".
    model_component: str                          # Model component name used to build the scenario, e.g. "a_woodii" or "e_coli_core".
    environment_component: str                    # Environment component name used to build the scenario, e.g. "a_woodii_env".
    substrate_name: str                           # Active substrate condition name, e.g. "co2_h2" or "glucose_anaerobic".
    product_name: str                             # Active product name, e.g. "acetate" or "isopropanol".

    model: cobra.Model                            # Fully prepared COBRA model for this scenario, e.g. a cobrapy Model with applied bounds and adjustments.
    cobrak_model: Optional[CobrakModel] = None     # Optional annotated COBRA-k representation prepared for COBRA-k solver runs.
    concentrations: dict[str, dict[str, float]]  # Resolved metabolite concentration ranges, e.g. {"nad_c": {"min": 1e-6, "max": 1e-3}}.
    dG0: Optional[dict[str, Any]]                 # Optional resolved thermodynamic dataset, e.g. {"R1": {"dG0": -20.1}}.

    model_conf: ModelConfig                       # Typed model configuration used to construct the scenario, e.g. ModelConfig(model_file="A_woodii_cleaned.xml.gz", ...).
    environment_conf: EnvironmentConfig           # Typed environment configuration used to construct the scenario, e.g. EnvironmentConfig(substrates={...}, products={...}).
    diagnostics_environment: dict[str, Any] = Field(default_factory=dict)  # Effective-medium record created while applying the environment, including open uptake reactions and changed bounds.


# =============================================================================
# Execution Containers (Run Metadata)
# =============================================================================


# Shared result object for thermo and stoichiometric optimization solves.
class SolverResult(BaseModel):
    status: str                                  # e.g. "Optimal", "Infeasible", or "TimeLimit".
    objective_value: Optional[float] = None      # The final objective score, e.g. 8.42.
    values: dict[str, float] = Field(default_factory=dict)  # Variable names mapped to solved values, e.g. {"ATPM": 3.0, "var_B": 0.2}.
    metrics: dict[str, Any] = Field(default_factory=dict)   # Compact backend-neutral metrics such as MDF, protein use, and stage timings.
    metadata: dict[str, Any] = Field(default_factory=dict)  # Solver/formulation/version and termination details used for reproducibility.
    native_result: Optional[dict[str, Any]] = Field(default=None, exclude=True)  # Package-native result retained in memory and exported only as an optional sidecar.
    message: str = ""                            # Optional solver termination details, e.g. "optimal solution found".
    error_message: Optional[str] = None          # Populated if an exception occurred, e.g. "CPLEX numerical issue".


# =============================================================================
# Experiment and Campaign Containers
# =============================================================================

# =====================================================================
# 1. The Unified Analysis Parameters (Matrix Rules + Post-Analysis)
# =====================================================================


class EscherConfig(BaseModel):
    map_identifier: str                                                   # Built-in Escher map name or path to a local map JSON, e.g. "e_coli_core.Core metabolism" or "/maps/acetogen_map.json".
    reaction_data_source: Literal["flux", "force"] = "flux"              # Values written into Escher reaction tooltips, e.g. "flux" or "force".
    size_source: Literal["flux", "force", "none"] = "force"              # Quantity used to scale arrow thickness, e.g. "force".
    filename_suffix: str = "visualization"                               # Suffix for the exported HTML file, e.g. "visualization" or "escher_growth".
    min_arrow_thickness: int = Field(default=4, ge=1)                    # Minimum arrow thickness in the rendered map, e.g. 4.
    max_arrow_thickness: int = Field(default=1000, ge=1)                 # Maximum arrow thickness in the rendered map, e.g. 1000.
    active_color: str = "#ff0000"                                        # Color for active reactions, e.g. "#ff0000".
    inactive_color: str = "#808080"                                      # Color for inactive reactions, e.g. "#808080".
    activity_threshold: float = Field(default=1e-6, ge=0.0)              # Absolute threshold used to classify reactions as active, e.g. 1e-6.
    exclude_exchange_reactions_from_scaling: bool = True                 # Exclude exchange reactions from the scaling range, e.g. True.


class CVAConfig(BaseModel):
    metabolites_to_check: list[str] = Field(default_factory=list)               # Metabolites to analyze via CVA, e.g. ["nad_c", "nadh_c"].
    target_successes: int = Field(default=5, ge=1)                              # Number of successful min/max samples to collect per metabolite, e.g. 5.
    max_attempts: int = Field(default=20, ge=1)                                 # Hard stop for adaptive CVA retries, e.g. 20.


class FVAConfig(BaseModel):
    """Configure numerical FVA targets and retry limits.

    Entries in ``reactions_to_check`` are normally model reaction IDs. A local
    Escher ``.json`` path is an additional shorthand that expands to every map
    reaction available in the active model. Explicit IDs and map paths may be
    mixed in the same list.

    Examples:
        ```python
        FVAConfig(
            reactions_to_check=["ATPM", "./Models/Maps/core.json"],
            target_successes=1,
            max_attempts=1,
        )
        ```
    """

    model_config = ConfigDict(extra="forbid")

    reactions_to_check: list[str] = Field(default_factory=list)                 # Reaction IDs or local Escher map JSON paths expanded into reaction IDs, e.g. ["ATPM", "./maps/core.json"].
    target_successes: int = Field(default=5, ge=1)                              # Number of successful min/max samples to collect per reaction, e.g. 5.
    max_attempts: int = Field(default=20, ge=1)                                 # Hard stop for adaptive FVA retries, e.g. 20.


class CobrakNLPVariabilityConfig(BaseModel):
    """Configure COBRA-k variability around a solved kinetic active set."""

    model_config = ConfigDict(extra="forbid")
    calculate_reacs: bool = True                                               # Calculate directional reaction-flux ranges.
    calculate_concs: bool = True                                               # Calculate logarithmic metabolite-concentration ranges.
    calculate_rest: bool = True                                                # Calculate enzyme, kappa, gamma, and remaining native-variable ranges.
    warm_start: bool = True                                                    # Initialize each native NLP from the successful primary kinetic solution.
    active_reactions: list[str] = Field(default_factory=list)                  # Native reaction variants that must remain active, e.g. ["ATPM_FWD"].
    min_active_flux: float = Field(default=1e-5, ge=0.0)                       # Minimum flux imposed on explicitly active reaction variants.
    extra_tested_vars: list[str] = Field(default_factory=list)                 # Additional native COBRA-k variables tested in both directions.
    min_flux_cutoff: float = Field(default=1e-8, ge=0.0)                       # Small reaction minima reported as zero.
    do_not_delete_with_z_var_one: bool = False                                 # Preserve reactions marked active by a z variable in the supplied best result.


class VariableOptimizationConfig(BaseModel):
    variables_to_check: list[str] = Field(default_factory=list)                 # Solver variables to re-optimize under the locked base state; must be fluxes/reactions or var_B, e.g. ["var_B", "ATPM"].
    target_successes: int = Field(default=1, ge=1)                              # Number of successful samples to collect per variable, e.g. 1.
    max_attempts: int = Field(default=1, ge=1)                                  # Hard stop for adaptive optimization retries, e.g. 3.


class MetaboliteRatiosConfig(BaseModel):
    ratios: list[dict[str, Any]] = Field(default_factory=list)                  # Ratio definitions to evaluate, e.g. [{"name": "NADH/NAD", "metabolites": ["nadh_c", "nad_c"], "multipliers": [1.0, -1.0]}].
    target_successes: int = Field(default=5, ge=1)                              # Number of successful min/max samples to collect per ratio, e.g. 5.
    max_attempts: int = Field(default=25, ge=1)                                 # Hard stop for adaptive ratio-analysis retries, e.g. 25.


class BottleneckAnalysisConfig(BaseModel):
    dG0_change: float = 0.5                                                  # dG0 perturbation applied per reaction during bottleneck screening, e.g. 10.0.
    test_active_reactions_only: bool = False                                    # Restrict screening to reactions active in the base solution, e.g. True.


class BottleneckSetAnalysisConfig(BaseModel):
    target_variable: Optional[str] = None                                      # Variable that should reach the target; defaults to the active MDF variable ("var_B" or "var_B2"), e.g. "biomass".
    target_mode: Literal["base_plus_epsilon", "base_fraction", "absolute"] = "base_plus_epsilon"  # How the required MDF target is derived from the base solution, e.g. "base_plus_epsilon".
    target_value: float = 0.1                                                   # Meaning depends on target_mode: additive epsilon, multiplicative fraction, or absolute MDF target, e.g. 0.1.
    target_constraint_sign: Literal["geq", "leq", "eq"] = "geq"                 # Constraint sense for the target variable, e.g. "geq" for reaching at least a growth rate.
    topology_lock_mode: Literal["auto", "none", "variants_only", "active_set"] = "auto"  # How strongly the base solution topology is preserved during the bottleneck-set solve, e.g. "variants_only".
    enumerate_all_minimal_sets: bool = False                                    # Enumerate alternative minimal-cardinality bottleneck sets instead of only the first one, e.g. True.
    max_sets: int = Field(default=10, ge=1)                                     # Maximum number of minimal sets to collect when enumeration is enabled, e.g. 10.


class TrueBottleneckCheckConfig(BaseModel):
    target_reaction_ids: Optional[list[str]] = None                             # Optional reaction ids to verify explicitly instead of auto-detected candidates, e.g. ["HDCR_FWD", "CODH4_FWD"].


class MetaboliteBottleneckAnalysisConfig(BaseModel):
    relax_factor_concentrations: float                                          # Relative concentration-range expansion used for the relaxed comparison solve, e.g. 0.2.
    test_active_metabolites_only: bool = False                                  # Restrict screening to metabolites touched by active reactions, e.g. True.


class Dg0AdjustmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str                                                                  # Reaction id whose dG0 should change; use the concrete split id when available, e.g. "PFOR_FWD".
    mode: Literal["add", "set", "scale"] = "add"                                # How value is applied: "add" -> old + value, "set" -> value, "scale" -> old * value.
    value: float                                                                 # Numeric dG0 change, absolute dG0, or scaling factor depending on mode, e.g. -5.0.
    pair_policy: Literal["none", "auto_reverse"] = "auto_reverse"               # Keep the paired split reaction thermodynamically opposite when present, e.g. PFOR_REV = -PFOR_FWD.


class ConcentrationAdjustmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str                                                                  # Metabolite id whose concentration bounds should change, e.g. "nadh_c".
    min: Optional[float] = None                                                  # Optional lower concentration bound; None keeps the loaded scenario value, e.g. 1e-6.
    max: Optional[float] = None                                                  # Optional upper concentration bound; None keeps the loaded scenario value, e.g. 1e-3.


class ThermoAdjustmentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dG0: list[Dg0AdjustmentConfig] = Field(default_factory=list)                 # Experiment-wide dG0 changes applied to solver-local copies after scenario loading, e.g. [{"target": "PFOR_FWD", "mode": "add", "value": -5.0}].
    concentrations: list[ConcentrationAdjustmentConfig] = Field(default_factory=list)  # Experiment-wide concentration overrides applied to solver-local copies, e.g. [{"target": "nadh_c", "max": 1e-4}].


class AnalysisParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Shared setup / metadata.
    subsystem_map_file: Optional[str] = None                                 # Optional subsystem mapping file, e.g. "subsystems_awoodii.json".

    # Shared biological thresholds and reporting selectors.
    min_growth: float = 0.0                                                  # Global minimum growth threshold, e.g. 0.05.
    fraction_prod: float = 0.0                                               # Fraction of the maximum product yield used in shorthand constraints, e.g. 0.8.
    key_reactions: list[str] = []                                            # Reaction prefixes to annotate explicitly in outputs, e.g. ["HDCR", "RNF"].
    key_metabolites: list[str] = []                                          # Metabolite IDs to annotate explicitly in outputs, e.g. ["nadh_c", "atp_c"].

    # Shared retry / variant-selection controls.
    num_numerical_retries: int = 5                                            # Retry count for numerically unstable solves, e.g. 5.
    num_swap_attempts: int = 5                                                # Retry count for swap-minimization solves, e.g. 5.
    num_reoptimization_attempts_sandwich: int = 5                             # Retry count for the sandwich re-optimization phase, e.g. 5.
    minimize_swaps: Optional[Any] = None                                      # Optional swap-minimization configuration, e.g. {"reaction_groups": ["HDCR", "RNF"]}.
    
    # Shared output / artifact controls.
    sorting_key_column: Optional[str] = None                                  # Optional sorting key expected by downstream plotting/reporting, e.g. "H2" or "atpm_target".
    json_output: bool = True                                                  # Whether the experiment should write its aggregate JSON output, e.g. True.
    csv_output: bool = True                                                   # Whether the experiment should write CSV output, e.g. True.
    excel_output: bool = False                                                # Export one workbook with standardized Summary and available Fluxes sheets, e.g. True.
    save_raw_solutions: bool = False                                          # Export raw solver values as JSON, e.g. True.
    save_realized_adjustments: bool = False                                   # Export realized pathway adjustments, e.g. True.
    escher: Optional[EscherConfig] = None                                     # Optional Escher export configuration, e.g. EscherConfig(map_identifier="core_map").
    generate_cnapy_files: bool = False                                        # Export CNApy scenario files, e.g. True.
    
    # Optional post-analysis blocks.
    run_cva: Optional[CVAConfig] = None                                       # CVA configuration block, e.g. CVAConfig(metabolites_to_check=["nad_c", "nadh_c"]).
    run_fva: Optional[FVAConfig] = None                                       # FVA computation block; reactions_to_check accepts reaction IDs and optional Escher map JSON paths.
    run_cobrak_nlp_variability: Optional[CobrakNLPVariabilityConfig] = None    # Native kinetic variability around the COBRA-k best-result active set.
    run_min_optimization: Optional[VariableOptimizationConfig] = None         # Generic locked-state minimization for flux variables or MDF variables, e.g. VariableOptimizationConfig(variables_to_check=["ATPM"]).
    run_max_optimization: Optional[VariableOptimizationConfig] = None         # Generic locked-state maximization for flux variables or MDF variables, e.g. VariableOptimizationConfig(variables_to_check=["var_B"]).
    run_metabolite_ratios: Optional[MetaboliteRatiosConfig] = None            # Metabolite-ratio configuration block, e.g. MetaboliteRatiosConfig(ratios=[{"name": "NADH/NAD", "metabolites": ["nadh_c", "nad_c"], "multipliers": [1.0, -1.0]}]).
    run_bottleneck_analysis: Optional[BottleneckAnalysisConfig] = None        # Bottleneck-analysis configuration block, e.g. BottleneckAnalysisConfig(dG0_change=5.0).
    run_bottleneck_set_analysis: Optional[BottleneckSetAnalysisConfig] = None  # MILP-based bottleneck-set configuration block, e.g. BottleneckSetAnalysisConfig(target_mode="base_plus_epsilon", target_value=0.1).
    run_true_bottleneck_check: Optional[TrueBottleneckCheckConfig] = None     # True bottleneck verification block, e.g. TrueBottleneckCheckConfig(target_reaction_ids=["HDCR_FWD"]).
    run_metabolite_bottleneck_analysis: Optional[MetaboliteBottleneckAnalysisConfig] = None  # Metabolite bottleneck-analysis configuration block, e.g. MetaboliteBottleneckAnalysisConfig(relax_factor_concentrations=0.2).

    apply_specificity_constraints: bool = False                                # Apply specificity constraints during post-analysis, e.g. True.

    # Thermo-only parameters.
    min_mdf: float = -1000                                                    # Minimum MDF threshold; -1000 means effectively off.
    subnetwork: list[str] = Field(default_factory=list)                       # Reaction ids for subMDF / var_B2 calculations.
    thermo_unconstrained_reactions: list[str] = Field(default_factory=list)    # Reactions left thermodynamically unconstrained.
    thermo_adjustments: Optional[ThermoAdjustmentsConfig] = None               # Per-analysis dG0/concentration overrides.

    # MCS-only parameters. Kept here for now so MCS uses the same shared
    # `analysis_params` container as the other experiment types.
    max_cost: int = Field(default=12, ge=0)                                  # Max MCS intervention cost / size.
    max_solutions: int = Field(default=100, ge=1)                            # Max number of MCS sets to enumerate.
    time_limit: int = Field(default=600, ge=1)                               # Enumeration timeout in seconds.
    yield_fraction: float = Field(default=0.8, gt=0.0, le=1.0)               # Fraction of max carbon yield used as MCS target.
    calculate_yield_metrics: bool = False                                    # Calculate product/yield reporting metrics even when custom MCS suppress constraints are supplied.
    exchange_protection: bool = True                                         # Exclude exchange reactions from KO candidates.
    transport_suffixes: list[str] = Field(default_factory=lambda: ["tex", "texi", "tiex", "tpp", "tipp", "tppi", "trpp", "rpp"])  # Reaction id suffixes treated as transport reactions and excluded from default KO candidates.
    save_sds_files: bool = True                                              # Save successful MCS sets as .sds artifacts.
    export_ko_models_yml_for: Optional[str] = None                           # Export KO model YAML entries for "used_model" or for a target model file, e.g. "ECC2_Isopropanol_irrev.xml.gz"; None disables export.
    ko_models_yml_filename: Optional[str] = None                              # Optional KO model YAML filename; defaults to model_<campaign name>.yml.
    cobra_solver: str = Field(default="cplex", validation_alias=AliasChoices("cobra_solver", "solver"))  # COBRA backend; accepts legacy `solver`.

    @field_validator("export_ko_models_yml_for", mode="before")
    @classmethod
    def _validate_export_ko_models_yml_for(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("export_ko_models_yml_for must be 'used_model' or a target model filename, not a boolean.")
        if not isinstance(value, str):
            raise ValueError("export_ko_models_yml_for must be 'used_model' or a target model filename.")
        normalized_value = value.strip()
        if not normalized_value:
            return None
        if normalized_value.lower() in {"true", "false"}:
            raise ValueError("export_ko_models_yml_for must be 'used_model' or a target model filename, not a boolean-like string.")
        if normalized_value.lower() == "used_model":
            return "used_model"
        return normalized_value

# =====================================================================
# 2. The Experiment Workflow Configs
# =====================================================================
class OptimizationStep(BaseModel):
    name: str                                           # Human-readable step name, e.g. "find_max_growth" or "optMDF_minProd".
    objective_var_ids: list[str]                        # Objective variable ids optimized in this step, e.g. ["ATPM"] or ["biomass", "EX_ac_e"].
    objective_multipliers: Optional[list[float]] = None  # Objective coefficients aligned with objective_var_ids, e.g. [1.0] or [1.0, 0.5].
    objective_direction: str = "max"                   # Optimization direction, typically "max" or "min", e.g. "max".
    constraints: list[Any] = Field(default_factory=list)  # Raw constraint definitions for the step, e.g. ["min_growth", ["ATPM", 2.0, "min_atpm", "geq"]].


class CharacterizationConfig(BaseModel):
    type: Literal["Characterization"] = "Characterization"                  # Experiment type discriminator, e.g. "Characterization".
    analysis_component: Optional[str] = None                               # Analysis component name injected by config loading, e.g. "advanced_wildtype_characterization".
    analysis_params: AnalysisParamsConfig = Field(default_factory=AnalysisParamsConfig)  # Nested shared analysis parameters, e.g. AnalysisParamsConfig(minimize_swaps={"reaction_groups": ["HDCR"]}).
    optimization_steps: Optional[list[OptimizationStep]] = None            # Ordered characterization workflow steps, e.g. [OptimizationStep(name="optATPM", objective_var_ids=["ATPM"])].

class VariableScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["VariableScan"] = "VariableScan"                         # Experiment type discriminator, e.g. "VariableScan".
    analysis_component: Optional[str] = None                              # Analysis component name injected by config loading, e.g. "test_variablescan_ecoli".
    analysis_params: AnalysisParamsConfig = Field(default_factory=AnalysisParamsConfig)  # Nested shared analysis parameters, e.g. AnalysisParamsConfig(min_growth=0.0, min_mdf=0.1, fva=FVAConfig(reactions_to_check=["ATPM"])).

    scan_axis: "TwoVariableAxisConfig"                                     # Single typed scan axis; replaces old top-level scan_variable/scan_values fields, e.g. {"type": "flux_variable", "target": "ATPM", "scan_values": [1.0, 2.0]}.
    objective_var_ids: list[str] = Field(default_factory=lambda: ["var_B"])  # Objective variables optimized at each scan point, e.g. ["var_B"] or ["BIOMASS_Ecoli_core_w_GAM"].
    objective_multipliers: Optional[list[float]] = None                   # Objective coefficients aligned with objective_var_ids, e.g. [1.0] or [1.0, 0.1].
    objective_direction: Literal["max", "min"] = "max"                    # Optimization direction at each scan point, e.g. "max".

    base_constraints: list[Any] = Field(default_factory=list)             # Constraint block used for the benchmark solve and carried into every scan point, e.g. ["min_growth"].
    point_constraints: list[Any] = Field(default_factory=list)            # Additional constraint block applied only during the point-by-point scan solves, e.g. ["min_mdf", "min_product"].
    execution_mode: Literal["standard", "parallel_scan_points", "parallel_scan_chunks"] = "standard"  # Local full scan or SLURM-expanded point/chunk execution.
    chunk_size: int = Field(default=10, ge=1)                             # Number of adjacent 1D points assigned to one chunk worker.

    num_priming_attempts: int = Field(default=10, ge=1)                   # Multistart attempts for the first scan point, e.g. 10.
    num_improvement_attempts: int = Field(default=7, ge=1)                # Retry attempts when a point looks unexpectedly worse, e.g. 7.
    scan_zero_epsilon: float = Field(default=1e-4, gt=0.0)                # Small replacement value when zero should be nudged positive, e.g. 1e-4.


class TwoVariableAxisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    type: Literal["flux_variable", "metabolite_concentration", "dG0", "constraint"] = "flux_variable"  # Axis kind: solver variable, concentration context edit, dG0 context edit, or named solver constraint.
    target: str = Field(validation_alias=AliasChoices("target", "name"))           # Concrete axis target; symbolic "product"/"biomass" is only resolved for flux axes, e.g. "ATPM", "nadh_c", "PFOR_FWD", or a constraint name.
    scan_values: Optional[list[float]] = None                                       # Explicit axis values used directly as constraints, concentration bounds, or dG0 values/changes, e.g. [0.0, 1.0, 2.0].
    scan_range: Optional[Any] = Field(default=None, validation_alias=AliasChoices("scan_range", "range"))  # Axis range spec converted into scan_values, e.g. {"start": 0.0, "stop": 2.0, "num_points": 5} or [-8, -3, 11, "log"].
    scan_fractions: Optional[list[float]] = None                                    # VariableScan-only flux-axis fractions relative to the benchmark maximum, e.g. [0.25, 0.5, 1.0].
    scan_step_size: Optional[float] = Field(default=None, gt=0.0)                   # VariableScan-only flux-axis step from zero to benchmark maximum, e.g. 0.2.
    constraint_sign: Literal["geq", "leq", "eq"] = "eq"                            # Constraint sense used only for flux-variable axes, e.g. "eq" for fixed uptake points.
    bound_type: Literal["fixed", "max_only", "min_only"] = "fixed"                 # Concentration-axis application mode: fixed sets min=max, max_only changes max, min_only changes min.
    mode: Literal["add", "set", "scale"] = "set"                                   # dG0-axis application mode: "add" perturbs loaded dG0, "set" uses absolute values, "scale" multiplies.
    pair_policy: Literal["none", "auto_reverse"] = "auto_reverse"                  # dG0-axis split-pair handling; auto_reverse updates the opposite split reaction if present.
    constraint: Optional[list[Any]] = None                                         # Constraint-axis template; when omitted, the template is taken from a base/point constraint whose name matches target.
    constraint_value_mode: Literal["set", "range_lower", "range_upper", "range_symmetric"] = "set"  # Constraint-axis value mapping: replace the value, replace one range side, or set [-abs(value), abs(value)].
    label: Optional[str] = None                                                     # Optional display label for reports and plots, e.g. "NADH" or "PFOR dG0".


class FluxValidationConfig(BaseModel):
    type: Literal["FluxValidation"] = "FluxValidation"                    # Experiment type discriminator, e.g. "FluxValidation".
    analysis_component: Optional[str] = None                              # Analysis component name injected by config loading, e.g. "test_flux_validation".
    analysis_params: AnalysisParamsConfig = Field(default_factory=AnalysisParamsConfig)  # Nested shared analysis parameters, e.g. AnalysisParamsConfig(run_fva=FVAConfig(reactions_to_check=["ATPM"])).

    optimization_step: Optional[OptimizationStep] = None                  # Validation optimization step definition, e.g. OptimizationStep(name="opt_atpm", objective_var_ids=["ATPM"]).

    default_measurement_variance: Optional[float] = Field(default=None, ge=0.0)  # Optional fallback variance; without one, CNApy uses its standard unweighted calculation.
    default_measurement_fit_bounds: tuple[float, float] = (-1000.0, 1000.0)  # Default fit bounds for measured fluxes, e.g. (-1000.0, 1000.0).
    generate_corrected_scenario_file: bool = False                         # Export a corrected scenario artifact, e.g. False.
    generate_fitted_environment_yaml: bool = False                         # Export a reusable fitted environment YAML, e.g. True.


# =====================================================================
# 3. The Master Union
# =====================================================================
AnyAnalysisConfig = Annotated[
    Union[
        CharacterizationConfig,
        VariableScanConfig,
        FluxValidationConfig,
    ],
    Field(discriminator="type"),
]

class ExperimentDefinition(BaseModel):
    """Central executable unit of the framework.
    An `ExperimentDefinition` binds:
    - one experiment definition,
    - one scenario context,
    - one execution context,
    - one selected solver.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    experiment_name: str              # e.g. "MDF_Scan_ATPM_withGrowth"
    experiment_type: str              # e.g. "MDFScan", "Characterization", "MCS"
    solver_config: SolverConfig        # Declarative solver recipe used to instantiate the live solver, e.g. ThermoSolverCPLEXConfig(to_solve="thermo").
    biological_scenarios: list[BiologicalScenario]      # Resolved biological scenarios for the experiment batch, e.g. [BiologicalScenario(scenario_id="e_coli_core__e_coli_env__glucose_anaerobic__acetate", ...)].
    analysis_config: AnyAnalysisConfig                  # Typed analysis configuration selected for this experiment, e.g. VariableScanConfig(scan_axis=TwoVariableAxisConfig(target="ATPM", ...)).


class CampaignDefinition(BaseModel):
    campaign_name: str                                           # Campaign identifier, e.g. "Test_Characterization_Campaign".
    output_dir: str = "./Results"                                # Root output directory for campaign artifacts, e.g. "./Results".
    experiments: list[ExperimentDefinition] = Field(default_factory=list)  # Flattened experiment list to execute, e.g. [ExperimentDefinition(experiment_name="Run_1", ...)].
