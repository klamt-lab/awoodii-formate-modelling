import copy
import logging
from typing import Any, Literal, Optional


from mymodelingpackage.docplex_core_functions import docplex_optmdf
from mymodelingpackage.structures import(   
                                            BiologicalScenario, 
                                            ThermoSolverCPLEXConfig, 
                                            AnalysisParamsConfig,
                                            SolverResult
)


logger = logging.getLogger(__name__)


class ThermoSolverCPLEX:
    """Thermo-specific solve solver wrapper.

    Attributes:
        solver_config:
            Static thermo configuration (constants and CPLEX parameter overrides).
        biological_scenario:
            The currently loaded biological scenario (model, concentrations, dG0).
    """

    def __init__(self, solver_config: ThermoSolverCPLEXConfig) -> None:
        self.solver_config = solver_config
        self.config = solver_config
        self.biological_scenario: BiologicalScenario | None = None
        self.analysis_config: AnalysisParamsConfig | None = None
        logger.debug("Initialized ThermoSolverCPLEX with fixed docplex/CPLEX backend.")

    def load_context(self, biological_scenario: BiologicalScenario, analysis_config: AnalysisParamsConfig) -> None:
        """Load a biological scenario into the solver.

        This must be called before solve(). It resets any previous scenario state.

        Args:
            biological_scenario: The resolved BiologicalScenario from ScenariosBuilder.
        """
        self.biological_scenario = biological_scenario
        self.analysis_config = analysis_config
        logger.debug("ThermoSolverCPLEX loaded scenario='%s', model='%s', analysis_type='%s'.", biological_scenario.scenario_id, biological_scenario.model.id, getattr(analysis_config, "type", type(analysis_config).__name__))

    def supports(self, capability: str) -> bool:
        """Report optional solver capabilities used by dispatchers and workflows."""
        supported = capability == "thermo_driving_force"
        logger.debug("ThermoSolverCPLEX capability query: capability='%s' -> %s", capability, supported)
        return supported

    def solve(
        self,
        objective_var_ids: list[str],
        objective_multipliers: list[float] | None = None,
        objective_direction: Literal["max", "min"] = "max",
        constraints: list[list[Any]] | None = None,
        bottleneck_analysis: bool = False,
        warm_start_solution: Optional[dict] = None,
        subnetwork_ids: list[str] | None = None,
        thermo_unconstrained_reactions: list[str] | None = None,
    ) -> SolverResult:
        """Execute a thermodynamic optimization on the loaded scenario.

        Args:
            objective_var_ids:
                List of variable IDs to optimize (e.g., ["var_B"], ["ATPM"]).
            objective_multipliers:
                Coefficients for the objective variables. Defaults to 1.0.
            objective_direction:
                Optimization direction ("max" or "min").
            constraints:
                List of custom constraints [var_id, value, name, type].
            warm_start_solution:
                Optional dumped incumbent solution used as a solver warm start.
            subnetwork_ids:
                Optional per-call subnetwork override. If omitted, the solver
                falls back to ``analysis_config.analysis_params.subnetwork``.
            thermo_unconstrained_reactions:
                Optional per-call override for reactions that should be treated
                as thermodynamically unconstrained. If omitted, the solver falls
                back to ``analysis_config.analysis_params.thermo_unconstrained_reactions``.

        Returns:
            Dictionary containing 'status', 'values', and 'objective_value'.

        Raises:
            RuntimeError: If no scenario is loaded.
        """
        # 1. Validate Context and Inputs
        if self.biological_scenario is None:
            raise RuntimeError("ThermoSolverCPLEX has no loaded scenario. Call load_context() first.")
        if self.analysis_config is None:
            raise RuntimeError("ThermoSolverCPLEX has no loaded analysis config. Call load_context() first.")
        if not objective_var_ids:
            raise ValueError("ThermoSolverCPLEX.solve requires at least one objective variable id.")
        if objective_direction not in {"max", "min"}:
            raise ValueError(f"Unsupported objective_direction '{objective_direction}'. Expected 'max' or 'min'.")
        if objective_multipliers is not None and len(objective_multipliers) != len(objective_var_ids):
            raise ValueError("Length mismatch between objective_var_ids and objective_multipliers.")

        # 2. Prepare Parameters
        if objective_multipliers is None:
            objective_multipliers = [1] * len(objective_var_ids)

        analysis_params = self.analysis_config.analysis_params
        active_subnetwork_ids = (
            subnetwork_ids
            if subnetwork_ids is not None
            else analysis_params.subnetwork
        )
        active_thermo_unconstrained_reactions = (
            thermo_unconstrained_reactions
            if thermo_unconstrained_reactions is not None
            else analysis_params.thermo_unconstrained_reactions
        )

        # 3. Prepare Solver Data
        # Work on deep copies so global analysis-level overrides cannot leak
        # back into the reusable BiologicalScenario or later scan points.
        active_concentrations = copy.deepcopy(self.biological_scenario.concentrations or {})
        if "DEFAULT" not in active_concentrations:
            # Fallback values: 1 µM to 20 mM
            active_concentrations["DEFAULT"] = {"min": 1e-6, "max": 1e-2}
            logger.warning("Scenario '%s' has no DEFAULT concentration. Using fallback: %s", self.biological_scenario.scenario_id, active_concentrations["DEFAULT"])
        active_dG0 = copy.deepcopy(self.biological_scenario.dG0 or {})

        thermo_adjustments = analysis_params.thermo_adjustments
        if thermo_adjustments:
            logger.debug("Applying thermo adjustments: concentrations=%d, dG0=%d.", len(thermo_adjustments.concentrations), len(thermo_adjustments.dG0))
            for concentration_adjustment in thermo_adjustments.concentrations:
                # A concentration adjustment may introduce a metabolite that
                # only used DEFAULT bounds before. In that case we seed the
                # metabolite-specific entry from DEFAULT and then overwrite the
                # requested side(s) of the range.
                default_bounds = active_concentrations.get(concentration_adjustment.target) or active_concentrations.get("DEFAULT") or {"min": 1e-6, "max": 1e-2}
                active_concentrations.setdefault(concentration_adjustment.target, copy.deepcopy(default_bounds))
                if concentration_adjustment.min is not None:
                    active_concentrations[concentration_adjustment.target]["min"] = float(concentration_adjustment.min)
                if concentration_adjustment.max is not None:
                    active_concentrations[concentration_adjustment.target]["max"] = float(concentration_adjustment.max)
                if active_concentrations[concentration_adjustment.target]["min"] > active_concentrations[concentration_adjustment.target]["max"]:
                    raise ValueError(f"Invalid concentration adjustment for '{concentration_adjustment.target}': min exceeds max.")
                logger.debug("Applied concentration adjustment: target='%s', min=%s, max=%s.", concentration_adjustment.target, active_concentrations[concentration_adjustment.target]["min"], active_concentrations[concentration_adjustment.target]["max"])

            for dG0_adjustment in thermo_adjustments.dG0:
                # dG0 targets must already exist. Silently creating new dG0
                # entries would hide reaction-id typos and make results harder
                # to interpret.
                if dG0_adjustment.target not in active_dG0:
                    raise ValueError(f"dG0 adjustment target '{dG0_adjustment.target}' is not present in the loaded dG0 dataset.")
                if "dG0" not in active_dG0[dG0_adjustment.target]:
                    raise ValueError(f"dG0 adjustment target '{dG0_adjustment.target}' has no 'dG0' value.")

                # Convert the configured mode into the actual dG0 value that
                # the MILP receives for this solve.
                old_dG0 = float(active_dG0[dG0_adjustment.target]["dG0"])
                if dG0_adjustment.mode == "add":
                    new_dG0 = old_dG0 + float(dG0_adjustment.value)
                elif dG0_adjustment.mode == "scale":
                    new_dG0 = old_dG0 * float(dG0_adjustment.value)
                else:
                    new_dG0 = float(dG0_adjustment.value)
                active_dG0[dG0_adjustment.target]["dG0"] = new_dG0
                logger.debug("Applied dG0 adjustment: target='%s', mode='%s', value=%s, dG0=%s -> %s.", dG0_adjustment.target, dG0_adjustment.mode, dG0_adjustment.value, old_dG0, new_dG0)

                # Split forward/reverse reactions should remain thermodynamic
                # opposites when the paired entry exists in the dG0 data.
                if dG0_adjustment.pair_policy == "auto_reverse":
                    forward_suffix = self.biological_scenario.model_conf.forward_reaction_suffix
                    reverse_suffix = self.biological_scenario.model_conf.reverse_reaction_suffix
                    paired_reaction_id = None
                    if forward_suffix and dG0_adjustment.target.endswith(forward_suffix):
                        paired_reaction_id = dG0_adjustment.target[: -len(forward_suffix)] + reverse_suffix
                    elif reverse_suffix and dG0_adjustment.target.endswith(reverse_suffix):
                        paired_reaction_id = dG0_adjustment.target[: -len(reverse_suffix)] + forward_suffix
                    if paired_reaction_id and paired_reaction_id in active_dG0:
                        active_dG0[paired_reaction_id]["dG0"] = -new_dG0
                        logger.debug("Applied paired dG0 adjustment: target='%s', dG0=%s.", paired_reaction_id, active_dG0[paired_reaction_id]["dG0"])

        # 4. Build Solver-Specific Problem
        # docplex_optmdf owns construction of the thermodynamic MILP from the
        # active data, objective request, constraints, and solver configuration.

        # 5. Execute Core Solver
        try:
            solution = docplex_optmdf(
                # Passed inputs
                objective_var_ids=objective_var_ids,
                objective_direction=objective_direction,
                objective_multipliers=objective_multipliers,
                constraints=constraints or [],
                bottleneck_analysis=bottleneck_analysis,
                warm_start_solution=warm_start_solution,

                # From biological_scenario combining information from ModelConfig and EnvironmentConfig
                model=self.biological_scenario.model,
                used_concentration_values=active_concentrations,
                dG0_values=active_dG0,
                co_factor_alternatives=self.biological_scenario.model_conf.exclusive_reaction_groups,

                # From analysis_config
                subnetwork_ids=active_subnetwork_ids,
                thermo_unconstrained_reactions=active_thermo_unconstrained_reactions,

                # Execution solver_config
                solver_config=self.solver_config
            )

            # 6. Validate and Package Result
            return solution
        except Exception as e:
            logger.error("Error during thermo solve for scenario='%s', objective=%s: %s", getattr(self.biological_scenario, "scenario_id", "unknown"), objective_var_ids, str(e), exc_info=True)
            return SolverResult(status="Error", error_message=str(e))


