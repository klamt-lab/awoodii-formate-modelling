import copy
import logging
import math
import time
from typing import Any

import cobra

import mymodelingpackage.structures as structs
from mymodelingpackage.experiments.base import BaseExperiment
from mymodelingpackage.experiments.utils import (
    is_optimal,
    resolve_constraint_block,
    resolve_symbolic_vars,
    sanitize_token,
    solve_step_robustly,
)
from mymodelingpackage.solvers.interface import BiologicalSolver


logger = logging.getLogger(__name__)
Scenario: Any = None
make_scenario_feasible: Any = None


def fit_flux_measurements(
    model: cobra.Model,
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit measured fluxes to the active model with CNApy feasibility repair.

    Args:
        model: Active COBRA model containing the measured reactions.
        measurements: Measured flux definitions, fitting bounds, and optional
            variances.

    Returns:
        Fitted fluxes and absolute and relative adjustment diagnostics.

    Raises:
        ValueError: If no measurements are supplied.
        RuntimeError: If CNApy does not return an optimal, finite fitted flux
            for every measured reaction.
    """
    # CNApy starts JPype while importing its feasibility helpers. Keep that
    # optional JVM dependency isolated to FluxValidation runs while retaining
    # module-level injection points used by tests and downstream callers.
    global Scenario, make_scenario_feasible
    if Scenario is None:
        from cnapy.appdata import Scenario as CnapyScenario

        Scenario = CnapyScenario
    if make_scenario_feasible is None:
        from cnapy.core import make_scenario_feasible as cnapy_make_scenario_feasible

        make_scenario_feasible = cnapy_make_scenario_feasible

    if not measurements:
        raise ValueError("Flux validation requires at least one measurement.")

    # CNApy's feasibility repair expects its Scenario container, not a plain dict,
    # because it attaches helper methods such as add_scenario_reactions_to_model().
    # Example `measurements`:
    # [
    #   {"reaction_id": "EX_h2_e", "measured_value": -120.0, "variance": 0.0001, "bounds": (-1000.0, 1000.0)},
    #   {"reaction_id": "EX_ac_e", "measured_value": 30.0, "variance": 0.0001, "bounds": (-1000.0, 1000.0)},
    #   {"reaction_id": "biomass", "measured_value": 0.08, "variance": 0.002, "bounds": (0.08, 0.08)},
    # ]
    # Example `scenario_values` after conversion to the CNApy Scenario object:
    # {"EX_h2_e": (-120.0, -120.0), "EX_ac_e": (30.0, 30.0), "biomass": (0.08, 0.08)}
    scenario_values = Scenario()
    measurements_without_variance = [
        str(measurement["reaction_id"])
        for measurement in measurements
        if measurement.get("variance") is None
    ]
    weights_key = None if measurements_without_variance else "variance"
    if weights_key is not None:
        logger.info("Configured CNApy flux reconciliation with weights_key='variance' for %d measurement(s).", len(measurements))
    else:
        logger.info("Configured CNApy flux reconciliation with weights_key=None (standard unweighted calculation); measurements without variance: %s.", ", ".join(measurements_without_variance))

    for measurement in measurements:
        reaction_id = str(measurement["reaction_id"])
        measured_value = float(measurement["measured_value"])
        variance = measurement.get("variance")
        bounds = tuple(measurement["bounds"])
        reaction = model.reactions.get_by_id(reaction_id)
        if variance is not None:
            reaction.annotation["variance"] = float(variance)
        else:
            reaction.annotation.pop("variance", None)
        reaction.bounds = bounds
        scenario_values[reaction_id] = (measured_value, measured_value)
        logger.info("Prepared flux measurement '%s': measured_value=%s, bounds=%s, variance=%s.", reaction_id, measured_value, bounds, variance)

    # CNApy's default calculation is unweighted. Variance weighting is valid
    # only when the scenario provides a variance for every measurement.
    if weights_key is not None:
        solution, reactions_in_objective, _, _, _ = make_scenario_feasible(model, scenario_values, weights_key=weights_key, use_QP=False, abs_flux_weights=True)
    else:
        solution, reactions_in_objective, _, _, _ = make_scenario_feasible(model, scenario_values, use_QP=False, abs_flux_weights=True)
    solution_status = str(getattr(solution, "status", "unknown") or "unknown")
    logger.info("CNApy flux reconciliation finished with status=%s, weights_key=%s, and %d reaction(s) in the objective.", solution_status, repr(weights_key), len(reactions_in_objective or []))
    if not is_optimal(solution_status):
        logger.error("CNApy flux reconciliation failed with status=%s; no corrected measurements will be used.", solution_status)
        raise RuntimeError(f"CNApy flux reconciliation failed with status '{solution_status}'.")

    corrected_fluxes: dict[str, float] = {}
    adjustments: list[dict[str, float | str | None]] = []
    abs_adjustments: list[float] = []
    absolute_relative_adjustments_percent: list[float] = []

    for measurement in measurements:
        reaction_id = str(measurement["reaction_id"])
        measured_value = float(measurement["measured_value"])
        variance = measurement.get("variance")
        try:
            corrected_value = float(solution[reaction_id])
        except Exception as exc:
            logger.error("CNApy returned an optimal solution without a usable flux for measured reaction '%s'.", reaction_id)
            raise RuntimeError(f"CNApy returned no usable corrected flux for measured reaction '{reaction_id}'.") from exc
        if not math.isfinite(corrected_value):
            logger.error("CNApy returned a non-finite corrected flux for measured reaction '%s': %s.", reaction_id, corrected_value)
            raise RuntimeError(f"CNApy returned a non-finite corrected flux for measured reaction '{reaction_id}'.")
        corrected_fluxes[reaction_id] = corrected_value

        delta = corrected_value - measured_value; abs_delta = abs(delta); abs_adjustments.append(abs_delta)
        absolute_relative_adjustment_percent = abs_delta / abs(measured_value) * 100.0 if measured_value != 0.0 else None
        if absolute_relative_adjustment_percent is not None:
            absolute_relative_adjustments_percent.append(absolute_relative_adjustment_percent)

        if abs_delta > 1e-9:
            adjustments.append(
                {
                    "reaction_id": reaction_id,
                    "measured_value": measured_value,
                    "corrected_value": corrected_value,
                    "delta": delta,
                    "absolute_delta": abs_delta,
                    "absolute_relative_adjustment_percent": absolute_relative_adjustment_percent,
                    "variance": float(variance) if variance is not None else None,
                }
            )

    weighted_absolute_adjustment_sum = sum(
        abs(corrected_fluxes[str(measurement["reaction_id"])] - float(measurement["measured_value"]))
        * (float(measurement["variance"]) if weights_key is not None else 1.0)
        for measurement in measurements
    )

    return {
        "status": solution_status,
        "corrected_fluxes": corrected_fluxes,
        "adjustments": adjustments,
        "adjusted_reaction_ids": [item["reaction_id"] for item in adjustments],
        "reactions_in_objective": list(reactions_in_objective or []),
        "measurement_count": len(measurements),
        "adjusted_count": len(adjustments),
        "mean_absolute_adjustment": sum(abs_adjustments) / len(abs_adjustments) if abs_adjustments else 0.0,
        "max_absolute_adjustment": max(abs_adjustments) if abs_adjustments else 0.0,
        "mean_absolute_relative_adjustment_percent": sum(absolute_relative_adjustments_percent) / len(absolute_relative_adjustments_percent) if absolute_relative_adjustments_percent else None,
        "max_absolute_relative_adjustment_percent": max(absolute_relative_adjustments_percent) if absolute_relative_adjustments_percent else None,
        "weighted_absolute_adjustment_sum": weighted_absolute_adjustment_sum,
        "weights_key": weights_key,
    }


class FluxValidationExperiment(BaseExperiment):
    """Fit experimental fluxes, then validate the resulting phenotype stoichiometrically.

    This experiment combines two layers of logic:
    - a stoichiometric reconciliation phase that adjusts measured fluxes with
      ``make_scenario_feasible`` under configured variances
    - a stoichiometric validation phase that locks the corrected phenotype on
      the model and optimizes a user-defined target on that fitted state
    """

    def __init__(
        self,
        experiment_def: structs.ExperimentDefinition,
        solver: BiologicalSolver,
        output_dir: str = "./Results",
    ) -> None:
        """Initialize a scenario-wise flux validation workflow.

        Args:
            experiment_def: Parsed experiment definition including the
                validation config and biological scenarios.
            solver: Solver implementation used for the stoichiometric
                validation step.
            output_dir: Directory where CSV, JSON, and optional artifacts are
                written.
        """
        super().__init__(experiment_def=experiment_def, output_dir=output_dir)
        self.solver = solver
        self.aggregate_results: list[dict[str, Any]] = []

        cfg = self.experiment_def.analysis_config
        self.analysis_params = cfg.analysis_params

        # Experiment specific
        self.default_measurement_variance = cfg.default_measurement_variance
        self.default_measurement_fit_bounds = tuple(cfg.default_measurement_fit_bounds)
        self.generate_corrected_scenario_file = cfg.generate_corrected_scenario_file
        self.generate_fitted_environment_yaml = cfg.generate_fitted_environment_yaml

        self.optimization_step = cfg.optimization_step

        self.min_growth = self.analysis_params.min_growth
        self.min_mdf = self.analysis_params.min_mdf
        self.fraction_prod = self.analysis_params.fraction_prod
        self.num_numerical_retries = self.analysis_params.num_numerical_retries

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #
    # MAIN RUN
    #
    # Shared experiment lifecycle used across the refactored experiment classes:
    #   PHASE 1: Scenario preparation.
    #   PHASE 2: Experiment core logic.
    #   PHASE 3: Artifact generation.
    #   PHASE 4: Failure handling.
    #   PHASE 5: Scenario finalization.
    #   PHASE 6: Experiment finalization.
    #
    # FluxValidation follows the same high-level skeleton as Characterization
    # and VariableScan, but it intentionally stops after the validation solve
    # and optional export generation. Any broader phenotype characterization
    # should continue from the exported fitted environment in a dedicated
    # Characterization run.
    #
    # Its experiment-specific workflow units are:
    #   PHASE 2.1: Execute the fit workflow.
    #   PHASE 2.2: Prepare validation inputs.
    #   PHASE 2.3: Execute the validation objective.
    #   PHASE 2.4: Build the validation payload.
    #   PHASE 2.5: Update successful optimization state.
    #
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self) -> list[dict[str, Any]]:
        """Execute flux validation across all configured biological scenarios.

        Returns:
            list[dict[str, Any]]: One standardized scenario payload per
            validation run, including step-level results and summary metrics.
        """
        start_time = time.time()
        logger.debug("================= STARTING RUN() ====================")
        logger.info(f"Starting FluxValidationExperiment: {self.experiment_def.experiment_name}")

        scenarios = self.experiment_def.biological_scenarios
        logger.info(f"Processing {len(scenarios)} scenarios...")
        cfg = self.experiment_def.analysis_config

        for scenario_idx, scenario in enumerate(scenarios):
            # -------------------------------------------------------------
            # PHASE 1: SCENARIO PREPARATION
            # Start timing, emit the scenario banner, and create the fresh
            # payload container that will collect this validation run.
            # -------------------------------------------------------------
            scenario_start_time = time.time()
            scenario_id = scenario.scenario_id
            logger.info(f" [scenario {scenario_idx+1}]     --- Processing Scenario: {scenario.scenario_id} ---")
            run_payload = self.build_scenario_payload(scenario)

            try:
                # -------------------------------------------------------------
                # PHASE 1.1: LOAD SOLVER CONTEXT
                # Load the biological context into the active validation solver before any
                # phenotype-specific bounds are applied to the active model.
                # -------------------------------------------------------------
                logger.debug(f" [scenario {scenario_idx+1}] Loading solver context...")
                self.solver.load_context(biological_scenario=scenario, analysis_config=cfg)

                # -------------------------------------------------------------
                # PHASE 1.2: PREPARE OPTIONAL EXPERIMENT FEATURES
                # Initialize optional scenario-local features such as swap
                # minimization helpers without mixing them into the core scan
                # logic below.
                # -------------------------------------------------------------
                # Not implemented 

                # -------------------------------------------------------------
                # PHASE 1.3: RESOLVE SCENARIO METADATA
                # Resolve scenario-dependent symbolic ids once so the fit and
                # validation phases operate on the same reaction identifiers.
                # -------------------------------------------------------------
                scenario_metadata = self.resolve_scenario_metadata(scenario)
                biomass_id = scenario_metadata["biomass_id"]
                product_name = scenario_metadata["product_name"]
                product_id = scenario_metadata["product_id"]
                if product_id: logger.info(f" [scenario {scenario_idx+1}] Resolved Product ID: '{product_id}' (from name '{product_name}')")
                else:          logger.warning(f"Could not resolve Product ID for '{product_name}'. Steps relying on 'product' may fail.")

                max_prod = 0.0 # Initialize safely to 0.0

                # -------------------------------------------------------------
                # PHASE 1.4: RESOLVE VALIDATION INPUTS
                # Collect the active substrate, product, and adjustment blocks
                # for this scenario so the validation fit sees only the
                # currently selected phenotype inputs.
                # -------------------------------------------------------------
                
                substrate_entries = copy.deepcopy(scenario.environment_conf.substrates.get(scenario.substrate_name, []))
                product_entry = copy.deepcopy(scenario.environment_conf.products.get(scenario.product_name, {}))
                adjustment_entries = copy.deepcopy(scenario.environment_conf.substrate_product_adjustments.get(scenario.substrate_name, {}).get(scenario.product_name, []))

                # Convert the active environment entries into validation measurements and 
                # strict hard bounds after resolving symbolic ids such as `product` and `biomass`.
                # Example `substrate_entries`:
                # [{"id": "EX_h2_e", "lb": -1000.0, "ub": 1000.0, "measured_value": -120.0, "variance": 0.0001}]
                # Example `product_entry`:
                # {"id": "EX_ac_e", "lb": -1000.0, "ub": 1000.0, "measured_value": 30.0, "variance": 0.0001}
                # Example `adjustment_entries`:
                # [{"id": "biomass", "lb": 0.08, "ub": 0.08, "measured_value": 0.08, "variance": 0.002}]
                resolved_measurements: list[dict[str, Any]] = []
                resolved_hard_bounds: list[dict[str, Any]] = []
                seen_measurements: set[str] = set()
                seen_hard_bounds: set[str] = set()

                # Example `active_environment_entries`:
                # [
                #   {"id": "EX_h2_e", "lb": -1000.0, "ub": 1000.0, "measured_value": -120.0, "variance": 0.0001},
                #   {"id": "EX_ac_e", "lb": -1000.0, "ub": 1000.0, "measured_value": 30.0, "variance": 0.0001},
                #   {"id": "biomass", "lb": 0.08, "ub": 0.08, "measured_value": 0.08, "variance": 0.002},
                # ]
                active_environment_entries = substrate_entries + ([product_entry] if product_entry else []) + adjustment_entries
                logger.debug(f" [scenario {scenario_idx+1}] active_environment_entries: {active_environment_entries}")

                logger.info(" [scenario %d] Resolving %d active environment entrie(s) into measurements and hard bounds...", scenario_idx + 1, len(active_environment_entries))

                for entry in active_environment_entries:
                    # Resolve the symbolic entry id to the concrete reaction id
                    # that is present on the active scenario model.
                    reaction_token = entry.get("id")
                    reaction_id = resolve_symbolic_vars(reaction_token, product_id=product_id, biomass_id=biomass_id)
                    if not reaction_id:
                        raise ValueError(f"Could not resolve environment reaction '{reaction_token}'.")

                    # Decide whether the entry should be interpreted as a fitted measurement or as a strict hard bound.
                    # FluxValidation now keeps this contract minimal: an entry becomes a measurement when it carries
                    # either a measured point estimate or a variance-weighted reconciliation target.
                    is_measurement_entry = any(key in entry for key in ("variance", "measured_value"))

                    if is_measurement_entry:
                        # Resolve the target measurement value. When no explicit `measured_value` is given, identical `lb`
                        # and `ub` act as the measured point estimate.
                        measured_value = float(entry["measured_value"]) if "measured_value" in entry else None
                        if measured_value is None:
                            if "lb" not in entry or "ub" not in entry or float(entry["lb"]) != float(entry["ub"]):
                                raise ValueError(f"Measured environment entry '{reaction_id}' requires either `measured_value` or identical `lb`/`ub`.")
                            measured_value = float(entry["lb"])

                        # Resolve the allowed fitting bounds directly from the normal environment bounds. This keeps the
                        # validation config aligned with the standard scenario builder contract: `lb`/`ub` already define
                        # the allowable range, so no extra `fit_bounds` field is needed.
                        fit_bounds = (
                            float(entry.get("lb", self.default_measurement_fit_bounds[0])),
                            float(entry.get("ub", self.default_measurement_fit_bounds[1])),
                        )

                        if reaction_id in seen_measurements:
                            raise ValueError(f"Duplicate measurement reaction '{reaction_id}' after scenario resolution.")
                        seen_measurements.add(reaction_id)

                        # Example `measurement_payload` with optional variance:
                        # {"reaction_id": "EX_h2_e", "measured_value": -120.0, "variance": 0.0001, "bounds": (-1000.0, 1000.0)}
                        measurement_payload = {
                            "reaction_id": reaction_id,
                            "measured_value": measured_value,
                            "bounds": fit_bounds,
                        }
                        variance = entry.get("variance", self.default_measurement_variance)
                        if variance is not None:
                            measurement_payload["variance"] = float(variance)
                        resolved_measurements.append(measurement_payload)
                        logger.info(" [scenario %d] Measurement '%s' resolved with target=%s, bounds=%s, variance=%s.", scenario_idx + 1, reaction_id, measured_value, fit_bounds, measurement_payload.get("variance"))
                        continue

                    # Non-measurement entries become strict hard bounds. Their currently active model bounds are preserved as the
                    # constraint state that should already hold before fitting.
                    if reaction_id in seen_hard_bounds:
                        raise ValueError(f"Duplicate hard-bound reaction '{reaction_id}' after scenario resolution.")
                    seen_hard_bounds.add(reaction_id)

                    # Example `hard_bound_payload`:
                    # {"reaction_id": "EX_o2_e", "bounds": (0.0, 0.0)}
                    hard_bound_payload = {
                        "reaction_id": reaction_id,
                        "bounds": tuple(scenario.model.reactions.get_by_id(reaction_id).bounds),
                    }
                    resolved_hard_bounds.append(hard_bound_payload)
                    logger.info(" [scenario %d] Hard bound '%s' resolved with active bounds=%s.", scenario_idx + 1, reaction_id, hard_bound_payload["bounds"])

                # Final validation of the resolved inputs: one reaction may not simultaneously be a fitted measurement and a hard bound, and
                # every validation scenario must contribute at least one measurable flux to reconcile.
                overlap = {measurement["reaction_id"] for measurement in resolved_measurements} & {bound["reaction_id"] for bound in resolved_hard_bounds}
                if overlap:
                    raise ValueError(f"Validation reactions cannot appear in both `measurements` and `hard_bounds`: {sorted(overlap)}")
                if not resolved_measurements:
                    raise ValueError(f"FluxValidation scenario '{scenario_id}' did not resolve any measured environment entries.")
                logger.info(" [scenario %d] Resolved %d measurement(s) and %d hard bound(s) for validation.", scenario_idx + 1, len(resolved_measurements), len(resolved_hard_bounds))

                run_payload["metrics"]["measurement_count"] = len(resolved_measurements)
                run_payload["metrics"]["hard_bound_count"] = len(resolved_hard_bounds)

                with scenario.model:
                    # ---------------------------------------------------------
                    # PHASE 1.5: PRECOMPUTATIONS
                    # ---------------------------------------------------------
                    # not implemented

                    # ---------------------------------------------------------
                    # PHASE 2: EXPERIMENT CORE LOGIC
                    # FluxValidation follows the same general rhythm as the
                    # other experiment classes:
                    #   prepare inputs -> solve -> build payload -> update state
                    #
                    # It applies that pattern twice:
                    #   1. once for the fit/reconciliation solve
                    #   2. once for the validation optimization solve
                    # ---------------------------------------------------------

                    # ---------------------------------------------------------
                    # PHASE 2.1: EXECUTE FIT WORKFLOW
                    #
                    #   PHASE 2.1.1: Prepare fit inputs.
                    #   PHASE 2.1.2: Solve the fit problem.
                    #   PHASE 2.1.3: Build the fit payload.
                    #   PHASE 2.1.4: Update fit state and lock the phenotype.
                    # ---------------------------------------------------------

                    # ---------------------------------------------------------
                    # PHASE 2.1.1: PREPARE FIT INPUTS
                    # Apply strict hard intervention bounds before the
                    # feasibility-repair solve starts. This mirrors the
                    # "prepare inputs" subphase used by the other workflows.
                    # ---------------------------------------------------------
                    logger.debug(f" [scenario {scenario_idx+1}] Applying {len(resolved_hard_bounds)} hard bound(s) before fit...")
                    for bound in resolved_hard_bounds: scenario.model.reactions.get_by_id(bound["reaction_id"]).bounds = tuple(bound["bounds"])

                    # ---------------------------------------------------------
                    # PHASE 2.1.2: SOLVE FIT PROBLEM
                    # CNApy adjusts the measured fluxes only as much as needed
                    # within the configured fitting bounds and variances.
                    # The resulting corrected flux set is the phenotype that
                    # the downstream stoichiometric validation solve will
                    # validate.
                    # Example `fit_result`:
                    # {
                    #   "status": "Optimal",
                    #   "corrected_fluxes": {"EX_h2_e": -118.7, "EX_ac_e": 29.6, "biomass": 0.08},
                    #   "adjustments": [{"reaction_id": "EX_h2_e", "measured_value": -120.0, "corrected_value": -118.7, "delta": 1.3, "absolute_delta": 1.3, "variance": 0.0001}],
                    #   "adjusted_reaction_ids": ["EX_h2_e"],
                    #   "measurement_count": 3,
                    #   "adjusted_count": 1,
                    # }
                    # ---------------------------------------------------------
                    logger.info(f" [scenario {scenario_idx+1}] Running measurement reconciliation for {len(resolved_measurements)} flux(es)...")
                    fit_result = fit_flux_measurements(scenario.model, resolved_measurements)

                    
                    # ---------------------------------------------------------
                    # PHASE 2.1.3: BUILD FIT PAYLOAD
                    # Convert the feasibility-repair result into the same
                    # step-payload structure used by the other experiment
                    # classes so the fit solve is reported in a comparable way.
                    # ---------------------------------------------------------
                    fit_payload = self.build_standard_step_payload(
                        step_name="fit_measurements",
                        objective_var_ids=["make_scenario_feasible"],
                        objective_direction="min",
                        constraints=[],
                        solution=structs.SolverResult(
                            status=fit_result["status"],
                            objective_value=float(fit_result["weighted_absolute_adjustment_sum"]),
                            values={},
                        ),
                        biomass_id=biomass_id,
                        product_id=product_id,
                        substrate_ids=scenario_metadata["substrate_ids"],
                    )
                    adjusted_reaction_descriptions: list[str] = []
                    relative_adjustment_descriptions: list[str] = []
                    for adjustment in fit_result["adjustments"]:
                        reaction_id = str(adjustment.get("reaction_id", "unknown"))
                        measured_value = adjustment.get("measured_value")
                        corrected_value = adjustment.get("corrected_value")
                        if measured_value is None or corrected_value is None:
                            adjusted_reaction_descriptions.append(reaction_id)
                            continue
                        adjusted_reaction_descriptions.append(
                            f"{reaction_id}: {float(measured_value):.8g} -> {float(corrected_value):.8g}"
                        )
                        relative_adjustment_percent = adjustment.get("absolute_relative_adjustment_percent")
                        if relative_adjustment_percent is None and float(measured_value) != 0.0:
                            relative_adjustment_percent = abs(float(corrected_value) - float(measured_value)) / abs(float(measured_value)) * 100.0
                        relative_adjustment_descriptions.append(
                            f"{reaction_id}: {float(relative_adjustment_percent):.8g}%" if relative_adjustment_percent is not None else f"{reaction_id}: undefined (zero baseline)"
                        )
                    if not adjusted_reaction_descriptions and fit_result["adjusted_reaction_ids"]:
                        adjusted_reaction_descriptions = [str(reaction_id) for reaction_id in fit_result["adjusted_reaction_ids"]]
                    fit_payload.update(
                        {
                            "Constraints": f"{fit_result['measurement_count']} measurements; {len(resolved_hard_bounds)} hard bounds",
                            "Measurement_Count": fit_result["measurement_count"],
                            "Adjusted_Reactions_Count": fit_result["adjusted_count"],
                            "Adjusted_Reactions": "; ".join(adjusted_reaction_descriptions),
                            "Mean_Absolute_Adjustment": round(float(fit_result["mean_absolute_adjustment"]), 8),
                            "Max_Absolute_Adjustment": round(float(fit_result["max_absolute_adjustment"]), 8),
                            "Relative_Adjustments_Percent": "; ".join(relative_adjustment_descriptions),
                            "Mean_Absolute_Relative_Adjustment_Percent": round(float(fit_result["mean_absolute_relative_adjustment_percent"]), 8) if fit_result.get("mean_absolute_relative_adjustment_percent") is not None else None,
                            "Max_Absolute_Relative_Adjustment_Percent": round(float(fit_result["max_absolute_relative_adjustment_percent"]), 8) if fit_result.get("max_absolute_relative_adjustment_percent") is not None else None,
                        }
                    )
                    run_payload["steps"].append(fit_payload)

                    # Store high-level fit diagnostics on the scenario payload so they remain easy to access without 
                    # drilling into the individual fit step record.
                    run_payload["metrics"]["adjusted_measurement_count"] = (fit_result["adjusted_count"])
                    run_payload["metrics"]["max_absolute_adjustment"] = round(float(fit_result["max_absolute_adjustment"]), 8)

                    # ---------------------------------------------------------
                    # PHASE 2.1.4: UPDATE FIT STATE AND LOCK PHENOTYPE
                    # Persist the fit outcome as the active scenario state
                    # before the validation optimization starts. This includes
                    # locking corrected fluxes on the model. The corresponding
                    # fitted-phenotype files are written later in the shared
                    # artifact-generation phase.
                    # ---------------------------------------------------------
                    logger.debug(f" [scenario {scenario_idx+1}] Fixing corrected phenotype fluxes on the model...")
                    for reaction_id, value in fit_result["corrected_fluxes"].items(): scenario.model.reactions.get_by_id(reaction_id).bounds = (value, value)

                    # ---------------------------------------------------------
                    # PHASE 2.2: PREPARE VALIDATION INPUTS
                    # Resolve symbolic objective ids and shorthand constraints
                    # against the current scenario so the solver sees only
                    # concrete reaction ids and numeric thresholds.
                    # ---------------------------------------------------------
                    step_name = self.optimization_step.name
                    resolved_objective_var_ids = resolve_symbolic_vars(
                        list(self.optimization_step.objective_var_ids),
                        product_id=product_id,
                        biomass_id=biomass_id,
                    )
                    if not resolved_objective_var_ids:
                        raise ValueError(f"Validation step '{step_name}' has no resolvable objective ids.")

                    objective_multipliers = (
                        list(self.optimization_step.objective_multipliers)
                        if self.optimization_step.objective_multipliers
                        else [1.0] * len(resolved_objective_var_ids)
                    )
                    objective_direction = self.optimization_step.objective_direction
                    resolved_constraints = resolve_constraint_block(
                        list(self.optimization_step.constraints or []),
                        product_id=product_id,
                        biomass_id=biomass_id,
                        max_prod=max_prod,
                        min_growth=self.min_growth,
                        min_mdf=self.min_mdf,
                        fraction_prod=self.fraction_prod,
                        objective_var_ids=resolved_objective_var_ids,
                    )

                    # ---------------------------------------------------------
                    # PHASE 2.3: EXECUTE VALIDATION OBJECTIVE
                    # Run the validation optimization on the already
                    # locked corrected phenotype using the same robust solve
                    # helper that drives the other experiment classes.
                    # ---------------------------------------------------------
                    logger.info(f" [scenario {scenario_idx+1}] Solving validation target '{step_name}' with objective {resolved_objective_var_ids} ({objective_direction})...")
                    solution = solve_step_robustly(
                        solver=self.solver,
                        objective_var_ids=resolved_objective_var_ids,
                        objective_multipliers=objective_multipliers,
                        objective_direction=objective_direction,
                        constraints=resolved_constraints,
                        num_retries=self.num_numerical_retries,
                        logger_=logger,
                    )

                    # ---------------------------------------------------------
                    # PHASE 2.4: BUILD VALIDATION PAYLOAD
                    # Convert the validation solve into the shared step-level
                    # payload schema used by the other experiments. This keeps
                    # the "solve -> build payload" pattern aligned with
                    # VariableScan and Characterization.
                    # ---------------------------------------------------------
                    step_payload = self.build_standard_step_payload(
                        step_name=step_name,
                        objective_var_ids=resolved_objective_var_ids,
                        objective_direction=objective_direction,
                        constraints=resolved_constraints,
                        solution=solution,
                        biomass_id=biomass_id,
                        product_id=product_id,
                        substrate_ids=scenario_metadata["substrate_ids"],
                    )

                    if is_optimal(solution.status):
                         # -----------------------------------------------------
                        # PHASE 2.5: UPDATE SUCCESSFUL OPTIMIZATION STATE
                        #
                        #   PHASE 2.5.1: Persist optimization state and metrics.
                        #   PHASE 2.5.2: Enrich the payload annotations.
                        # -----------------------------------------------------
                        run_payload["status"] = "Success"

                        # -----------------------------------------------------
                        # PHASE 2.5.1: PERSIST OPTIMIZATION STATE AND METRICS
                        # Record realized objective values on the scenario
                        # payload before optional exports and diagnostics.
                        # -----------------------------------------------------
                        for objective_var_id in resolved_objective_var_ids:
                            objective_value = solution.values.get(objective_var_id)
                            if objective_value is not None:
                                run_payload["metrics"][f"{step_name}_{objective_var_id}"] = objective_value

                        # -----------------------------------------------------
                        # PHASE 2.5.2: ENRICH PAYLOAD ANNOTATIONS
                        # Keep the primary validation step payload aligned with
                        # the standard experiment output contract.
                        # -----------------------------------------------------
                        run_payload["steps"].append(step_payload)

                        # -----------------------------------------------------
                        # PHASE 3: ARTIFACT GENERATION
                        #
                        # PHASE 3.1: WRITE OPTIONAL ARTIFACTS
                        # Persist the fitted-phenotype exports and the same
                        # validation-solve artifacts used by the other
                        # experiments for downstream inspection.
                        # -----------------------------------------------------
                        artifact_id = sanitize_token(f"{scenario.scenario_id}_{step_name}")

                        if self.analysis_params.excel_output:
                            self.save_to_excel(
                                model=scenario.model,
                                raw_solution=solution,
                                scenario_id=artifact_id,
                            )

                        if self.generate_corrected_scenario_file:
                            # Export the locked corrected phenotype for manual inspection in CNApy. Hard bounds are merged in so the
                            # scenario file reflects the full realized validation state, not just the corrected measurements.
                            logger.info(f" [scenario {scenario_idx+1}] Saving corrected validation scenario file...")

                            # Example `corrected_flux_bounds`:
                            # {"EX_h2_e": (-118.7, -118.7), "EX_ac_e": (29.6, 29.6), "biomass": (0.08, 0.08), "EX_o2_e": (0.0, 0.0)}
                            corrected_flux_bounds = {reaction_id: (value, value) for reaction_id, value in fit_result["corrected_fluxes"].items()}
                            corrected_flux_bounds.update(
                                {
                                    bound["reaction_id"]: tuple(bound["bounds"])
                                    for bound in resolved_hard_bounds
                                    if bound["reaction_id"] not in corrected_flux_bounds
                                }
                            )
                            self.generate_cnapy_files(
                                model=scenario.model,
                                raw_solution=None,
                                scenario_id=scenario_id,
                                filename_suffix="corrected_measurements",
                                flux_bounds=corrected_flux_bounds,
                            )

                        if self.generate_fitted_environment_yaml:
                            logger.info(f" [scenario {scenario_idx+1}] Saving fitted environment YAML...")

                            # Build the fitted environment component directly from the active scenario entries so 
                            # the exported YAML can be reused as a standalone environment definition.
                            # The source environment component already identifies
                            # the biological condition. Keeping only a Fitted_
                            # prefix avoids redundant substrate/product suffixes
                            # and allows scenario files to merge directly.
                            environment_name = f"Fitted_{sanitize_token(scenario.environment_component)}"

                            # Transform the active substrate entries into fitted export form. Measured entries are replaced by their
                            # corrected flux values, and validation-only metadata is removed from the reusable YAML.
                            # Example `fitted_substrates`:
                            # [{"id": "EX_h2_e", "lb": -118.7, "ub": -118.7}]
                            fitted_substrates: list[dict[str, Any]] = []
                            for item in substrate_entries:
                                transformed = copy.deepcopy(item)
                                reaction_id = resolve_symbolic_vars(transformed.get("id"), product_id=product_id, biomass_id=biomass_id)
                                is_measurement_entry = any(key in transformed for key in ("variance", "measured_value"))
                                if reaction_id and is_measurement_entry and reaction_id in fit_result["corrected_fluxes"]:
                                    fitted_value = float(fit_result["corrected_fluxes"][reaction_id]); transformed["lb"] = fitted_value; transformed["ub"] = fitted_value
                                for key in ("measured_value", "variance"):
                                    transformed.pop(key, None)
                                fitted_substrates.append(transformed)

                            # Transform the active product entry with the same fitted-value replacement and metadata cleanup.
                            # Example `fitted_product`:
                            # {"id": "EX_ac_e", "lb": 29.6, "ub": 29.6}
                            fitted_product = copy.deepcopy(product_entry)
                            reaction_id = resolve_symbolic_vars(fitted_product.get("id"), product_id=product_id, biomass_id=biomass_id)
                            is_measurement_entry = any(key in fitted_product for key in ("variance", "measured_value"))
                            if reaction_id and is_measurement_entry and reaction_id in fit_result["corrected_fluxes"]:
                                fitted_value = float(fit_result["corrected_fluxes"][reaction_id]); fitted_product["lb"] = fitted_value; fitted_product["ub"] = fitted_value
                            for key in ("measured_value", "variance"):
                                fitted_product.pop(key, None)

                            # Transform any active substrate-product adjustments so the exported environment matches 
                            # the realized fitted phenotype everywhere validation touched it.
                            # Example `fitted_adjustments`:
                            # [{"id": "biomass", "lb": 0.08, "ub": 0.08}]
                            fitted_adjustments: list[dict[str, Any]] = []
                            for item in adjustment_entries:
                                transformed = copy.deepcopy(item)
                                reaction_id = resolve_symbolic_vars(transformed.get("id"), product_id=product_id, biomass_id=biomass_id)
                                is_measurement_entry = any(key in transformed for key in ("variance", "measured_value"))
                                if reaction_id and is_measurement_entry and reaction_id in fit_result["corrected_fluxes"]:
                                    fitted_value = float(fit_result["corrected_fluxes"][reaction_id]); transformed["lb"] = fitted_value; transformed["ub"] = fitted_value
                                for key in ("measured_value", "variance"):
                                    transformed.pop(key, None)
                                fitted_adjustments.append(transformed)

                            # Preserve the complete validated operating point.
                            # Corrected measurements alone do not constrain the
                            # optimized validation target, such as ATPM, in a
                            # later FVA run.
                            fitted_reaction_ids = {
                                str(item.get("id"))
                                for item in [*fitted_substrates, fitted_product, *fitted_adjustments]
                                if isinstance(item, dict) and item.get("id")
                            }
                            for objective_var_id in resolved_objective_var_ids:
                                objective_value = solution.values.get(objective_var_id)
                                if (
                                    objective_value is None
                                    or objective_var_id in fitted_reaction_ids
                                    or not scenario.model.reactions.has_id(objective_var_id)
                                ):
                                    continue
                                fitted_value = float(objective_value)
                                fitted_adjustments.append(
                                    {"id": objective_var_id, "lb": fitted_value, "ub": fitted_value}
                                )
                                fitted_reaction_ids.add(objective_var_id)

                            active_concentration_adjustments = copy.deepcopy(scenario.environment_conf.concentration_adjustments.get(scenario.substrate_name, {}).get(scenario.product_name, {}))

                            # Example `environment_payload`:
                            # {
                            #   "concentrations_file": "concentrations_awoodii_h2_co2_medium.json",
                            #   "substrates": {"co2_h2": [{"id": "EX_h2_e", "lb": -118.7, "ub": -118.7}]},
                            #   "products": {"acetate": {"id": "EX_ac_e", "lb": 29.6, "ub": 29.6}},
                            #   "substrate_product_adjustments": {"co2_h2": {"acetate": [{"id": "biomass", "lb": 0.08, "ub": 0.08}]}},
                            #   "concentration_adjustments": {},
                            # }
                            environment_payload: dict[str, Any] = {
                                "concentrations_file": scenario.environment_conf.concentrations_file,
                                "substrates": {scenario.substrate_name: fitted_substrates},
                                "products": {scenario.product_name: fitted_product},
                                "substrate_product_adjustments": (
                                    {scenario.substrate_name: {scenario.product_name: fitted_adjustments}}
                                    if fitted_adjustments
                                    else {}
                                ),
                                "concentration_adjustments": (
                                    {scenario.substrate_name: {scenario.product_name: active_concentration_adjustments}}
                                    if active_concentration_adjustments
                                    else {}
                                ),
                            }
                            self.save_environment_component_yaml(
                                environment_name=environment_name,
                                environment_payload=environment_payload,
                                scenario_id=scenario_id,
                                filename_suffix="fitted_environment",
                            )

                        if self.analysis_params.save_raw_solutions:
                            self.save_raw_solutions(solution.values, artifact_id)

                        if self.analysis_params.save_realized_adjustments:
                            self.save_realized_pathway_adjustments(
                                solution.values,
                                artifact_id,
                                forward_reaction_suffix=scenario.model_conf.forward_reaction_suffix,
                                reverse_reaction_suffix=scenario.model_conf.reverse_reaction_suffix,
                            )

                        if self.analysis_params.generate_cnapy_files:
                            self.generate_cnapy_files(
                                model=scenario.model,
                                raw_solution=solution,
                                scenario_id=artifact_id,
                                objective_direction=objective_direction,
                            )
                            
                        # -------------------------------------------------
                        # PHASE 4: POST-ANALYSES
                        # -------------------------------------------------
                        # not implemented

                    else:
                        # -----------------------------------------------------
                        # PHASE 5.1: HANDLE FAILURE AND STOP EARLY
                        # Keep the failed optimization payload and stop this
                        # validation scenario after the primary solve.
                        # -----------------------------------------------------
                        logger.warning(f" [scenario {scenario_idx+1}] Validation step '{step_name}' failed with status={solution.status}.")
                        run_payload["status"] = f"Failed at {step_name}: {solution.status}"
                        run_payload["steps"].append(step_payload)
                    
                       

            except Exception as exc:
                # -------------------------------------------------------------
                # PHASE 5.2: HANDLE UNEXPECTED SCENARIO EXCEPTION
                # Convert unexpected scenario-level errors into the standard
                # scenario failure payload without dropping final timing data.
                # -------------------------------------------------------------
                logger.error(f" [scenario {scenario_idx+1}] Unhandled exception during validation scenario '{scenario_id}': {exc}", exc_info=True)
                run_payload["status"] = f"Failed: {exc}"

            finally:
                # -------------------------------------------------------------
                # PHASE 6: SCENARIO FINALIZATION
                # Persist timing metadata and keep the aggregate results aligned
                # with the standard experiment output contract.
                # -------------------------------------------------------------
                elapsed = time.time() - scenario_start_time
                run_payload["metrics"]["execution_time_seconds"] = round(elapsed, 2)
                logger.info(f" [scenario {scenario_idx+1}]         Finished scenario {scenario_id} in {elapsed:.2f} seconds.")
                self.aggregate_results.append(run_payload)

        # -------------------------------------------------------------
        # PHASE 7: EXPERIMENT FINALIZATION
        # Emit the experiment-level completion log and return the full
        # aggregated payload for JSON/CSV export.
        # -------------------------------------------------------------
        total_elapsed = time.time() - start_time
        logger.info(f"FluxValidationExperiment '{self.experiment_def.experiment_name}' finished {len(scenarios)} scenario(s) in {total_elapsed:.2f}s.")
        return self.aggregate_results
