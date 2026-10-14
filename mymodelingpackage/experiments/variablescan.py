import copy
import math
import logging
import os
import random
import time
from typing import Any


import mymodelingpackage.structures as structs
from mymodelingpackage.experiments.base import BaseExperiment
from mymodelingpackage.post_analyses.dispatcher import PostAnalysisDispatcher
from mymodelingpackage.solvers.interface import BiologicalSolver
from mymodelingpackage.experiments.utils import build_constraint_axis_point, build_quality_lock_constraints, apply_concentration_axis_point, apply_dg0_axis_point, collect_swap_ids, dedupe_constraints, is_optimal, objective_score, resolve_constraint_block, resolve_constraint_axis_template, resolve_symbolic_vars, sanitize_token, set_solver_random_seed, solve_step_robustly, solve_with_swap_minimization

logger = logging.getLogger(__name__)


class VariableScanExperiment(BaseExperiment):
    """
    Scan one variable across a 1D grid and optimize a configurable objective at each point.

    This generalizes the old MDFScan behavior, allowing systematic variation of a single 
    model variable (e.g., ATPM demand, biomass, or a specific flux) across a specified range. 
    At each point, it optimizes for a target objective (e.g., product yield, MDF). 
    The result is a trade-off curve essential for understanding performance limits.
    """
    def __init__(
        self,
        experiment_def: structs.ExperimentDefinition,
        solver: BiologicalSolver,
        output_dir: str = "./Results",
    ):
        """Initialize a one-dimensional scan experiment.

        Args:
            experiment_def: Parsed experiment definition including scan config
                and biological scenarios.
            solver: Solver implementation used for benchmarking and point
                solves.
            output_dir: Directory where CSV, JSON, and optional artifacts are
                written.
        """
        super().__init__(experiment_def=experiment_def, output_dir=output_dir)
        self.solver = solver
        self.dispatcher = PostAnalysisDispatcher()
        self.aggregate_results: list[dict[str, Any]] = []

        # Experiment specific
        cfg = self.experiment_def.analysis_config
        self.analysis_params = cfg.analysis_params

        # `scan_axis` is the single source of truth for what changes across
        # scan points. Flux and constraint axes become constraints;
        # concentration and dG0 axes become temporary thermo-context edits.
        self.scan_axis = cfg.scan_axis
        if not self.scan_axis.target:
            raise ValueError("VariableScanExperiment requires `scan_axis.target`.")
        if self.scan_axis.type in {"metabolite_concentration", "dG0"} and getattr(self.experiment_def.solver_config, "to_solve", None) not in {"thermo", "cobrak"}:
            raise ValueError("VariableScan concentration and dG0 axes require the thermo solver or COBRA-k.")

        self.objective_var_ids = list(cfg.objective_var_ids or ["var_B"])
        self.objective_direction = cfg.objective_direction
        if self.objective_direction not in {"max", "min"}:
            raise ValueError(f"Invalid `objective_direction`: {self.objective_direction}. Allowed values are 'max' and 'min'.")

        self.objective_multipliers = (list(cfg.objective_multipliers) if cfg.objective_multipliers else [1.0] * len(self.objective_var_ids))
        if len(self.objective_multipliers) != len(self.objective_var_ids):
            raise ValueError("Length mismatch between `objective_var_ids` and `objective_multipliers`.")

        # The constraint sign only affects flux axes. It is still read from
        # scan_axis for all axis types so the config shape stays uniform.
        self.scan_axis_constraint_sign = self.scan_axis.constraint_sign
        if self.scan_axis_constraint_sign not in {"geq", "leq", "eq"}:
            raise ValueError(f"Invalid `scan_axis.constraint_sign`: {self.scan_axis_constraint_sign}. Allowed values are 'geq', 'leq', and 'eq'.")

        # Pull the point-generation settings out of the axis object once so the
        # rest of the scan logic can operate on one normalized internal form.
        self.scan_fractions = list(self.scan_axis.scan_fractions) if self.scan_axis.scan_fractions else None
        self.scan_values = list(self.scan_axis.scan_values) if self.scan_axis.scan_values else None
        self.scan_range = self.scan_axis.scan_range
        self.scan_step_size = self.scan_axis.scan_step_size
        if self.scan_step_size is not None and self.scan_step_size <= 0:
            raise ValueError("`scan_axis.scan_step_size` must be > 0.")

        configured_scan_modes = [
            field_name
            for field_name, value in (
                ("scan_values", self.scan_values),
                ("scan_range", self.scan_range),
                ("scan_step_size", self.scan_step_size),
                ("scan_fractions", self.scan_fractions),
            )
            if value not in (None, [])
        ]
        if len(configured_scan_modes) > 1:
            raise ValueError(f"VariableScan accepts at most one explicit scan-point definition inside `scan_axis`. Received: {configured_scan_modes}")
        if self.scan_axis.type != "flux_variable" and any(mode in configured_scan_modes for mode in ("scan_step_size", "scan_fractions")):
            raise ValueError("VariableScan dG0, concentration, and constraint axes require explicit `scan_values` or `scan_range`; benchmark-relative scans only apply to flux-variable axes.")

        # Only flux axes have a natural benchmark maximum, so only flux axes can
        # default to benchmark-relative fractions when the user omits points.
        if (
            self.scan_fractions is None
            and self.scan_values is None
            and self.scan_range is None
            and self.scan_step_size is None
        ):
            if self.scan_axis.type != "flux_variable":
                raise ValueError("VariableScan dG0, concentration, and constraint axes require explicit `scan_values` or `scan_range`.")
            self.scan_fractions = [round(i / 100, 2) for i in range(1, 100)]
            logger.info(f"No explicit scan points provided. Defaulting to fractions: {self.scan_fractions}")

        self.base_constraints = list(cfg.base_constraints or [])
        self.point_constraints = list(cfg.point_constraints or [])

        # Base biological thresholds
        self.min_growth = self.analysis_params.min_growth
        self.min_mdf = self.analysis_params.min_mdf
        self.fraction_prod = self.analysis_params.fraction_prod
        self.key_reactions = self.analysis_params.key_reactions
        self.key_metabolites = self.analysis_params.key_metabolites

        # Robust + swap controls
        self.num_numerical_retries = self.analysis_params.num_numerical_retries
        self.num_swap_attempts = self.analysis_params.num_swap_attempts
        self.num_reoptimization_attempts_sandwich = self.analysis_params.num_reoptimization_attempts_sandwich
        self.minimize_swaps_config = self.analysis_params.minimize_swaps

        self.num_priming_attempts = cfg.num_priming_attempts
        self.num_improvement_attempts = cfg.num_improvement_attempts
        self.scan_zero_epsilon = cfg.scan_zero_epsilon
        if self.num_priming_attempts < 1:
            raise ValueError("`num_priming_attempts` must be >= 1.")
        if self.num_improvement_attempts < 1:
            raise ValueError("`num_improvement_attempts` must be >= 1.")

        # --- Runtime Scratch State ---
        # These variables hold data that needs to persist across sequential steps *within* a single scenario.
        self.swap_ids: list[str] = []

    def _parallel_fragment_path(self, scenario: structs.BiologicalScenario, point_index: int) -> str:
        """Build the deterministic CSV path for one parallel scan point.

        Args:
            scenario: Biological scenario owning the point result.
            point_index: Zero-based index in the resolved one-dimensional scan
                grid.

        Returns:
            Absolute or output-directory-relative fragment CSV path. Repeated
            calls with the same scenario and point index return the same path.
        """
        # STEP 1: Ensure every worker writes into the experiment-local fragment
        # directory instead of the final aggregated CSV namespace.
        fragment_dir = os.path.join(self.output_dir, "fragments")
        os.makedirs(fragment_dir, exist_ok=True)

        # STEP 2: Encode scenario and grid coordinate into a stable filename so
        # restart checks and independent SLURM workers address the same artifact.
        scenario_label = self._scenario_filename_label(scenario.scenario_id)
        fragment_path = os.path.join(fragment_dir, f"fragment__{scenario_label}__p{point_index}.csv")
        logger.debug("Resolved VariableScan fragment path: scenario='%s', point_index=%d, path='%s'.", scenario.scenario_id, point_index, fragment_path)
        return fragment_path

    def _save_parallel_fragment(self, scenario: structs.BiologicalScenario, step_payload: dict[str, Any], point_index: int) -> str:
        """Write one parallel VariableScan point atomically.

        Args:
            scenario: Biological scenario associated with the point.
            step_payload: Standard experiment step row produced by the point
                solve and optional post-analyses.
            point_index: Zero-based coordinate in the resolved scan grid.

        Returns:
            Path to the completed fragment CSV.

        Raises:
            OSError: If temporary-file creation or atomic replacement fails.
        """
        logger.debug("Preparing VariableScan fragment: scenario='%s', point_index=%d, status='%s'.", scenario.scenario_id, point_index, step_payload.get("Status", "Unknown"))

        # STEP 1: Wrap the point row in the same scenario payload shape used by
        # normal CSV generation so serial and parallel column schemas match.
        fragment_payload = {
            "experiment_name": self.experiment_def.experiment_name,
            "scenario_id": scenario.scenario_id,
            "substrate": scenario.substrate_name,
            "product": scenario.product_name,
            "status": step_payload.get("Status", "Unknown"),
            "steps": [step_payload],
        }

        # STEP 2: Convert through BaseExperiment's canonical CSV pipeline and
        # add helper columns used only during deterministic aggregation.
        rows = self._build_csv_rows(fragment_payload)
        for row in rows:
            row["scenario_id"] = scenario.scenario_id
            row["fragment_point_index"] = point_index
        dataframe = self._build_csv_dataframe(rows)

        # STEP 3: Write to a temporary sibling and atomically replace the final
        # path. A worker interruption therefore cannot leave a partial CSV that
        # a restarted job would mistake for a completed point.
        fragment_path = self._parallel_fragment_path(scenario, point_index)
        temporary_path = f"{fragment_path}.tmp"
        logger.debug("Writing VariableScan temporary fragment '%s' with %d row(s) and %d column(s).", temporary_path, len(dataframe), len(dataframe.columns))
        dataframe.to_csv(temporary_path, sep=";", index=False)
        os.replace(temporary_path, fragment_path)
        logger.info("Saved atomic VariableScan fragment: scenario='%s', point_index=%d, rows=%d, path='%s'.", scenario.scenario_id, point_index, len(dataframe), fragment_path)
        return fragment_path


    # ------------------------------------------------------------------
    # scan setup
    # ------------------------------------------------------------------

    def _normalize_scan_target(self, target: float) -> float:
        """Apply the zero-target policy consistently across scan modes."""
        if self.scan_axis.type == "flux_variable" and abs(target) < 1e-9 and self.scan_axis_constraint_sign in {"geq", "eq"}:
            return self.scan_zero_epsilon
        return float(target)

    def _format_scan_number(self, value: float) -> str:
        """Create readable, stable numeric tokens for scan labels."""
        formatted = format(float(value), ".12g")
        if formatted == "-0":
            return "0"
        return formatted

    def _resolve_scan_points(self, max_theoretical: float | None = None) -> list[dict[str, float | str]]:
        """Translate the scan config into concrete point targets.

        Flux-variable scans may use benchmark-relative point definitions
        (`scan_step_size` or `scan_fractions`). Concentration and dG0 scans
        use explicit values/ranges because there is no solver benchmark that
        defines a meaningful upper endpoint for those axes.

        Args:
            max_theoretical: Optional benchmark value of a flux-variable axis.
                It acts as the upper limit for fraction/step scans and as a cap
                for explicit flux ranges.

        Returns:
            list[dict[str, float | str]]: Ordered scan points in the form
            ``{"label": "...", "target": 1.23}``.

        Example:
            ``scan_values=[0.1, 0.2]`` becomes
            ``[{"label": "value_0.1", "target": 0.1}, ...]``.

            ``scan_range={"start": 0.0, "stop": 0.2, "step": 0.1}`` becomes
            three points from ``scan_zero_epsilon`` up to ``0.2``.
        """
        if max_theoretical is not None:
            logger.info("    Step 2: Resolving scan points from benchmark value %.6f...", max_theoretical)
        else:
            logger.info("    Step 2: Resolving scan points from explicit %s axis configuration...", self.scan_axis.type)
        points: list[dict[str, float | str]] = []
        
        # 1. Explicit Values
        if self.scan_values:
            logger.info("    -> Using explicit scan values: %s", self.scan_values)
            for value in self.scan_values:
                if max_theoretical is None or value <= max_theoretical + 1e-9:
                    target = self._normalize_scan_target(float(value))
                    points.append({"label": f"value_{self._format_scan_number(target)}", "target": target})
                else:
                    logger.warning("Skipping explicit scan value %s because it exceeds benchmark %.6f", value, max_theoretical)
        
        # 2. Explicit Range {start, stop, step|num_points, scale}
        elif self.scan_range:
            range_spec = self.scan_range
            scale = "linear"
            values: list[float]
            if isinstance(range_spec, dict):
                start = float(range_spec.get("start", 0.0))
                stop = float(range_spec.get("stop", max_theoretical if max_theoretical is not None else start))
                scale = str(range_spec.get("scale", "linear")).lower()
                if "step" in range_spec:
                    step = float(range_spec["step"])
                    logger.info("    -> Using explicit range start=%.6f stop=%.6f step=%.6f", start, stop, step)
                    if step <= 0:
                        raise ValueError("`scan_axis.scan_range.step` must be > 0.")
                    values = []
                    current = start
                    while current <= stop + 1e-12:
                        values.append(float(current))
                        current += step
                    if not values or abs(values[-1] - stop) > 1e-12:
                        values.append(float(stop))
                else:
                    num_points = int(range_spec.get("num_points", 0))
                    if num_points < 1:
                        raise ValueError("`scan_axis.scan_range` requires `num_points` >= 1 when `step` is omitted.")
                    if num_points == 1:
                        values = [float(start)]
                    else:
                        increment = (stop - start) / float(num_points - 1)
                        values = [start + increment * index for index in range(num_points)]
                    if scale == "log":
                        values = [math.pow(10.0, exponent) for exponent in values]
            elif isinstance(range_spec, (list, tuple)) and len(range_spec) == 4:
                start = float(range_spec[0])
                stop = float(range_spec[1])
                num_points = int(range_spec[2])
                scale = str(range_spec[3]).lower()
                if num_points < 1:
                    raise ValueError("`scan_axis.scan_range` list form requires `num_points` >= 1.")
                if num_points == 1:
                    values = [float(start)]
                else:
                    increment = (stop - start) / float(num_points - 1)
                    values = [start + increment * index for index in range(num_points)]
                if scale == "log":
                    values = [math.pow(10.0, exponent) for exponent in values]
            else:
                raise ValueError(f"Unsupported `scan_axis.scan_range` format: {range_spec!r}")

            if max_theoretical is not None and values and max(values) > max_theoretical:
                logger.warning("Requested scan stop %.6f exceeds benchmark %.6f. Capping to benchmark.", stop, max_theoretical)
                values = [value for value in values if value <= max_theoretical + 1e-9]
                if not values or abs(values[-1] - max_theoretical) > 1e-9:
                    values.append(float(max_theoretical))
            for value in values:
                target = self._normalize_scan_target(value)
                points.append({"label": f"value_{self._format_scan_number(target)}", "target": float(target)})
            logger.info("    -> Generated %d points from range config.", len(points))

        # 3. Dynamic Step Size (0 to Max with fixed step)
        elif self.scan_step_size:
            if max_theoretical is None:
                raise ValueError("`scan_axis.scan_step_size` requires a flux-variable benchmark value.")
            logger.info("    -> Using step size %s from %s to max.", self.scan_step_size, self.scan_zero_epsilon)
            current = 0.0
            while current <= max_theoretical + 1e-9:
                target = self._normalize_scan_target(current)
                points.append({"label": f"value_{self._format_scan_number(target)}", "target": float(target)})
                current += self.scan_step_size
            final_target = self._normalize_scan_target(max_theoretical)
            if not points or abs(float(points[-1]["target"]) - final_target) > 1e-9:
                points.append({"label": f"value_{self._format_scan_number(final_target)}", "target": float(final_target)})

        # 4. Fractions (Default behavior)
        else:
            if max_theoretical is None:
                raise ValueError("`scan_axis.scan_fractions` requires a flux-variable benchmark value.")
            logger.info("    -> Using fractional scan (default).")
            fractions = self.scan_fractions or [round(i / 100, 2) for i in range(1, 100)]
            for frac in fractions:
                target = self._normalize_scan_target(frac * max_theoretical)
                points.append({"label": f"fraction_{self._format_scan_number(frac)}", "target": target})

        # Flux scans are ordered by constraint sign. Direct thermo axes keep
        # the configured order because "harder" is not universally defined.
        if self.scan_axis.type == "flux_variable" and self.scan_axis_constraint_sign == "geq":
            points.sort(key=lambda item: float(item["target"]), reverse=True)
        elif self.scan_axis.type == "flux_variable" and self.scan_axis_constraint_sign == "leq":
            points.sort(key=lambda item: float(item["target"]))

        # Deduplicate targets after zero-normalization and endpoint handling so
        # equivalent numerical points do not get solved twice.
        deduped_points: list[dict[str, float | str]] = []
        for point in points:
            target = float(point["target"])
            if any(abs(target - float(existing["target"])) <= 1e-9 for existing in deduped_points):
                logger.warning("Skipping duplicate scan target %.12g after normalization.", target)
                continue
            deduped_points.append({"label": str(point["label"]), "target": target})

        # Make labels unique after deduplication so step names, metric keys,
        # and artifact ids cannot collide even if different inputs format to
        # the same human-readable token.
        finalized_points: list[dict[str, float | str]] = []
        seen_labels: dict[str, int] = {}
        for point in deduped_points:
            label = str(point["label"])
            seen_labels[label] = seen_labels.get(label, 0) + 1
            if seen_labels[label] > 1:
                unique_label = f"{label}__{seen_labels[label]}"
                logger.warning("Adjusted colliding scan label '%s' to '%s'.", label, unique_label)
            else:
                unique_label = label
            finalized_points.append({"label": unique_label, "target": float(point["target"])})

        return finalized_points
    # ------------------------------------------------------------------
    # solver helpers
    # ------------------------------------------------------------------
   

    def _solve_first_scan_point(
        self,
        objective_var_ids: list[str],
        objective_multipliers: list[float],
        objective_direction: str,
        constraints: list[list[Any]],
        logger_: logging.Logger = logger,
    ) -> structs.SolverResult:
        """Solve the first scan point using multi-start priming.

        The first point is usually the hardest one numerically because it lies
        closest to the benchmark extreme. This helper samples several seeds and
        keeps the best solution before falling back to the generic robust solve.

        Args:
            objective_var_ids: Objective variable ids optimized at the point.
            objective_multipliers: Objective coefficients aligned with
                ``objective_var_ids``.
            objective_direction: ``"max"`` or ``"min"``.
            constraints: Solver-ready point constraints.
            logger_: Logger used for progress and debug output.

        Returns:
            structs.SolverResult: Best point solution found during priming or
            the robust fallback solve.
        """
        logger_.info("    -> This is the first and hardest fraction. Performing priming multistart to find a robust starting point.")
        logger_.debug("      -> Priming constraints: %s", constraints)
        best_solution = None
        best_score = None

        for attempt in range(self.num_priming_attempts):
            logger_.debug(f"      -> Priming attempt {attempt+1}/{self.num_priming_attempts}...")

            seed = random.randint(10000, 99999)
            set_solver_random_seed(self.solver, seed)

            priming_sol = self.solver.solve(
                objective_var_ids=objective_var_ids,
                objective_multipliers=objective_multipliers,
                objective_direction=objective_direction,
                constraints=constraints,
            )
            priming_score = objective_score(priming_sol)

            if priming_score is None:
                continue

            is_better = ( best_score is None or ( priming_score > best_score
                                                    if objective_direction == "max"
                                                    else priming_score < best_score
                                                )
            )
            
            if is_better:
                logger_.info(f"      -> Found new best starting solution with score: {priming_score:.6f}")
                best_solution = priming_sol
                best_score = priming_score

        if best_solution is not None and is_optimal(best_solution.status):
            logger_.info("    -> Priming finished successfully. Best initial objective=%s", best_score)
            return best_solution

        logger_.warning("    -> Priming did not produce an optimal result. Falling back to robust retry solve.")
        return solve_step_robustly(
            solver=self.solver,
            objective_var_ids=objective_var_ids,
            objective_multipliers=objective_multipliers,
            objective_direction=objective_direction,
            constraints=constraints,
            num_retries=self.num_numerical_retries,
            logger_=logger_,
        )


    def _solve_scan_point_objective(
        self,
        point_index: int,
        objective_var_ids: list[str],
        objective_multipliers: list[float],
        objective_direction: str,
        constraints: list[list[Any]],
        previous_solution: Any | None,
        previous_score: float | None,
    ) -> structs.SolverResult:
        """Solve one scan point, optionally using warm start and retries.

        Args:
            point_index: Zero-based index of the current scan point.
            objective_var_ids: Objective variable ids optimized at the point.
            objective_multipliers: Objective coefficients aligned with
                ``objective_var_ids``.
            objective_direction: ``"max"`` or ``"min"``.
            constraints: Solver-ready constraints for the current point.
            previous_solution: Previously solved optimal point, used as warm
                start when available.
            previous_score: Objective score of the previous successful point.

        Returns:
            structs.SolverResult: Best solution accepted for the current point.

        Example:
            For ``scan_axis.constraint_sign="geq"`` and ``objective_direction="max"``,
            a point that drops below the previous objective triggers the
            improvement loop before it is accepted.
        """
        # Solve first and hardest point
        if point_index == 0:
            return self._solve_first_scan_point(
                objective_var_ids=objective_var_ids,
                objective_multipliers=objective_multipliers,
                objective_direction=objective_direction,
                constraints=constraints,
                logger_=logger,
            )

        # Solve other points
        warm_start_solution = None
        if previous_solution is not None and is_optimal(previous_solution.status):
            warm_start_solution = previous_solution.model_dump()
            logger.debug("      -> Using previous optimal scan point as warm start.")
        else:
            logger.debug("      -> No warm start available for this scan point.")


        best_solution = solve_step_robustly(
            solver=self.solver,
            objective_var_ids=objective_var_ids,
            objective_multipliers=objective_multipliers,
            objective_direction=objective_direction,
            constraints=constraints,
            warm_start_solution=warm_start_solution,
            num_retries=self.num_numerical_retries,
            logger_=logger
        )

        best_score = objective_score(best_solution)
        logger.debug("      -> Initial solve for this point returned status=%s objective=%s previous_objective=%s", best_solution.status, best_score, previous_score)
        
        is_suboptimal = (
            best_score is not None
            and previous_score is not None
            and self.scan_axis.type == "flux_variable"
            and self.scan_axis_constraint_sign != "eq"
            and (
                best_score < previous_score - 1e-6
                if objective_direction == "max"
                else best_score > previous_score + 1e-6
            )
        )
        if not is_suboptimal:
            return best_solution
            
        logger.warning("A scan point returned a worse-than-expected objective value. Launching %d improvement attempt(s).", self.num_improvement_attempts)

        # This loop is intentionally not routed through solve_step_robustly().
        # The robust helper only retries when the solve fails numerically or
        # does not reach an optimal status. Here the solve already returned an
        # optimal solution, but its objective is unexpectedly worse than the
        # previous scan point. We therefore try a few additional seeds, keep
        # the best objective found across those retries, and stop early once
        # the result is no longer worse than the previous successful point.
        for attempt in range(self.num_improvement_attempts):
            seed = random.randint(10000, 99999)
            set_solver_random_seed(self.solver, seed)
            logger.debug("      -> Improvement attempt %d/%d with seed=%d", attempt + 1, self.num_improvement_attempts, seed)
            retry_solution = self.solver.solve(
                objective_var_ids=objective_var_ids,
                objective_multipliers=objective_multipliers,
                objective_direction=objective_direction,
                constraints=constraints,
                warm_start_solution=warm_start_solution,
            )
            retry_score = objective_score(retry_solution)
            logger.debug("      -> Improvement attempt %d returned status=%s objective=%s", attempt + 1, retry_solution.status, retry_score)
            if retry_score is None:
                continue
            is_better = (best_score is None or (
                                                    retry_score > best_score
                                                    if objective_direction == "max"
                                                    else retry_score < best_score
                                                )
            )
            if is_better:
                best_solution = retry_solution
                best_score = retry_score
                logger.info("      -> Improved solution found on retry %d with objective=%s", attempt + 1, best_score)
            still_suboptimal = (
                best_score is not None
                and previous_score is not None
                and self.scan_axis.type == "flux_variable"
                and self.scan_axis_constraint_sign != "eq"
                and (
                    best_score < previous_score - 1e-6
                    if objective_direction == "max"
                    else best_score > previous_score + 1e-6
                )
            )
            if not still_suboptimal:
                break
        return best_solution

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
    #   PHASE 4: Post-analyses.
    #   PHASE 5: Failure handling.
    #   PHASE 6: Scenario finalization.
    #   PHASE 7: Experiment finalization.
    #
    # VariableScan follows the same skeleton as other experiments, but its
    # experiment-specific core logic is:
    #   PHASE 2.1: Build workflow inputs.
    #   PHASE 2.2: Benchmark scan axis when needed.
    #   PHASE 2.3: Resolve scan grid.
    #   PHASE 2.4: Execute scan loop.
    #
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, parallel_coordinates: dict[str, int] | None = None) -> list[dict[str, Any]]:
        """Execute the scan workflow across all configured scenarios.

        Returns:
            list[dict[str, Any]]: One scenario payload per biological scenario.
            Each payload contains scan metadata in ``metrics``, one entry per
            point in ``steps``, and a top-level scenario ``status``.

        Example:
            A successful point payload contains fields such as ``Step_Name``,
            ``Objective_Value``, ``Scan_Label``, and ``Scan_Axis_Value``.
        """
        start_time = time.time()
        logger.debug("================= STARTING RUN() ====================")
        logger.info("Starting VariableScanExperiment: %s", self.experiment_def.experiment_name)

        scenarios = self.experiment_def.biological_scenarios
        logger.info(f"Processing {len(scenarios)} scenarios...")
        cfg = self.experiment_def.analysis_config
        parallel_mode = parallel_coordinates is not None
        if parallel_mode:
            has_point = "point_index" in parallel_coordinates
            has_chunk = "point_start" in parallel_coordinates and "point_stop" in parallel_coordinates
            if not has_point and not has_chunk:
                raise ValueError("VariableScan parallel mode requires `point_index` or (`point_start`, `point_stop`).")
            scenario_index = int(parallel_coordinates.get("scenario_index", 0))
            if scenario_index < 0 or scenario_index >= len(scenarios):
                raise IndexError(f"Parallel scenario_index {scenario_index} is out of bounds for {len(scenarios)} scenario(s).")
            scenario_items = [(scenario_index, scenarios[scenario_index])]
        else:
            scenario_items = list(enumerate(scenarios))

        for scenario_idx, scenario in scenario_items:
            if parallel_mode and (self.scan_values or self.scan_range):
                configured_points = self._resolve_scan_points(None)
                if "point_index" in parallel_coordinates:
                    selected_indices = [int(parallel_coordinates["point_index"])]
                else:
                    selected_indices = list(range(int(parallel_coordinates["point_start"]), int(parallel_coordinates["point_stop"])))
                if any(index < 0 or index >= len(configured_points) for index in selected_indices):
                    raise IndexError(f"VariableScan parallel coordinates are out of bounds for {len(configured_points)} configured point(s).")
                if selected_indices and all(os.path.exists(self._parallel_fragment_path(scenario, index)) for index in selected_indices):
                    logger.info(" [scenario parallel] Skipping completed VariableScan assignment for '%s' with point indices %s.", scenario.scenario_id, selected_indices)
                    continue
            # -------------------------------------------------------------
            # PHASE 1: SCENARIO PREPARATION
            # Start timing, emit the scenario banner, and create the fresh
            # payload container that will accumulate this scan run.
            # -------------------------------------------------------------
            scenario_start_time = time.time()
            logger.info(f" [scenario {scenario_idx+1}]     --- Processing Scenario: {scenario.scenario_id} ---")
            run_payload = self.build_scenario_payload(scenario)

            try:
                # -------------------------------------------------------------
                # PHASE 1.1: LOAD SOLVER CONTEXT
                # Load the solver context before any scenario-specific
                # features, metadata, or core optimization logic are prepared.
                # -------------------------------------------------------------
                logger.debug(f" [scenario {scenario_idx+1}] Loading biological scenario and analysis config into solver context...")
                self.solver.load_context(biological_scenario=scenario, analysis_config=cfg)

                # -------------------------------------------------------------
                # PHASE 1.2: PREPARE OPTIONAL EXPERIMENT FEATURES
                # Initialize optional scenario-local features such as swap
                # minimization helpers without mixing them into the core scan
                # logic below.
                # -------------------------------------------------------------
                if self.minimize_swaps_config:
                    self.swap_ids = collect_swap_ids(scenario, self.minimize_swaps_config)
                    if self.swap_ids:
                        logger.info(" [scenario %d] Swap minimization active with %d candidate swap variable(s).", scenario_idx + 1, len(self.swap_ids))
                        logger.debug(" [scenario %d] Swap IDs: %s", scenario_idx + 1, self.swap_ids)
                    else:
                        logger.warning(" [scenario %d] Swap minimization requested, but no z-vars were found.", scenario_idx + 1)
                else:
                    self.swap_ids = []
                    logger.debug(" [scenario %d] Swap minimization disabled.", scenario_idx + 1)


                # -------------------------------------------------------------
                # PHASE 1.3: RESOLVE SCENARIO METADATA
                # Resolve scenario-specific identifiers that the later scan
                # workflow will need in solver-ready form.
                # -------------------------------------------------------------
                scenario_metadata = self.resolve_scenario_metadata(scenario)
                biomass_id = scenario_metadata["biomass_id"]
                product_name = scenario_metadata["product_name"]
                product_id = scenario_metadata["product_id"]
                if product_id: logger.info(f" [scenario {scenario_idx+1}] Resolved Product ID: '{product_id}' (from name '{product_name}')")
                else:          logger.warning(f"Could not resolve Product ID for '{product_name}'. Steps relying on 'product' may fail.")

                substrate_name = scenario_metadata["substrate_name"]
                substrate_ids = scenario_metadata["substrate_ids"]
                if substrate_ids: logger.info(f" [scenario {scenario_idx+1}] Resolved Substrate IDs: {substrate_ids} (from condition '{substrate_name}')")
                else:             logger.warning(f"Could not resolve any Substrate IDs for condition '{substrate_name}'.")

                max_prod = 0.0 # Initialize safely to 0.0

                # -------------------------------------------------------------
                # PHASE 1.4: PRECOMPUTATIONS
                # Compute shared scenario-wide limits once and reuse them
                # during scan setup and constraint resolution.
                # -------------------------------------------------------------
                if product_id:
                    try:
                        # Max Yield
                        logger.info(f" [scenario {scenario_idx+1}]   -> Calculating Max Theoretical Yield...")
                        max_prod = self.compute_max_theoretical_yield(scenario, product_id)
                        run_payload["metrics"]["max_theoretical_yield"] = max_prod
                        logger.info(f" [scenario {scenario_idx+1}]   -> Max Theoretical Yield for '{product_id}': {max_prod:.4f}")
                    
                    except Exception as exc:
                        logger.warning("Could not compute max theoretical yield for scenario '%s': %s", scenario.scenario_id, exc)

                # ------------------------------------------------------------------
                # PHASE 2: EXPERIMENT CORE LOGIC
                # Translate the VariableScan configuration into solver-ready
                # inputs, benchmark the scanned variable, resolve the scan
                # grid, and execute the point-by-point scan.
                # ------------------------------------------------------------------

                # ------------------------------------------------------------------
                # PHASE 2.1: BUILD WORKFLOW INPUTS
                # Normalize the user-facing scan configuration into concrete
                # solver inputs. The axis types diverge in two places:
                #
                #   1. Target resolution here:
                #      - flux_variable: may use symbolic ids such as
                #        "product" or "biomass", so it must be resolved
                #        against this scenario.
                #      - metabolite_concentration: already names a metabolite
                #        in the concentration dictionary.
                #      - dG0: already names a reaction in the dG0 dictionary.
                #      - constraint: names an existing constraint or provides
                #        a full constraint template.
                #
                #   2. Point application later in PHASE 2.4.1:
                #      - flux_variable adds a point-specific solver
                #        constraint.
                #      - metabolite_concentration and dG0 reload the thermo
                #        context with copied, point-specific data.
                #      - constraint replaces one named static constraint with
                #        a point-specific version.
                # ------------------------------------------------------------------
                # PHASE 2.1.1: Resolve the scan-axis target.
                benchmark_required = self.scan_axis.type == "flux_variable" and (
                    not parallel_mode or self.scan_axis.scan_fractions or self.scan_axis.scan_step_size is not None
                )
                if benchmark_required:
                    resolved_scan_axis_target = resolve_symbolic_vars(self.scan_axis.target, product_id=product_id, biomass_id=biomass_id)
                    if not resolved_scan_axis_target:
                        raise ValueError(f"Could not resolve flux-variable scan axis target '{self.scan_axis.target}' for scenario '{scenario.scenario_id}'.")
                else:
                    resolved_scan_axis_target = self.scan_axis.target

                # PHASE 2.1.2: Resolve the objective optimized at each scan point.
                resolved_objective_var_ids = resolve_symbolic_vars(list(self.objective_var_ids), product_id=product_id, biomass_id=biomass_id)
                if not resolved_objective_var_ids:
                    raise ValueError(f"Could not resolve objective variables {self.objective_var_ids} for scenario '{scenario.scenario_id}'.")

                run_payload["metrics"]["resolved_scan_axis_target"] = resolved_scan_axis_target
                run_payload["metrics"]["scan_axis_type"] = self.scan_axis.type
                run_payload["metrics"]["resolved_objective_var_ids"] = resolved_objective_var_ids

                # PHASE 2.1.3: Resolve constraints used during scan-axis benchmarking.
                # Resolve the benchmark-only constraint block against the
                # scanned flux variable when a benchmark is needed. Direct
                # thermo axes do not have a benchmark objective, so their base
                # constraints are resolved against the point objective.
                #
                # Example:
                #   scan_axis.target == "product"
                #   base_constraints == ["min_growth"]
                # becomes something like:
                #   benchmark objective -> max EX_ac_e
                #   benchmark constraints -> [["BIOMASS_AWOODII", 0.05, "min_growth", "geq"]]
                benchmark_constraints = resolve_constraint_block(
                    self.base_constraints,
                    product_id=product_id,
                    biomass_id=biomass_id,
                    max_prod=max_prod,
                    min_growth=self.min_growth,
                    min_mdf=self.min_mdf,
                    fraction_prod=self.fraction_prod,
                    objective_var_ids=([resolved_scan_axis_target] if self.scan_axis.type == "flux_variable" else resolved_objective_var_ids),
                )
                logger.debug(" [scenario %d] Resolved %d benchmark constraint(s): %s", scenario_idx + 1, len(benchmark_constraints), benchmark_constraints)

                # PHASE 2.1.4: Resolve constraints shared by every point solve.
                # Resolve the static point-solve constraint block against the
                # real optimization objective. These constraints are shared by
                # every scan point before the point-specific scan-axis setting
                # is added later inside the loop.
                #
                # Example:
                #   objective_var_ids == ["var_B"]
                #   point_constraints == ["min_growth", "min_product"]
                # becomes something like:
                #   [["BIOMASS_AWOODII", 0.05, "min_growth", "geq"],
                #    ["EX_ac_e", 12.3, "min_product", "geq"]]
                scan_static_constraints = resolve_constraint_block(
                    self.point_constraints,
                    product_id=product_id,
                    biomass_id=biomass_id,
                    max_prod=max_prod,
                    min_growth=self.min_growth,
                    min_mdf=self.min_mdf,
                    fraction_prod=self.fraction_prod,
                    objective_var_ids=resolved_objective_var_ids,
                )
                logger.debug(" [scenario %d] Resolved %d static scan constraint(s): %s", scenario_idx + 1, len(scan_static_constraints), scan_static_constraints)

                # PHASE 2.1.5: Prepare constraint-axis replacement, if requested.
                # A constraint scan point must replace the named base/point
                # constraint rather than stack another constraint on top of it.
                # This matters for range constraints such as cofactor ratios:
                # widening a ratio would have no effect if the original
                # narrower ratio remained in the static constraint block.
                constraint_axis_template: list[Any] | None = None
                if self.scan_axis.type == "constraint":
                    constraint_axis_template = resolve_constraint_axis_template(
                        self.scan_axis,
                        candidate_constraints=benchmark_constraints + scan_static_constraints,
                    )
                    constraint_axis_names = {
                        self.scan_axis.target,
                        str(constraint_axis_template[2]),
                    }
                    benchmark_constraints = [constraint for constraint in benchmark_constraints if not (isinstance(constraint, list) and len(constraint) >= 3 and str(constraint[2]) in constraint_axis_names)]
                    scan_static_constraints = [constraint for constraint in scan_static_constraints if not (isinstance(constraint, list) and len(constraint) >= 3 and str(constraint[2]) in constraint_axis_names)]
                    logger.debug(" [scenario %d] Constraint scan axis will rewrite constraint '%s' from template: %s", scenario_idx + 1, constraint_axis_template[2], constraint_axis_template)

                # Keep the fully resolved constraint lists visible in debug
                # output so scenario-specific symbolic substitutions can be
                # inspected directly during scan troubleshooting.
                logger.debug(" [scenario %d] Benchmark constraints: %s", scenario_idx + 1, benchmark_constraints)
                logger.debug(" [scenario %d] Static scan constraints: %s", scenario_idx + 1, scan_static_constraints)

                # ------------------------------------------------------------------
                # PHASE 2.2: BENCHMARK SCAN AXIS WHEN NEEDED
                # Flux-variable axes can use benchmark-relative grids. Direct
                # concentration/dG0/constraint axes are already explicit and
                # therefore skip the benchmark solve entirely.
                # ------------------------------------------------------------------
                scan_ready = True
                points_to_scan: list[dict[str, float | str]] = []

                if benchmark_required:
                    logger.info(" [scenario %d] Step 1: Benchmarking maximum feasible value of '%s'...", scenario_idx + 1, resolved_scan_axis_target)
                    benchmark_solution = solve_step_robustly(
                        solver=self.solver,
                        objective_var_ids=[resolved_scan_axis_target],
                        objective_multipliers=[1.0],
                        objective_direction="max",
                        constraints=benchmark_constraints,
                        num_retries=self.num_numerical_retries,
                        logger_=logger
                    )
                    logger.info(" [scenario %d] Benchmark result: status=%s objective=%s", scenario_idx + 1, benchmark_solution.status, benchmark_solution.objective_value)
                    scan_ready = is_optimal(benchmark_solution.status)
                    if scan_ready:
                        benchmark_value = benchmark_solution.values.get(resolved_scan_axis_target, benchmark_solution.objective_value)
                        if benchmark_value is None:
                            raise ValueError(f"Benchmark solution for scan axis '{resolved_scan_axis_target}' did not report a value or objective.")
                        run_payload["metrics"]["max_scan_value"] = float(benchmark_value)
                        logger.info(" [scenario %d] Step 2: Generating scan grid from benchmark value %.6f", scenario_idx + 1, float(benchmark_value))
                        points_to_scan = self._resolve_scan_points(float(benchmark_value))
                    else:
                        run_payload["status"] = f"Failed at benchmark: {benchmark_solution.status}"
                else:
                    logger.info(" [scenario %d] Step 1: Skipping benchmark for explicit %s scan axis '%s'.", scenario_idx + 1, self.scan_axis.type, resolved_scan_axis_target)
                    points_to_scan = self._resolve_scan_points(None)

                if scan_ready:
                    run_payload["metrics"]["scan_point_count"] = len(points_to_scan)
                    if not points_to_scan:
                        logger.warning("No valid scan points resolved. Check configuration.")
                        run_payload["status"] = "Failed: no valid scan points resolved."
                        scan_ready = False

                indexed_points = list(enumerate(points_to_scan))
                if parallel_mode:
                    if "point_index" in parallel_coordinates:
                        point_index = int(parallel_coordinates["point_index"])
                        indexed_points = [item for item in indexed_points if item[0] == point_index]
                    else:
                        point_start = int(parallel_coordinates["point_start"])
                        point_stop = int(parallel_coordinates["point_stop"])
                        indexed_points = [item for item in indexed_points if point_start <= item[0] < point_stop]
                    if not indexed_points:
                        raise IndexError(f"VariableScan parallel coordinates selected no points from a grid of size {len(points_to_scan)}.")

                previous_solution = None
                previous_score = None
                successful_point_count = 0
                failed_point_count = 0

                # ------------------------------------------------------------------
                # PHASE 2.4: EXECUTE SCAN LOOP
                # Iterate over the scan points, solve the configured objective,
                # optionally run swap minimization, and record both successful
                # and failed points across the full scan grid.
                # ------------------------------------------------------------------
                if scan_ready:
                    for point_index, point in indexed_points:
                        if parallel_mode and os.path.exists(self._parallel_fragment_path(scenario, point_index)):
                            logger.info(" [scenario parallel] Skipping completed VariableScan point %d for '%s'.", point_index, scenario.scenario_id)
                            continue
                        step_name = f"scan_{point['label']}"
                        logger.info(" [scenario %d] Step 3: Scan point %d/%d | label=%s | target=%.6f", scenario_idx + 1, point_index + 1, len(points_to_scan), point["label"], point["target"])

                        # ---------------------------------------------------------
                        # PHASE 2.4.1: PREPARE POINT INPUTS
                        # Build the point-specific constraints or thermo
                        # context for the current scan position before
                        # executing the solve.
                        # ---------------------------------------------------------
                        # Static constraints always apply. Axis-specific
                        # handling below then either adds one flux constraint or
                        # reloads the solver with a copied thermo scenario.
                        point_constraints = []
                        point_constraints.extend(scan_static_constraints)

                        if self.scan_axis.type == "flux_variable":
                            # Flux axes are ordinary solver variables, so the
                            # scan target is represented as a point constraint.
                            point_constraints.append([resolved_scan_axis_target, point["target"], "scan_axis_constraint", self.scan_axis_constraint_sign])
                            point_constraints.extend(benchmark_constraints)
                        elif self.scan_axis.type == "constraint":
                            # Constraint axes rewrite one named static
                            # constraint. The original same-named constraint
                            # was removed above so wider and narrower point
                            # ranges both behave as actual replacements.
                            point_constraints.extend(benchmark_constraints)
                            if constraint_axis_template is None:
                                raise ValueError(f"Constraint scan axis '{self.scan_axis.target}' has no resolved template.")
                            point_constraints.append(
                                build_constraint_axis_point(
                                    axis=self.scan_axis,
                                    template_constraint=constraint_axis_template,
                                    target_value=float(point["target"]),
                                    product_id=product_id,
                                    biomass_id=biomass_id,
                                )
                            )
                        elif self.scan_axis.type == "metabolite_concentration":
                            # Concentration axes do not appear as linear
                            # constraints. They are represented by reloading the
                            # thermo solver with copied, point-specific bounds.
                            point_constraints.extend(benchmark_constraints)
                            point_concentrations = copy.deepcopy(scenario.concentrations)
                            apply_concentration_axis_point(
                                point_concentrations=point_concentrations,
                                axis=self.scan_axis,
                                target_value=float(point["target"]),
                            )
                            self.solver.load_context(biological_scenario=scenario.model_copy(update={"concentrations": point_concentrations}), analysis_config=cfg)
                        elif self.scan_axis.type == "dG0":
                            # dG0 axes are also thermo-context edits. The scan
                            # point is first converted according to axis.mode
                            # and then written into a copied dG0 dictionary.
                            point_constraints.extend(benchmark_constraints)
                            if scenario.dG0 is None:
                                raise ValueError(f"Scenario '{scenario.scenario_id}' has no dG0 dataset for dG0 scan axis '{self.scan_axis.target}'.")
                            point_dG0 = copy.deepcopy(scenario.dG0)
                            apply_dg0_axis_point(
                                point_dG0=point_dG0,
                                axis=self.scan_axis,
                                target_value=float(point["target"]),
                                forward_suffix=scenario.model_conf.forward_reaction_suffix,
                                reverse_suffix=scenario.model_conf.reverse_reaction_suffix,
                                axis_name="scan axis",
                            )
                            self.solver.load_context(biological_scenario=scenario.model_copy(update={"dG0": point_dG0}), analysis_config=cfg)
                        
                        if "var_B2" in resolved_objective_var_ids and not any(
                            isinstance(constraint, list) and constraint and constraint[0] == "var_B"
                            for constraint in point_constraints
                        ):
                            logger.debug(" [scenario %d] Point %d: auto-injecting fallback `var_B >= %.6f` because objective contains `var_B2`.", scenario_idx + 1, point_index + 1, self.min_mdf)
                            point_constraints.append(["var_B", self.min_mdf, "min_mdf", "geq"])

                        point_constraints = dedupe_constraints(point_constraints)
                        logger.debug(" [scenario %d] Point %d constraints: %s", scenario_idx + 1, point_index + 1, point_constraints)

                        # ---------------------------------------------------------
                        # PHASE 2.4.2: SOLVE POINT OBJECTIVE
                        # Solve the current scan point, optionally warm-started
                        # from the previous successful point.
                        # ---------------------------------------------------------
                        solution = self._solve_scan_point_objective(
                            point_index=point_index,
                            objective_var_ids=resolved_objective_var_ids,
                            objective_multipliers=self.objective_multipliers,
                            objective_direction=self.objective_direction,
                            constraints=point_constraints,
                            previous_solution=previous_solution,
                            previous_score=previous_score,
                        )

                        current_swaps: Any = "N/A"
                        final_constraints = point_constraints

                        # ---------------------------------------------------------
                        # PHASE 2.4.3: OPTIONAL SWAP MINIMIZATION
                        # Optionally apply the old MDF-style swap sandwich after
                        # the primary point objective has reached an optimum.
                        # ---------------------------------------------------------
                        if is_optimal(solution.status) and self.swap_ids:
                            logger.info(" [scenario %d] Point %d reached optimal objective. Starting swap minimization.", scenario_idx + 1, point_index + 1)
                            solution, current_swaps, final_constraints = (
                                solve_with_swap_minimization(
                                    solver=self.solver,
                                    ideal_solution=solution,
                                    objective_var_ids=resolved_objective_var_ids,
                                    objective_multipliers=self.objective_multipliers,
                                    objective_direction=self.objective_direction,
                                    constraints=point_constraints,
                                    swap_ids=self.swap_ids,
                                    num_numerical_retries=self.num_numerical_retries,
                                    num_swap_attempts=self.num_swap_attempts,
                                    num_reoptimization_attempts_sandwich=self.num_reoptimization_attempts_sandwich,
                                    quality_prefix="maintain_scan_quality",
                                    raise_on_swap_failure=False,
                                    logger_=logger,
                                )
                            )
                        
                        logger.info(" [scenario %d] Point %d result: status=%s objective=%s swaps=%s", scenario_idx + 1, point_index + 1, solution.status, solution.objective_value, current_swaps)

                        # ---------------------------------------------------------
                        # PHASE 2.4.4: BUILD POINT PAYLOAD
                        # Convert the raw point result into the shared tabular
                        # output schema used by all experiment classes.
                        # ---------------------------------------------------------
                        step_payload = self.build_standard_step_payload(
                            step_name=step_name,
                            objective_var_ids=resolved_objective_var_ids,
                            objective_direction=self.objective_direction,
                            constraints=final_constraints,
                            solution=solution,
                            biomass_id=biomass_id,
                            product_id=product_id,
                            swaps_required=current_swaps,
                            substrate_ids=substrate_ids,
                        )
                        step_payload.update(
                            {
                                "Scan_Role": "point",
                                "Scan_Axis_Type": self.scan_axis.type,
                                "Scan_Axis": self.scan_axis.label or self.scan_axis.target,
                                "Scan_Axis_Target": resolved_scan_axis_target,
                                "Scan_Axis_Value": round(float(point["target"]), 12),
                                "Scan_Axis_Realized_Value": "N/A",
                                "Scan_Objective": ", ".join(resolved_objective_var_ids),
                                f"{self.scan_axis.target}_target": round(float(point["target"]), 6),
                                "Scan_Label": point["label"],
                                "Model": (f"{scenario.model_component}_{self.scan_axis.target}_{point['label']}"),
                                "Point_Index": point_index + 1,
                            }
                        )

                        if self.key_reactions:
                            self.add_key_reaction_placeholders(step_payload, self.key_reactions)
                        if self.key_metabolites:
                            self.add_key_metabolite_placeholders(step_payload, self.key_metabolites)

                        if is_optimal(solution.status):
                            # -------------------------------------------------
                            # PHASE 2.4.5: UPDATE SUCCESSFUL OPTIMIZATION STATE
                            # Store the realized point state and enrich the
                            # payload before artifacts or post-analyses run.
                            # -------------------------------------------------

                            # -------------------------------------------------
                            # PHASE 2.4.5.1: PERSIST OPTIMIZATION STATE AND METRICS
                            # Store the current point result so later points can
                            # warm-start and monotonicity can be evaluated.
                            # -------------------------------------------------
                            previous_solution = solution
                            previous_score = objective_score(solution)
                            run_payload["status"] = "Success"
                            run_payload["metrics"][f"{step_name}_objective_value"] = (solution.objective_value)

                            # Keep the configured bound and the realized target separate. COBRA-k's
                            # native result is needed for directional targets such as EX_glc_e_REV.
                            realized_scan_value = solution.values.get(resolved_scan_axis_target)
                            native_primary = (solution.native_result or {}).get("primary", {})
                            if isinstance(native_primary, dict):
                                realized_scan_value = native_primary.get(resolved_scan_axis_target, realized_scan_value)
                            if isinstance(realized_scan_value, (int, float)):
                                step_payload["Scan_Axis_Realized_Value"] = round(float(realized_scan_value), 12)
                            logger.debug(" [scenario %d] Point %d realized scan target '%s'=%s", scenario_idx + 1, point_index + 1, resolved_scan_axis_target, realized_scan_value)

                            # -------------------------------------------------
                            # PHASE 2.4.5.2: ENRICH PAYLOAD ANNOTATIONS
                            # Add optional diagnostic annotations such as key
                            # fluxes and the realized active swap list.
                            # -------------------------------------------------
                            if self.key_reactions:
                                key_flux_data = self._get_key_fluxes(solution, self.key_reactions)
                                logger.debug(f" [scenario {scenario_idx+1}] Got {len(key_flux_data)} key reaction fluxes")
                                for prefix, (variant_name, net_flux) in key_flux_data.items():
                                    step_payload[f"variant_{prefix}"] = variant_name
                                    step_payload[f"flux_{prefix}"] = net_flux

                            if self.key_metabolites:
                                key_conc_data = self._get_key_concentrations(solution, self.key_metabolites)
                                for met_id, conc in key_conc_data.items():
                                    step_payload[f"conc_{met_id}"] = conc

                            if self.swap_ids:
                                active_swaps = [
                                    swap_id
                                    for swap_id in self.swap_ids
                                    if solution.values.get(swap_id, 0.0) > 0.5
                                ]
                                step_payload["Active_Swaps_List"] = str(active_swaps)
                                logger.debug(f" [scenario {scenario_idx+1}]   Recorded active swaps for payload: {active_swaps}")

                            # -------------------------------------------------
                            # PHASE 3: ARTIFACT GENERATION
                            #
                            # PHASE 3.1: WRITE OPTIONAL ARTIFACTS
                            # Persist optional raw/debug/visualization files
                            # associated with this successful scan point.
                            # -------------------------------------------------
                            artifact_label = f"point_{point_index + 1:03d}_{point['label']}" if parallel_mode else str(point["label"])
                            artifact_id = sanitize_token(f"{scenario.scenario_id}_{artifact_label}")

                            # Persist optional debug and visualization artifacts directly
                            # here so the run flow fully describes what happens after a
                            # successful scan point.
                            if self.analysis_params.excel_output:
                                self.save_to_excel(
                                    model=scenario.model,
                                    raw_solution=solution,
                                    scenario_id=artifact_id,
                                    workbook_id=artifact_id if parallel_mode else None,
                                )

                            if self.analysis_params.save_raw_solutions:
                                self.save_raw_solutions(solution, artifact_id)

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
                                    objective_direction=self.objective_direction,
                                )


                            # -------------------------------------------------
                            # PHASE 4: POST-ANALYSES
                            #
                            # PHASE 4.1: BUILD LOCKED POINT STATE
                            # PHASE 4.2: DISPATCH POST-ANALYSES
                            #
                            # Lock the optimum tightly and run secondary
                            # analyses so diagnostics stay anchored to the
                            # realized scan solution.
                            # -------------------------------------------------
                            if solution.objective_value is not None:
                                # ---------------------------------------------
                                # PHASE 4.1: BUILD LOCKED POINT STATE
                                # Convert the realized objective into tight
                                # lock constraints for downstream analyses.
                                # ---------------------------------------------
                                post_constraints = list(final_constraints) + build_quality_lock_constraints(
                                    solution=solution,
                                    objective_var_ids=resolved_objective_var_ids,
                                    objective_direction=self.objective_direction,
                                    prefix=f"lock_{step_name}",
                                )

                                # ---------------------------------------------
                                # PHASE 4.2: DISPATCH POST-ANALYSES
                                # Run the configured secondary analyses on the
                                # locked feasible region for this scan point.
                                # ---------------------------------------------
                                logger.debug(f" [scenario {scenario_idx+1}]     -> Dispatching post-analyses with locked bounds...")
                                post_analysis_results = self.dispatcher.run(
                                    solver=self.solver,
                                    base_solution=solution,
                                    analysis_params=self.analysis_params,
                                    constraints=post_constraints,
                                    objective_vars=resolved_objective_var_ids,
                                    subnetwork_ids=self.analysis_params.subnetwork,
                                    thermo_unconstrained_reactions=self.analysis_params.thermo_unconstrained_reactions,
                                )
                                self.attach_post_analysis_results(step_payload, post_analysis_results)
                        else:
                            # -------------------------------------------------
                            # PHASE 5.1: HANDLE FAILURE AND CONTINUE SCANNING
                            # Keep the failed payload, preserve the last
                            # successful point state, and continue scanning so
                            # later easier targets can still be evaluated.
                            # -------------------------------------------------
                            failed_point_count += 1
                            logger.warning(f"    -> Step '{step_name}' failed ({solution.status}). Continuing with remaining scan points.")
                            if parallel_mode:
                                self._save_parallel_fragment(scenario, step_payload, point_index)
                            else:
                                run_payload["steps"].append(step_payload)
                            continue

                        successful_point_count += 1
                        if parallel_mode:
                            self._save_parallel_fragment(scenario, step_payload, point_index)
                        else:
                            run_payload["steps"].append(step_payload)

                    run_payload["metrics"]["successful_scan_point_count"] = successful_point_count
                    run_payload["metrics"]["failed_scan_point_count"] = failed_point_count

                    if points_to_scan:
                        if failed_point_count == 0 and successful_point_count == len(points_to_scan):
                            run_payload["status"] = "Success"
                        elif successful_point_count > 0:
                            run_payload["status"] = (
                                f"Partial Success: {successful_point_count}/{len(points_to_scan)} "
                                "scan points solved optimally"
                            )
                        else:
                            run_payload["status"] = "Failed: no scan points solved optimally"

            except Exception as exc:
                # -------------------------------------------------------------
                # PHASE 5.2: HANDLE UNEXPECTED SCENARIO EXCEPTION
                # Convert unexpected scenario-level errors into the standard
                # scenario failure payload without dropping execution metadata.
                # -------------------------------------------------------------
                logger.error("Unhandled exception during variable scan scenario '%s': %s", scenario.scenario_id, exc, exc_info=True)
                run_payload["status"] = f"Failed: {exc}"

            finally:
                # -------------------------------------------------------------
                # PHASE 6: SCENARIO FINALIZATION
                # Persist final execution metadata and store the scenario result
                # in the experiment-wide aggregate list.
                # -------------------------------------------------------------
                elapsed = time.time() - scenario_start_time
                run_payload["metrics"]["execution_time_seconds"] = round(elapsed, 2)
                logger.info(f" [scenario {scenario_idx+1}]         Finished scenario {scenario.scenario_id} in {elapsed:.2f} seconds.")
                if not parallel_mode:
                    self.aggregate_results.append(run_payload)

        # -------------------------------------------------------------
        # PHASE 7: EXPERIMENT FINALIZATION
        # Emit the experiment-level completion log and return the full
        # aggregated payload for JSON/CSV export.
        # -------------------------------------------------------------
        total_elapsed = time.time() - start_time
        logger.info("VariableScanExperiment '%s' finished %d scenario(s) in %.2fs.", self.experiment_def.experiment_name, len(scenarios), total_elapsed)
        return [] if parallel_mode else self.aggregate_results
