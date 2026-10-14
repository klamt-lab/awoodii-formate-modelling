import copy
import logging
from math import exp
from typing import Any

from mymodelingpackage.structures import SolverResult
from mymodelingpackage.solvers.interface import BiologicalSolver
from mymodelingpackage.utilities import (
    z_var_rea_constraints,
    z_var_variant_and_original_rea_constraints,
    generate_pathway_constraints,
    get_exclusive_reaction_groups,
)


logger = logging.getLogger(__name__)


def _solution_to_dict(base_solution: SolverResult | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize solver-result inputs into one dictionary representation.

    Args:
        base_solution: Solver result provided either as a typed
            ``SolverResult``, a legacy dictionary payload, or ``None`` for
            exploration attempts without a warm start.

    Returns:
        dict[str, Any]: Dictionary-form solver result. Returns an empty dict
        when ``base_solution`` is ``None``.
    """
    if base_solution is None:
        return {}
    if isinstance(base_solution, SolverResult):
        return base_solution.model_dump()
    return dict(base_solution)


def _model_settings_from_solver(solver: BiologicalSolver) -> Any:
    """Return the loaded model settings object from the active solver context.

    Args:
        solver: Solver whose loaded biological scenario should be inspected.

    Returns:
        Any: The scenario's ``model_conf`` object.

    Raises:
        RuntimeError: If no biological scenario is currently loaded on the
            solver.
    """
    biological_scenario = getattr(solver, "biological_scenario", None)
    if biological_scenario is None:
        raise RuntimeError("Thermo post-analysis requires a loaded biological scenario on the solver.")
    return biological_scenario.model_conf


def _scenario_from_solver(solver: BiologicalSolver):
    """Return the loaded biological scenario from the active solver context.

    Args:
        solver: Solver whose loaded biological scenario should be returned.

    Returns:
        BiologicalScenario: The scenario currently loaded on the solver.

    Raises:
        RuntimeError: If no biological scenario is currently loaded on the
            solver.
    """
    biological_scenario = getattr(solver, "biological_scenario", None)
    if biological_scenario is None:
        raise RuntimeError("Thermo post-analysis requires a loaded biological scenario on the solver.")
    return biological_scenario


def run_cva(
    solver: BiologicalSolver,
    base_solution: SolverResult | dict[str, Any] | None,
    constraints: list | None,
    metabolites_to_check: list[str],
    subnetwork_ids: list[str] = None,
    thermo_unconstrained_reactions: list[str] = None,
    apply_pathway_constraints: bool = True,
    apply_specificity_constraints: bool = False,
) -> dict[str, dict[str, any]]:
    """Run concentration variability analysis for one or more metabolites.

    This helper solves the loaded thermo problem twice per metabolite, once to
    minimize and once to maximize the corresponding log-concentration variable
    ``x_<metabolite_id>``. The returned values are transformed back from
    log-space into concentration space.

    Args:
        solver: Solver with an already loaded thermo scenario.
        base_solution: Optional reference solution used to derive topology or
            specificity constraints and to warm-start the solve.
        constraints: Additional constraints defining the feasible state that
            CVA should preserve.
        metabolites_to_check: Metabolite ids to analyze.
        subnetwork_ids: Optional subnetwork restriction forwarded to the
            solver.
        thermo_unconstrained_reactions: Optional thermo exclusions forwarded to
            the solver.
        apply_pathway_constraints: Legacy flag kept for compatibility. The
            current implementation relies on solver settings plus
            ``apply_specificity_constraints``.
        apply_specificity_constraints: Whether exclusive-group pathway
            specificity should be enforced when a base solution is available.

    Returns:
        dict[str, dict[str, Any]]: Mapping from metabolite id to ``min`` and
        ``max`` concentration values or solver-status strings when infeasible.
    """
    base_solution_dict = _solution_to_dict(base_solution)
    settings = _model_settings_from_solver(solver)
    z_var_constraints = []
    if getattr(settings, "exact", False) and base_solution_dict:
        z_var_constraints, _, _ = z_var_rea_constraints(base_solution_dict)
    elif get_exclusive_reaction_groups(settings):
        if apply_specificity_constraints and base_solution_dict:
            z_var_constraints = generate_pathway_constraints(base_solution_dict, settings)
        elif apply_specificity_constraints and not base_solution_dict:
            logging.info("  [CVA] Specificity requested but no base solution provided; skipping pathway constraints.")
        else:
            logging.info("  [CVA] Mode: UNCONSTRAINED (Specificity constraints ignored).")

    full_constraints = (constraints or []) + z_var_constraints

    min_max_values = {}
    for met_id in metabolites_to_check:
        min_max_values[met_id] = {}
        objective_var = "x_" + met_id

        for opt_dir in ["min", "max"]:
            logging.debug(f"--- Running CVA for '{met_id}' (direction: {opt_dir}) ---")
            cva_sol = solver.solve(
                objective_var_ids=[objective_var],
                objective_direction=opt_dir,
                objective_multipliers=[1],
                constraints=full_constraints,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                warm_start_solution=base_solution_dict or None,
            )

            if cva_sol.status == "Optimal":
                log_conc = cva_sol.values.get(objective_var)
                if log_conc is not None:
                    min_max_values[met_id][opt_dir] = round(exp(log_conc), 8)
                else:
                    min_max_values[met_id][opt_dir] = "value_not_found"
            else:
                min_max_values[met_id][opt_dir] = cva_sol.status or "Error"

    return min_max_values


def run_metabolite_ratios(
    solver: BiologicalSolver,
    base_solution: SolverResult | dict[str, Any] | None,
    constraints: list | None,
    ratio_definitions: list[dict],
    subnetwork_ids: list[str] = None,
    thermo_unconstrained_reactions: list[str] = None,
    apply_specificity_constraints: bool = False,
) -> dict[str, dict[str, any]]:
    """Run metabolite-ratio variability analysis for configured ratios.

    Each ratio definition is translated into a linear objective on the
    corresponding log-concentration variables. The resulting objective values
    are exponentiated so the final output is reported in ratio space.

    Args:
        solver: Solver with an already loaded thermo scenario.
        base_solution: Optional reference solution used to derive topology or
            specificity constraints and to warm-start the solve.
        constraints: Additional constraints defining the feasible state that
            ratio analysis should preserve.
        ratio_definitions: Ratio specifications containing ``name``,
            ``metabolites``, and ``multipliers`` entries.
        subnetwork_ids: Optional subnetwork restriction forwarded to the
            solver.
        thermo_unconstrained_reactions: Optional thermo exclusions forwarded to
            the solver.
        apply_specificity_constraints: Whether exclusive-group pathway
            specificity should be enforced when a base solution is available.

    Returns:
        dict[str, dict[str, Any]]: Mapping from ratio name to ``min`` and
        ``max`` values or solver-status strings when infeasible.
    """
    base_solution_dict = _solution_to_dict(base_solution)
    settings = _model_settings_from_solver(solver)
    z_var_constraints = []
    if getattr(settings, "exact", False) and base_solution_dict:
        z_var_constraints, _, _ = z_var_rea_constraints(base_solution_dict)
    elif get_exclusive_reaction_groups(settings):
        if apply_specificity_constraints and base_solution_dict:
            z_var_constraints = generate_pathway_constraints(base_solution_dict, settings)
        elif apply_specificity_constraints and not base_solution_dict:
            logging.info("  [Ratios] Specificity requested but no base solution provided; skipping pathway constraints.")
        else:
            logging.info("  [Ratios] Mode: UNCONSTRAINED (Specificity constraints skipped).")

    full_constraints = (constraints or []) + z_var_constraints

    final_results = {}
    for ratio_def in ratio_definitions:
        name = ratio_def.get("name")
        metabolites = ratio_def.get("metabolites")
        multipliers = ratio_def.get("multipliers")

        if not all([name, metabolites, multipliers]):
            logging.warning(f"Skipping invalid metabolite ratio definition: {ratio_def}")
            continue

        objective_vars = ["x_" + met for met in metabolites]
        final_results[name] = {}

        for opt_dir in ["min", "max"]:
            logging.debug(f"--- Calculating ratio for '{name}' (direction: {opt_dir}) ---")
            ratio_sol = solver.solve(
                objective_var_ids=objective_vars,
                objective_direction=opt_dir,
                objective_multipliers=multipliers,
                constraints=full_constraints,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                warm_start_solution=base_solution_dict or None,
            )

            if ratio_sol.status == "Optimal":
                obj_val = ratio_sol.objective_value
                try:
                    final_results[name][opt_dir] = round(exp(obj_val), 8)
                except Exception:
                    final_results[name][opt_dir] = "value_not_found"
            else:
                final_results[name][opt_dir] = ratio_sol.status or "Error"

    return final_results



def run_bottleneck_analysis(
    solver: BiologicalSolver,
    base_solution: SolverResult | dict[str, Any],
    constraints: list | None,
    dG0_change: float,
    test_active_reactions_only: bool = False,
    subnetwork_ids: list[str] = None,
    thermo_unconstrained_reactions: list[str] = None,
) -> list[str]:
    """Identify reaction bottlenecks by perturbing reaction dG0 values.

    The method temporarily relaxes one reaction's thermodynamic driving force
    at a time and checks whether the optimized MDF-like objective improves. If
    it does, that reaction is reported as a potential bottleneck.

    Args:
        solver: Solver with an already loaded thermo scenario.
        base_solution: Optimal reference solution that defines the baseline MDF
            objective value.
        constraints: Additional constraints defining the feasible state that
            the bottleneck analysis should preserve.
        dG0_change: Thermodynamic perturbation applied to one candidate
            reaction at a time.
        test_active_reactions_only: Whether to test only active reactions from
            the base solution instead of the full model.
        subnetwork_ids: Optional subnetwork restriction forwarded to the
            solver.
        thermo_unconstrained_reactions: Optional list of reactions that should
            be skipped during bottleneck testing.

    Returns:
        list[str]: Reaction ids whose dG0 relaxation improves the target MDF
        objective, or an error marker when the setup is invalid.
    """
    scenario = _scenario_from_solver(solver)
    base_solution_dict = _solution_to_dict(base_solution)
    bottlenecks = []

    objective_var_to_check = "var_B2" if subnetwork_ids else "var_B"
    initial_objective_value = base_solution_dict.get("values", {}).get(objective_var_to_check)
    if initial_objective_value is None:
        logging.error("Could not find initial objective value for bottleneck analysis. Aborting.")
        return ["Error: initial objective not found"]

    settings = scenario.model_conf
    z_var_constraints = []
    active_reactions = []
    if getattr(settings, "exact", False):
        z_var_constraints, active_reactions, _ = z_var_rea_constraints(base_solution_dict)
    elif get_exclusive_reaction_groups(settings):
        z_var_constraints, active_reactions, _ = z_var_variant_and_original_rea_constraints(base_solution_dict)
    full_constraints = (constraints or []) + z_var_constraints

    if test_active_reactions_only:
        reactions_to_test = active_reactions
        logging.info(f"      -> Running bottleneck analysis on {len(reactions_to_test)} active reactions...")
    else:
        reactions_to_test = [r.id for r in scenario.model.reactions]
        logging.info(f"      -> Running bottleneck analysis on all {len(reactions_to_test)} model reactions...")

    unconstrained_bases = set(thermo_unconstrained_reactions or [])
    if scenario.dG0 is None:
        logging.error("Bottleneck analysis requires dG0 data on the loaded biological scenario.")
        return ["Error: dG0 data unavailable"]

    total_reactions = len(reactions_to_test)
    for reaction_index, rea_id in enumerate(reactions_to_test, start=1):
        base_id = rea_id.replace("_FWD", "").replace("_REV", "")
        if base_id in unconstrained_bases:
            continue

        if rea_id not in scenario.dG0:
            continue

        if abs(scenario.dG0[rea_id]["dG0"]) == 100:
            continue

        paired_rea_id = None
        if rea_id.endswith("_FWD"):
            paired_rea_id = rea_id.replace("_FWD", "_REV")
        elif rea_id.endswith("_REV"):
            paired_rea_id = rea_id.replace("_REV", "_FWD")

        if paired_rea_id and paired_rea_id not in scenario.dG0:
            paired_rea_id = None

        try:
            scenario.dG0[rea_id]["dG0"] -= dG0_change
            if paired_rea_id:
                scenario.dG0[paired_rea_id]["dG0"] += dG0_change

            logger.info("      -> Testing bottleneck reaction %d/%d: %s", reaction_index, total_reactions, rea_id)
            test_sol = solver.solve(
                objective_var_ids=[objective_var_to_check],
                objective_direction="max",
                objective_multipliers=[1],
                constraints=full_constraints,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            )

            if test_sol.status == "Optimal":
                new_objective_value = test_sol.values.get(objective_var_to_check)
                if new_objective_value > initial_objective_value + 1e-6:
                    bottlenecks.append(rea_id)
                    logger.info("      -> MDF improved for reaction %s: %.6g -> %.6g (delta=%.6g)", rea_id, initial_objective_value, new_objective_value, new_objective_value - initial_objective_value)

        finally:
            scenario.dG0[rea_id]["dG0"] += dG0_change
            if paired_rea_id:
                scenario.dG0[paired_rea_id]["dG0"] -= dG0_change

    return bottlenecks


def run_bottleneck_set_analysis(
    solver: BiologicalSolver,
    base_solution: SolverResult | dict[str, Any],
    constraints: list | None,
    target_variable: str | None = None,
    target_mode: str = "base_plus_epsilon",
    target_value: float = 0.1,
    target_constraint_sign: str = "geq",
    topology_lock_mode: str = "auto",
    enumerate_all_minimal_sets: bool = False,
    max_sets: int = 10,
    subnetwork_ids: list[str] = None,
    thermo_unconstrained_reactions: list[str] = None,
) -> dict[str, Any]:
    """Identify minimal bottleneck-reaction sets using the MILP slack formulation.

    This helper re-solves the current thermo problem with
    ``bottleneck_analysis=True``. Instead of perturbing one reaction at a time,
    it asks for the smallest set of reactions whose thermodynamic constraints
    must be relaxed to reach a requested target such as a higher MDF or growth
    rate.

    Args:
        solver: Solver with an already loaded thermo scenario.
        base_solution: Optimal thermo reference solution.
        constraints: Additional constraints defining the feasible state that
            should be preserved.
        target_variable: Variable that should reach the requested target. When
            omitted, the active MDF variable (``var_B`` or ``var_B2``) is used.
        target_mode: Strategy used to derive the target. Supported values:
            ``"base_plus_epsilon"``, ``"base_fraction"``, and ``"absolute"``.
        target_value: Numeric value interpreted according to ``target_mode``.
        target_constraint_sign: Constraint sense used for the target variable.
            ``"geq"`` asks for at least the target, which is the usual growth
            or MDF threshold use case.
        topology_lock_mode: How strongly the base solution topology should be
            preserved. Supported values: ``"auto"``, ``"none"``,
            ``"variants_only"``, and ``"active_set"``.
        enumerate_all_minimal_sets: Whether to enumerate alternative
            minimal-cardinality bottleneck sets using no-good cuts.
        max_sets: Maximum number of minimal sets to collect when enumeration is
            enabled.
        subnetwork_ids: Optional subnetwork restriction forwarded to the
            solver.
        thermo_unconstrained_reactions: Optional thermo exclusions forwarded to
            the solver.

    Returns:
        dict[str, Any]: Structured payload describing the minimal bottleneck
        set analysis, including the identified sets and their intersection.
    """
    # Normalize inputs and pick the MDF variable that matches the active scope.
    # Bottleneck relaxations always act on the MDF constraints, but the target
    # variable can now be a non-MDF variable such as biomass.
    scenario = _scenario_from_solver(solver)
    base_solution_dict = _solution_to_dict(base_solution)
    mdf_variable = "var_B2" if subnetwork_ids else "var_B"
    target_variable_to_check = target_variable or mdf_variable
    base_target_value = base_solution_dict.get("values", {}).get(target_variable_to_check)

    # This analysis only makes sense when anchored to a real optimal MDF-like
    # solution, because the target and optional topology locks are derived from it.
    if base_solution_dict.get("status") != "Optimal":
        return {"status": "Error", "error": "Base solution not optimal"}
    if base_target_value is None:
        return {"status": "Error", "error": f"Base solution is missing '{target_variable_to_check}'"}
    if max_sets < 1:
        return {"status": "Error", "error": "max_sets must be at least 1"}
    if target_constraint_sign not in {"geq", "leq", "eq"}:
        return {"status": "Error", "error": f"Unsupported target_constraint_sign '{target_constraint_sign}'"}

    # Convert the user-facing target rule into one concrete target bound.
    # The bottleneck solve then asks: which thermo constraints must be relaxed
    # to make this target feasible?
    if target_mode == "base_plus_epsilon":
        target_threshold = base_target_value + target_value
    elif target_mode == "base_fraction":
        target_threshold = base_target_value * target_value
    elif target_mode == "absolute":
        target_threshold = target_value
    else:
        return {"status": "Error", "error": f"Unsupported target_mode '{target_mode}'"}

    settings = scenario.model_conf
    resolved_topology_lock_mode = topology_lock_mode
    z_var_constraints = []
    # "auto" mirrors the existing post-analysis behavior: exact mode locks all
    # activity indicators, while TCOSA-style alternative groups lock variants only.
    if topology_lock_mode == "auto":
        if getattr(settings, "exact", False):
            resolved_topology_lock_mode = "active_set"
        elif get_exclusive_reaction_groups(settings):
            resolved_topology_lock_mode = "variants_only"
        else:
            resolved_topology_lock_mode = "none"

    # Topology locks keep the bottleneck set tied to the same biological
    # interpretation as the base solution instead of allowing pathway switching.
    if resolved_topology_lock_mode == "active_set":
        z_var_constraints, _, _ = z_var_rea_constraints(base_solution_dict)
    elif resolved_topology_lock_mode == "variants_only":
        z_var_constraints, _, _ = z_var_variant_and_original_rea_constraints(base_solution_dict)
    elif resolved_topology_lock_mode != "none":
        return {"status": "Error", "error": f"Unsupported topology_lock_mode '{topology_lock_mode}'"}

    # Start from the experiment-provided locked feasible region, add any topology
    # locks, then require the requested target threshold.
    base_constraints = list(constraints or []) + z_var_constraints + [
        [target_variable_to_check, target_threshold, "bottleneck_set_target", target_constraint_sign]
    ]

    # The loop is used for both the first solve and optional enumeration. The
    # first optimal solve establishes the minimal cardinality k; later solves fix
    # zb_sum_var == k and add no-good cuts to find alternative sets.
    minimal_sets: list[list[str]] = []
    seen_sets: set[tuple[str, ...]] = set()
    minimal_relaxation_count: int | None = None
    stopped_early = False
    termination_message = ""
    current_constraints = list(base_constraints)

    while True:
        bottleneck_sol = solver.solve(
            objective_var_ids=[mdf_variable],
            objective_direction="max",
            objective_multipliers=[1.0],
            constraints=current_constraints,
            bottleneck_analysis=True,
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            warm_start_solution=base_solution_dict,
        )

        if bottleneck_sol.status != "Optimal":
            # After at least one set has been found, a non-optimal solve usually
            # means all minimal alternatives have been excluded by no-good cuts.
            if minimal_sets:
                termination_message = bottleneck_sol.status
                break
            return {
                "status": bottleneck_sol.status,
                "error": "No feasible bottleneck-set solution found for the requested target",
                "target_variable": target_variable_to_check,
                "mdf_variable": mdf_variable,
                "base_value": base_target_value,
                "target_mode": target_mode,
                "target_value": target_value,
                "target_threshold": target_threshold,
                "target_constraint_sign": target_constraint_sign,
                "topology_lock_mode": resolved_topology_lock_mode,
            }

        # Active zb variables are the reactions whose thermo constraints had to
        # be relaxed to satisfy the target at minimal cardinality.
        bottleneck_set = sorted(
            key.replace("zb_var_", "")
            for key, value in bottleneck_sol.values.items()
            if key.startswith("zb_var_") and value > 0.5
        )
        bottleneck_tuple = tuple(bottleneck_set)
        if bottleneck_tuple in seen_sets:
            # This guard prevents an accidental infinite loop if numerical
            # tolerances or duplicate cuts make CPLEX return the same set again.
            stopped_early = True
            termination_message = "Duplicate bottleneck set encountered during enumeration"
            break

        seen_sets.add(bottleneck_tuple)
        minimal_sets.append(bottleneck_set)

        if minimal_relaxation_count is None:
            # The bottleneck formulation minimizes zb_sum_var, so the first
            # objective value is the minimal number of relaxed reactions.
            if bottleneck_sol.objective_value is not None:
                minimal_relaxation_count = int(round(float(bottleneck_sol.objective_value)))
            else:
                minimal_relaxation_count = len(bottleneck_set)
            current_constraints = list(base_constraints) + [
                ["zb_sum_var", float(minimal_relaxation_count), "fix_minimal_bottleneck_count", "eq"]
            ]

        # Stop after the representative set unless explicit enumeration was
        # requested. If k is zero, no relaxed reaction is required and there is
        # no meaningful alternative bottleneck set to enumerate.
        if not enumerate_all_minimal_sets or minimal_relaxation_count == 0:
            break
        if len(minimal_sets) >= max_sets:
            stopped_early = True
            termination_message = f"Stopped after reaching max_sets={max_sets}"
            break
        if not bottleneck_set:
            break

        # Exclude exactly the current minimal set while keeping zb_sum_var fixed
        # at k. This finds overlapping alternatives without banning individual
        # reactions globally.
        current_constraints.append(
            [
                [[f"zb_var_{reaction_id}", 1.0] for reaction_id in bottleneck_set],
                float(minimal_relaxation_count) - 1.0,
                f"exclude_bottleneck_set_{len(minimal_sets)}",
                "leq",
            ]
        )

    # Summarize the set family: core reactions appear in every minimal set,
    # optional reactions appear only in some alternatives.
    if minimal_sets:
        core_reactions = sorted(set(minimal_sets[0]).intersection(*map(set, minimal_sets[1:]))) if len(minimal_sets) > 1 else list(minimal_sets[0])
        optional_reactions = sorted(set().union(*map(set, minimal_sets)) - set(core_reactions))
    else:
        core_reactions = []
        optional_reactions = []

    return {
        "status": "Optimal" if minimal_sets else "Infeasible",
        "target_variable": target_variable_to_check,
        "mdf_variable": mdf_variable,
        "base_value": base_target_value,
        "target_mode": target_mode,
        "target_value": target_value,
        "target_threshold": target_threshold,
        "target_constraint_sign": target_constraint_sign,
        "topology_lock_mode": resolved_topology_lock_mode,
        "minimal_relaxation_count": minimal_relaxation_count if minimal_relaxation_count is not None else 0,
        "minimal_bottleneck_sets": minimal_sets,
        "core_reactions": core_reactions,
        "optional_reactions": optional_reactions,
        "enumeration_requested": bool(enumerate_all_minimal_sets),
        "enumerated_set_count": len(minimal_sets),
        "enumeration_stopped_early": stopped_early,
        "termination_message": termination_message,
    }


def run_driving_force_variability(
    solver: BiologicalSolver,
    base_solution: SolverResult | dict[str, Any],
    constraints: list | None,
    target_reaction_ids: list[str] = None,
    subnetwork_ids: list[str] = None,
    thermo_unconstrained_reactions: list[str] = None,
) -> dict[str, any]:
    """Verify true bottlenecks by maximizing per-reaction driving forces.

    Starting from an optimal MDF solution, this helper fixes the optimal MDF
    and then asks how much each candidate reaction's driving force can be
    increased. Reactions whose maximal force stays near the optimal MDF are
    reported as true bottlenecks.

    Args:
        solver: Solver with an already loaded thermo scenario.
        base_solution: Optimal MDF reference solution.
        constraints: Additional constraints defining the feasible state that
            should be preserved.
        target_reaction_ids: Optional explicit candidate set. When omitted,
            candidates are inferred from the active solution.
        subnetwork_ids: Optional subnetwork restriction forwarded to the
            solver.
        thermo_unconstrained_reactions: Optional list of reactions that should
            be excluded from automatic candidate generation.

    Returns:
        dict[str, Any]: A payload containing the sorted list of
        ``true_bottlenecks`` and the per-reaction ``max_driving_forces``, or
        an ``error`` entry when the baseline solve is invalid.
    """
    base_solution_dict = _solution_to_dict(base_solution)
    mdf_var = "var_B2" if subnetwork_ids else "var_B"

    if base_solution_dict.get("status") != "Optimal":
        return {"error": "Base solution not optimal"}

    optimal_mdf = base_solution_dict["values"].get(mdf_var)
    if optimal_mdf is None:
        return {"error": "MDF variable not found"}

    z_var_constraints, active_reactions_in_solution, _ = z_var_rea_constraints(base_solution_dict)

    candidates = []
    if target_reaction_ids:
        active_set = set(active_reactions_in_solution)
        candidates = [rid for rid in target_reaction_ids if rid in active_set]
    else:
        for rid in active_reactions_in_solution:
            base_id = rid.replace("_FWD", "").replace("_REV", "")
            if thermo_unconstrained_reactions and base_id in thermo_unconstrained_reactions:
                continue
            val = base_solution_dict["values"].get(f"f_var_{rid}")
            if val is not None and abs(val - optimal_mdf) < 1e-4:
                candidates.append(rid)

    if not candidates:
        return {"true_bottlenecks": [], "slack_values": {}}

    verification_constraints = (constraints or []) + z_var_constraints
    verification_constraints.append([mdf_var, optimal_mdf - 1e-5, "fix_optimal_mdf", "geq"])

    true_bottlenecks = []
    slack_results = {}

    for rid in candidates:
        sol = solver.solve(
            objective_var_ids=[f"f_var_{rid}"],
            objective_direction="max",
            objective_multipliers=[1.0],
            constraints=verification_constraints,
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            warm_start_solution=base_solution_dict,
        )

        if sol.status == "Optimal":
            max_df = sol.values.get(f"f_var_{rid}")
            slack_results[rid] = max_df
            if max_df <= optimal_mdf + 1e-3:
                true_bottlenecks.append(rid)

    return {
        "true_bottlenecks": sorted(true_bottlenecks),
        "max_driving_forces": slack_results,
    }


def run_metabolite_bottleneck_analysis(
    solver: BiologicalSolver,
    base_solution: SolverResult | dict[str, Any],
    constraints: list | None,
    relax_factor: float,
    test_active_metabolites_only: bool = False,
    subnetwork_ids: list[str] = None,
    thermo_unconstrained_reactions: list[str] = None,
) -> list[str]:
    """Identify metabolite bottlenecks by relaxing and re-tightening ranges.

    The method first solves a relaxed concentration scenario, then tightens one
    metabolite at a time back to its original range. Metabolites whose
    tightening lowers the optimized MDF-like objective are reported as
    bottlenecks.

    Args:
        solver: Solver with an already loaded thermo scenario.
        base_solution: Optimal reference solution used for topology-preserving
            constraints and warm starts.
        constraints: Additional constraints defining the feasible state that
            the bottleneck analysis should preserve.
        relax_factor: Relative factor used to widen concentration ranges in the
            relaxed baseline solve.
        test_active_metabolites_only: Whether to test only metabolites touched
            by active reactions instead of the full metabolite set.
        subnetwork_ids: Optional subnetwork restriction forwarded to the
            solver.
        thermo_unconstrained_reactions: Optional thermo exclusions forwarded to
            the solver.

    Returns:
        list[str]: Metabolite ids identified as bottlenecks, or an error marker
        when the relaxed comparison solve fails.
    """
    scenario = _scenario_from_solver(solver)
    base_solution_dict = _solution_to_dict(base_solution)

    bottleneck_metabolites = []
    standard_concentrations = copy.deepcopy(scenario.concentrations)

    if relax_factor is None:
        logging.error("Metabolite bottleneck analysis requires relax_factor.")
        return ["Error: relax_factor not configured"]

    multiplier = 1.0 + relax_factor

    def create_relaxed_concentrations(base_conc, mult):
        relaxed = copy.deepcopy(base_conc)
        for met_id, bounds in relaxed.items():
            if bounds["min"] >= bounds["max"]:
                continue
            new_min = bounds["min"] / mult
            new_max = bounds["max"] * mult
            relaxed[met_id] = {"min": max(1e-12, new_min), "max": new_max}
        return relaxed

    relaxed_concentrations = create_relaxed_concentrations(standard_concentrations, multiplier)
    objective_var = "var_B2" if subnetwork_ids else "var_B"

    settings = scenario.model_conf
    z_var_constraints, active_reactions = [], []
    if getattr(settings, "exact", False):
        z_var_constraints, active_reactions, _ = z_var_rea_constraints(base_solution_dict)
    elif get_exclusive_reaction_groups(settings):
        z_var_constraints, active_reactions, _ = z_var_variant_and_original_rea_constraints(base_solution_dict)
    full_constraints = (constraints or []) + z_var_constraints

    if test_active_metabolites_only:
        active_metabolites = set()
        for rea_id in active_reactions:
            try:
                reaction = scenario.model.reactions.get_by_id(rea_id)
                active_metabolites.update(met.id for met in reaction.metabolites)
            except KeyError:
                pass
        metabolites_to_test = sorted(list(active_metabolites))
    else:
        metabolites_to_test = [met.id for met in scenario.model.metabolites]

    logger.info("  [Metabolite Bottlenecks] scope=%s, metabolites=%d, relax_factor=%s, objective=%s.", "active" if test_active_metabolites_only else "all", len(metabolites_to_test), relax_factor, objective_var)

    relaxed_solver = copy.deepcopy(solver)
    relaxed_solver.biological_scenario = copy.deepcopy(scenario)
    relaxed_solver.biological_scenario.concentrations = relaxed_concentrations
    sol_relaxed = relaxed_solver.solve(
        objective_var_ids=[objective_var],
        objective_direction="max",
        objective_multipliers=[1],
        constraints=full_constraints,
        subnetwork_ids=subnetwork_ids,
        thermo_unconstrained_reactions=thermo_unconstrained_reactions,
        warm_start_solution=base_solution_dict,
    )

    if sol_relaxed.status != "Optimal":
        logger.warning("  [Metabolite Bottlenecks] relaxed solve failed for scenario='%s' with status='%s'.", scenario.scenario_id, sol_relaxed.status)
        return [f"Relaxed solve failed: {sol_relaxed.status}"]

    mdf_relaxed = sol_relaxed.values[objective_var]

    total_metabolites = len(metabolites_to_test)
    for metabolite_index, met_id in enumerate(metabolites_to_test, start=1):
        standard_bounds = standard_concentrations.get(met_id, standard_concentrations["DEFAULT"])
        temp_conc = copy.deepcopy(relaxed_concentrations)
        temp_conc[met_id] = standard_bounds

        temp_solver = copy.deepcopy(solver)
        temp_solver.biological_scenario = copy.deepcopy(scenario)
        temp_solver.biological_scenario.concentrations = temp_conc
        logger.info("  [Metabolite Bottlenecks] Testing metabolite %d/%d: %s", metabolite_index, total_metabolites, met_id)
        sol_tightened = temp_solver.solve(
            objective_var_ids=[objective_var],
            objective_direction="max",
            objective_multipliers=[1],
            constraints=full_constraints,
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            warm_start_solution=base_solution_dict,
        )

        if sol_tightened.status == "Optimal":
            mdf_tightened = sol_tightened.values[objective_var]
            if mdf_tightened < mdf_relaxed - 1e-6:
                bottleneck_metabolites.append(met_id)
                logger.info("  [Metabolite Bottlenecks] MDF improved for metabolite %s: %.6g -> %.6g (delta=%.6g)", met_id, mdf_tightened, mdf_relaxed, mdf_relaxed - mdf_tightened)

    logger.info("  [Metabolite Bottlenecks] scenario='%s', relaxed_%s=%.4g, bottlenecks=%d/%d.", scenario.scenario_id, objective_var, mdf_relaxed, len(bottleneck_metabolites), len(metabolites_to_test))
    return bottleneck_metabolites
