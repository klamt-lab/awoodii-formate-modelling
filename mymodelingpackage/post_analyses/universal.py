import logging
from typing import Any

from mymodelingpackage.solvers.interface import BiologicalSolver
from mymodelingpackage.utilities import generate_pathway_constraints, z_var_rea_constraints

logger = logging.getLogger(__name__)



def run_fva(
    solver: BiologicalSolver,
    reactions_to_check: list[str],
    constraints: list[list[Any]] | None = None,
    base_solution: dict[str, Any] | None = None,
    apply_specificity_constraints: bool = False,
    exact_topology: bool = False,
    subnetwork_ids: list[str] | None = None,
    thermo_unconstrained_reactions: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Perform Flux Variability Analysis (FVA) for a loaded solver context.

    This helper re-solves the current scenario for each target reaction twice:
    once with the reaction objective minimized and once maximized. It can also
    reconstruct topology constraints from a previous base solution.

    Args:
        solver:
            Loaded biological solver. The solver must already contain a
            scenario and analysis context.
        reactions_to_check:
            Reaction ids to analyze.
        constraints:
            Additional numeric constraints that define the solved state to
            preserve during FVA.
        base_solution:
            Optional dumped solution used to derive specificity or exact
            topology constraints.
        apply_specificity_constraints:
            If True, uses base_solution to enforce exclusive-group specificity logic.
        exact_topology:
            If True, fixes binary variables (z_var) to the base_solution values.
        subnetwork_ids:
            Optional per-call subnetwork override forwarded into
            ``solver.solve(...)``. If omitted, the solver's loaded analysis
            config is used.
        thermo_unconstrained_reactions:
            Optional per-call override forwarded into ``solver.solve(...)``.

    Returns:
        dict[str, dict[str, Any]]: Mapping of reaction id to min/max fluxes.

    Example:
        ``run_fva(solver, ["ATPM"], constraints=[["var_B", 0.1, "min_mdf", "geq"]])``
        returns a structure like ``{"ATPM": {"min": 2.0, "max": 10.0}}``.
    """

    # PHASE 0: Validate the loaded solver context.
    if not solver.biological_scenario:
        raise RuntimeError("Engine has no loaded scenario. Cannot run FVA.")

    # PHASE 1: Build optional topology-preserving constraints from the base
    # solution. These are appended to the user-provided numeric constraints.
    z_var_constraints: list[list[Any]] = []

    if exact_topology:
        if base_solution:
            z_var_constraints, _, _ = z_var_rea_constraints(base_solution)
            logger.info("  [FVA] Applied exact z_var constraints from base solution.")
        else:
            logger.info("  [FVA] exact=True but no base_solution provided; skipping z_var constraints.")

    elif solver.biological_scenario.model_conf.exclusive_reaction_groups:
        if apply_specificity_constraints and base_solution:
            z_var_constraints = generate_pathway_constraints(base_solution, solver.biological_scenario.model_conf)
            logger.info("  [FVA] Applied specificity constraints.")
        elif apply_specificity_constraints and not base_solution:
            logger.info("  [FVA] Specificity requested but no base_solution provided; skipping.")
        else:
            logger.info("  [FVA] Mode: UNCONSTRAINED (specificity constraints skipped).")

    # PHASE 2: Combine user constraints with the topology-specific additions.
    full_constraints = (constraints or []) + z_var_constraints

    # PHASE 3A: Prefer a solver-native variability implementation when one is
    # advertised. COBRA-k constructs exact net-flux watches and solves the
    # formulation directly instead of rebuilding one model per min/max pair.
    if hasattr(solver, "supports") and solver.supports("native_variability") and hasattr(solver, "native_variability"):
        logger.info("  [FVA] Dispatching %d reaction(s) to solver-native variability with %d constraint(s).", len(reactions_to_check), len(full_constraints))
        native_results = solver.native_variability(reactions_to_check, constraints=full_constraints)
        normalized_native_results = {reaction_id: {"min": round(float(bounds[0]), 6), "max": round(float(bounds[1]), 6)} for reaction_id, bounds in native_results.items()}
        logger.debug("  [FVA] Solver-native variability returned: %s", normalized_native_results)
        return normalized_native_results

    # PHASE 3B: Generic fallback. Solve a min/max pair for every requested
    # reaction through the common BiologicalSolver interface.
    fva_results = {}

    for rid in reactions_to_check:
        fva_results[rid] = {}

        if rid not in solver.biological_scenario.model.reactions:
            logger.warning("  [FVA] Reaction '%s' not found in model.", rid)
            fva_results[rid] = {"min": "not_in_model", "max": "not_in_model"}
            continue

        for direction in ["min", "max"]:
            logger.debug("  [FVA] Solving %s (%s)", rid, direction)

            result = solver.solve(
                objective_var_ids=[rid],
                objective_direction=direction,
                constraints=full_constraints,
                warm_start_solution=base_solution,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            )

            if result.status == "Optimal":
                val = result.values.get(rid, 0.0)
                z_val = result.values.get(f"z_var_{rid}")

                if val is None:
                    fva_results[rid][direction] = "value_not_found"
                elif z_val is None or z_val > 0.5:
                    fva_results[rid][direction] = round(float(val), 6)
                else:
                    fva_results[rid][direction] = 0.0
            else:
                fva_results[rid][direction] = result.status
                logger.debug(f"  [FVA] {rid} {direction}.")

    return fva_results


def run_variable_optimization(
    solver: BiologicalSolver,
    variables_to_check: list[str],
    objective_direction: str,
    constraints: list[list[Any]] | None = None,
    base_solution: dict[str, Any] | None = None,
    apply_specificity_constraints: bool = False,
    exact_topology: bool = False,
    subnetwork_ids: list[str] | None = None,
    thermo_unconstrained_reactions: list[str] | None = None,
) -> dict[str, Any]:
    """Optimize configured solver variables under a locked post-analysis state.

    This helper re-solves the currently loaded scenario for each configured
    target variable while preserving the constraints that define the parent
    solution. Targets are intentionally limited to model reactions, meaning
    flux variables, plus the MDF variable ``var_B``.

    Args:
        solver:
            Loaded biological solver. The solver must already contain a
            scenario and analysis context.
        variables_to_check:
            Solver variable ids to optimize. Accepted targets are model
            reaction ids and ``var_B``.
        objective_direction:
            Optimization direction, either ``"min"`` or ``"max"``.
        constraints:
            Additional numeric constraints that define the solved state to
            preserve during this optimization.
        base_solution:
            Optional dumped solution used to derive specificity or exact
            topology constraints and to warm-start the solver.
        apply_specificity_constraints:
            If True, uses ``base_solution`` to enforce exclusive-group
            specificity logic.
        exact_topology:
            If True, fixes binary variables (z_var) to the base solution
            values.
        subnetwork_ids:
            Optional per-call subnetwork override forwarded into
            ``solver.solve(...)``.
        thermo_unconstrained_reactions:
            Optional per-call thermo exclusion override forwarded into
            ``solver.solve(...)``.

    Returns:
        Mapping of variable id to the optimized value or a status marker such
        as ``"not_allowed"`` or the solver status.

    Raises:
        ValueError:
            If ``objective_direction`` is not ``"min"`` or ``"max"``.
        RuntimeError:
            If the solver has no loaded biological scenario.
    """
    # PHASE 0: Validate the requested direction and loaded solver context.
    if objective_direction not in {"min", "max"}:
        raise ValueError(f"objective_direction must be 'min' or 'max', got {objective_direction!r}.")

    if not solver.biological_scenario:
        raise RuntimeError("Engine has no loaded scenario. Cannot run variable optimization.")

    # PHASE 1: Build optional topology-preserving constraints from the base
    # solution. These mirror the FVA behavior so both post-analysis types can
    # preserve pathway specificity in the same way.
    z_var_constraints: list[list[Any]] = []
    if exact_topology:
        if base_solution:
            z_var_constraints, _, _ = z_var_rea_constraints(base_solution)
            logger.info("  [VarOpt] Applied exact z_var constraints from base solution.")
        else:
            logger.info("  [VarOpt] exact=True but no base_solution provided; skipping z_var constraints.")
    elif solver.biological_scenario.model_conf.exclusive_reaction_groups:
        if apply_specificity_constraints and base_solution:
            z_var_constraints = generate_pathway_constraints(base_solution, solver.biological_scenario.model_conf)
            logger.info("  [VarOpt] Applied specificity constraints.")
        elif apply_specificity_constraints and not base_solution:
            logger.info("  [VarOpt] Specificity requested but no base_solution provided; skipping.")
        else:
            logger.info("  [VarOpt] Mode: UNCONSTRAINED (specificity constraints skipped).")

    # PHASE 2: Combine user constraints with topology-specific additions.
    full_constraints = (constraints or []) + z_var_constraints

    # PHASE 3: Build the target allowlist. Flux variables are identified from
    # model reactions; the only allowed internal thermo variable is var_B.
    model_reaction_ids = {reaction.id for reaction in solver.biological_scenario.model.reactions}
    allowed_internal_vars = {"var_B"}
    optimization_results: dict[str, Any] = {}

    # PHASE 4: Solve each requested target and store either its optimized value
    # or an explicit status marker for reporting.
    for variable_id in variables_to_check:
        if variable_id not in model_reaction_ids and variable_id not in allowed_internal_vars:
            logger.warning("  [VarOpt] Variable '%s' is neither a model reaction nor one of %s.", variable_id, sorted(allowed_internal_vars))
            optimization_results[variable_id] = "not_allowed"
            continue

        logger.debug("  [VarOpt] Solving %s (%s)", variable_id, objective_direction)
        result = solver.solve(
            objective_var_ids=[variable_id],
            objective_direction=objective_direction,
            constraints=full_constraints,
            warm_start_solution=base_solution,
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
        )

        if result.status == "Optimal":
            value = result.values.get(variable_id, result.objective_value)
            optimization_results[variable_id] = round(float(value), 6) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        else:
            optimization_results[variable_id] = result.status
            logger.debug("  [VarOpt] %s %s returned status=%s.", variable_id, objective_direction, result.status)

    return optimization_results


def run_min_optimization(
    solver: BiologicalSolver,
    variables_to_check: list[str],
    constraints: list[list[Any]] | None = None,
    base_solution: dict[str, Any] | None = None,
    apply_specificity_constraints: bool = False,
    exact_topology: bool = False,
    subnetwork_ids: list[str] | None = None,
    thermo_unconstrained_reactions: list[str] | None = None,
) -> dict[str, Any]:
    """Minimize configured flux variables or ``var_B`` under a locked state.

    Args:
        solver:
            Loaded biological solver.
        variables_to_check:
            Model reaction ids or ``var_B`` to minimize.
        constraints:
            Additional state-preserving constraints.
        base_solution:
            Optional dumped solution used for warm-starting and topology locks.
        apply_specificity_constraints:
            Whether to preserve exclusive-group specificity from the base
            solution.
        exact_topology:
            Whether to fix z_var topology exactly from the base solution.
        subnetwork_ids:
            Optional per-call subnetwork override.
        thermo_unconstrained_reactions:
            Optional per-call thermo exclusion override.

    Returns:
        Mapping of variable id to minimized value or status marker.
    """
    return run_variable_optimization(
        solver=solver,
        variables_to_check=variables_to_check,
        objective_direction="min",
        constraints=constraints,
        base_solution=base_solution,
        apply_specificity_constraints=apply_specificity_constraints,
        exact_topology=exact_topology,
        subnetwork_ids=subnetwork_ids,
        thermo_unconstrained_reactions=thermo_unconstrained_reactions,
    )


def run_max_optimization(
    solver: BiologicalSolver,
    variables_to_check: list[str],
    constraints: list[list[Any]] | None = None,
    base_solution: dict[str, Any] | None = None,
    apply_specificity_constraints: bool = False,
    exact_topology: bool = False,
    subnetwork_ids: list[str] | None = None,
    thermo_unconstrained_reactions: list[str] | None = None,
) -> dict[str, Any]:
    """Maximize configured flux variables or ``var_B`` under a locked state.

    Args:
        solver:
            Loaded biological solver.
        variables_to_check:
            Model reaction ids or ``var_B`` to maximize.
        constraints:
            Additional state-preserving constraints.
        base_solution:
            Optional dumped solution used for warm-starting and topology locks.
        apply_specificity_constraints:
            Whether to preserve exclusive-group specificity from the base
            solution.
        exact_topology:
            Whether to fix z_var topology exactly from the base solution.
        subnetwork_ids:
            Optional per-call subnetwork override.
        thermo_unconstrained_reactions:
            Optional per-call thermo exclusion override.

    Returns:
        Mapping of variable id to maximized value or status marker.
    """
    return run_variable_optimization(
        solver=solver,
        variables_to_check=variables_to_check,
        objective_direction="max",
        constraints=constraints,
        base_solution=base_solution,
        apply_specificity_constraints=apply_specificity_constraints,
        exact_topology=exact_topology,
        subnetwork_ids=subnetwork_ids,
        thermo_unconstrained_reactions=thermo_unconstrained_reactions,
    )
