from __future__ import annotations

import logging
from typing import Any, Literal

import cobra

from mymodelingpackage.structures import BiologicalScenario, SolverResult, StoichiometricSolverConfig


logger = logging.getLogger(__name__)


class StoichiometricSolver:
    """Lightweight cobrapy LP solver wrapper for stoichiometric workflows."""

    def __init__(self, solver_config: StoichiometricSolverConfig) -> None:
        self.solver_config = solver_config
        self.config = solver_config
        self.interface_name = solver_config.interface_name
        self.solver_name = solver_config.solver_name
        self.biological_scenario: BiologicalScenario | None = None
        self.analysis_config: Any | None = None
        logger.debug("Initialized StoichiometricSolver with interface='%s' and backend='%s'.", self.interface_name, self.solver_name)

    def load_context(self, biological_scenario: BiologicalScenario, analysis_config: Any) -> None:
        self.biological_scenario = biological_scenario
        self.analysis_config = analysis_config
        logger.debug("StoichiometricSolver loaded scenario='%s', model='%s', analysis_type='%s'.", biological_scenario.scenario_id, biological_scenario.model.id, getattr(analysis_config, "type", type(analysis_config).__name__))

    def supports(self, capability: str) -> bool:
        logger.debug("StoichiometricSolver capability query: capability='%s' -> False", capability)
        return False

    def solve(
        self,
        objective_var_ids: list[str],
        objective_multipliers: list[float] | None = None,
        objective_direction: Literal["max", "min"] = "max",
        constraints: list[list[Any]] | None = None,
        warm_start_solution: dict[str, Any] | None = None,
        subnetwork_ids: list[str] | None = None,
        thermo_unconstrained_reactions: list[str] | None = None,
    ) -> SolverResult:
        # Stoichiometric runs are intentionally strict: the solver must have a
        # loaded biological context, and unsupported thermo-style overrides
        # must fail loudly instead of being ignored.
        if self.biological_scenario is None:
            raise RuntimeError("StoichiometricSolver has no loaded scenario. Call load_context() first.")
        if self.analysis_config is None:
            raise RuntimeError("StoichiometricSolver has no loaded analysis config. Call load_context() first.")

        unsupported_arguments: list[str] = []
        if warm_start_solution is not None:
            unsupported_arguments.append("warm_start_solution")
        if subnetwork_ids is not None:
            unsupported_arguments.append("subnetwork_ids")
        if thermo_unconstrained_reactions is not None:
            unsupported_arguments.append("thermo_unconstrained_reactions")
        if unsupported_arguments:
            unsupported_values = {"warm_start_solution": warm_start_solution, "subnetwork_ids": subnetwork_ids, "thermo_unconstrained_reactions": thermo_unconstrained_reactions}
            message = f"StoichiometricSolver does not support per-call override(s): {unsupported_arguments}. Received values: { {key: unsupported_values[key] for key in unsupported_arguments} }"
            logger.error(message)
            raise ValueError(message)

        # Validate the optimization request before touching the cobra model so
        # configuration errors surface as simple argument errors.
        if not objective_var_ids:
            raise ValueError("StoichiometricSolver.solve requires at least one objective variable id.")
        if objective_direction not in {"max", "min"}:
            raise ValueError(f"Unsupported objective_direction '{objective_direction}'. Expected 'max' or 'min'.")

        objective_multipliers = list(objective_multipliers or [1.0] * len(objective_var_ids))
        if len(objective_multipliers) != len(objective_var_ids):
            raise ValueError("Length mismatch between objective_var_ids and objective_multipliers.")

        logger.info("Stoichiometric solve start: scenario='%s', backend='%s', objective=%s, direction='%s', constraints=%d.", self.biological_scenario.scenario_id, self.solver_name, objective_var_ids, objective_direction, len(constraints or []))
        logger.debug("Stoichiometric solve details: interface='%s', multipliers=%s, analysis_type='%s'.", self.interface_name, objective_multipliers, getattr(self.analysis_config, "type", type(self.analysis_config).__name__))

        with self.biological_scenario.model as model:
            try:
                logger.debug("Switching cobra model solver backend to '%s'.", self.solver_name)
                model.solver = self.solver_name
            except Exception as exc:
                message = f"Could not switch model solver to '{self.solver_name}': {exc}"
                logger.error(message)
                raise ValueError(message) from exc

            # Build the linear objective directly from reaction flux
            # expressions so multi-objective weighted sums remain explicit.
            objective_expression = 0
            for reaction_id, multiplier in zip(objective_var_ids, objective_multipliers):
                logger.debug("Adding objective term: reaction_id='%s', multiplier=%s.", reaction_id, multiplier)
                objective_expression += float(multiplier) * model.reactions.get_by_id(reaction_id).flux_expression
            model.objective = objective_expression
            model.objective_direction = objective_direction
            logger.debug("Objective configured successfully for %d reaction term(s).", len(objective_var_ids))

            # Translate each normalized campaign constraint into a cobra
            # linear expression plus the correct bound semantics.
            for idx, constraint in enumerate(constraints or []):
                if not (isinstance(constraint, list) and len(constraint) >= 4):
                    raise ValueError(f"Unsupported constraint format: {constraint!r}")
                lhs, target_value, constraint_name, relation = constraint[:4]
                logger.debug("Applying stoichiometric constraint %d: name='%s', relation='%s', target=%s, lhs=%s.", idx, constraint_name, relation, target_value, lhs)
                if isinstance(lhs, str):
                    expression = model.reactions.get_by_id(lhs).flux_expression
                elif isinstance(lhs, list):
                    expression = 0
                    for item in lhs:
                        if not (isinstance(item, list) and len(item) >= 2):
                            raise ValueError(f"Unsupported linear-expression constraint item: {item!r}")
                        reaction_id = str(item[0])
                        coefficient = float(item[1])
                        expression += coefficient * model.reactions.get_by_id(reaction_id).flux_expression
                else:
                    raise ValueError(f"Unsupported constraint lhs type: {type(lhs).__name__}")

                kwargs: dict[str, float] = {}
                if relation == "eq":
                    kwargs["lb"] = float(target_value)
                    kwargs["ub"] = float(target_value)
                elif relation == "geq":
                    kwargs["lb"] = float(target_value)
                elif relation == "leq":
                    kwargs["ub"] = float(target_value)
                elif relation == "range":
                    lower_bound, upper_bound = target_value
                    kwargs["lb"] = float(lower_bound)
                    kwargs["ub"] = float(upper_bound)
                else:
                    raise ValueError(f"Unsupported constraint relation '{relation}' in {constraint!r}")

                model.add_cons_vars(model.problem.Constraint(expression, name=f"{constraint_name}_{idx}", **kwargs))

            # Let cobra execute the LP once the objective and every requested
            # linear constraint have been materialized on the copied model.
            logger.debug("Calling cobra optimize() with backend='%s'.", self.solver_name)
            solution = model.optimize()
            status = (solution.status or "").lower()
            mapped_status = "Optimal" if status == "optimal" else (solution.status or "Unknown")
            objective_value = float(solution.objective_value) if solution.objective_value is not None else None
            values = {reaction_id: float(value) for reaction_id, value in solution.fluxes.to_dict().items()}
            logger.info("Stoichiometric solve finished: scenario='%s', status='%s', objective_value=%s, n_fluxes=%d.", self.biological_scenario.scenario_id, mapped_status, objective_value, len(values))
            logger.debug("Stoichiometric solve raw status='%s'. Returning normalized status='%s'.", solution.status, mapped_status)

            return SolverResult(status=mapped_status, objective_value=objective_value, values=values)
