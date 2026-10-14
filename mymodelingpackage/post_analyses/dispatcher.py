import logging
import random
from pathlib import Path
from typing import Any

from mymodelingpackage.structures import SolverResult, AnalysisParamsConfig
from mymodelingpackage.solvers.interface import BiologicalSolver

from mymodelingpackage.post_analyses.universal import (
    run_fva,
    run_variable_optimization,
)
from mymodelingpackage.post_analyses.thermo_specific import (
    run_bottleneck_analysis,
    run_bottleneck_set_analysis,
    run_driving_force_variability,
    run_metabolite_bottleneck_analysis,
    run_metabolite_ratios,
    run_cva,
)
from mymodelingpackage.utilities import resolve_model_reactions_from_escher_map

logger = logging.getLogger(__name__)


class PostAnalysisDispatcher:
    """Dispatch configured post-analyses for one solved experiment step.

    The dispatcher is intentionally small and declarative: experiment classes
    provide the solved state and the shared analysis configuration, while this
    class decides which post-analyses should run and collects their outputs.

    The analyses are grouped into two categories:

    - Universal analyses:
      CVA, FVA, and metabolite-ratio checks can run for any suitable solution.
    - Thermo + MDF-specific analyses:
      bottleneck-style checks only run when the solver reports thermo support
      and at least one active objective is MDF-like (``var_B`` or ``var_B2``).
    """

    def _perform_cva(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
        apply_specificity: bool,
    ) -> dict[str, dict[str, Any]]:
        """Run adaptive CVA sampling for the configured metabolite targets.

        Args:
            solver: Loaded solver instance used for all CVA solves.
            base_solution: Optimal parent solution that anchors the first CVA attempt.
            analysis_params: Parsed analysis config containing the CVA settings.
            constraints: Additional constraints that define the solved base state.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for CVA solves.
            apply_specificity: Whether pathway specificity constraints should be applied.

        Returns:
            Mapping from metabolite id to aggregated ``min`` and ``max`` values.
        """
        # Build the CVA target list and the min/max accumulator that collects
        # successful samples across adaptive multi-start attempts.
        metabolites_to_check = list(analysis_params.run_cva.metabolites_to_check)
        accumulator = {metabolite_id: {"mins": [], "maxs": []} for metabolite_id in metabolites_to_check}
        logger.info("  [CVA] Starting adaptive CVA for %d metabolite(s).", len(metabolites_to_check))

        # Re-run CVA until every metabolite has enough min/max samples or the
        # configured attempt budget is exhausted.
        for attempt in range(1, analysis_params.run_cva.max_attempts + 1):
            pending_metabolites = [
                metabolite_id
                for metabolite_id in metabolites_to_check
                if len(accumulator[metabolite_id]["mins"]) < analysis_params.run_cva.target_successes
                or len(accumulator[metabolite_id]["maxs"]) < analysis_params.run_cva.target_successes
            ]
            if not pending_metabolites:
                logger.info("  [CVA] All metabolites reached the sample goal after %d attempt(s).", attempt - 1)
                break

            current_seed = 12345 if attempt == 1 else random.randint(1, 50000)
            warm_start_solution: SolverResult | dict[str, Any] | None = base_solution if attempt == 1 else None
            attempt_type = "ANCHOR" if attempt == 1 else "EXPLORATION"
            cplex_params = getattr(getattr(solver, "solver_config", None), "cplex_params", None)
            if isinstance(cplex_params, dict):
                cplex_params["randomseed"] = current_seed
            logger.debug("  [CVA] Attempt %d/%d (%s, seed=%d, pending=%d)", attempt, analysis_params.run_cva.max_attempts, attempt_type, current_seed, len(pending_metabolites))

            # Run one CVA batch for the currently unsatisfied metabolite set.
            batch_results = run_cva(
                solver=solver,
                base_solution=warm_start_solution,
                constraints=constraints,
                metabolites_to_check=pending_metabolites,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                apply_specificity_constraints=apply_specificity,
            )
            logger.debug("  [CVA] Attempt %d returned results for %d metabolite(s).", attempt, len(batch_results))

            # Store every numeric min/max returned by this attempt so the final
            # aggregate can widen over multiple feasible seeds.
            for metabolite_id in pending_metabolites:
                result = batch_results.get(metabolite_id)
                if not isinstance(result, dict):
                    continue
                value_min = result.get("min")
                value_max = result.get("max")
                if isinstance(value_min, (int, float)) and not isinstance(value_min, bool):
                    accumulator[metabolite_id]["mins"].append(float(value_min))
                if isinstance(value_max, (int, float)) and not isinstance(value_max, bool):
                    accumulator[metabolite_id]["maxs"].append(float(value_max))

        # Collapse the collected samples into one final min/max pair per
        # metabolite and mark permanently infeasible targets explicitly.
        aggregated_results: dict[str, dict[str, Any]] = {}
        for metabolite_id in metabolites_to_check:
            mins = accumulator[metabolite_id]["mins"]
            maxs = accumulator[metabolite_id]["maxs"]
            aggregated_results[metabolite_id] = {"min": min(mins) if mins else "Infeasible", "max": max(maxs) if maxs else "Infeasible"}
            if not mins and not maxs:
                logger.warning("  [CVA] %s remained infeasible in all attempts.", metabolite_id)
            elif not mins or not maxs:
                logger.warning("  [CVA] %s returned only partial samples (mins=%d, maxs=%d).", metabolite_id, len(mins), len(maxs))

        return aggregated_results

    def _perform_fva(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        resolved_fva_reactions: list[str],
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
        apply_specificity: bool,
    ) -> dict[str, dict[str, Any]]:
        """Run adaptive FVA sampling for the configured reaction targets.

        Args:
            solver: Loaded solver instance used for all FVA solves.
            base_solution: Optimal parent solution that anchors the first FVA attempt.
            analysis_params: Parsed analysis config containing the FVA settings.
            constraints: Additional constraints that define the solved base state.
            resolved_fva_reactions: Final list of FVA targets after map expansion.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for FVA solves.
            apply_specificity: Whether pathway specificity constraints should be applied.

        Returns:
            Mapping from reaction id to aggregated ``min`` and ``max`` values.
        """
        # Freeze the resolved FVA target set and prepare the multi-attempt
        # accumulator that stores successful min/max samples per reaction.
        fva_config = analysis_params.run_fva.model_copy(update={"reactions_to_check": resolved_fva_reactions})
        base_solution_dump = base_solution.model_dump()
        if hasattr(solver, "supports") and solver.supports("native_variability"):
            logger.info("  [FVA] Using solver-native variability for %d reaction(s).", len(fva_config.reactions_to_check))
            return run_fva(
                solver=solver,
                reactions_to_check=fva_config.reactions_to_check,
                constraints=constraints,
                base_solution=base_solution_dump,
                apply_specificity_constraints=apply_specificity,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            )
        accumulator = {reaction_id: {"mins": [], "maxs": []} for reaction_id in fva_config.reactions_to_check}
        logger.info("  [FVA] Starting adaptive FVA for %d reaction(s).", len(fva_config.reactions_to_check))

        # Re-run FVA until every reaction has enough min/max samples or the
        # configured attempt budget is exhausted.
        for attempt in range(1, fva_config.max_attempts + 1):
            pending_reactions = [
                reaction_id
                for reaction_id in fva_config.reactions_to_check
                if len(accumulator[reaction_id]["mins"]) < fva_config.target_successes
                or len(accumulator[reaction_id]["maxs"]) < fva_config.target_successes
            ]
            if not pending_reactions:
                logger.info("  [FVA] All reactions reached the sample goal after %d attempt(s).", attempt - 1)
                break

            current_seed = 12345 if attempt == 1 else random.randint(1, 100000)
            warm_start_solution = base_solution_dump if attempt == 1 else None
            attempt_type = "ANCHOR" if attempt == 1 else "EXPLORATION"
            cplex_params = getattr(getattr(solver, "solver_config", None), "cplex_params", None)
            if isinstance(cplex_params, dict):
                cplex_params["randomseed"] = current_seed
            logger.debug("  [FVA] Attempt %d/%d (%s, seed=%d, pending=%d)", attempt, fva_config.max_attempts, attempt_type, current_seed, len(pending_reactions))

            # Run one FVA batch for the currently unsatisfied reaction set.
            batch_results = run_fva(
                solver=solver,
                reactions_to_check=pending_reactions,
                constraints=constraints,
                base_solution=warm_start_solution,
                apply_specificity_constraints=apply_specificity,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            )
            logger.debug("  [FVA] Attempt %d returned results for %d reaction(s).", attempt, len(batch_results))

            # Store every numeric min/max returned by this attempt so the final
            # aggregate can widen over multiple feasible seeds.
            for reaction_id in pending_reactions:
                result = batch_results.get(reaction_id)
                if not isinstance(result, dict):
                    continue
                value_min = result.get("min")
                value_max = result.get("max")
                if isinstance(value_min, (int, float)) and not isinstance(value_min, bool):
                    accumulator[reaction_id]["mins"].append(float(value_min))
                if isinstance(value_max, (int, float)) and not isinstance(value_max, bool):
                    accumulator[reaction_id]["maxs"].append(float(value_max))

        # Collapse the collected samples into one final min/max pair per
        # reaction and mark permanently infeasible targets explicitly.
        aggregated_results: dict[str, dict[str, Any]] = {}
        for reaction_id in fva_config.reactions_to_check:
            mins = accumulator[reaction_id]["mins"]
            maxs = accumulator[reaction_id]["maxs"]
            aggregated_results[reaction_id] = {"min": min(mins) if mins else "Infeasible", "max": max(maxs) if maxs else "Infeasible"}
            if not mins and not maxs:
                logger.warning("  [FVA] %s remained infeasible in all attempts.", reaction_id)
            elif not mins or not maxs:
                logger.warning("  [FVA] %s returned only partial samples (mins=%d, maxs=%d).", reaction_id, len(mins), len(maxs))

        return aggregated_results

    def _perform_variable_optimization(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        optimization_config: Any,
        objective_direction: str,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
        apply_specificity: bool,
    ) -> dict[str, Any]:
        """Run adaptive locked-state min/max optimization for configured variables."""
        variables_to_check = list(optimization_config.variables_to_check)
        accumulator = {variable_id: [] for variable_id in variables_to_check}
        base_solution_dump = base_solution.model_dump()
        label = "MinOptimization" if objective_direction == "min" else "MaxOptimization"
        logger.info("  [%s] Starting adaptive %s optimization for %d variable(s).", label, objective_direction, len(variables_to_check))

        for attempt in range(1, optimization_config.max_attempts + 1):
            pending_variables = [
                variable_id
                for variable_id in variables_to_check
                if len(accumulator[variable_id]) < optimization_config.target_successes
            ]
            if not pending_variables:
                logger.info("  [%s] All variables reached the sample goal after %d attempt(s).", label, attempt - 1)
                break

            current_seed = 12345 if attempt == 1 else random.randint(1, 100000)
            warm_start_solution = base_solution_dump if attempt == 1 else None
            attempt_type = "ANCHOR" if attempt == 1 else "EXPLORATION"
            cplex_params = getattr(getattr(solver, "solver_config", None), "cplex_params", None)
            if isinstance(cplex_params, dict):
                cplex_params["randomseed"] = current_seed
            logger.debug("  [%s] Attempt %d/%d (%s, seed=%d, pending=%d)", label, attempt, optimization_config.max_attempts, attempt_type, current_seed, len(pending_variables))

            batch_results = run_variable_optimization(
                solver=solver,
                variables_to_check=pending_variables,
                objective_direction=objective_direction,
                constraints=constraints,
                base_solution=warm_start_solution,
                apply_specificity_constraints=apply_specificity,
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
            )
            logger.debug("  [%s] Attempt %d returned results for %d variable(s).", label, attempt, len(batch_results))

            for variable_id in pending_variables:
                value = batch_results.get(variable_id)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    accumulator[variable_id].append(float(value))
                elif value == "not_allowed":
                    accumulator[variable_id].append(value)

        aggregated_results: dict[str, Any] = {}
        for variable_id in variables_to_check:
            values = accumulator[variable_id]
            numeric_values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if numeric_values:
                aggregated_results[variable_id] = min(numeric_values) if objective_direction == "min" else max(numeric_values)
            elif "not_allowed" in values:
                aggregated_results[variable_id] = "not_allowed"
            else:
                aggregated_results[variable_id] = "Infeasible"
                logger.warning("  [%s] %s remained infeasible in all attempts.", label, variable_id)

        return aggregated_results

    def _perform_metabolite_ratios(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
        apply_specificity: bool,
    ) -> dict[str, dict[str, Any]]:
        """Run adaptive metabolite-ratio sampling for the configured ratios.

        Args:
            solver: Loaded solver instance used for all ratio solves.
            base_solution: Optimal parent solution that anchors the first ratio attempt.
            analysis_params: Parsed analysis config containing the ratio settings.
            constraints: Additional constraints that define the solved base state.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for ratio solves.
            apply_specificity: Whether pathway specificity constraints should be applied.

        Returns:
            Mapping from ratio name to aggregated ``min`` and ``max`` values.
        """
        # Normalize the configured ratio definitions into name-keyed lookups
        # and prepare the min/max accumulator for adaptive retries.
        ratio_definitions = [dict(ratio_definition) for ratio_definition in analysis_params.run_metabolite_ratios.ratios]
        ratio_names = [ratio_definition["name"] for ratio_definition in ratio_definitions if ratio_definition.get("name")]
        ratio_by_name = {ratio_definition["name"]: ratio_definition for ratio_definition in ratio_definitions if ratio_definition.get("name")}
        accumulator = {ratio_name: {"mins": [], "maxs": []} for ratio_name in ratio_names}
        logger.info("  [Ratios] Starting adaptive ratio analysis for %d ratio(s).", len(ratio_names))

        # Re-run ratio analysis until every ratio has enough min/max samples or
        # the configured attempt budget is exhausted.
        for attempt in range(1, analysis_params.run_metabolite_ratios.max_attempts + 1):
            pending_ratios = [
                ratio_name
                for ratio_name in ratio_names
                if len(accumulator[ratio_name]["mins"]) < analysis_params.run_metabolite_ratios.target_successes
                or len(accumulator[ratio_name]["maxs"]) < analysis_params.run_metabolite_ratios.target_successes
            ]
            if not pending_ratios:
                logger.info("  [Ratios] All ratios reached the sample goal after %d attempt(s).", attempt - 1)
                break

            current_seed = 12345 if attempt == 1 else random.randint(1, 50000)
            warm_start_solution: SolverResult | dict[str, Any] | None = base_solution if attempt == 1 else None
            attempt_type = "ANCHOR" if attempt == 1 else "EXPLORATION"
            cplex_params = getattr(getattr(solver, "solver_config", None), "cplex_params", None)
            if isinstance(cplex_params, dict):
                cplex_params["randomseed"] = current_seed
            logger.debug("  [Ratios] Attempt %d/%d (%s, seed=%d, pending=%d)", attempt, analysis_params.run_metabolite_ratios.max_attempts, attempt_type, current_seed, len(pending_ratios))

            # Run one ratio-analysis batch for the currently unsatisfied ratio set.
            batch_results = run_metabolite_ratios(
                solver=solver,
                base_solution=warm_start_solution,
                constraints=constraints,
                ratio_definitions=[ratio_by_name[ratio_name] for ratio_name in pending_ratios],
                subnetwork_ids=subnetwork_ids,
                thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                apply_specificity_constraints=apply_specificity,
            )
            logger.debug("  [Ratios] Attempt %d returned results for %d ratio(s).", attempt, len(batch_results))

            # Store every numeric min/max returned by this attempt so the final
            # aggregate can widen over multiple feasible seeds.
            for ratio_name in pending_ratios:
                result = batch_results.get(ratio_name)
                if not isinstance(result, dict):
                    continue
                value_min = result.get("min")
                value_max = result.get("max")
                if isinstance(value_min, (int, float)) and not isinstance(value_min, bool):
                    accumulator[ratio_name]["mins"].append(float(value_min))
                if isinstance(value_max, (int, float)) and not isinstance(value_max, bool):
                    accumulator[ratio_name]["maxs"].append(float(value_max))

        # Collapse the collected samples into one final min/max pair per ratio
        # and mark permanently infeasible targets explicitly.
        aggregated_results: dict[str, dict[str, Any]] = {}
        for ratio_name in ratio_names:
            mins = accumulator[ratio_name]["mins"]
            maxs = accumulator[ratio_name]["maxs"]
            aggregated_results[ratio_name] = {"min": min(mins) if mins else "Infeasible", "max": max(maxs) if maxs else "Infeasible"}
            if not mins and not maxs:
                logger.warning("  [Ratios] %s remained infeasible in all attempts.", ratio_name)
            elif not mins or not maxs:
                logger.warning("  [Ratios] %s returned only partial samples (mins=%d, maxs=%d).", ratio_name, len(mins), len(maxs))

        return aggregated_results

    def _perform_bottleneck_reactions(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
    ) -> list[str]:
        """Run bottleneck-reaction analysis for one thermo MDF solution.

        Args:
            solver: Loaded thermo-capable solver instance.
            base_solution: Optimal MDF-like base solution to perturb from.
            analysis_params: Parsed analysis config containing bottleneck settings.
            constraints: Additional constraints that define the solved base state.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for the analysis.

        Returns:
            List of bottleneck reaction ids detected by the low-level analysis.
        """
        # Delegate to the thermo-specific bottleneck helper with the settings
        # already normalized out of the shared analysis configuration.
        logger.info("  [Bottlenecks] Running reaction bottleneck analysis.")
        return run_bottleneck_analysis(
            solver=solver,
            base_solution=base_solution,
            constraints=constraints,
            dG0_change=analysis_params.run_bottleneck_analysis.dG0_change,
            test_active_reactions_only=bool(analysis_params.run_bottleneck_analysis.test_active_reactions_only),
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
        )

    def _perform_true_bottleneck_check(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
    ) -> dict[str, Any]:
        """Run true-bottleneck variability analysis for one thermo MDF solution.

        Args:
            solver: Loaded thermo-capable solver instance.
            base_solution: Optimal MDF-like base solution to analyze.
            analysis_params: Parsed analysis config containing target reactions.
            constraints: Additional constraints that define the solved base state.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for the analysis.

        Returns:
            Result payload returned by the driving-force variability helper.
        """
        # Delegate to the thermo-specific driving-force variability helper with
        # the target reaction ids already resolved from config.
        logger.info("  [True Bottlenecks] Running driving-force variability analysis.")
        return run_driving_force_variability(
            solver=solver,
            base_solution=base_solution,
            constraints=constraints,
            target_reaction_ids=analysis_params.run_true_bottleneck_check.target_reaction_ids,
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
        )

    def _perform_bottleneck_sets(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
    ) -> dict[str, Any]:
        """Run MILP-based bottleneck-set analysis for one thermo MDF solution.

        Args:
            solver: Loaded thermo-capable solver instance.
            base_solution: Optimal MDF-like base solution to analyze.
            analysis_params: Parsed analysis config containing bottleneck-set settings.
            constraints: Additional constraints that define the solved base state.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for the analysis.

        Returns:
            Result payload returned by the bottleneck-set helper.
        """
        logger.info("  [Bottleneck Sets] Running MILP-based bottleneck-set analysis.")
        return run_bottleneck_set_analysis(
            solver=solver,
            base_solution=base_solution,
            constraints=constraints,
            target_variable=analysis_params.run_bottleneck_set_analysis.target_variable,
            target_mode=analysis_params.run_bottleneck_set_analysis.target_mode,
            target_value=analysis_params.run_bottleneck_set_analysis.target_value,
            target_constraint_sign=analysis_params.run_bottleneck_set_analysis.target_constraint_sign,
            topology_lock_mode=analysis_params.run_bottleneck_set_analysis.topology_lock_mode,
            enumerate_all_minimal_sets=bool(analysis_params.run_bottleneck_set_analysis.enumerate_all_minimal_sets),
            max_sets=analysis_params.run_bottleneck_set_analysis.max_sets,
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
        )

    def _perform_metabolite_bottlenecks(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None,
        subnetwork_ids: list[str] | None,
        thermo_unconstrained_reactions: list[str] | None,
    ) -> dict[str, Any]:
        """Run metabolite-bottleneck analysis for one thermo MDF solution.

        Args:
            solver: Loaded thermo-capable solver instance.
            base_solution: Optimal MDF-like base solution to analyze.
            analysis_params: Parsed analysis config containing relaxation settings.
            constraints: Additional constraints that define the solved base state.
            subnetwork_ids: Optional subnetwork restriction forwarded to the solver.
            thermo_unconstrained_reactions: Optional thermo exclusions for the analysis.

        Returns:
            Result payload returned by the metabolite-bottleneck helper.
        """
        # Delegate to the thermo-specific metabolite bottleneck helper with the
        # configured relaxation settings already normalized.
        logger.debug("  [Metabolite Bottlenecks] Dispatching metabolite bottleneck analysis.")
        return run_metabolite_bottleneck_analysis(
            solver=solver,
            base_solution=base_solution,
            constraints=constraints,
            relax_factor=analysis_params.run_metabolite_bottleneck_analysis.relax_factor_concentrations,
            test_active_metabolites_only=bool(analysis_params.run_metabolite_bottleneck_analysis.test_active_metabolites_only),
            subnetwork_ids=subnetwork_ids,
            thermo_unconstrained_reactions=thermo_unconstrained_reactions,
        )

    def run(
        self,
        *,
        solver: BiologicalSolver,
        base_solution: SolverResult,
        analysis_params: AnalysisParamsConfig,
        constraints: list[list[Any]] | None = None,
        objective_vars: str | list[str] | None = None,
        subnetwork_ids: list[str] | None = None,
        thermo_unconstrained_reactions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all configured post-analyses for one optimal base solution.

        Args:
            solver: Active solver instance used for analyses that require
                re-solving, such as FVA.
            base_solution: Optimal solution produced by the experiment step.
            analysis_params: Shared analysis configuration containing the
                enabled post-analyses and their settings.
            constraints: Active constraint set that defines the solved state.
            objective_vars: Objective identifier(s) of the solved step.
                These are used to decide whether MDF-specific analyses should
                run.
            subnetwork_ids: Optional subnetwork reaction ids used by thermo
                post-analyses.
            thermo_unconstrained_reactions: Optional list of reactions that
                should be excluded from thermo constraints in post-analyses.

        Returns:
            dict[str, Any]: Mapping from post-analysis output name to result
            payload. Analyses that fail are recorded as ``"Error: ..."``. If a
            thermo-specific analysis is requested for a non-thermo or non-MDF
            step, the result is set to ``"['N/A - non-thermo or non-MDF objective']"``.

        Example:
            A thermo MDF step with ``objective_vars=["ATPM", "var_B2"]`` may
            return keys such as ``"CVA_Results"``, ``"FVA_Results"``, and
            ``"Bottleneck_Reactions"``.
        """
        results = {}

        # PHASE 0: Guard against invalid dispatcher inputs.
        # Post-analyses are only meaningful for an available optimal base
        # solution and a parsed analysis configuration.
        if not analysis_params or (base_solution.status or "").lower() != "optimal":
            return results

        # Normalize shared dispatcher state that is reused across the different
        # post-analysis branches below.
        apply_specificity = bool(analysis_params.apply_specificity_constraints)
        normalized_objective_vars = [objective_vars] if isinstance(objective_vars, str) else list(objective_vars or [])
        is_mdf_objective = any(var_id in {"var_B", "var_B2"} for var_id in normalized_objective_vars) if normalized_objective_vars else True
        logger.info("Starting post-analysis dispatch for objective vars: %s", normalized_objective_vars or ["<unspecified>"])

        # Resolve numerical FVA targets independently from visualization. Plain
        # entries remain reaction IDs; JSON entries contribute all map reactions.
        configured_fva_targets = list(analysis_params.run_fva.reactions_to_check) if analysis_params.run_fva else []
        resolved_fva_reactions: list[str] = []
        map_paths: list[str] = []
        for target in configured_fva_targets:
            if Path(target).suffix.lower() == ".json":
                map_paths.append(target)
            else:
                resolved_fva_reactions.append(target)

        if map_paths:
            biological_scenario = getattr(solver, "biological_scenario", None)
            if not biological_scenario:
                raise RuntimeError("Map-driven FVA requires a loaded biological scenario on the solver.")

            for map_path in map_paths:
                map_reactions = resolve_model_reactions_from_escher_map(
                    map_identifier=map_path,
                    model=biological_scenario.model,
                    forward_reaction_suffix=biological_scenario.model_conf.forward_reaction_suffix,
                    reverse_reaction_suffix=biological_scenario.model_conf.reverse_reaction_suffix,
                )
                resolved_fva_reactions.extend(map_reactions)
                logger.info("Expanded FVA targets with %d model reaction(s) from map '%s'.", len(map_reactions), map_path)

        resolved_fva_reactions = list(dict.fromkeys(resolved_fva_reactions))
        if analysis_params.run_fva:
            logger.info(
                "Resolved %d FVA reaction target(s) from %d explicit ID(s) and %d map file(s).",
                len(resolved_fva_reactions),
                len(configured_fva_targets) - len(map_paths),
                len(map_paths),
            )

        # PHASE 1: Run universal post-analyses.
        # FVA is solver-agnostic and can run without thermo-specific state.
        if analysis_params.run_fva and resolved_fva_reactions:
            try:
                results["FVA_Results"] = self._perform_fva(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    resolved_fva_reactions=resolved_fva_reactions,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                    apply_specificity=apply_specificity,
                )
            except Exception as exc:
                logger.error("FVA failed: %s", exc, exc_info=True)
                results["FVA_Results"] = f"Error: {exc}"

        if analysis_params.run_min_optimization and analysis_params.run_min_optimization.variables_to_check:
            try:
                results["MinOptimization_Results"] = self._perform_variable_optimization(
                    solver=solver,
                    base_solution=base_solution,
                    optimization_config=analysis_params.run_min_optimization,
                    objective_direction="min",
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                    apply_specificity=apply_specificity,
                )
            except Exception as exc:
                logger.error("Min optimization failed: %s", exc, exc_info=True)
                results["MinOptimization_Results"] = f"Error: {exc}"

        if analysis_params.run_max_optimization and analysis_params.run_max_optimization.variables_to_check:
            try:
                results["MaxOptimization_Results"] = self._perform_variable_optimization(
                    solver=solver,
                    base_solution=base_solution,
                    optimization_config=analysis_params.run_max_optimization,
                    objective_direction="max",
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                    apply_specificity=apply_specificity,
                )
            except Exception as exc:
                logger.error("Max optimization failed: %s", exc, exc_info=True)
                results["MaxOptimization_Results"] = f"Error: {exc}"

        # PHASE 2: Run COBRA-k-specific post-analyses.
        if analysis_params.run_cobrak_nlp_variability:
            if hasattr(solver, "supports") and solver.supports("native_nlp_variability"):
                try:
                    results["COBRAk_NLP_Variability"] = run_cobrak_nlp_variability(
                        solver=solver,
                        base_solution=base_solution,
                        variability_config=analysis_params.run_cobrak_nlp_variability,
                        constraints=constraints,
                    )
                except Exception as exc:
                    logger.error("COBRA-k NLP variability failed: %s", exc, exc_info=True)
                    results["COBRAk_NLP_Variability"] = f"Error: {exc}"
            else:
                results["COBRAk_NLP_Variability"] = "N/A - requires COBRA-k kinetic_nlp"

        # PHASE 3: Run thermo concentration post-analyses.
        # CVA and metabolite ratios depend on thermo variables, but they do not
        # require the active objective to be MDF-like.
        if analysis_params.run_cva and analysis_params.run_cva.metabolites_to_check:
            try:
                results["CVA_Results"] = self._perform_cva(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                    apply_specificity=apply_specificity,
                )
            except Exception as exc:
                logger.error("CVA failed: %s", exc, exc_info=True)
                results["CVA_Results"] = f"Error: {exc}"

        if analysis_params.run_metabolite_ratios and analysis_params.run_metabolite_ratios.ratios:
            try:
                results["Metabolite_Ratios"] = self._perform_metabolite_ratios(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                    apply_specificity=apply_specificity,
                )
            except Exception as exc:
                logger.error("Ratio analysis failed: %s", exc, exc_info=True)
                results["Metabolite_Ratios"] = f"Error: {exc}"

        # PHASE 4: Determine whether thermo-specific analyses are allowed.
        # Most bottleneck checks require an MDF-like active objective. The
        # bottleneck-set helper may also target an explicit non-MDF variable,
        # such as biomass, while still relaxing thermodynamic MDF constraints.
        thermo_capable = hasattr(solver, "supports") and solver.supports("thermo_driving_force")

        # PHASE 5: Run thermo-specific post-analyses.
        # When the step is non-thermo or not MDF-like, store an explicit N/A
        # marker instead of silently skipping the requested analysis.
        if analysis_params.run_bottleneck_analysis:
            if thermo_capable and is_mdf_objective:
                results["Bottleneck_Reactions"] = self._perform_bottleneck_reactions(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                )
            else:
                results["Bottleneck_Reactions"] = "['N/A - non-thermo or non-MDF objective']"

        if analysis_params.run_bottleneck_set_analysis:
            explicit_bottleneck_set_target = bool(analysis_params.run_bottleneck_set_analysis.target_variable)
            if thermo_capable and (is_mdf_objective or explicit_bottleneck_set_target):
                results["Minimal_Bottleneck_Sets"] = self._perform_bottleneck_sets(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                )
            else:
                results["Minimal_Bottleneck_Sets"] = "['N/A - non-thermo or non-MDF objective']"

        if analysis_params.run_true_bottleneck_check:
            if thermo_capable and is_mdf_objective:
                results["True_Bottlenecks"] = self._perform_true_bottleneck_check(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                )
            else:
                results["True_Bottlenecks"] = "['N/A - non-thermo or non-MDF objective']"

        if analysis_params.run_metabolite_bottleneck_analysis:
            if thermo_capable and is_mdf_objective:
                results["Metabolite_Bottlenecks"] = self._perform_metabolite_bottlenecks(
                    solver=solver,
                    base_solution=base_solution,
                    analysis_params=analysis_params,
                    constraints=constraints,
                    subnetwork_ids=subnetwork_ids,
                    thermo_unconstrained_reactions=thermo_unconstrained_reactions,
                )
            else:
                results["Metabolite_Bottlenecks"] = "['N/A - non-thermo or non-MDF objective']"

        return results
