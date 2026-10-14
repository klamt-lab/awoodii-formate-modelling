import logging
import time
from typing import Any

import mymodelingpackage.structures as structs
from mymodelingpackage.solvers.interface import BiologicalSolver
from mymodelingpackage.post_analyses.dispatcher import PostAnalysisDispatcher
from mymodelingpackage.experiments.base import BaseExperiment

from mymodelingpackage.utilities import (
    count_carbon_atoms,
)

from mymodelingpackage.experiments.utils import (
    build_quality_lock_constraints,
    collect_swap_ids,
    dedupe_constraints,
    is_optimal,
    resolve_constraint_block,
    resolve_symbolic_vars,
    sanitize_token,
    solve_step_robustly,
    solve_with_swap_minimization,
)

logger = logging.getLogger(__name__)


class CharacterizationExperiment(BaseExperiment):
    """
    Executes a comprehensive characterization workflow across a batch of biological scenarios.

    This experiment sequentially optimizes for different biological and thermodynamic 
    objectives (e.g., Max Growth, Max ATPM, Max MDF), allowing later steps to dynamically 
    constrain themselves based on the results of earlier steps.

    Key Features:
    -------------
    - **Robust Solving**: Automatically utilizes multi-start CPLEX seed cycling if an 
      initial solve fails or returns infeasible.
    - **Swap Minimization Sandwich**: If enabled, executes a 3-phase optimization for 
      specific steps to find the minimal topological changes required:
        1) Phase A: Find the absolute theoretical limit of the primary objective.
        2) Phase B: Lock that limit (with a slight relaxation) and minimize total network swaps.
        3) Phase C: Lock the minimal swap count and re-optimize the primary objective.
    - **Dynamic Chaining**: Supports `use_result_from` and `use_swap_result_from` to pass 
      thresholds between sequential workflow steps.
    """

    def __init__(
        self,
        experiment_def: structs.ExperimentDefinition,
        solver: BiologicalSolver,
        output_dir: str = "./Results",
    ):
        """Initialize a scenario-wise characterization workflow.

        Args:
            experiment_def: Parsed experiment definition including the
                characterization config and biological scenarios.
            solver: Solver implementation used for all optimization steps.
            output_dir: Directory where CSV, JSON, and optional artifacts are
                written.
        """
        super().__init__(experiment_def=experiment_def, output_dir=output_dir)
        self.solver = solver
        self.dispatcher = PostAnalysisDispatcher()
        self.aggregate_results: list[dict[str, Any]] = []

        cfg = self.experiment_def.analysis_config
        # Validate that step names are unique to prevent silent overwriting
        step_names = [step.name for step in cfg.optimization_steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Duplicate step names found in optimization_steps. Step names must be unique.")
        # Optional list of user-configured "key" reaction ids.
        self.analysis_params = cfg.analysis_params

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

        # --- Runtime Scratch State ---
        # These variables hold data that needs to persist across sequential steps *within* a single scenario.
        self.swap_ids: list[str] = []
        self._current_primary_objective: list[str] = []
        self.step_results: dict[str, dict[str, float]] = {}   # Stores objective values per step (e.g., {"Step_1": {"var_B": 0.85}})
        self.step_swap_counts: dict[str, Any] = {}      # Stores realized swap limits (e.g., {"Step_1": 3})

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _resolve_constraints(
        self,
        raw_constraints: list[Any],
        optimization_steps: list[Any],
        product_id: str | None,
        biomass_id: str | None,
        max_prod: float,
    ) -> tuple[list[list[Any]], Any]:
        """Resolve one step's raw constraint config into solver-ready constraints.

        This method handles both static shorthand constraints and dynamic
        references to previously executed characterization steps.

        Args:
            raw_constraints: Raw step constraint configuration. Supported forms
                include strings like ``"min_growth"``, dictionaries like
                ``{"use_result_from": "optBio", "fraction": 0.5, "type": "geq"}``,
                and ``{"use_swap_result_from": "optMDF"}``.
            optimization_steps: Ordered workflow definition used to resolve
                step-name references.
            product_id: Scenario-specific product reaction id, if available.
            biomass_id: Scenario-specific biomass reaction id.
            max_prod: Precomputed maximum theoretical product yield for the
                current scenario.

        Returns:
            tuple[list[list[Any]], Any]: A tuple containing the resolved
            constraint list and an inherited swap count if a previous step
            passed down a fixed swap budget.

        Example:
            A raw block like ``["min_growth", {"use_result_from": "optATPM",
            "fraction": 0.8, "type": "geq"}]`` becomes concrete numeric
            constraints once the earlier step result is known. If the source
            step has multiple objectives, the dict must also provide an
            ``"objective_var"`` entry.
        """
        resolved: list[list[Any]] = []
        inherited_swap_count: Any = None
        step_lookup = {s.name: s for s in optimization_steps}

        for raw in raw_constraints:
            logger.debug(f"Parsing raw constraint: {raw}")

            # 1. Dynamic Dictionary Constraints (Dependent on previous steps)
            if isinstance(raw, dict):

                # Inherit objective threshold (e.g., Growth >= 50% of Step 1)
                if "use_result_from" in raw:
                    source_name = raw.get("use_result_from")
                    if source_name not in self.step_results:
                        raise ValueError(f"Constraint references '{source_name}' but that step has no stored value.")

                    source_step = step_lookup.get(source_name)
                    if source_step is None:
                        raise ValueError(f"Constraint references unknown step '{source_name}'.")

                    resolved_source_objectives = resolve_symbolic_vars(
                        list(source_step.objective_var_ids),
                        product_id=product_id,
                        biomass_id=biomass_id,
                    )
                    requested_objective = raw.get("objective_var")
                    if requested_objective is not None:
                        source_objective = resolve_symbolic_vars(
                            requested_objective,
                            product_id=product_id,
                            biomass_id=biomass_id,
                        )
                    elif len(resolved_source_objectives) == 1:
                        source_objective = resolved_source_objectives[0]
                    else:
                        raise ValueError(f"Constraint references step '{source_name}' with multiple objectives but does not specify 'objective_var'.")

                    if not source_objective:
                        raise ValueError(f"Cannot resolve objective variable for source step '{source_name}'.")
                    if source_objective not in self.step_results[source_name]:
                        raise ValueError(f"Constraint requests objective '{source_objective}' from step '{source_name}', but that value was not stored.")

                    fraction = raw.get("fraction", 1.0)
                    ctype = raw.get("type", "geq")
                    value = self.step_results[source_name][source_objective] * fraction

                    resolved.append([source_objective, value, f"from_{source_name}", ctype])
                    logger.debug(f" -> Dynamic constraint resolved: {source_objective} {ctype} {value:.4f}")

                # Inherit swap budget (e.g., Total Swaps <= Swaps required in Step 2)
                if "use_swap_result_from" in raw:
                    source_name = raw["use_swap_result_from"]
                    inherited = self.step_swap_counts.get(source_name)

                    if inherited is not None and inherited != "N/A":
                        if not self.swap_ids:
                            logger.warning("Cannot inherit swaps from '%s': no swap IDs initialized.", source_name)
                        else:
                            swap_sum = [[z, 1.0] for z in self.swap_ids]
                            resolved.append([swap_sum, inherited + 0.5, f"swaps_from_{source_name}", "leq"])
                            inherited_swap_count = inherited
                            logger.debug(f" -> Inherited swap budget resolved: <= {inherited} swaps.")
                    else:
                        logger.warning("Could not inherit swaps from '%s': no stored count.", source_name)

            # 2. Shared static constraints
            else:
                resolved.extend(
                    resolve_constraint_block(
                        [raw],
                        product_id=product_id,
                        biomass_id=biomass_id,
                        max_prod=max_prod,
                        min_growth=self.min_growth,
                        min_mdf=self.min_mdf,
                        fraction_prod=self.fraction_prod,
                        objective_var_ids=self._current_primary_objective,
                    )
                )

        return dedupe_constraints(resolved), inherited_swap_count


    def _get_default_steps(
        self,
        product_id: str,
        biomass_id: str,
        max_prod: float,
    ) -> list[structs.OptimizationStep]:
        """Build the built-in fallback characterization workflow.

        Args:
            product_id: Resolved product reaction id for the current scenario.
            biomass_id: Resolved biomass reaction id for the current scenario.
            max_prod: Precomputed maximum product yield. Included for symmetry
                with the caller and future extensions.

        Returns:
            list[structs.OptimizationStep]: The default ordered workflow used
            when the config does not define explicit optimization steps.

        Example:
            The fallback sequence includes steps such as ``optATPM``,
            ``optBio_minMDF``, and ``optProd_minMDF_minGrowth``.
        """
        logger.info("  - Using fallback 7-step characterization workflow.")
        return [
            structs.OptimizationStep(name="optATPM", objective_var_ids=["ATPM"], objective_direction="max", constraints=["min_mdf"]),
            structs.OptimizationStep(name="optATPM_minMDF", objective_var_ids=["ATPM"], objective_direction="max", constraints=["min_growth"]),
            structs.OptimizationStep(name="optBio_minMDF", objective_var_ids=[biomass_id], objective_direction="max", constraints=["min_mdf"]),
            structs.OptimizationStep(name="optMDF_minProd_minGrowth", objective_var_ids=["var_B"], objective_direction="max", constraints=["min_product", "min_growth"]),
            structs.OptimizationStep(name="optMDF_minProd", objective_var_ids=["var_B"], objective_direction="max", constraints=["min_product"]),
            structs.OptimizationStep(name="optProd_minMDF_minGrowth", objective_var_ids=[product_id] if product_id else [], objective_direction="max", constraints=["min_mdf", "min_growth"]),
            structs.OptimizationStep(name="optProd_minMDF", objective_var_ids=[product_id] if product_id else [], objective_direction="max", constraints=["min_mdf"]),
        ]

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
    # Characterization follows the same skeleton as other experiments, but its
    # experiment-specific core logic is:
    #   PHASE 2.1: Build workflow definition.
    #   PHASE 2.2: Execute workflow loop.
    #
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self) -> list[dict[str, Any]]:
        """Execute the full characterization workflow across all scenarios.

        Returns:
            list[dict[str, Any]]: One scenario payload per biological scenario.
            Each payload contains a ``metrics`` dictionary, a ``steps`` list,
            and a top-level scenario ``status``.

        Example:
            A single returned scenario payload has the rough structure
            ``{"scenario_id": "...", "metrics": {...}, "steps": [...],
            "status": "Success"}``.
        """
        start_time = time.time()
        logger.debug("================= STARTING RUN() ====================")
        logger.info(f"Starting CharacterizationExperiment: {self.experiment_def.experiment_name}")

        scenarios = self.experiment_def.biological_scenarios
        logger.info(f"Processing {len(scenarios)} scenarios...")
        cfg = self.experiment_def.analysis_config

        for scenario_idx, scenario in enumerate(scenarios):
            # -------------------------------------------------------------
            # PHASE 1: SCENARIO PREPARATION
            # Start timing, isolate scenario-local chained state, and create
            # the fresh payload container that will collect this scenario run.
            # -------------------------------------------------------------
            scenario_start_time = time.time()
            logger.info(f" [scenario {scenario_idx+1}]     --- Processing Scenario: {scenario.scenario_id} ---")
            self.step_results = {}
            self.step_swap_counts = {}
            self._current_primary_objective = []
            run_payload = self.build_scenario_payload(scenario)

            try:
                # -------------------------------------------------------------
                # PHASE 1.1: LOAD SOLVER CONTEXT
                # Load the solver context before scenario-specific optional
                # features, metadata, or workflow logic are prepared.
                # -------------------------------------------------------------
                logger.debug(f" [scenario {scenario_idx+1}] Loading biological scenario and analysis config into solver context...")
                self.solver.load_context(biological_scenario=scenario, analysis_config=cfg)

                # -------------------------------------------------------------
                # PHASE 1.2: PREPARE OPTIONAL EXPERIMENT FEATURES
                # Initialize optional scenario-local features such as swap
                # minimization helpers that may affect later workflow steps.
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
                # Resolve scenario-specific identifiers that the later
                # workflow loop will need in solver-ready form.
                # -------------------------------------------------------------
                scenario_metadata = self.resolve_scenario_metadata(scenario)
                biomass_id = scenario_metadata["biomass_id"]
                product_name = scenario_metadata["product_name"]
                product_id = scenario_metadata["product_id"]
                if product_id:
                    logger.info(f" [scenario {scenario_idx+1}] Resolved Product ID: '{product_id}' (from name '{product_name}')")
                else:
                    logger.warning(f"Could not resolve Product ID for '{product_name}'. Steps relying on 'product' may fail.")

                substrate_name = scenario_metadata["substrate_name"]
                substrate_ids = scenario_metadata["substrate_ids"]
                if substrate_ids:
                    logger.info(f" [scenario {scenario_idx+1}] Resolved Substrate IDs: {substrate_ids} (from condition '{substrate_name}')")
                else:
                    logger.warning(f"Could not resolve any Substrate IDs for condition '{substrate_name}'.")

                max_prod = 0.0  # Initialize safely to 0.0

                # -------------------------------------------------------------
                # PHASE 1.4: PRECOMPUTATIONS
                # This block mirrors VariableScan's pre-computation phase:
                # resolve carbon bookkeeping and theoretical yield once and
                # reuse it throughout the scenario workflow.
                # -------------------------------------------------------------
                logger.info(f" [scenario {scenario_idx+1}]   -> Starting pre-computation (Carbon counts and Max Yield)...")
                if substrate_ids and product_id:
                    try:
                        # Sum carbon atoms across all substrates (naturally filters out non-C sources like O2)
                        c_atoms_sub_total = 0
                        for sub_id in substrate_ids:
                            reaction = scenario.model.reactions.get_by_id(sub_id)
                            metabolite = list(reaction.metabolites.keys())[0]
                            c_count = count_carbon_atoms(metabolite.formula)
                            logger.debug(f" [scenario {scenario_idx+1}]      Substrate '{sub_id}' contains {c_count} carbon atoms.")
                            c_atoms_sub_total += c_count

                        reaction = scenario.model.reactions.get_by_id(product_id)
                        metabolite = list(reaction.metabolites.keys())[0]
                        c_atoms_prod = count_carbon_atoms(metabolite.formula)

                        logger.debug(f" [scenario {scenario_idx+1}]      Product '{product_id}' contains {c_atoms_prod} carbon atoms.")

                        run_payload["metrics"]["carbon_atoms_substrate_total"] = c_atoms_sub_total
                        run_payload["metrics"]["carbon_atoms_product"] = c_atoms_prod

                        # Max Yield
                        logger.info(f" [scenario {scenario_idx+1}]   -> Calculating Max Theoretical Yield...")
                        max_prod = self.compute_max_theoretical_yield(scenario, product_id)
                        run_payload["metrics"]["max_theoretical_yield"] = max_prod
                        logger.info(f" [scenario {scenario_idx+1}]   -> Max Theoretical Yield for '{product_id}': {max_prod:.4f}")

                        if max_prod == 0.0:
                            logger.warning(f"Max product yield for '{product_id}' is ZERO. MDF constraints relying on this will likely be infeasible.")

                    except Exception as e:
                        logger.warning(f"Failed during pre-computation (carbon counts or yield): {e}")
                        logger.debug(f" [scenario {scenario_idx+1}] Exception details:", exc_info=True)
                else:
                    logger.info(f" [scenario {scenario_idx+1}]   -> Skipping carbon/yield pre-computation due to missing substrate or product IDs.")

                # -------------------------------------------------------------
                # PHASE 2: EXPERIMENT CORE LOGIC
                # Convert the Characterization configuration into the ordered
                # workflow that will be executed below.
                # -------------------------------------------------------------

                # -------------------------------------------------------------
                # PHASE 2.1: BUILD WORKFLOW DEFINITION
                # Resolve the list of optimization steps, either from the user
                # configuration or from the built-in fallback workflow.
                # -------------------------------------------------------------
                custom_steps = cfg.optimization_steps
                workflow_steps = custom_steps if custom_steps else self._get_default_steps(product_id, biomass_id, max_prod)

                logger.info(f" [scenario {scenario_idx+1}]   START: Optimization Phase:  Using workflow with {len(workflow_steps)} step(s).")

                # Loop over each optimization step
                # ---------------------------------------------------------
                # PHASE 2.2: EXECUTE WORKFLOW LOOP
                # Resolve, solve, and record each workflow unit in sequence
                # while allowing later steps to reuse earlier results.
                # ---------------------------------------------------------
                for step_idx, step in enumerate(workflow_steps):
                    # ---------------------------------------------------------
                    # PHASE 2.2.1: PREPARE WORKFLOW UNIT INPUTS
                    # Characterization resolves one optimization step at a time.
                    # Each step can depend on results from earlier steps through
                    # dynamically resolved constraints.
                    # ---------------------------------------------------------
                    step_name = str(step.name or f"Unnamed_Step_{step_idx+1}")

                    logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]     >> Setting up Step {step_idx+1}/{len(workflow_steps)}: '{step_name}'")

                    # Setup objective ids
                    step_objective_var_ids = resolve_symbolic_vars(
                        list(step.objective_var_ids),
                        product_id=product_id,
                        biomass_id=biomass_id,
                    )
                    if not step_objective_var_ids:
                        raise ValueError(f"Step '{step_name}' has no resolvable objective ids.")
                    self._current_primary_objective = step_objective_var_ids

                    # Setup the multipliers
                    objective_multipliers = (
                        list(step.objective_multipliers)
                        if step.objective_multipliers
                        else [1.0] * len(step_objective_var_ids)
                    )

                    # Setup optimization direction
                    objective_direction = step.objective_direction
                    
                    # Setup constraints
                    constraints_raw = list(step.constraints or [])
                    logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Step objective ids: {step_objective_var_ids}, dir: {objective_direction}, raw_constraints count: {len(constraints_raw)}")

                    resolved_constraints, inherited_swap_count = self._resolve_constraints(
                        raw_constraints=constraints_raw,
                        optimization_steps=workflow_steps,
                        product_id=product_id,
                        biomass_id=biomass_id,
                        max_prod=max_prod,
                    )

                    current_swaps: Any = "N/A"
                    final_constraints = resolved_constraints

                    logger.info(f" [scenario {scenario_idx+1}] [step {step_idx+1}]    --- Step: {step_name} ---")
                    logger.info(f" [scenario {scenario_idx+1}] [step {step_idx+1}]      Objective: {step_objective_var_ids} ({objective_direction})")

                    # -------------------------------------------------------------
                    # PHASE 2.2.2: EXECUTE CURRENT WORKFLOW UNIT
                    # Solve the current step either with a standard robust solve
                    # or with the swap-minimization sandwich if configured.
                    # -------------------------------------------------------------
                    logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Execution block entered for '{step_name}'...")

                    if inherited_swap_count is not None:
                        # Directly use standard solve since swaps are strictly locked
                        logger.info(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Executing STANDARD robust solve (inherited swaps locked at {inherited_swap_count})")
                        current_swaps = inherited_swap_count
                        solution = solve_step_robustly(
                            solver=self.solver,
                            objective_var_ids=step_objective_var_ids,
                            objective_multipliers=objective_multipliers,
                            objective_direction=objective_direction,
                            constraints=resolved_constraints,
                            num_retries=self.num_numerical_retries,
                            logger_=logger
                        )
                    elif self.minimize_swaps_config and self.swap_ids:
                        logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Executing SANDWICH solve (Minimization Enabled)")
                        solution, current_swaps, final_constraints = solve_with_swap_minimization(
                            solver=self.solver,
                            objective_var_ids=step_objective_var_ids,
                            objective_multipliers=objective_multipliers,
                            objective_direction=objective_direction,
                            constraints=resolved_constraints,
                            swap_ids=self.swap_ids,
                            num_numerical_retries=self.num_numerical_retries,
                            num_swap_attempts=self.num_swap_attempts,
                            num_reoptimization_attempts_sandwich=self.num_reoptimization_attempts_sandwich,
                            quality_prefix="maintain_step_quality",
                            debug_name_filter="1Swap",
                            raise_on_swap_failure=True,
                            logger_=logger,
                        )
                    else:
                        logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Executing STANDARD robust solve (No active sandwich limits)")
                        solution = solve_step_robustly(
                            solver=self.solver, 
                            objective_var_ids=step_objective_var_ids,
                            objective_multipliers=objective_multipliers,
                            objective_direction=objective_direction,
                            constraints=resolved_constraints,
                            num_retries=self.num_numerical_retries,
                            logger_=logger
                        )

                    logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Solve Complete. Status: {solution.status}, Swaps: {current_swaps}")
                    
                    # -------------------------------------------------------------
                    # PHASE 2.2.3: BUILD STANDARDIZED PAYLOAD
                    # Convert the raw solver result into the shared tabular
                    # output schema used by all experiment classes.
                    # -------------------------------------------------------------
                    step_payload = self.build_standard_step_payload(
                        step_name=step_name,
                        objective_var_ids=step_objective_var_ids,
                        objective_direction=objective_direction,
                        constraints=final_constraints,
                        solution=solution,
                        biomass_id=biomass_id,
                        product_id=product_id,
                        swaps_required=current_swaps,
                        substrate_ids=substrate_ids,
                    )

                    if self.key_reactions:
                        self.add_key_reaction_placeholders(step_payload, self.key_reactions)
                    if self.key_metabolites:
                        self.add_key_metabolite_placeholders(step_payload, self.key_metabolites)

                    if is_optimal(solution.status):
                        # ---------------------------------------------------------
                        # PHASE 2.2.4: UPDATE SUCCESSFUL OPTIMIZATION STATE
                        # Store the realized step state and enrich the payload
                        # before any artifacts or secondary analyses run.
                        # ---------------------------------------------------------
                        run_payload["status"] = "Success"
                        objective_values = {
                            objective_var_id: objective_value
                            for objective_var_id in step_objective_var_ids
                            if (objective_value := solution.values.get(objective_var_id)) is not None
                        }
                        
                        # ---------------------------------------------------------
                        # PHASE 2.2.4.1: PERSIST OPTIMIZATION STATE AND METRICS
                        # Store all resolved objective values so downstream
                        # steps can reuse them via dynamic constraints.
                        # ---------------------------------------------------------
                        if objective_values:
                            self.step_results[step_name] = objective_values
                            for objective_var_id, objective_value in objective_values.items():
                                run_payload["metrics"][f"{step_name}_{objective_var_id}"] = objective_value
                            logger.debug(" [scenario %d] [step %d]   Saved step objectives: %s = %s", scenario_idx + 1, step_idx + 1, step_name, objective_values)

                        if current_swaps != "N/A":
                            self.step_swap_counts[step_name] = current_swaps
                            logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Saved step swap count: {step_name} = {current_swaps}")

                        # ---------------------------------------------------------
                        # PHASE 2.2.4.2: ENRICH PAYLOAD ANNOTATIONS
                        # Add optional diagnostic annotations such as key fluxes
                        # and the realized active swap list.
                        # ---------------------------------------------------------
                        if self.key_reactions:
                            key_flux_data = self._get_key_fluxes(solution, self.key_reactions)
                            logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}] Got {len(key_flux_data)} key reaction fluxes")
                            for prefix, (variant_name, net_flux) in key_flux_data.items():
                                step_payload[f"variant_{prefix}"] = variant_name
                                step_payload[f"flux_{prefix}"] = net_flux

                        if self.key_metabolites:
                            key_conc_data = self._get_key_concentrations(solution, self.key_metabolites)
                            for met_id, conc in key_conc_data.items():
                                step_payload[f"conc_{met_id}"] = conc

                        # Log active swaps
                        if self.minimize_swaps_config and self.swap_ids and is_optimal(solution.status):
                            active = [z for z in self.swap_ids if solution.values.get(z, 0.0) > 0.5]
                            step_payload["Active_Swaps_List"] = str(active)
                            logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Recorded active swaps for payload: {active}")

                        # ---------------------------------------------------------
                        # PHASE 3: ARTIFACT GENERATION
                        #
                        # PHASE 3.1: WRITE OPTIONAL ARTIFACTS
                        # Persist optional raw/debug/visualization files that
                        # belong to this successful optimization step.
                        # ---------------------------------------------------------
                        artifact_id = sanitize_token(f"{scenario.scenario_id}_{step_name}")

                        # Persist optional debug and visualization artifacts directly
                        # in the main workflow so the successful-step side effects stay
                        # visible alongside the optimization logic.
                        if self.analysis_params.excel_output:
                            self.save_to_excel(
                                model=scenario.model,
                                raw_solution=solution,
                                scenario_id=artifact_id,
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
                                objective_direction=objective_direction,
                            )

                        if self.analysis_params.escher:
                            self.save_escher_visualization(
                                model=scenario.model,
                                raw_solution=solution,
                                scenario_id=artifact_id,
                                escher_config=self.analysis_params.escher,
                            )

                        # ---------------------------------------------------------
                        # PHASE 4: POST-ANALYSES
                        #
                        # PHASE 4.1: BUILD LOCKED OPTIMUM STATE
                        # PHASE 4.2: DISPATCH POST-ANALYSES
                        #
                        # Lock the optimum tightly and run secondary analyses so
                        # diagnostics stay anchored to the realized solution.
                        # ---------------------------------------------------------
                        if objective_values:
                            # -------------------------------------------------
                            # PHASE 4.1: BUILD LOCKED OPTIMUM STATE
                            # Convert the realized objective values into tight
                            # lock constraints for downstream analyses.
                            # -------------------------------------------------
                            post_constraints = list(final_constraints) + build_quality_lock_constraints(
                                solution,
                                objective_var_ids=step_objective_var_ids,
                                objective_direction=objective_direction,
                                prefix=f"lock_{step_name}",
                            )

                            # -------------------------------------------------
                            # PHASE 4.2: DISPATCH POST-ANALYSES
                            # Run the configured secondary analyses on the
                            # locked feasible region for this solved step.
                            # -------------------------------------------------
                            logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]     -> Dispatching post-analyses with locked bounds...")
                            post_analysis_results = self.dispatcher.run(
                                solver=self.solver,
                                base_solution=solution,
                                analysis_params=self.analysis_params,
                                constraints=post_constraints,
                                objective_vars=step_objective_var_ids,
                                subnetwork_ids=self.analysis_params.subnetwork,
                                thermo_unconstrained_reactions=self.analysis_params.thermo_unconstrained_reactions,
                            )
                            self.attach_post_analysis_results(step_payload, post_analysis_results)
                    else:
                        # ---------------------------------------------------------
                        # PHASE 5.1: HANDLE FAILURE AND STOP EARLY
                        # Keep the failed payload, mark the scenario as failed,
                        # and stop the sequential workflow for this scenario.
                        # ---------------------------------------------------------
                        logger.warning(f"    -> Step '{step_name}' failed ({solution.status}). Halting workflow for this scenario.")
                        logger.debug(f" [scenario {scenario_idx+1}] [step {step_idx+1}]   Aborting further steps for '{scenario.scenario_id}' due to non-optimal status.")
                        run_payload["status"] = f"Failed at {step_name}: {solution.status}"
                        run_payload["steps"].append(step_payload)
                        break

                    run_payload["steps"].append(step_payload)

            except Exception as exc:
                # -------------------------------------------------------------
                # PHASE 5.2: HANDLE UNEXPECTED SCENARIO EXCEPTION
                # Convert unexpected scenario-level errors into the standard
                # scenario failure payload without dropping final timing data.
                # -------------------------------------------------------------
                logger.error(f"CRITICAL: Unhandled exception during scenario {scenario.scenario_id}: {exc}", exc_info=True)
                run_payload["status"] = f"Failed: {exc}"

            finally:
                # -------------------------------------------------------------
                # PHASE 6: SCENARIO FINALIZATION
                # Persist final execution metadata for this scenario and keep
                # the aggregate results aligned across experiment classes.
                # -------------------------------------------------------------
                elapsed = time.time() - scenario_start_time
                run_payload["metrics"]["execution_time_seconds"] = round(elapsed, 2)
                logger.info(f" [scenario {scenario_idx+1}]         Finished scenario {scenario.scenario_id} in {elapsed:.2f} seconds.")
                self.aggregate_results.append(run_payload)

        # -------------------------------------------------------------
        # PHASE 7: EXPERIMENT FINALIZATION
        # Emit the experiment-level completion log and return the full
        # aggregated payload for JSON/CSV export.
        # -------------------------------------------------------------
        total_elapsed = time.time() - start_time
        logger.info(
            "CharacterizationExperiment '%s' finished %d scenario(s) in %.2fs.",
            self.experiment_def.experiment_name,
            len(scenarios),
            total_elapsed,
        )

        return self.aggregate_results
