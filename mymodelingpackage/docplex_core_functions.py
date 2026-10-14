#################################################################################
#################################################################################
###
###                          Docplex core function for pulp FBA and Opt(sub)MDF
###
#################################################################################
#################################################################################
from math import log, isclose
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException

import logging
logger = logging.getLogger(__name__)

# # Silence noisy low-level docplex debug logs (for example, solve detail dumps)
# for noisy_name in (
#     "docplex",
#     "docplex.util",
#     "docplex.util.environment",
#     "docplex.util.environment.logger",
# ):
#     logging.getLogger(noisy_name).setLevel(logging.WARNING)

from typing import Any, Optional
import cobra

from collections import defaultdict
from mymodelingpackage.utilities import matches_prefix 
from mymodelingpackage.structures import ThermoSolverCPLEXConfig, SolverResult, DEFAULT_CPLEX_PARAMS


def docplex_optmdf(
    *,
    # 1. Objective / Engine instructions
    objective_var_ids: Optional[list[str]] = None,
    objective_direction: str = "max",
    objective_multipliers: Optional[list[float]] = None,
    constraints: Optional[list[list[Any]]] = None,
    bottleneck_analysis: bool = False,
    warm_start_solution: Optional[dict] = None,

    # 2. Biological Data (Required, but can now safely follow defaults!)
    model: cobra.Model,
    used_concentration_values: dict[str, dict[str, float]],
    dG0_values: dict[str, Any],
    co_factor_alternatives: Optional[list[str]] = None,

    # 3. Analysis Scope
    subnetwork_ids: Optional[list[str]] = None,
    thermo_unconstrained_reactions: list[str] = None,

    # 4. Solver Config
    solver_config: ThermoSolverCPLEXConfig = None
):
    """
    Core MILP implementation for Thermodynamic analysis (MDF/Flux) using CPLEX.

    Parameters
    ----------
    -- BIOLOGICAL INPUTS --
    model : cobra.Model
        The metabolic network model containing reactions and metabolites.
    used_concentration_values : dict
        Map of {metabolite_id: {'min': float, 'max': float}}. 
        Must contain a 'DEFAULT' key for fallback.
    dG0_values : dict
        Map of {reaction_id: {'dG0': float, 'uncertainty': float}}.
    objective_var_ids : list[str], optional
        List of variable IDs (fluxes or thermodynamic vars) to optimize.
    objective_direction : str, default 'max'
        Optimization direction: 'max' or 'min'.
    objective_multipliers : list[float], optional
        Coefficients for the objective variables. Defaults to 1.0.
    constraints : list, optional
        Additional linear constraints: [[var_id, value, name, relation_type], ...].
    subnetwork_ids : list[str], optional
        List of reaction prefixes to include in secondary subMDF calculation.
    co_factor_alternatives : list, optional
        Definitions for cofactor swapping logic (e.g., NADH vs NADPH variants).
    thermo_unconstrained_reactions : list[str], optional
        Override list of reaction IDs to exclude from DeltaG constraints. 
       


    -- EXECUTION PARAMETERS --
    solver_config : ThermoSolverCPLEXConfig
        Configuration object containing physics constants (R, T), solver constants
        (Big M, MAX_FLUX), and CPLEX parameters.
    bottleneck_analysis : bool, default False
        If True, minimizes the slack variables (zb) required to satisfy thermodynamics,
        rather than maximizing global MDF.
    warm_start_solution : dict, optional
        Previous solution dictionary to seed the solver (MIP start).

    Returns
    -------
    dict
        Result dictionary with keys: 'status', 'objective_value', 'values', 'message'.
    """

    # --- 0. Initialization & Safety Defaults ---
    if solver_config is None:
        logger.error('No config provided!')
        raise ValueError
    
    # Unpack commonly used solver_config for readability
    M_VALUE = solver_config.M_VALUE
    MAX_FLUX = solver_config.MAX_FLUX
    RT = solver_config.STANDARD_R * solver_config.STANDARD_T

    # --- SAFETY DEFAULTS ---
    dG0_values = dG0_values or {}
    objective_var_ids = objective_var_ids or []
    objective_multipliers = objective_multipliers or [1.0] * len(objective_var_ids)
    constraints = constraints or []
    co_factor_alternatives = co_factor_alternatives or []

    # 1.) Initialize docplex Model
    milp_model = Model(name="OptMDF_BigM")
    # Ensure solver parameters always include defaults, then override with provided values.
    merged_cplex_params = {**DEFAULT_CPLEX_PARAMS, **(solver_config.cplex_params or {})}
    milp_model.log_output = merged_cplex_params.get("mip.display", 0) > 0
    
    for param_path, value in merged_cplex_params.items():
        try:
            param_group = milp_model.parameters
            path_parts = param_path.split('.')
            for part in path_parts: param_group = getattr(param_group, part)
            param_group.set(value)
        except AttributeError:
            logger.warning(f"Warning: Could not set CPLEX parameter '{param_path}'.")

    # ________________________________________________________________________________________
    # 2.) Add reaction flux variables and set lower and upper fluxes for them
    #     αi  ≤  ri  ≤ βi
    #     -MAX_FLUX <= v <= MAX_FLUX (clamped by model bounds)

    flux_variables: dict[str, Any] = {}
    metabolite_expressions = defaultdict(lambda: 0)

    for reaction in model.reactions:
        #             ( -1000   , LowerBoundOfCobrapy )
        lowBound = max(-MAX_FLUX, reaction.lower_bound)
        #             ( lowBound, 1000   )
        lowBound = min(lowBound, MAX_FLUX)

        upBound = min(MAX_FLUX, reaction.upper_bound)
        upBound = max(-MAX_FLUX, upBound)

        ## Reaction variable
        flux_variables[reaction.id] = milp_model.continuous_var(
            lb=lowBound, ub=upBound, name=reaction.id
        )
        # Prepare list of metabolites with corresponding reactions for each metabolite for the next step
        for metabolite, stoichiometry in reaction.metabolites.items():
            metabolite_expressions[metabolite.id] += stoichiometry * flux_variables[reaction.id]

    # ________________________________________________________________________________________
    # 3.) Add steady-state constraints by adding a constraint for each metabolite that all reactions where metabolite is involved sum to 0
    #     N • r = 0
    for metabolite_id, expression in metabolite_expressions.items():
        milp_model.add_constraint(expression == 0, ctname=metabolite_id)

    # ________________________________________________________________________________________
    # 4.) Add metabolite concentration pulp variables
    #     ln(c_min) ≤ x ≤ ln(c_max)
    # metabolite_concentration_values = used_concentration_values

    concentration_vars: dict[str, Any] = {}
    for metabolite in model.metabolites:
        id_key = metabolite.id if metabolite.id in used_concentration_values else "DEFAULT"
        try:
            min_conc = log(used_concentration_values[id_key]["min"])
            max_conc = log(used_concentration_values[id_key]["max"])
        except KeyError:
            min_conc = log(used_concentration_values[id_key]["lb"])
            max_conc = log(used_concentration_values[id_key]["ub"])
        
        concentration_vars[metabolite.id] = milp_model.continuous_var(
            lb=min_conc, ub=max_conc, name=f"x_{metabolite.id}"
        )

    # ________________________________________________________________________________________
    # 5.) Add variables for B and B2
    var_B = milp_model.continuous_var(name="var_B", lb=-milp_model.infinity, ub=milp_model.infinity)
    var_B2 = milp_model.continuous_var(name="var_B2", lb=-milp_model.infinity, ub=milp_model.infinity)

    # ________________________________________________________________________________________
    # 6.) Add dG0 terms: fixed values if known, variables if unknown.
    # This logic prioritizes user-provided dG0 values.
    # 1. If dG0 is provided for a reaction, that value is used directly.
    # 2. If dG0 is missing for a reaction, it checks its forward/reverse pair.
    #    - If the pair has a known dG0, the missing value is calculated from it.
    #    - If the pair's dG0 is also missing, a linked variable is created for the pair.
    # 3. A warning is issued if the user provides inconsistent dG0 values for a pair.
    
    dG0_terms: dict[str, Any] = {} # Will store floats, variables, or expressions
    processed_reactions = set()
    reaction_ids = {r.id for r in model.reactions}

    for reaction in model.reactions:
        if reaction.id in processed_reactions:
            continue

        # --- Case 1: Handle Forward/Reverse Pairs ---
        if "_FWD" in reaction.id:
            fwd_key = reaction.id
            rev_key = fwd_key.replace("_FWD", "_REV")

            if rev_key in reaction_ids:
                # This is a confirmed pair.
                dG0_fwd_val = dG0_values.get(fwd_key, {}).get("dG0")
                dG0_rev_val = dG0_values.get(rev_key, {}).get("dG0")

                if dG0_fwd_val is not None and dG0_rev_val is not None:
                    # Both values are provided. Use them directly.
                    dG0_terms[fwd_key] = dG0_fwd_val
                    dG0_terms[rev_key] = dG0_rev_val
                    # Check for physical consistency and warn the user if needed.
                    if not isclose(dG0_fwd_val + dG0_rev_val, 0, abs_tol=1e-6):
                        logger.error(
                            f"Warning: Inconsistent dG0 values provided for pair. "
                            f"{fwd_key}: {dG0_fwd_val}, {rev_key}: {dG0_rev_val}. "
                            f"Sum is not zero. Using provided values directly."
                        )
                        raise(ValueError)

                elif dG0_fwd_val is not None:
                    # Only FWD is known. Use it for FWD and infer REV.
                    dG0_terms[fwd_key] = dG0_fwd_val
                    dG0_terms[rev_key] = -dG0_fwd_val
                elif dG0_rev_val is not None:
                    # Only REV is known. Use it for REV and infer FWD.
                    dG0_terms[rev_key] = dG0_rev_val
                    dG0_terms[fwd_key] = -dG0_rev_val
                else:
                    # NEITHER is known. Create linked variables.
                    base_name = fwd_key.replace("_FWD", "")
                    dG0_base_var = milp_model.continuous_var(
                        # lb=-milp_model.infinity, ub=milp_model.infinity, name=f"dG0_{base_name}"
                        lb=-1000, ub=1000, name=f"dG0_{base_name}"
                    )
                    dG0_terms[fwd_key] = dG0_base_var
                    dG0_terms[rev_key] = -dG0_base_var
                
                processed_reactions.add(fwd_key)
                processed_reactions.add(rev_key)
                continue # Pair handled, move to next reaction in list

        # --- Case 2: Handle Standalone Reactions (or orphaned FWD/REV) ---
        # This block is reached for reactions like 'R1', 'R2_FWD' (if R2_REV is missing), etc.
        dG0_val = dG0_values.get(reaction.id, {}).get("dG0")
        if dG0_val is not None:
            # The reaction has a known dG0 value.
            dG0_terms[reaction.id] = dG0_val
        else:
            # The reaction has no known dG0 value. Create a variable for it.
            dG0_terms[reaction.id] = milp_model.continuous_var(
                # lb=-milp_model.infinity, ub=milp_model.infinity, name=f"dG0_{reaction.id}"
                lb=-1000, ub=1000, name=f"dG0_{reaction.id}"
            )
        processed_reactions.add(reaction.id)

    # ------------------ REVISED BIG-M BLOCK ------------------
    # 7 & 8.) Add thermodynamic constraints using the BIG-M METHOD
    z_variables, zb_variables = {}, {}
    fixed_z_count = 0
    unconstrained_bases = thermo_unconstrained_reactions
    if unconstrained_bases:
        logger.debug(f" [SOLVER] Filtering active constraints against: {unconstrained_bases}")
    
    for reaction in model.reactions:
        # --- A. ALWAYS Create z_var (Activity) and Link to Flux ---
        # This ensures the reaction exists in the model's logic, regardless of thermodynamics.
        z_var = milp_model.binary_var(name=f"z_var_{reaction.id}")
        z_variables[reaction.id] = z_var

        # --- BIG-M CONSTRAINT FOR FLUX ---
        # Replaces: add_indicator(z_var, flux_var == 0, active_value=0)
        # Logic: -M*z_var <= flux_var <= M*z_var
        milp_model.add_constraint(flux_variables[reaction.id] <= M_VALUE * z_var)
        milp_model.add_constraint(flux_variables[reaction.id] >= -M_VALUE * z_var)

        # If a reaction is fixed to zero flux, its z_var MUST be zero.
        flux_var = flux_variables[reaction.id]
        if flux_var.lb == 0 and flux_var.ub == 0:
            z_var.ub = 0 # Fix the z_var to 0
            fixed_z_count += 1

        # --- B. CHECK: Should we enforce thermodynamic feasibility? ---
        # Check if the reaction's base ID is in the exclusion list.
        # This handles both standalone IDs ('ATPM') and group IDs ('RNF_').
        reaction_base_id = reaction.id.replace("_FWD", "").replace("_REV", "")
        if reaction_base_id in unconstrained_bases:
            # If the reaction is unconstrained, we don't create a z_var
            # or any thermodynamic constraints for it. It only participates
            # in the mass balance constraints.
            logger.debug(f"  - Skipping thermodynamic constraints for unconstrained reaction: {reaction.id}")
            continue # Skip to the next reaction

        # --- C. Add Thermodynamic Constraints (Only for non-ignored reactions) ---
        dG0_term = dG0_terms[reaction.id]
        f_var = milp_model.continuous_var(name=f"f_var_{reaction.id}", lb=-milp_model.infinity, ub=milp_model.infinity)
        # Calculate Delta G' (f_var)
        conc_term = milp_model.sum(s * concentration_vars[m.id] for m, s in reaction.metabolites.items())
        milp_model.add_constraint(f_var == -dG0_term - (RT * conc_term))

        # --- BIG-M CONSTRAINT FOR DRIVING FORCE ---
        # Replaces: add_indicator(z_var, var_B <= f_var, active_value=1)
        # Logic: var_B <= f_var + M * (1 - z_var)
        if not bottleneck_analysis:
            milp_model.add_constraint(var_B <= f_var + M_VALUE * (1 - z_var))
        else:
            zb_var = milp_model.binary_var(name=f"zb_var_{reaction.id}")
            zb_variables[reaction.id] = zb_var
            # The bottleneck variable adds another M term to relax the constraint
            milp_model.add_constraint(var_B <= f_var + M_VALUE * (1 - z_var) + 5_000 * zb_var)
        
        # --- BIG-M FOR SUBNETWORK B2 ---
        if subnetwork_ids and any(reaction.id.startswith(p) for p in subnetwork_ids):
            milp_model.add_constraint(var_B2 <= f_var + M_VALUE * (1 - z_var))
    # -----------------------------------------------------------
    # if fixed_z_count > 0:
    #     print(f"--- Presolve Improvement: Fixed {fixed_z_count} z_variables to 0 based on reaction bounds. ---")

    if bottleneck_analysis:
        zb_sum_var = milp_model.continuous_var(name="zb_sum_var")
        milp_model.add_constraint(zb_sum_var == milp_model.sum(zb_variables.values()))

    # 9.) Constraints for co-factor alternatives and FWD/REV exclusivity    
    processed_reactions = set()
    for reaction in model.reactions:
        key = reaction.id
        if key in processed_reactions:
            continue
        
        cluster_keys = {key}
        
        # Step 1: Conditionally add the reverse reaction based on the flag
        if solver_config.with_loop_constraints:
            if "_FWD" in key:
                rev_key = key.replace("_FWD", "_REV")
                if rev_key in z_variables: cluster_keys.add(rev_key)
            elif "_REV" in key:
                fwd_key = key.replace("_REV", "_FWD")
                if fwd_key in z_variables: cluster_keys.add(fwd_key)
        
        # Step 2: Always add co-factor alternatives to the cluster
        for alt_group in co_factor_alternatives:
            if matches_prefix(key, alt_group):
                for r in model.reactions:
                    if matches_prefix(r.id, alt_group):
                        cluster_keys.add(r.id)

        # Step 3: Create the constraint if the cluster has more than one member
        valid_keys = [k for k in cluster_keys if k in z_variables]
        if len(valid_keys) > 1:
            milp_model.add_constraint(milp_model.sum(z_variables[k] for k in valid_keys) <= 1,
                                      ctname=f"exclusive_cluster_{valid_keys[0]}")
            processed_reactions.update(valid_keys)
    # ________________________________________________________________________________________
    # 10.) Add additional constraints (such as minimal growth)
    for variable_def, variable_value, constraint_name, sign in constraints:
        # Get the target variables from the docplex model
        target_vars = {v.name: v for v in milp_model.iter_variables()}
        
        if isinstance(variable_def, str):
            expression = target_vars[variable_def]
        elif isinstance(variable_def, list):
            expression = milp_model.sum(
                coeff * target_vars[var_id] for var_id, coeff in variable_def if var_id in target_vars
            )

        if sign == "geq": milp_model.add_constraint(expression >= variable_value, ctname=constraint_name)
        elif sign == "leq": milp_model.add_constraint(expression <= variable_value, ctname=constraint_name)
        elif sign == "eq": milp_model.add_constraint(expression == variable_value, ctname=constraint_name)
        elif sign == "range":
            milp_model.add_range(variable_value[0], expression, variable_value[1])


    # ________________________________________________________________________________________
    # 12.) Define objective expression 
    if bottleneck_analysis:
        milp_model.minimize(zb_sum_var)
    elif len(objective_var_ids) > 0:
        all_model_vars = {v.name: v for v in milp_model.iter_variables()}
        objective_expression = milp_model.sum(
            mult * all_model_vars[var_id] for mult, var_id in zip(objective_multipliers, objective_var_ids)
        )
        if objective_direction == "max":
            milp_model.maximize(objective_expression)
        else:
            milp_model.minimize(objective_expression)
    elif subnetwork_ids:
        milp_model.maximize(var_B2)
    else: # Default is maximizing driving force
        milp_model.maximize(var_B)

    # 12.5) Export the model to an LP file if a filename is provided
    # logging.info('----------------------------------------------------------------------------------')
    # try:
    #     logging.info(f"--- Exporting model to LP file: {'solution.lp'} ---")
    #     milp_model.export_as_lp('solution.lp')
    # except IOError as e:
    #     logging.info(f"Error: Could not write LP file to '{'solution.lp'}'. Reason: {e}")

    # ________________________________________________________________________________________
    # 13.) Apply the warm start if provided, right before solving
    if warm_start_solution and warm_start_solution.get("values"):
        try:
            # logger.debug("--- Applying warm start solution from previous step ---")
            start_solution = milp_model.new_solution()
            all_model_vars = {v.name: v for v in milp_model.iter_variables()}
            
            for var_name, var_value in warm_start_solution["values"].items():
                if var_name in all_model_vars and var_value is not None:
                    start_solution.add_var_value(all_model_vars[var_name], var_value)
            
            milp_model.add_mip_start(start_solution)
        except Exception as e:
            logger.warning(f"Could not apply warm start. Error: {e}")
    
    # 14.) Solve
    solution = milp_model.solve()

    # 15.) STORE RESULTS

    if solution:
        # Don't distinguish between different types of optimal such as 'integer optimal solution' or 'optimal solution'
        status_str = 'Optimal' if 'optimal' in milp_model.solve_details.status else milp_model.solve_details.status
        
        # Safely extract values
        solved_values = {}
        for var in milp_model.iter_variables():
            try:
                solved_values[var.name] = solution.get_value(var)
            except DOcplexException:
                # This correctly handles cases where a variable might not have a value
                pass 

        return SolverResult(
            status=status_str,
            objective_value=solution.objective_value,
            values=solved_values,
            message=f"Termination status: {status_str}"
        )

    else:
        # If no solution was found
        return SolverResult(
            status="Infeasible",
            message=f"No feasible solution found. Status: {milp_model.solve_details.status}"
        )
