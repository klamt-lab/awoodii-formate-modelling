import logging
from pydantic import TypeAdapter
from mymodelingpackage.structures import AnyAnalysisConfig

logger = logging.getLogger(__name__)

MCS_ONLY_ANALYSIS_PARAMS = frozenset(
    {
        "max_cost",
        "max_solutions",
        "time_limit",
        "yield_fraction",
        "calculate_yield_metrics",
        "exchange_protection",
        "transport_suffixes",
        "save_sds_files",
        "export_ko_models_yml_for",
        "ko_models_yml_filename",
        "cobra_solver",
        "solver",
    }
)

MCS_ONLY_CONFIG_PARAMS = frozenset(
    {
        "suppress_constraints",
        "protect_constraints",
        "targetable_reactions",
    }
)

THERMO_ONLY_ANALYSIS_PARAMS = frozenset(
    {
        "min_mdf",
        "subnetwork",
        "thermo_unconstrained_reactions",
        "thermo_adjustments",
    }
)

POST_ANALYSIS_PARAMS = frozenset(
    {
        "run_cva",
        "run_fva",
        "run_min_optimization",
        "run_max_optimization",
        "run_metabolite_ratios",
        "run_bottleneck_analysis",
        "run_bottleneck_set_analysis",
        "run_true_bottleneck_check",
        "run_metabolite_bottleneck_analysis",
        "apply_specificity_constraints",
    }
)

def build_analysis_config(raw_components: dict) -> dict[str, AnyAnalysisConfig]:
    """
    Parses the 'analyses' block of the YAML into strictly typed Pydantic models.
    Automatically flattens nested experiment type declarations.
    """
    analysis_dict = raw_components.get("analyses", {})
    adapter = TypeAdapter(AnyAnalysisConfig)
    
    configs = {}
    for component_name, data in analysis_dict.items():
        
        # 1. Handle Nested vs Flat structures
        if "type" in data:
            # Case A: Flat structure (e.g., pure standalone thermo/mcs)
            parsed_data = dict(data)
        else:
            # Case B: Nested structure (e.g., 'Characterization': { ... })
            if len(data) != 1:
                raise ValueError(f"Ambiguous analysis structure for '{component_name}'. Expected a single experiment type key.")
            
            exp_type = list(data.keys())[0]       # Extracts "Characterization"
            parsed_data = dict(data[exp_type])    # Grabs the inner dictionary
            parsed_data["type"] = exp_type        # Explicitly adds the 'type' key for Pydantic

        # Solver selection is consumed by the campaign builder before analysis
        # config validation. It is not part of the executable experiment schema.
        parsed_data.pop("solver", None)
        parsed_data.pop("solver_settings", None)

        # 2. Inject the master YAML key (e.g., 'test_characterization')
        parsed_data["analysis_component"] = component_name

        # AnalysisParamsConfig is intentionally shared by all experiment types,
        # so catch explicitly misplaced analysis_params keys before Pydantic
        # applies defaults and the original YAML intent is lost.
        raw_analysis_params = parsed_data.get("analysis_params", {})
        if isinstance(raw_analysis_params, dict):
            raw_analysis_param_keys = set(raw_analysis_params)
            analysis_param_keys = raw_analysis_param_keys | (set(parsed_data) - {"analysis_params", "analysis_component", "type"})
            if parsed_data["type"] == "MCS":
                misplaced_config_keys = sorted(raw_analysis_param_keys & MCS_ONLY_CONFIG_PARAMS)
                if misplaced_config_keys:
                    raise ValueError(
                        f"Analysis config '{component_name}' has MCS config key(s) under analysis_params; "
                        f"move them directly under the MCS config: {', '.join(misplaced_config_keys)}"
                    )
                invalid_keys = sorted(analysis_param_keys & (THERMO_ONLY_ANALYSIS_PARAMS | POST_ANALYSIS_PARAMS))
                if invalid_keys:
                    raise ValueError(
                        f"Analysis config '{component_name}' has unsupported analysis_params "
                        f"key(s) in an MCS experiment: {', '.join(invalid_keys)}"
                    )
            else:
                invalid_keys = sorted(analysis_param_keys & (MCS_ONLY_ANALYSIS_PARAMS | MCS_ONLY_CONFIG_PARAMS))
                if invalid_keys:
                    raise ValueError(
                        f"Analysis config '{component_name}' has MCS-only analysis_params "
                        f"key(s) in a {parsed_data['type']} experiment: {', '.join(invalid_keys)}"
                    )
            
        # 3. Validate and Build Data Object
        try:
            configs[component_name] = adapter.validate_python(parsed_data)
        except Exception as e:
            logger.error(f"Validation failed for analysis config '{component_name}': {e}")
            raise
            
    return configs
