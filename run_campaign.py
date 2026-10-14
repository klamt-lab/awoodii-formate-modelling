# Command for running:
# uv run --no-sync python run_campaign.py

# Import packages
from __future__ import annotations
import logging
from pathlib import Path
import cobra
import yaml
from mymodelingpackage.campaign.campaign import Campaign
from mymodelingpackage.experiments.utils import write_fitted_environment_library
from mymodelingpackage.utilities import load_settings_components

# Setup paths and logger
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
CONFIG_DIR = PROJECT_ROOT / "config"
logger = logging.getLogger()


# ============================================================================
# USER CONFIGURATION
# Run the ten raw fits, ten fitted characterizations, then ten formate-feed scans.
# ============================================================================
CAMPAIGNS_TO_RUN: dict[str, list[dict[str, list[str]]]] = {
    "01_Flux_Validation": [
        {"components": ["Wildtype", "MeasurementValidation_DStat_09_average", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_10_t668h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_11_t674h", "FluxValidation_ATPM"]},    
        {"components": ["Wildtype", "MeasurementValidation_DStat_12_t677h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_13_t692h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_14_t698h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_15_t701h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_16_t715h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_17_t717h", "FluxValidation_ATPM"]},
        {"components": ["Wildtype", "MeasurementValidation_DStat_18_t743h", "FluxValidation_ATPM"]},
    ],
    "02_Fitted_Characterizations": [
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_09_average", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_10_t668h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_11_t674h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_12_t677h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_13_t692h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_14_t698h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_15_t701h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_16_t715h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_17_t717h", "MDF_Characterization"]},
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_18_t743h", "MDF_Characterization"]},
    ],
    "03_NAD_Pool_Scan_with_Bottlenecks_PGCD0": [
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_17_t717h_formate_feed", "MDF_Bottlenecks_NADH_NAD_Lower_Bound_Scan_PGCD0"]},
    ],
    "04_NADH_NAD_Ratio_Exact_Value": [
        {"components": ["Wildtype", "Fitted_MeasurementValidation_DStat_17_t717h_formate_feed", "NADH_NAD_Ratio_Characterization"]},
    ],

}

COBRA_SOLVER = "cplex"
OVERWRITE_OUTPUT = False
GENERATE_SUMMARY = True
GENERATE_SUMMARY_EXCEL = True
CONTINUE_ON_ERROR = False
VERBOSE = False
GENERATE_FITTED_ENVIRONMENT_LIBRARY = True
FITTED_ENVIRONMENT_SOURCE_CAMPAIGN = "01_Flux_Validation"
FITTED_ENVIRONMENT_OUTPUT_PATH = CONFIG_DIR / "environments_d_stat_fitted_generated.yml"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if VERBOSE else logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logging.getLogger("cobra").setLevel(logging.WARNING)
    logging.getLogger("docplex").setLevel(logging.WARNING)

    total_experiments = sum(len(experiments) for experiments in CAMPAIGNS_TO_RUN.values())
    logger.info("Script configuration | campaigns=%d | experiments=%d | config_dir=%s | cobra_solver=%s | overwrite_output=%s | fitted_environment_output=%s", len(CAMPAIGNS_TO_RUN), total_experiments, CONFIG_DIR, COBRA_SOLVER, OVERWRITE_OUTPUT, FITTED_ENVIRONMENT_OUTPUT_PATH)

    # Load the raw and formate-feed environments before running the campaigns.
    configured_components = load_settings_components(str(CONFIG_DIR))
    configured_raw_names = [
        name
        for name in configured_components.get("environments", {})
        if name.startswith("MeasurementValidation_DStat_")
    ]
    if len(configured_raw_names) != 10:
        raise ValueError(f"Expected 10 configured raw D-stat environments, received {len(configured_raw_names)}.")
    configured_components["Campaigns"] = CAMPAIGNS_TO_RUN

    try:
        cobra.Configuration().solver = COBRA_SOLVER
        logger.info("COBRA solver set to %s.", COBRA_SOLVER)
    except Exception as exc:
        logger.warning("Could not set COBRA solver '%s': %s", COBRA_SOLVER, exc)

    # Fit raw measurements, characterize fitted states, then scan formate-feed states.
    campaign_names = list(CAMPAIGNS_TO_RUN)
    for index, campaign_name in enumerate(campaign_names, start=1):
        logger.info("Starting campaign %d/%d: %s", index, len(campaign_names), campaign_name)
        campaign = Campaign.from_components(
            campaign_to_run=campaign_name,
            all_components=configured_components,
            models_dir=str(MODELS_DIR),
            data_dir=str(DATA_DIR),
            output_root=str(RESULTS_DIR),
            overwrite_output=OVERWRITE_OUTPUT,
            generate_summary=GENERATE_SUMMARY,
            generate_summary_excel=GENERATE_SUMMARY_EXCEL,
            continue_on_error=CONTINUE_ON_ERROR,
        )
        results = campaign.run_all()
        logger.info("Finished campaign '%s' with %d experiment payload(s).", campaign_name, len(results))

        if GENERATE_FITTED_ENVIRONMENT_LIBRARY and campaign_name == FITTED_ENVIRONMENT_SOURCE_CAMPAIGN:
            failed_fits = [f"{name}: {payload.get('status')}" for name, payloads in results.items() for payload in payloads if payload.get("status") != "Success"]
            if failed_fits:
                raise RuntimeError(f"Flux validation failed; no fitted environments can be generated. Check {campaign.campaign_output_dir}: {'; '.join(failed_fits)}")
            fitted_library_path = write_fitted_environment_library(
                campaign_dir=campaign.campaign_output_dir,
                output_path=FITTED_ENVIRONMENT_OUTPUT_PATH,
            )
            fitted_document = yaml.safe_load(fitted_library_path.read_text(encoding="utf-8"))
            fitted_environments = fitted_document.get("environments")
            if not isinstance(fitted_environments, dict) or len(fitted_environments) != 10:
                raise ValueError(f"Expected 10 generated fitted environments: {fitted_library_path}")
            configured_components["environments"].update(fitted_environments)
            logger.info("Generated and loaded %d fitted environments: %s", len(fitted_environments), fitted_library_path)

            # Keep each formate-feed template's concentration bounds and use
            # the fresh fit's flux bounds for the following scan campaign.
            for fitted_name, fitted in fitted_environments.items():
                feed = configured_components["environments"][f"{fitted_name}_formate_feed"]
                for field in ("substrates", "products", "substrate_product_adjustments"):
                    feed[field] = fitted[field]
            logger.info("Updated %d formate-feed environments with fitted fluxes.", len(fitted_environments))
