"""Scenario construction utilities for metabolic modeling.

This module implements `ScenariosBuilder`, a factory class that transforms 
raw configuration objects into fully prepared, engine-agnostic biological 
scenarios (`BiologicalScenario`).

Key architectural behavior and method mapping (The Prototype Pattern):

1. **Initialization (`__init__`)**: 
   Sets up the directory paths and initializes the empty `_scenario_templates` 
   memory cache.

2. **Template Generation & Caching (`_get_or_create_template`)**: 
   Loads massive template assets exactly once per model/environment pair. 
   - Reads the heavy SBML model (`cobra.io.read_sbml_model`) and JSON files (dG0, concentrations).
   - Applies static base model adjustments (e.g., global gene knockouts) to create the Template.
   - Caches this fully formed Template in memory to bypass future disk I/O.

3. **Cloning & Customization (`build_scenario`)**: 
   Creates a strictly independent instance for a specific substrate/product pair.
   - Clones (deep-copies) the cached Template.
   - Calls `_apply_scenario_bounds` to enforce the specific substrate/product 
     exchange bounds and interaction rules.
   - Calls `_apply_concentration_adjustments` to overlay specific metabolite limits.
   - Returns a single, fully-resolved `BiologicalScenario`.

4. **Batch Execution (`build_all_scenarios`)**: 
   Automates the cloning process. It iterates through the Cartesian product of 
   all defined substrates and products in the environment config, rapidly stamping 
   out a complete list of ready-to-solve `BiologicalScenario` objects.
"""

import copy
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any

import cobra
from cobrak.dataclasses import Model as CobrakModel
from cobrak.io import json_load as cobrak_json_load
from cobrak.io import load_annotated_sbml_model_as_cobrak_model
from cobrak.utilities import get_base_id

import mymodelingpackage.structures as structs
from mymodelingpackage.utilities import json_load

_MODEL_SETTINGS_FIELDS = set(structs.ModelConfig.model_fields.keys()) - {"extra"}
_ENV_SETTINGS_FIELDS = set(structs.EnvironmentConfig.model_fields.keys()) - {"extra"}
logger = logging.getLogger(__name__)



class ScenariosBuilder:
    """Builds and manages independent biological scenario contexts.

    The builder receives a `ModelConfig` and `EnvironmentConfig`, retrieves 
    or creates a base template, and generates scenario-specific states for 
    requested substrate/product pairs.

    By utilizing an internal template dictionary (`_scenario_templates`), it 
    avoids heavy disk I/O and redundant constraint parsing when multiple 
    scenarios share the same foundational biological model and environment.

    Attributes:
        models_dir (str): Directory path containing SBML model files.
        data_dir (str): Directory path containing JSON data (dG0, concentrations).
        apply_base_adjustments (bool): Flag to apply static base adjustments to templates.
    """

    def __init__(
        self,
        models_dir: str = "./Models",
        data_dir: str = "./Data",
        apply_base_adjustments: bool = True,
    ) -> None:
        """Initializes the ScenariosBuilder with component definitions and paths.

        Args:
            components: Dictionary returned by the component loader, typically 
                containing 'models' and 'environments' top-level keys.
            models_dir: Directory containing the `.xml` or `.xml.gz` model files. 
                Defaults to "./Models".
            data_dir: Directory containing the `.json` thermodynamic and 
                concentration files. Defaults to "./Data".
            apply_base_adjustments: If True, applies `model_adjustments` 
                (like base gene KOs) to the template model once before caching. 
                Defaults to True.
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.apply_base_adjustments = apply_base_adjustments
        # Cache holding the prototype templates to clone from.
        # Key: (model_component_name, environment_component_name)
        # Value: Dictionary containing the loaded model, configs, and JSON data.
        self._scenario_templates: dict[tuple[str, str, bool], dict[str, Any]] = {}
        
        logger.info("   ScenariosBuilder initialized (models_dir=%s, data_dir=%s, apply_base_adjustments=%s)", self.models_dir, self.data_dir, self.apply_base_adjustments)


    def build_all_scenarios(
        self,
        model_component: str,
        environment_component: str,
        model_config: structs.ModelConfig,
        environment_config: structs.EnvironmentConfig,
        include_cobrak_model: bool = False,
        scenario_indices: set[int] | None = None,
    ) -> list[structs.BiologicalScenario]:
        """Builds all valid substrate/product scenarios for a given configuration.

        Iterates through the Cartesian product of all defined substrates and products, 
        generating a discrete `BiologicalScenario` for each combination.

        Args:
            model_component: String identifier for the model.
            environment_component: String identifier for the environment.
            model_config: The parsed ModelConfig struct.
            environment_config: The parsed EnvironmentConfig struct.

        Returns:
            A list of fully prepared, independent `BiologicalScenario` instances.
        """
        logger.info("Building all scenarios for model_component=%s, environment_component=%s", 
            model_component, environment_component)

        self._get_or_create_template(
            model_component=model_component,
            environment_component=environment_component,
            model_config=model_config,
            environment_config=environment_config,
            include_cobrak_model=include_cobrak_model,
        )

        substrates = environment_config.substrates
        products = environment_config.products
        total = len(substrates) * len(products)

        logger.info("   Found %d substrates, %d products => %d scenario combinations", len(substrates), len(products), total)

        scenarios: list[structs.BiologicalScenario] = []
        built = 0
        for substrate_name in substrates:
            for product_name in products:
                built += 1
                if scenario_indices is not None and built - 1 not in scenario_indices:
                    continue
                logger.debug("   Building scenario %d/%d: substrate=%s, product=%s", built, total, substrate_name, product_name)
                scenarios.append(
                    self.build_scenario(
                        model_component=model_component,
                        environment_component=environment_component,
                        model_config=model_config,
                        environment_config=environment_config,
                        substrate_name=substrate_name, 
                        product_name=product_name,
                        include_cobrak_model=include_cobrak_model,
                    )
                )
        logger.info("   Finished building %d BiologicalScenario objects", len(scenarios))
        return scenarios

    def build_scenario(
        self,
        model_component: str,
        environment_component: str,
        model_config: structs.ModelConfig,
        environment_config: structs.EnvironmentConfig,
        substrate_name: str,
        product_name: str,
        include_cobrak_model: bool = False,
    ) -> structs.BiologicalScenario:
        """Builds a single scenario context for a specific substrate/product pair.

        Args:
            model_component: String identifier for the model.
            environment_component: String identifier for the environment.
            model_config: The parsed ModelConfig struct.
            environment_config: The parsed EnvironmentConfig struct.
            substrate_name: The specific substrate key to apply.
            product_name: The specific product key to apply.

        Returns:
            A prepared, engine-agnostic `BiologicalScenario` data object.
        """
        template = self._get_or_create_template(
            model_component=model_component,
            environment_component=environment_component,
            model_config=model_config,
            environment_config=environment_config,
            include_cobrak_model=include_cobrak_model,
        )

        # Deep-copy template assets so each scenario stays independent.
        scenario_model = copy.deepcopy(template["model"])
        scenario_cobrak_model = copy.deepcopy(template["cobrak_model"])
        scenario_concentrations = copy.deepcopy(template["concentrations"])

        # Step 1: apply reaction bound changes.
        diagnostics_environment = self._apply_scenario_bounds(
            model=scenario_model,
            environment_config=environment_config,
            substrate_name=substrate_name,
            product_name=product_name,
        )

        # Step 2: apply concentration changes.
        self._apply_concentration_adjustments(
            concentrations=scenario_concentrations,
            environment_config=environment_config,
            substrate_name=substrate_name,
            product_name=product_name,
        )

        # The clean COBRApy model is authoritative for scenario bounds. Mirror
        # its final reversible bounds onto every matching directional and
        # enzyme-specific COBRA-k reaction variant after all environment rules
        # have been applied.
        if scenario_cobrak_model is not None and not model_config.cobrak_preserve_model_state:
            self._synchronize_cobrak_model(
                cobra_model=scenario_model,
                cobrak_model=scenario_cobrak_model,
                concentrations=scenario_concentrations,
                dG0=template["dG0"],
            )
        elif scenario_cobrak_model is not None:
            logger.info("Preserved saved COBRA-k model state while preparing scenario='%s__%s__%s__%s'.", model_component, environment_component, substrate_name, product_name)

        scenario_id = (
            f"{model_component}__{environment_component}__"
            f"{substrate_name}__{product_name}"
        )
        logger.debug("   Scenario id generated: %s", scenario_id)

        biological_scenario = structs.BiologicalScenario(
            scenario_id=scenario_id,
            model_component=model_component,
            environment_component=environment_component,
            substrate_name=substrate_name,
            product_name=product_name,
            model=scenario_model,
            cobrak_model=scenario_cobrak_model,
            concentrations=scenario_concentrations,
            dG0=copy.deepcopy(template["dG0"]),
            model_conf=model_config,
            environment_conf=environment_config,
            diagnostics_environment=diagnostics_environment,
        )
        
        logger.info("   BiologicalScenario ready: %s", scenario_id)
        return biological_scenario

    def _get_or_create_template(
        self, 
        model_component: str, 
        environment_component: str,
        model_config: structs.ModelConfig,
        environment_config: structs.EnvironmentConfig,
        include_cobrak_model: bool = False,
    ) -> dict[str, Any]:
        """Retrieves an existing template from memory, or creates one from disk.

        Args:
            model_component: Target model component string.
            environment_component: Target environment component string.
            model_config: The parsed model configuration struct.
            environment_config: The parsed environment configuration struct.

        Returns:
            A dictionary containing the loaded COBRA model and JSON data.
        """
        _scenario_key = (model_component, environment_component, include_cobrak_model)
        if _scenario_key in self._scenario_templates:
            logger.debug("   Scenario template found for model=%s, env=%s", model_component, environment_component)
            return self._scenario_templates[_scenario_key]

        logger.info("   Scenario template not found for model=%s, env=%s; loading template assets", model_component, environment_component)
        logger.info("   Loading settings for model_component='%s', environment_component='%s'.", model_component, environment_component)

        # ---- loading model ----
        model_path = os.path.join(self.models_dir, model_config.model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info("Loading model from %s", model_path)
        if include_cobrak_model:
            cobra.Configuration().solver = "cplex"
        model = cobra.io.read_sbml_model(model_path)
        certificate_validation_model_cobrapy_pathway = model_config.extra.get("certificate_validation_model_cobrapy_pathway")
        if certificate_validation_model_cobrapy_pathway is not None:
            if not isinstance(certificate_validation_model_cobrapy_pathway, dict) or certificate_validation_model_cobrapy_pathway.get("status") != "passed":
                raise ValueError(f"Model component '{model_component}' has an invalid COBRApy validation certificate.")
            if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != certificate_validation_model_cobrapy_pathway.get("sha256"):
                raise ValueError(f"COBRApy model file for component '{model_component}' no longer matches its validation certificate. Rebuild and re-register the model.")

        # COBRA-k assets are loaded only for experiments that explicitly use
        # the COBRA-k solver. Other campaigns therefore retain their previous
        # model-loading cost and dependency behavior.
        cobrak_model = None
        if include_cobrak_model:
            if not model_config.cobrak_model_file:
                raise ValueError(f"Model component '{model_component}' requires 'cobrak_model_file' for a COBRA-k run.")
            cobrak_model_path = os.path.join(self.models_dir, model_config.cobrak_model_file)
            if not os.path.exists(cobrak_model_path):
                raise FileNotFoundError(f"COBRA-k model file not found: {cobrak_model_path}")
            certificate_validation_model_cobrak_pathway = model_config.extra.get("certificate_validation_model_cobrak_pathway")
            id_profile_cobrak_kinetics = model_config.extra.get("id_profile_cobrak_kinetics")
            if id_profile_cobrak_kinetics and not isinstance(certificate_validation_model_cobrak_pathway, dict):
                raise ValueError(f"Model component '{model_component}' uses reviewed COBRA-k profile '{id_profile_cobrak_kinetics}' but has no COBRA-k validation certificate.")
            if certificate_validation_model_cobrak_pathway is not None:
                if certificate_validation_model_cobrak_pathway.get("status") != "passed":
                    raise ValueError(f"Model component '{model_component}' has an invalid COBRA-k validation certificate.")
                actual_sha256_model_cobrak_pathway = hashlib.sha256(Path(cobrak_model_path).read_bytes()).hexdigest()
                if actual_sha256_model_cobrak_pathway != certificate_validation_model_cobrak_pathway.get("sha256_model_cobrak_pathway"):
                    raise ValueError(f"COBRA-k model file for component '{model_component}' no longer matches its validation certificate. Rebuild and re-register the model.")
                if certificate_validation_model_cobrak_pathway.get("sha256_model_cobrapy_pathway") != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
                    raise ValueError(f"COBRA-k model for component '{model_component}' was certified against a different COBRApy model.")
                if certificate_validation_model_cobrak_pathway.get("id_profile_cobrak_kinetics") != id_profile_cobrak_kinetics:
                    raise ValueError(f"COBRA-k profile ID for component '{model_component}' does not match its validation certificate.")
            logger.info("Loading COBRA-k model from %s", cobrak_model_path)
            if Path(cobrak_model_path).suffix.lower() == ".json":
                cobrak_model = cobrak_json_load(cobrak_model_path, CobrakModel)
            else:
                cobrak_model = load_annotated_sbml_model_as_cobrak_model(cobrak_model_path, do_model_fullsplit=False, exclude_enzyme_constraints=False)

        # ---- applying base adjustments ----
        if self.apply_base_adjustments:
            adjustments = model_config.model_adjustments
            logger.info("Applying %d base model adjustments", len(adjustments))
            unknown_adjustment_ids = sorted(set(adjustments) - {reaction.id for reaction in model.reactions})
            if certificate_validation_model_cobrapy_pathway is not None and unknown_adjustment_ids:
                raise ValueError(f"Model component '{model_component}' introduces unknown reaction IDs in model_adjustments: {unknown_adjustment_ids}.")
            for rid, bounds in adjustments.items():
                try:
                    model.reactions.get_by_id(rid).bounds = tuple(bounds)
                    logger.debug("Applied base adjustment: %s -> %s", rid, tuple(bounds))
                except KeyError:
                    logger.warning("Reaction '%s' from model_adjustments not found in model", rid)
        else:
            logger.debug("Base model adjustments are disabled")

        # ---- applying reaction string overrides ----
        reaction_string_overrides = model_config.reaction_string_overrides
        if reaction_string_overrides:
            logger.info("Applying %d reaction string override(s)", len(reaction_string_overrides))
            unknown_override_ids = sorted(set(reaction_string_overrides) - {reaction.id for reaction in model.reactions})
            if certificate_validation_model_cobrapy_pathway is not None and unknown_override_ids:
                raise ValueError(f"Model component '{model_component}' introduces unknown reaction IDs in reaction_string_overrides: {unknown_override_ids}.")
            for rid, reaction_string in reaction_string_overrides.items():
                try:
                    model.reactions.get_by_id(rid).build_reaction_from_string(reaction_string)
                    logger.debug("Applied reaction string override: %s -> %s", rid, reaction_string)
                except KeyError:
                    logger.warning("Reaction '%s' from reaction_string_overrides not found in model", rid)
        else:
            logger.debug("No reaction string overrides configured")

        # ---- loading concentrations ----
        concentrations: dict[str, Any] = {}
        conc_file = environment_config.concentrations_file
        if conc_file:
            conc_path = os.path.join(self.data_dir, conc_file)
            if not os.path.exists(conc_path):
                raise FileNotFoundError(f"Concentration file not found: {conc_path}")
            logger.info("Loading concentrations from %s", conc_path)
            concentrations = json_load(conc_path)
            logger.debug("Loaded %d concentration entries", len(concentrations))
        else:
            logger.debug("No concentrations_file configured; using empty concentrations")


        # ---- loading dG0 ----
        dG0: dict[str, Any] | None = None
        dG0_file = model_config.dG0_file
        if dG0_file:
            dG0_path = os.path.join(self.data_dir, dG0_file)
            if not os.path.exists(dG0_path):
                raise FileNotFoundError(f"dG0 file not found: {dG0_path}")
            logger.info("Loading dG0 from %s", dG0_path)
            dG0 = json_load(dG0_path)
            logger.debug("Loaded %d dG0 entries", len(dG0))
        else:
            logger.debug("No dG0_file configured for this model")

        if cobrak_model is not None and not model_config.cobrak_preserve_model_state:
            if model_config.extra.get("certificate_validation_model_cobrak_pathway") is None:
                self._validate_dual_models(model, cobrak_model, model_config, environment_config)
            else:
                logger.info("Accepted hash-bound clean/COBRA-k structural validation for model component '%s'.", model_component)
            self._synchronize_cobrak_model(model, cobrak_model, concentrations, dG0)
        elif cobrak_model is not None:
            logger.info("Preserving saved COBRA-k model state without clean-model validation or synchronization: model='%s'.", cobrak_model_path)

        # ---- Store Template ----
        template = {
            "model": model,
            "cobrak_model": cobrak_model,
            "concentrations": concentrations,
            "dG0": dG0,
        }
        self._scenario_templates[_scenario_key] = template
        logger.info("   Cached template assets for model=%s, env=%s", model_component, environment_component)
        return template

    @staticmethod
    def _validate_dual_models(
        cobra_model: cobra.Model,
        cobrak_model: Any,
        model_config: structs.ModelConfig,
        environment_config: structs.EnvironmentConfig,
    ) -> None:
        """Validate that clean and annotated files describe one base network.

        The clean COBRApy model remains the authoritative user-facing model.
        The COBRA-k model may split a reversible base reaction into forward,
        reverse, and enzyme-specific variants, but every such variant must map
        back to the same metabolites, compartment assignments, and signed
        stoichiometry as its clean base reaction.

        Args:
            cobra_model: Clean reversible COBRApy model used by MMP2 and CNApy.
            cobrak_model: Annotated COBRA-k model containing directional and
                optionally enzyme-specific reaction variants.
            model_config: Model configuration containing biomass and model-file
                metadata.
            environment_config: Environment configuration containing configured
                product reaction IDs.

        Raises:
            ValueError: If reaction IDs, metabolites, compartments,
                stoichiometry, biomass, or product IDs differ between models.
        """
        logger.info("Starting clean/COBRA-k structural validation: clean_reactions=%d, cobrak_variants=%d, clean_metabolites=%d, cobrak_metabolites=%d.", len(cobra_model.reactions), len(cobrak_model.reactions), len(cobra_model.metabolites), len(cobrak_model.metabolites))

        # ------------------------------------------------------------------
        # STEP 1: Collapse every COBRA-k directional/enzyme reaction ID to its
        # clean base reaction ID and retain all variants for later checks.
        # ------------------------------------------------------------------
        base_to_variants: dict[str, list[str]] = {}
        for reaction_id in cobrak_model.reactions:
            base_id = get_base_id(reaction_id, cobrak_model.fwd_suffix, cobrak_model.rev_suffix, cobrak_model.reac_enz_separator)
            base_to_variants.setdefault(base_id, []).append(reaction_id)
        logger.debug("Built dual-model reaction mapping for %d base reaction(s); sample=%s.", len(base_to_variants), list(base_to_variants.items())[:5])

        # ------------------------------------------------------------------
        # STEP 2: Require exact base-reaction set equality.
        # Missing or extra base IDs indicate that bounds/objectives cannot be
        # translated unambiguously between the two models.
        # ------------------------------------------------------------------
        cobra_reaction_ids = {reaction.id for reaction in cobra_model.reactions if reaction.id != "cobrak_global_settings"}
        cobrak_base_ids = set(base_to_variants)
        missing_in_cobrak = sorted(cobra_reaction_ids - cobrak_base_ids)
        extra_in_cobrak = sorted(cobrak_base_ids - cobra_reaction_ids)
        if missing_in_cobrak or extra_in_cobrak:
            raise ValueError(f"Clean/COBRA-k reaction mismatch: missing_in_cobrak={missing_in_cobrak[:20]}, extra_in_cobrak={extra_in_cobrak[:20]}.")
        logger.debug("Dual-model reaction ID validation passed for %d base reaction(s).", len(cobra_reaction_ids))

        # ------------------------------------------------------------------
        # STEP 3: Require exact metabolite ID equality before compartment and
        # reaction-level stoichiometry checks.
        # ------------------------------------------------------------------
        cobra_metabolite_ids = {metabolite.id for metabolite in cobra_model.metabolites}
        cobrak_metabolite_ids = set(cobrak_model.metabolites)
        if cobra_metabolite_ids != cobrak_metabolite_ids:
            raise ValueError(f"Clean/COBRA-k metabolite mismatch: missing_in_cobrak={sorted(cobra_metabolite_ids - cobrak_metabolite_ids)[:20]}, extra_in_cobrak={sorted(cobrak_metabolite_ids - cobra_metabolite_ids)[:20]}.")
        logger.debug("Dual-model metabolite ID validation passed for %d metabolite(s).", len(cobra_metabolite_ids))

        # ------------------------------------------------------------------
        # STEP 4: Verify compartment labels one metabolite at a time.
        # ------------------------------------------------------------------
        tolerance = 1e-9
        for metabolite in cobra_model.metabolites:
            cobrak_metabolite = cobrak_model.metabolites[metabolite.id]
            if metabolite.compartment != cobrak_metabolite.compartment:
                raise ValueError(f"Compartment mismatch for metabolite '{metabolite.id}': clean='{metabolite.compartment}', cobrak='{cobrak_metabolite.compartment}'.")
        logger.debug("Dual-model compartment validation passed for %d metabolite(s).", len(cobra_metabolite_ids))

        # ------------------------------------------------------------------
        # STEP 5: Compare every COBRA-k variant against signed clean
        # stoichiometry. Reverse variants are multiplied by -1 before the
        # comparison; enzyme variants retain their directional sign.
        # ------------------------------------------------------------------
        for reaction in cobra_model.reactions:
            if reaction.id == "cobrak_global_settings":
                continue
            variant_ids = base_to_variants[reaction.id]
            expected = {metabolite.id: float(coefficient) for metabolite, coefficient in reaction.metabolites.items()}
            for variant_id in variant_ids:
                variant = cobrak_model.reactions[variant_id]
                direction = -1.0 if variant_id.endswith(cobrak_model.rev_suffix) else 1.0
                observed = {metabolite_id: direction * float(coefficient) for metabolite_id, coefficient in variant.stoichiometries.items()}
                if set(expected) != set(observed) or any(abs(expected[metabolite_id] - observed[metabolite_id]) > tolerance for metabolite_id in expected):
                    raise ValueError(f"Stoichiometry mismatch for base reaction '{reaction.id}' using COBRA-k variant '{variant_id}'.")
            logger.debug("Validated base reaction='%s' against %d COBRA-k variant(s).", reaction.id, len(variant_ids))

        # ------------------------------------------------------------------
        # STEP 6: Confirm configured biomass and product reaction IDs exist in
        # the shared base-reaction namespace used by experiment configs.
        # ------------------------------------------------------------------
        required_ids = set()
        if model_config.biomass_id:
            required_ids.add(model_config.biomass_id)
        required_ids.update(str(product.get("id")) for product in environment_config.products.values() if product.get("id"))
        missing_required = sorted(required_ids - cobra_reaction_ids)
        if missing_required:
            raise ValueError(f"Configured biomass/product reaction ids are absent from the dual model: {missing_required}.")
        logger.info("Completed clean/COBRA-k structural validation: base_reactions=%d, variants=%d, metabolites=%d, enzymes=%d, required_ids=%s.", len(cobra_reaction_ids), len(cobrak_model.reactions), len(cobra_metabolite_ids), len(cobrak_model.enzymes), sorted(required_ids))

    @staticmethod
    def _synchronize_cobrak_model(
        cobra_model: cobra.Model,
        cobrak_model: Any,
        concentrations: dict[str, Any],
        dG0: dict[str, Any] | None,
    ) -> None:
        """Synchronize scenario state from the clean model into COBRA-k.

        Clean reversible bounds are projected onto directional COBRA-k
        variants. Concentration values are converted from linear concentration
        units to natural-log bounds, and base-reaction dG0 values are copied to
        all mapped variants with reverse-direction sign inversion.

        Args:
            cobra_model: Scenario-specific clean model after all bound changes.
            cobrak_model: Scenario-specific COBRA-k model to mutate in place.
            concentrations: Metabolite concentration dictionary with optional
                ``DEFAULT`` bounds and metabolite-specific overrides.
            dG0: Optional reaction free-energy dictionary. Keys may be base
                reaction IDs or exact COBRA-k variant IDs.

        Raises:
            ValueError: If a clean reaction cannot be mapped, concentration
                bounds are invalid, or a dG0 key cannot be resolved.
        """
        logger.info("Synchronizing clean scenario state into COBRA-k: clean_reactions=%d, cobrak_variants=%d, concentration_entries=%d, dG0_entries=%d.", len(cobra_model.reactions), len(cobrak_model.reactions), len(concentrations), len(dG0 or {}))

        # ------------------------------------------------------------------
        # STEP 1: Build the same base-to-variant mapping used by validation.
        # ------------------------------------------------------------------
        base_to_variants: dict[str, list[str]] = {}
        for reaction_id in cobrak_model.reactions:
            base_id = get_base_id(reaction_id, cobrak_model.fwd_suffix, cobrak_model.rev_suffix, cobrak_model.reac_enz_separator)
            base_to_variants.setdefault(base_id, []).append(reaction_id)
        logger.debug("Synchronization mapping contains %d base reaction(s).", len(base_to_variants))

        # ------------------------------------------------------------------
        # STEP 2: Materialize a missing direction for boundary reactions. Some
        # native COBRA-k files omit a direction that was closed when exported.
        # A later medium change must still be able to open that exchange.
        # ------------------------------------------------------------------
        for reaction in cobra_model.reactions:
            if reaction.id == "cobrak_global_settings":
                continue
            lower_bound, upper_bound = reaction.bounds
            if reaction.id not in base_to_variants:
                raise ValueError(f"Cannot synchronize clean reaction '{reaction.id}': no COBRA-k reaction variant exists.")

            variants = base_to_variants[reaction.id]
            missing_direction = None
            source_suffix = None
            if lower_bound < 0.0 and not any(variant_id.endswith(cobrak_model.rev_suffix) for variant_id in variants):
                missing_direction = cobrak_model.rev_suffix
                source_suffix = cobrak_model.fwd_suffix
            elif upper_bound > 0.0 and not any(variant_id.endswith(cobrak_model.fwd_suffix) for variant_id in variants):
                missing_direction = cobrak_model.fwd_suffix
                source_suffix = cobrak_model.rev_suffix

            if missing_direction is not None:
                source_ids = [variant_id for variant_id in variants if variant_id.endswith(source_suffix)]
                if not reaction.boundary:
                    logger.debug("Kept one-direction COBRA-k reaction unchanged: base='%s', requested_missing_suffix='%s'.", reaction.id, missing_direction)
                    continue
                if len(source_ids) != 1:
                    logger.warning("Could not reconstruct missing COBRA-k boundary direction: base='%s', opposite_variants=%s.", reaction.id, source_ids)
                    continue
                source = cobrak_model.reactions[source_ids[0]]
                if source.enzyme_reaction_data is not None:
                    logger.warning("Could not reconstruct enzyme-constrained COBRA-k boundary direction: base='%s', source='%s'.", reaction.id, source_ids[0])
                    continue

                variant_id = f"{reaction.id}{missing_direction}"
                variant = copy.deepcopy(source)
                variant.stoichiometries = {metabolite_id: -coefficient for metabolite_id, coefficient in source.stoichiometries.items()}
                variant.dG0 = -source.dG0 if source.dG0 is not None else None
                variant.min_flux = 0.0
                variant.max_flux = max(0.0, -float(lower_bound)) if missing_direction == cobrak_model.rev_suffix else max(0.0, float(upper_bound))
                cobrak_model.reactions[variant_id] = variant
                variants.append(variant_id)
                logger.info("Created missing COBRA-k boundary direction: base='%s', source='%s', created='%s', max_flux=%g.", reaction.id, source_ids[0], variant_id, variant.max_flux)

        # ------------------------------------------------------------------
        # STEP 3: Project reversible clean bounds onto every directional and
        # enzyme-specific COBRA-k reaction variant.
        # ------------------------------------------------------------------
        for reaction in cobra_model.reactions:
            if reaction.id == "cobrak_global_settings":
                continue
            lower_bound, upper_bound = reaction.bounds
            for variant_id in base_to_variants[reaction.id]:
                variant = cobrak_model.reactions[variant_id]
                if variant_id.endswith(cobrak_model.rev_suffix):
                    variant.min_flux = 0.0
                    variant.max_flux = max(0.0, -float(lower_bound))
                elif variant_id.endswith(cobrak_model.fwd_suffix):
                    variant.min_flux = 0.0
                    variant.max_flux = max(0.0, float(upper_bound))
                else:
                    variant.min_flux = float(lower_bound)
                    variant.max_flux = float(upper_bound)
                logger.debug("Synchronized bounds: base='%s' clean=(%g,%g) variant='%s' cobrak=(%g,%g).", reaction.id, lower_bound, upper_bound, variant_id, variant.min_flux, variant.max_flux)

        # ------------------------------------------------------------------
        # STEP 4: Apply default or metabolite-specific concentration bounds.
        # COBRA-k stores natural logarithms, while MMP2 configuration uses
        # positive linear concentration values.
        # ------------------------------------------------------------------
        default_concentration = concentrations.get("DEFAULT", {})
        applied_concentration_count = 0
        for metabolite_id, metabolite in cobrak_model.metabolites.items():
            bounds = concentrations.get(metabolite_id, default_concentration)
            if not bounds:
                continue
            minimum = float(bounds.get("min", math.exp(metabolite.log_min_conc)))
            maximum = float(bounds.get("max", math.exp(metabolite.log_max_conc)))
            if minimum <= 0.0 or maximum <= 0.0 or minimum > maximum:
                raise ValueError(f"Invalid concentration bounds for '{metabolite_id}': min={minimum}, max={maximum}.")
            metabolite.log_min_conc = math.log(minimum)
            metabolite.log_max_conc = math.log(maximum)
            applied_concentration_count += 1
            logger.debug("Synchronized concentration: metabolite='%s' linear=(%g,%g) log=(%g,%g).", metabolite_id, minimum, maximum, metabolite.log_min_conc, metabolite.log_max_conc)
        logger.info("Applied concentration bounds to %d COBRA-k metabolite(s).", applied_concentration_count)

        # ------------------------------------------------------------------
        # STEP 5: Apply dG0 values and uncertainties to exact or mapped
        # reaction variants. Reverse mapped variants receive the opposite sign.
        # ------------------------------------------------------------------
        applied_dg0_count = 0
        for configured_id, configured_data in (dG0 or {}).items():
            if not isinstance(configured_data, dict) or "dG0" not in configured_data:
                continue
            targets = [configured_id] if configured_id in cobrak_model.reactions else base_to_variants.get(configured_id, [])
            if not targets:
                raise ValueError(f"dG0 entry '{configured_id}' cannot be mapped to the COBRA-k model.")
            for target_id in targets:
                sign = -1.0 if target_id.endswith(cobrak_model.rev_suffix) and configured_id not in cobrak_model.reactions else 1.0
                cobrak_model.reactions[target_id].dG0 = sign * float(configured_data["dG0"])
                if "uncertainty" in configured_data:
                    cobrak_model.reactions[target_id].dG0_uncertainty = float(configured_data["uncertainty"])
                applied_dg0_count += 1
                logger.debug("Synchronized dG0: configured_id='%s' target_variant='%s' sign=%g dG0=%g uncertainty=%s.", configured_id, target_id, sign, cobrak_model.reactions[target_id].dG0, cobrak_model.reactions[target_id].dG0_uncertainty)
        logger.info("Completed clean-to-COBRA-k synchronization: bound_variants=%d, concentration_metabolites=%d, dG0_variants=%d.", len(cobrak_model.reactions), applied_concentration_count, applied_dg0_count)

    @staticmethod
    def _apply_bounds_from_list(
        model: cobra.Model,
        bounds_list: list[dict[str, Any]],
        bounds_source: str,
    ) -> list[dict[str, Any]]:
        """Applies a list of reaction bound definitions to a COBRA model.

        Missing `lb` or `ub` keys in the dictionary will preserve the model's 
        current bounds for that direction.

        Args:
            model: The COBRA model to mutate.
            bounds_list: A list of bound dictionaries. Each dict must contain an 
                'id' key, and optionally 'lb' and 'ub' keys.
            bounds_source: Human-readable location of the bounds in the
                environment configuration.

        Returns:
            Bound changes whose old and new bounds differ.

        Raises:
            ValueError: If an item omits its reaction id or references a
                reaction that is absent from the model.
        """
        if not isinstance(bounds_list, list):
            raise TypeError(f"Expected list of bounds, got {type(bounds_list)}")

        bounds_changed = []
        for item in bounds_list:
            if "id" not in item:
                raise ValueError(f"Environment {bounds_source} contains a bound without an 'id': {item}")
            rid = item["id"]
            try:
                reaction = model.reactions.get_by_id(rid)
            except KeyError as error:
                raise ValueError(
                    f"Environment {bounds_source} references reaction '{rid}', "
                    f"but it is absent from model '{model.id}'."
                ) from error

            lb_old, ub_old = reaction.bounds
            lb_new = item.get("lb", lb_old)
            ub_new = item.get("ub", ub_old)
            reaction.bounds = (lb_new, ub_new)

            if (lb_old, ub_old) != reaction.bounds:
                bounds_changed.append(
                    {
                        "reaction_id": rid,
                        "source": bounds_source,
                        "bounds_before": [float(lb_old), float(ub_old)],
                        "bounds_after": [float(reaction.lower_bound), float(reaction.upper_bound)],
                    }
                )
            logger.debug("   Reaction %s bounds: (%s, %s) -> (%s, %s)", rid, lb_old, ub_old, lb_new, ub_new)
        return bounds_changed

    def _apply_scenario_bounds(
        self,
        model: cobra.Model,
        environment_config: structs.EnvironmentConfig,
        substrate_name: str,
        product_name: str,
    ) -> dict[str, Any]:
        """Applies all scenario-specific flux bounds to the model template.

        The application order prioritizes specific interactions over general ones:
        1. Substrate bounds (e.g., allow Glucose uptake).
        2. Product bounds (e.g., force Acetate excretion).
        3. Substrate-product specific adjustments (e.g., if Glucose AND Acetate, 
           enforce specific pathway knockouts).

        Args:
            model: The scenario-specific cloned model to mutate.
            environment_config: The parsed environment configuration object.
            substrate_name: The selected substrate key from the configuration.
            product_name: The selected product key from the configuration.

        Returns:
            Effective-medium diagnostics for logging and campaign metadata.
        """
        substrates = environment_config.substrates
        products = environment_config.products
        logger.info("   Applying scenario bounds for substrate=%s, product=%s", substrate_name, product_name)

        # COBRApy resolves active uptake directions from reaction stoichiometry,
        # so model.medium is more reliable than assuming every uptake is negative.
        medium_before = dict(sorted((reaction_id, float(rate)) for reaction_id, rate in model.medium.items()))
        uptake_before = {
            reaction_id: {
                "max_uptake_rate": rate,
                "bounds": [
                    float(model.reactions.get_by_id(reaction_id).lower_bound),
                    float(model.reactions.get_by_id(reaction_id).upper_bound),
                ],
            }
            for reaction_id, rate in medium_before.items()
        }
        logger.info("Environment uptake reactions open before application (%d): %s", len(uptake_before), list(uptake_before))

        substrate_bounds = substrates[substrate_name]
        logger.debug("   Substrate bounds count: %d", len(substrate_bounds))
        bounds_changed = self._apply_bounds_from_list(
            model,
            substrate_bounds,
            f"substrates.{substrate_name}",
        )

        product_bound = copy.deepcopy(products[product_name])
        if "lb" not in product_bound and "ub" not in product_bound:
            product_bound["ub"] = 1000
            logger.debug("   No product lb/ub provided; defaulted ub=1000 for %s", product_name)
        bounds_changed.extend(
            self._apply_bounds_from_list(
                model,
                [product_bound],
                f"products.{product_name}",
            )
        )

        sp_adjustments = (
            environment_config.substrate_product_adjustments
            .get(substrate_name, {})
            .get(product_name, [])
        )
        if sp_adjustments:
            logger.debug("   Applying %d substrate-product adjustments for %s -> %s", len(sp_adjustments), substrate_name, product_name)
            bounds_changed.extend(
                self._apply_bounds_from_list(
                    model,
                    sp_adjustments,
                    f"substrate_product_adjustments.{substrate_name}.{product_name}",
                )
            )
        else:
            logger.debug("   No substrate-product adjustments for %s -> %s", substrate_name, product_name)

        medium_after = dict(sorted((reaction_id, float(rate)) for reaction_id, rate in model.medium.items()))
        uptake_after = {
            reaction_id: {
                "max_uptake_rate": rate,
                "bounds": [
                    float(model.reactions.get_by_id(reaction_id).lower_bound),
                    float(model.reactions.get_by_id(reaction_id).upper_bound),
                ],
            }
            for reaction_id, rate in medium_after.items()
        }
        substrate_reaction_ids = [item["id"] for item in substrate_bounds]
        other_uptake_after = {
            reaction_id: details
            for reaction_id, details in uptake_after.items()
            if reaction_id not in substrate_reaction_ids
        }
        logger.info("Environment changed %d reaction bound(s); uptake reactions open afterward (%d): %s", len(bounds_changed), len(uptake_after), list(uptake_after))
        if other_uptake_after:
            logger.warning("Environment leaves %d other uptake reaction(s) open after application: %s", len(other_uptake_after), list(other_uptake_after))

        return {
            "substrate_reaction_ids": substrate_reaction_ids,
            "uptake_reactions_open_before": uptake_before,
            "bounds_changed": bounds_changed,
            "uptake_reactions_open_after": uptake_after,
            "other_uptake_reactions_open_after": other_uptake_after,
            "reaction_ids_missing": [],
        }

    @staticmethod
    def _apply_concentration_adjustments(
        concentrations: dict[str, dict[str, float]],
        environment_config: structs.EnvironmentConfig,
        substrate_name: str,
        product_name: str,
    ) -> None:
        """Applies concentration boundary overrides for a specific scenario.

        Modifies the base concentration dictionary in-place by overlaying 
        scenario-specific rules defined in the `EnvironmentConfig`.

        Args:
            concentrations: The cloned concentration dictionary to mutate.
            environment_config: Environment config containing adjustment rules.
            substrate_name: The selected substrate key.
            product_name: The selected product key.
        """
        scenario_adjustments = (
            environment_config.concentration_adjustments
            .get(substrate_name, {})
            .get(product_name, {})
        )
        if not scenario_adjustments:
            logger.debug("   No concentration adjustments for substrate=%s, product=%s", substrate_name, product_name)
            return

        logger.info("   Applying concentration adjustments for %d metabolites (%s -> %s)", len(scenario_adjustments), substrate_name, product_name)

        # Update metabolite concentration dicts in-place.
        for met_id, bounds in scenario_adjustments.items():
            concentrations.setdefault(met_id, {}).update(bounds)
            logger.debug("   Updated concentration bounds for metabolite %s: %s", met_id, bounds)
