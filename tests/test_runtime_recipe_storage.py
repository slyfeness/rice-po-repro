"""Runtime provenance must not mutate the research recipe used by holdout."""
from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from ricepo.config import load_config, resolve
from ricepo import hf_runtime


class RuntimeRecipeStorage(TestCase):
    def test_initial_evaluation_mode_only_allows_a_fresh_run(self):
        self.assertTrue(hasattr(hf_runtime, "initial_evaluation_mode"))
        self.assertTrue(hf_runtime.initial_evaluation_mode({"initial_evaluation_only": True},
                                                           parent=None, resume=False))
        with self.assertRaisesRegex(ValueError, "fresh"):
            hf_runtime.initial_evaluation_mode({"initial_evaluation_only": True}, parent=object(), resume=False)
        with self.assertRaisesRegex(ValueError, "fresh"):
            hf_runtime.initial_evaluation_mode({"initial_evaluation_only": True}, parent=None, resume=True)

    def test_asset_binding_does_not_mutate_research_recipe(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/controlled_biology.json")
        recipe = resolve(config, "train")
        expected = deepcopy(recipe)
        deployment = dict(output_parser_mode="strict_native_tags",
                          turn_protocol="native_user_observation_v1",
                          parser_settings=dict(structure_mode="tagged",
                                               boundary_rule="native_body_coverage_v1",
                                               allow_empty_reasoning=True))
        class AssetGate(Exception):
            pass
        with patch.object(hf_runtime, "resolve", return_value=recipe), patch.object(
                hf_runtime, "verify_bundle", side_effect=AssetGate):
            with self.assertRaises(AssetGate):
                hf_runtime._train(config, deployment, Path("unused"), Path("unused-attempt"), False,
                                  cleanup=None)
        self.assertEqual(recipe, expected)
