from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mapanything.models import model_factory


class ModelFactoryDependencyTests(unittest.TestCase):
    def test_pi3x_missing_dependency_reports_install_commands(self) -> None:
        with (
            patch(
                "mapanything.models.check_module_exists",
                return_value=False,
            ) as check_module_exists,
            patch("mapanything.models.importlib.import_module") as import_module,
        ):
            with self.assertRaises(ImportError) as context:
                model_factory("pi3x")

        message = str(context.exception)
        self.assertIn("Model 'pi3x'", message)
        self.assertIn("optional dependency module 'pi3'", message)
        self.assertIn("uv sync --extra pi3", message)
        self.assertIn(
            "uv run --extra pi3 python scripts/map_free_inference.py --model pi3x",
            message,
        )
        check_module_exists.assert_called_once_with("pi3")
        import_module.assert_not_called()

    def test_pi3x_available_dependency_uses_dynamic_wrapper_import(self) -> None:
        class StubPi3XWrapper:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        wrapper_module = SimpleNamespace(Pi3XWrapper=StubPi3XWrapper)
        with (
            patch(
                "mapanything.models.check_module_exists",
                return_value=True,
            ) as check_module_exists,
            patch(
                "mapanything.models.importlib.import_module",
                return_value=wrapper_module,
            ) as import_module,
        ):
            model = model_factory("pi3x", name="pi3x")

        self.assertIsInstance(model, StubPi3XWrapper)
        self.assertEqual(model.kwargs, {"name": "pi3x"})
        self.assertEqual(
            [call.args[0] for call in check_module_exists.call_args_list],
            ["pi3", "mapanything.models.external.pi3x"],
        )
        import_module.assert_called_once_with("mapanything.models.external.pi3x")

    def test_bundled_pi3_does_not_require_optional_pi3_extra(self) -> None:
        class StubPi3Wrapper:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        wrapper_module = SimpleNamespace(Pi3Wrapper=StubPi3Wrapper)
        with (
            patch(
                "mapanything.models.check_module_exists",
                return_value=True,
            ) as check_module_exists,
            patch(
                "mapanything.models.importlib.import_module",
                return_value=wrapper_module,
            ),
        ):
            model = model_factory("pi3", name="pi3")

        self.assertIsInstance(model, StubPi3Wrapper)
        self.assertEqual(model.kwargs, {"name": "pi3"})
        check_module_exists.assert_called_once_with(
            "mapanything.models.external.pi3"
        )


if __name__ == "__main__":
    unittest.main()
