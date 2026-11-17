# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import io
import os
import pathlib
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

import icontract

from icontract_hypothesis.pyicontract_hypothesis import _general, _ghostwrite, main


class TestParsingOfParameters(unittest.TestCase):
    def test_subcommand_ghostwrite(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "some_module",
            "--include", "include-something",
            "--exclude", "exclude-something",
            "--explicit",
            "--bare",
        ]
        # fmt: on
        parser = main._make_argument_parser()
        args, _out, err = main._parse_args(parser=parser, argv=argv)
        assert args is not None, f"Failed to parse argv {argv!r}: {err}"

        general, errs = _general.parse_params(args=args)

        self.assertListEqual([], errs)
        assert general is not None
        self.assertListEqual(
            [re.compile(pattern) for pattern in ["include-something"]], general.include
        )
        self.assertListEqual(
            [re.compile(pattern) for pattern in ["exclude-something"]], general.exclude
        )

        ghostwrite, errs = _ghostwrite.parse_params(args=args)

        self.assertListEqual([], errs)
        assert ghostwrite is not None
        self.assertEqual("some_module", ghostwrite.module_name)
        self.assertTrue(ghostwrite.explicit)
        self.assertTrue(ghostwrite.bare)

    def test_path_as_input(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--path", "path/to/some_module.py"
        ]
        # fmt: on
        parser = main._make_argument_parser()
        args, _out, err = main._parse_args(parser=parser, argv=argv)
        assert args is not None, f"Failed to parse argv {argv!r}: {err}"

        general, errs = _general.parse_params(args=args)

        self.assertListEqual([], errs)
        assert general is not None

        ghostwrite, errs = _ghostwrite.parse_params(args=args)

        self.assertListEqual([], errs)
        assert ghostwrite is not None
        assert ghostwrite.path is not None
        assert ghostwrite.module_name is None

        self.assertEqual("path/to/some_module.py", ghostwrite.path.as_posix())


class TestQualifiedModuleNameFromPath(unittest.TestCase):
    def test_non_identifier_stem(self) -> None:
        path = pathlib.Path("some_directory") / "some-script.py"

        qualified_name = _ghostwrite._qualified_module_name_from_path(
            path=path, sys_path=[]
        )

        self.assertEqual("some_script", qualified_name)

    def test_no_parent_in_sys_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            (tmpdir / "some_module" / "__init__.py").write_text('"""something"""')

            path = tmpdir / "some_module" / "some_submodule.py"
            path.write_text('"""something"""')

            # If tmpdir were added to sys_path, we would have a meaningful qualified name.
            # As sys_path is empty, we have to resort to a stem.
            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[]
            )

            self.assertEqual("some_submodule", qualified_name)

    def test_path_to_module_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            (tmpdir / "some_module" / "__init__.py").write_text('"""something"""')

            path = tmpdir / "some_module" / "some_submodule.py"
            path.write_text('"""something"""')

            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[tmpdir_str]
            )

            self.assertEqual("some_module.some_submodule", qualified_name)

    def test_path_to_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            (tmpdir / "some_module" / "__init__.py").write_text('"""something"""')

            (tmpdir / "some_module" / "some_submodule").mkdir()

            path = tmpdir / "some_module" / "some_submodule" / "__init__.py"
            path.write_text('"""something"""')

            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[tmpdir_str]
            )

            self.assertEqual("some_module.some_submodule", qualified_name)

    def test_path_to_module_file_with_ancestor_without_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            # No __init__.py in the directory ``some_module``.

            path = tmpdir / "some_module" / "some_submodule.py"
            path.write_text('"""something"""')

            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[tmpdir_str]
            )

            self.assertEqual("some_submodule", qualified_name)


class TestGhostwriteFallback(unittest.TestCase):
    def assert_generated_test_runs(self, func, fallback):
        module = types.ModuleType("condition_sample")
        setattr(module, func.__name__, func)
        point = _general.FunctionPoint(first_row=1, last_row=1, func=func)
        point.func_name = func.__name__
        code, errors = _ghostwrite._ghostwrite_for_function_points(
            points=[point], module_name=module.__name__, explicit=True, bare=False
        )
        self.assertEqual([], errors)
        if fallback:
            self.assertIn("icontract_hypothesis.test_with_inferred_strategy(", code)
            self.assertNotIn("_condition(", code)
        else:
            self.assertIn("@given(", code)
            self.assertIn(
                "from hypothesis.strategies import fixed_dictionaries, integers", code
            )

        namespace = {"__name__": "generated_test"}
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            exec(compile(code, "<generated-test>", "exec"), namespace)
            case = namespace["TestWithInferredStrategies"]()
            getattr(case, "test_" + func.__name__)()

    def test_source_free_condition(self):
        condition = eval(compile("lambda x, y: x < y", "<source-free>", "eval"))

        @icontract.require(condition)
        def func(x: int, y: int):
            pass

        self.assert_generated_test_runs(func, fallback=True)

    def test_closure(self):
        limit = 0

        @icontract.require(lambda x, y: limit < x < y)
        def func(x: int, y: int):
            pass

        self.assert_generated_test_runs(func, fallback=True)

    def test_named_condition(self):
        def condition(x, y):
            return x < y

        @icontract.require(condition)
        def func(x: int, y: int):
            pass

        self.assert_generated_test_runs(func, fallback=True)

    def test_explicit_condition(self):
        @icontract.require(lambda x, y: x < y)
        def func(x: int, y: int):
            pass

        self.assert_generated_test_runs(func, fallback=False)

    def test_nested_scope(self):
        @icontract.require(lambda x, y: (lambda x: x > 0)(y))
        def func(x: int, y: int):
            pass

        self.assert_generated_test_runs(func, fallback=False)

    def test_mixed_explicit_and_fallback_output(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        exit_code = main.run(
            argv=[
                "ghostwrite",
                "--module",
                "test_samples.pyicontract_hypothesis.sample_module",
                "--explicit",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        self.assertEqual(0, exit_code, stderr.getvalue())
        namespace = {"__name__": "generated_test"}
        exec(compile(stdout.getvalue(), "<generated-test>", "exec"), namespace)
        case = namespace["TestWithInferredStrategies"]()
        case.test_some_func()
        case.test_another_func()
        case.test_yet_another_func()

    def test_global_in_nested_scope(self):
        @icontract.require(lambda x, y: (lambda: sys.maxsize > x < y)())
        def func(x: int, y: int):
            pass

        self.assert_generated_test_runs(func, fallback=True)


class TestGhostwrite(unittest.TestCase):
    def test_path(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent

        path = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )
        # fmt: off
        argv = [
            "ghostwrite",
            "--path", str(path)
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        # The qualified name of the module depends on the sys.path which, in turn, is modified
        # by the test executor (*e.g.*, IDE can add temporarily directories to sys.path
        # automatically when executing tests).
        #
        # Since the qualified name of a module depends on ``sys.path``, we need to replace it with
        # a generic placeholder to make the test output uniform and reproducible.

        qualified_name = _ghostwrite._qualified_module_name_from_path(
            path=path, sys_path=sys.path
        )
        got = stdout.getvalue()
        got = got.replace(qualified_name, "placeholder_for_sample_module")

        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (f"for_{TestGhostwrite.test_path.__name__}.txt")
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, got)

    def test_bare_and_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
            "--bare",
            "--explicit"
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (f"for_{TestGhostwrite.test_bare_and_explicit.__name__}.txt")
        )

        expected = expected_pth.read_text().rstrip("\n")
        self.assertEqual(expected, stdout.getvalue())

    def test_bare_and_non_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
            "--bare",
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (f"for_{TestGhostwrite.test_bare_and_non_explicit.__name__}.txt")
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
            "--explicit"
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (f"for_{TestGhostwrite.test_non_bare_and_explicit.__name__}.py")
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_non_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (f"for_{TestGhostwrite.test_non_bare_and_non_explicit.__name__}.py")
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_well_formatted_with_two_arguments(self) -> None:
        # This test is related to the issue:
        # https://github.com/mristin/icontract-hypothesis/issues/29
        # fmt: off
        argv = [
            "ghostwrite",
            "--module",
            "test_samples.pyicontract_hypothesis.well_formatted_with_two_arguments",
            "--explicit"
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (
                f"for_{TestGhostwrite.test_well_formatted_with_two_arguments.__name__}.py"
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
