"""Provide functionality for ghostwriting test modules."""

import argparse
import ast
import builtins
import inspect
import pathlib
import re
import symtable
import sys
import textwrap
from typing import Any, Callable, List, Optional, Tuple, Union, overload

import hypothesis.strategies
import hypothesis.strategies._internal.collections
import hypothesis.strategies._internal.lazy
import icontract._checkers
import icontract._represent

import icontract_hypothesis
from icontract_hypothesis.pyicontract_hypothesis import _general


@icontract.invariant(lambda self: (self.module_name is None) ^ (self.path is None))
class Params:
    """Represent parameters of the command "ghostwrite"."""

    @icontract.require(lambda module_name, path: (module_name is None) ^ (path is None))
    def __init__(
        self,
        module_name: Optional[str],
        path: Optional[pathlib.Path],
        output: Optional[pathlib.Path],
        explicit: bool,
        bare: bool,
    ) -> None:
        """Initialize with the given values."""
        self.module_name = module_name
        self.path = path
        self.output = output
        self.explicit = explicit
        self.bare = bare


def parse_params(args: argparse.Namespace) -> Tuple[Optional[Params], List[str]]:
    """
    Try to parse the parameters of the command "ghostwrite".

    Return (parsed parameters, errors if any).
    """
    output = pathlib.Path(args.output) if args.output != "-" else None

    path = None  # type: Optional[pathlib.Path]

    if args.path is not None:
        try:
            path = pathlib.Path(args.path)
        except Exception as err:
            return None, [f"Failed to parse --path {args.path!r}: {err}"]

    return (
        Params(
            module_name=args.module,
            path=path,
            output=output,
            explicit=args.explicit,
            bare=args.bare,
        ),
        [],
    )


@overload
def _indent_but_first(lines: List[str], level: int = 1) -> List[str]:  # pylint: disable=all
    ...


@overload
def _indent_but_first(lines: str, level: int = 1) -> str:  # pylint: disable=all
    ...


def _indent_but_first(
    lines: Union[List[str], str], level: int = 1
) -> Union[str, List[str]]:
    r"""
    Indents the text by 4 spaces.

    >>> _indent_but_first([''], 0)
    ['']

    >>> _indent_but_first(['test'], 1)
    ['test']

    >>> _indent_but_first(['test', '', 'me'], 1)
    ['test', '', '    me']

    >>> _indent_but_first('test\n\nme', 1)
    'test\n\n    me'
    """
    if isinstance(lines, str):
        result = lines.splitlines()
        for i in range(1, len(result)):
            if len(result[i]) > 0:
                result[i] = "    " * level + result[i]

        return "\n".join(result)

    elif isinstance(lines, list):
        result = lines[:]
        for i in range(1, len(result)):
            if len(result[i]) > 0:
                result[i] = "    " * level + result[i]

        return result

    else:
        raise TypeError(f"Unhandled input: {lines}")


def _ghostwrite_condition_code(condition: Callable[..., Any]) -> str:
    """Ghostwrite the code representing the condition in an assumption."""
    if not icontract._represent.is_lambda(a_function=condition):
        sign = inspect.signature(condition)
        args = ", ".join(param for param in sign.parameters)

        return f"{condition.__name__}({args})"

    # We need to extract the source code corresponding to the decorator since
    # inspect.getsource() is broken with lambdas.

    # Find the line corresponding to the condition lambda
    lines, condition_lineno = inspect.findsource(condition)
    filename = inspect.getsourcefile(condition)
    assert filename is not None

    decorator_inspection = icontract._represent.inspect_decorator(
        lines=lines, lineno=condition_lineno, filename=filename
    )

    lambda_inspection = icontract._represent.find_lambda_condition(
        decorator_inspection=decorator_inspection
    )

    assert lambda_inspection is not None, (
        f"Expected lambda_inspection to be non-None if is_lambda is True on: {condition}"
    )

    return lambda_inspection.text


def _can_render_condition(condition: Callable[..., Any]) -> bool:
    """Check whether a lambda can be emitted without external bindings."""
    inspection = icontract_hypothesis._inspect_condition(condition)
    if inspection is None:
        return False

    bindings = inspect.getclosurevars(condition)
    if bindings.nonlocals or bindings.globals or bindings.unbound:
        return False

    # getclosurevars only checks the outer function. Inspect nested scopes too.
    source = inspection.atok.get_text(inspection.node)
    scopes = [symtable.symtable(source, "<condition>", "eval")]
    while scopes:
        scope = scopes.pop()
        for symbol in scope.get_symbols():
            if symbol.is_global() and (
                symbol.get_name() in condition.__globals__
                or not hasattr(builtins, symbol.get_name())
            ):
                return False
        scopes.extend(scope.get_children())

    return True


@icontract.ensure(lambda result: len(result[1]) > 0 or not result[0].endswith("\n"))
def _ghostwrite_test_function(
    module_name: str, point: _general.FunctionPoint, explicit: bool
) -> Tuple[str, List[str]]:
    """
    Ghostwrite a test function for the given function point.

    The result needs to be properly indented afterwards.

    Return (code, errors if any)
    """
    test_func = ""

    if explicit:
        checker = icontract._checkers.find_checker(point.func)
        conjunctions = (
            getattr(checker, "__preconditions__", []) if checker is not None else []
        )
        explicit = all(
            _can_render_condition(contract.condition)
            for conjunction in conjunctions
            for contract in conjunction
        )

    if not explicit:
        test_func = textwrap.dedent(
            f"""\
            def test_{point.func_name}(self) -> None:
                icontract_hypothesis.test_with_inferred_strategy(
                    {module_name}.{point.func_name})
            """
        ).strip()
        return test_func, []

    try:
        strategy = icontract_hypothesis.infer_strategy(func=point.func)
    except Exception as err:
        return "", [
            (
                f"Failed to infer a search strategy for the function "
                f"on line {point.first_row}: {point.func_name}. "
                f"The exception was:\n{err}"
            )
        ]

    # If there is no filtering on multiple arguments, unpack the argument strategies and
    # re-pack them into a hypothesis.given decorator with argument names as keyword arguments
    arguments_unpacked = False
    collections = hypothesis.strategies._internal.collections
    fixed_dict_type = getattr(
        collections,
        "FixedDictStrategy",
        getattr(collections, "FixedKeysDictStrategy", ()),
    )

    if isinstance(
        strategy, hypothesis.strategies._internal.lazy.LazyStrategy
    ) and isinstance(strategy.wrapped_strategy, fixed_dict_type):
        fixed_dict_st = strategy.wrapped_strategy
        if hasattr(fixed_dict_st, "mapping"):
            argument_strategies = list(fixed_dict_st.mapping.items())
        else:
            # Hypothesis releases compatible with Python 3.8 use this layout.
            argument_strategies = list(
                zip(
                    fixed_dict_st.keys, fixed_dict_st.mapped_strategy.element_strategies
                )
            )
    else:
        argument_strategies = []

    if argument_strategies:
        arguments_unpacked = True

        given_args_lines = []  # type: List[str]

        for arg_name, arg_strategy in argument_strategies:
            line = f"{arg_name}={arg_strategy}"
            assert not line.endswith("\n")
            given_args_lines.append(line)

        given = textwrap.dedent(
            """\
            @given(
                {}
            )"""
        ).format(_indent_but_first(",\n".join(given_args_lines), level=1))
    else:
        given = textwrap.dedent(
            f"""\
            @given(
                {strategy}
            )"""
        )

    assert not re.match(r"^\s", given), (
        f"Unexpected whitespace at the begining of given: {given!r}"
    )
    assert not given.endswith("\n")

    test_func = textwrap.dedent(
        """\
        def test_{func}(self) -> None:
            {given}
            def execute({execute_kwargs}) -> None:
                {module}.{func}(**kwargs)

            execute()"""
    ).format(
        module=module_name,
        func=point.func.__name__,
        given=_indent_but_first(given, level=1),
        # If we supply unpacked arguments in the ``given``, we have to forward them
        # as keyword arguments. Otherwise, the generative strategy will be fixed dictionary, so
        # we need to accept them as a single argument.
        execute_kwargs="**kwargs" if arguments_unpacked else "kwargs",
    )

    return test_func, []


def _ghostwrite_for_function_points(
    points: List[_general.FunctionPoint],
    module_name: str,
    explicit: bool,
    bare: bool,
) -> Tuple[str, List[str]]:
    """
    Ghostwrite a test case for the given function points.

    Return (generated code, errors if any).
    """
    errors = []  # type: List[str]
    test_funcs = []  # type: List[str]
    for point in points:
        test_func, test_func_errors = _ghostwrite_test_function(
            module_name=module_name, point=point, explicit=explicit
        )
        if test_func_errors:
            errors.extend(test_func_errors)
        else:
            test_funcs.append(test_func)

    if errors:
        return "", errors

    if bare:
        return "\n\n".join(test_funcs), []

    blocks = []  # type: List[str]
    strategy_names = sorted(
        {
            node.id
            for func in test_funcs
            for node in ast.walk(ast.parse(func))
            if isinstance(node, ast.Name) and node.id in hypothesis.strategies.__all__
        }
    )

    header = "\n\n".join(
        [
            f'"""Test {module_name} with inferred Hypothesis strategies."""',
            "import unittest",
        ]
        + (
            ["from hypothesis import given"]
            if explicit
            else ["import icontract_hypothesis"]
        )
        + (
            ["from hypothesis.strategies import {}".format(", ".join(strategy_names))]
            if strategy_names
            else []
        )
        + (
            ["import icontract_hypothesis"]
            if explicit
            and any(
                "icontract_hypothesis.test_with_inferred_strategy(" in func
                for func in test_funcs
            )
            else []
        )
        + [f"import {module_name}"]
    )
    blocks.append(header)

    if len(points) == 0:
        blocks.append(
            textwrap.dedent(
                f'''\
                class TestWithInferredStrategies(unittest.TestCase):
                    """Test all functions from {module_name} with inferred Hypothesis strategies."""
                    # Either there are no functions in {module_name} or all the functions were excluded.
                '''
            ).strip()
        )
    else:
        body = "\n\n".join(test_funcs)

        test_case = [
            textwrap.dedent(
                '''\
                class TestWithInferredStrategies(unittest.TestCase):
                    """Test all functions from {module} with inferred Hypothesis strategies."""

                    {body}'''
            ).format(
                module=module_name,
                body="\n".join(_indent_but_first(lines=body.splitlines(), level=1)),
            )
        ]
        blocks.append("".join(test_case))

    blocks.append(
        textwrap.dedent(
            """\
        if __name__ == '__main__':
            unittest.main()
        """
        )
    )

    return "\n\n\n".join(blocks), []


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z_0-9]*$")


def _qualified_module_name_from_path(path: pathlib.Path, sys_path: List[str]) -> str:
    """
    Infer the qualified (dotted) module name from its path.

    If the path does not point to a proper Python module, the module name is computed
    as the stem of the ``path`` with all the non-identifier characters replaced by an
    underscore ("_").

    :param path: to the module
    :param sys_path: the value of ``sys.path``
    """
    stem = path.stem
    if (
        not _IDENTIFIER_RE.match(stem)
        or path.parent == pathlib.Path()
        or path.suffix != ".py"
    ):
        qualified_name = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    else:
        sys_path_set = set(sys_path)
        ancestor_in_sys_path = False

        if path.name == "__init__.py":
            # Go one level up the directory path as the module name comes from the base directory
            reversed_parts = [path.parent.stem]
            ancestors = path.parent.parents
        else:
            # Start from the file as the base directory refers to the parent module
            reversed_parts = [stem]
            ancestors = path.parents

        for ancestor in ancestors:
            if str(ancestor) in sys_path_set:
                ancestor_in_sys_path = True
                break

            # Check to see if the ancestor is a proper Python parent module
            if (
                not _IDENTIFIER_RE.match(ancestor.stem)
                or not (ancestor / "__init__.py").exists()
            ):
                break

            # Continue to see if we can reach one of the paths in sys.path
            reversed_parts.append(ancestor.stem)

        if ancestor_in_sys_path:
            qualified_name = ".".join(reversed(reversed_parts))
        else:
            qualified_name = re.sub(r"[^A-Za-z0-9_]", "_", stem)

    return qualified_name


def ghostwrite(general: _general.Params, command: Params) -> Tuple[str, List[str]]:
    """
    Write a unit test module for the specified functions.

    Return (generated code, errors if any).
    """
    if command.module_name is not None:
        mod, errors = _general.load_module_with_name(command.module_name)
    elif command.path is not None:
        mod, errors = _general.load_module_from_source_file(path=command.path)
    else:
        raise AssertionError(
            f"Unexpected execution path. The command was: {command.__dict__!r}"
        )

    if errors:
        return "", errors

    assert mod is not None

    ##
    # Load the source code
    ##

    if command.module_name is not None:
        source_code = inspect.getsource(mod)
    elif command.path is not None:
        source_code = command.path.read_text()
    else:
        raise AssertionError(
            f"Unexpected execution path. The command was: {command.__dict__!r}"
        )

    ##
    # Select points
    ##

    points, errors = _general.select_function_points(
        source_code=source_code,
        mod=mod,
        include=general.include,
        exclude=general.exclude,
    )
    if errors:
        return "", errors

    ##
    # Figure out the qualified name of the module
    ##

    if command.module_name is not None:
        qualified_name = command.module_name
    elif command.path is not None:
        qualified_name = _qualified_module_name_from_path(
            path=command.path, sys_path=sys.path
        )
    else:
        raise AssertionError(
            f"Unexpected execution path. The command was: {command.__dict__!r}"
        )

    ##
    # Ghostwrite
    ##

    return _ghostwrite_for_function_points(
        points=points,
        module_name=qualified_name,
        explicit=command.explicit,
        bare=command.bare,
    )
