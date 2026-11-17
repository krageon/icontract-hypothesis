"""
Integrate icontract with Hypothesis.

You need to install ``hypothesis`` extras in order to use this module.
"""
# pylint: disable=unused-argument
# pylint: disable=protected-access
# pylint: disable=inconsistent-return-statements
# pylint: disable=too-many-lines

import ast
import datetime
import decimal
import fractions
import inspect
import keyword
import re
import typing
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Pattern,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import hypothesis.errors
import hypothesis.internal.reflection
import hypothesis.strategies
import hypothesis.strategies._internal.collections
import hypothesis.strategies._internal.types
import icontract._checkers
import icontract._metaclass
import icontract._recompute
import icontract._represent
import icontract._types

# Don't forget to update the version in setup.py!
__version__ = "1.1.8"
__author__ = "Marko Ristin"
__copyright__ = "Copyright 2020-2021 Marko Ristin"
__license__ = "MIT"
__status__ = "Production/Stable"

CallableT = TypeVar("CallableT", bound=Callable[..., Any])

T = TypeVar("T")  # pylint: disable=invalid-name


def _assume_preconditions_always_satisfied(
    *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
) -> None:
    """Assume that the preconditions are always satisfied (say, when there are no preconditions)."""


def make_assume_preconditions(func: CallableT) -> Callable[..., None]:
    """
    Create a function that assumes the preconditions are satisfied given the positional and keyword arguments.

    Here is an example test case which tests a function ``some_func``:

    >>> import unittest
    >>> import hypothesis.strategies as st
    >>> import icontract_hypothesis
    >>> import icontract

    >>> @icontract.require(lambda x: x > 0)
    ... def some_func(x: int) -> None:
    ...     ...

    >>> class TestSomething(unittest.TestCase):
    ...     def test_that_it_works(self) -> None:
    ...         assume_preconditions = icontract_hypothesis.make_assume_preconditions(
    ...             some_func)
    ...
    ...         @hypothesis.given(x=st.integers())
    ...         def run(x: int) -> None:
    ...             assume_preconditions(x)
    ...             some_func(x)
    ...         run()

    >>> TestSomething().test_that_it_works()
    """
    # The implementation follows tightly icontract._checkers.decorate_with_checker and
    # icontract._checkers._assert_precondition.

    checker = icontract._checkers.find_checker(func)
    if checker is None:
        return _assume_preconditions_always_satisfied

    preconditions = getattr(checker, "__preconditions__", None)
    if preconditions is None or len(preconditions) == 0:
        return _assume_preconditions_always_satisfied

    sign = inspect.signature(func)
    param_names = list(sign.parameters.keys())
    kwdefaults = icontract._checkers.resolve_kwdefaults(sign=sign)

    def assume_preconditions(*args, **kwargs) -> None:  # type: ignore
        """Accept only positional and keyword arguments that satisfy one of the precondition groups."""
        resolved_kwargs = icontract._checkers.kwargs_from_call(
            param_names=param_names, kwdefaults=kwdefaults, args=args, kwargs=kwargs
        )

        success = True

        for group in preconditions:  # pylint: disable=not-an-iterable
            success = True
            for contract in group:
                condition_kwargs = icontract._checkers.select_condition_kwargs(
                    contract=contract, resolved_kwargs=resolved_kwargs
                )

                check = contract.condition(**condition_kwargs)

                if icontract._checkers.not_check(check=check, contract=contract):
                    success = False
                    break

            if success:
                return

        if not success:
            raise hypothesis.errors.UnsatisfiedAssumption()

    return assume_preconditions


class _InferredMinMax:
    """Represent the inference result of boundaries on an argument."""

    def __init__(
        self,
        min_value: Optional[Any] = None,
        min_inclusive: bool = False,
        max_value: Optional[Any] = None,
        max_inclusive: bool = False,
    ) -> None:
        """Initialize with the given values."""
        self.min_value = min_value
        self.min_inclusive = min_inclusive
        self.max_value = max_value
        self.max_inclusive = max_inclusive


def _no_name_in_descendants(root: ast.expr, name: str) -> bool:
    """Check whether a ``ast.Name`` node with ``root`` identifier is present in the descendants of the node."""
    found = False

    class Visitor(ast.NodeVisitor):
        """Search for the name node."""

        def visit_Name(  # pylint: disable=invalid-name,no-self-use,missing-docstring
            self, node: ast.Name
        ) -> None:
            if node.id == name:
                nonlocal found
                found = True

        def generic_visit(self, node: Any) -> None:
            if not found:
                super().generic_visit(node)

    visitor = Visitor()
    visitor.visit(root)

    return not found


def _recompute(condition: Callable[..., Any], node: ast.expr) -> Tuple[Any, bool]:
    """Recompute the value corresponding to the node."""
    recompute_visitor = icontract._recompute.Visitor(
        variable_lookup=icontract._represent.collect_variable_lookup(
            condition=condition, resolved_kwargs=None
        )
    )

    recompute_visitor.visit(node=node)

    if node in recompute_visitor.recomputed_values:
        return recompute_visitor.recomputed_values[node], True

    return None, False


def _infer_min_max_from_node(
    condition: Callable[..., bool], node: ast.Compare, arg_name: str
) -> Optional[_InferredMinMax]:
    """Match one of the patterns against the AST compare node."""
    # pylint: disable=too-many-boolean-expressions
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    if len(node.comparators) == 1:
        comparator = node.comparators[0]
        operation = node.ops[0]

        # Match something like "x > 0" and "x < 100"
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == arg_name
            and _no_name_in_descendants(root=comparator, name=arg_name)
        ):
            value, recomputed = _recompute(condition=condition, node=comparator)

            # If we can not recompute the value, we also can not infer the bounds.
            if not recomputed:
                return None

            # Match something like "x < 100"
            if isinstance(operation, ast.Lt):
                return _InferredMinMax(max_value=value, max_inclusive=False)
            # Match something like "x =< 100"
            elif isinstance(operation, ast.LtE):
                return _InferredMinMax(max_value=value, max_inclusive=True)

            # Match something like "x > 0"
            elif isinstance(operation, ast.Gt):
                return _InferredMinMax(min_value=value, min_inclusive=False)
            # Match something like "x >= 0"
            elif isinstance(operation, ast.GtE):
                return _InferredMinMax(min_value=value, min_inclusive=True)

            # We could not infer any bound from this condition.
            else:
                return None

        # Match something like "0 < x" and "100 > x"
        if (
            _no_name_in_descendants(root=node.left, name=arg_name)
            and isinstance(comparator, ast.Name)
            and comparator.id == arg_name
        ):
            value, recomputed = _recompute(condition=condition, node=node.left)

            # If we can not recompute the value, we also can not infer the bounds.
            if not recomputed:
                return None

            # Match something like "0 < x"
            if isinstance(operation, ast.Lt):
                return _InferredMinMax(min_value=value, min_inclusive=False)

            # Match something like "0 =< x"
            if isinstance(operation, ast.LtE):
                return _InferredMinMax(min_value=value, min_inclusive=True)

            # Match something like "100 > x"
            elif isinstance(operation, ast.Gt):
                return _InferredMinMax(max_value=value, max_inclusive=False)

            # Match something like "100 >= x"
            elif isinstance(operation, ast.GtE):
                return _InferredMinMax(max_value=value, max_inclusive=True)

            # We could not infer any bound from this condition.
            else:
                pass

    # Match something like "0 < x < 100" and "0 > x > -100"
    elif (
        len(node.comparators) == 2
        and _no_name_in_descendants(root=node.left, name=arg_name)
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == arg_name
        and _no_name_in_descendants(root=node.comparators[1], name=arg_name)
    ):
        left_value, recomputed = _recompute(condition=condition, node=node.left)

        # If we can not recompute the left value, we also can not infer the bounds.
        if not recomputed:
            return None

        right_value, recomputed = _recompute(
            condition=condition, node=node.comparators[1]
        )

        # If we can not recompute the right value, we also can not infer the bounds.
        if not recomputed:
            return None

        op0, op1 = node.ops

        # Match something like "0 < x < 100"
        if isinstance(op0, (ast.Lt, ast.LtE)) and isinstance(op1, (ast.Lt, ast.LtE)):
            return _InferredMinMax(
                min_value=left_value,
                min_inclusive=isinstance(op0, ast.LtE),
                max_value=right_value,
                max_inclusive=isinstance(op1, ast.LtE),
            )

        # Match something like "0 > x > -100"
        elif isinstance(op0, (ast.Gt, ast.GtE)) and isinstance(op1, (ast.Gt, ast.GtE)):
            return _InferredMinMax(
                min_value=right_value,
                min_inclusive=isinstance(op1, ast.GtE),
                max_value=left_value,
                max_inclusive=isinstance(op0, ast.GtE),
            )

        # We could not infer any bound from this condition.
        else:
            pass

    return None


def _inspect_condition(
    condition: Callable[..., Any],
) -> Optional[icontract._represent.ConditionLambdaInspection]:
    """Return lambda source information when available."""
    if getattr(condition, "__name__", None) != "<lambda>":
        return None

    try:
        return icontract._represent.inspect_lambda_condition(condition)
    except (OSError, TypeError, SyntaxError, ValueError):
        return None


def _body_node_from_condition(condition: Callable[..., Any]) -> Optional[ast.expr]:
    """Try to extract the body node of the contract's lambda condition."""
    inspection = _inspect_condition(condition)
    return None if inspection is None else inspection.node.body


@icontract.require(
    lambda arg_name, contracts: all(
        len(contract.condition_args) == 1 and contract.condition_args[0] == arg_name
        for contract in contracts
    ),
    "All contracts are single-argument contracts related to this argument.",
)
def _infer_min_max_from_preconditions(
    arg_name: str, type_hint: Type[T], contracts: List[icontract._types.Contract]
) -> Tuple[_InferredMinMax, List[icontract._types.Contract]]:
    """
    Infer the min and max values for the given argument from all related preconditions.

    Return the contracts which could not be interpreted.
    """
    min_value: Optional[Union[int, float]] = None
    min_inclusive = False
    max_value: Optional[Union[int, float]] = None
    max_inclusive = False

    remaining_contracts = []  # type: List[icontract._types.Contract]

    for contract in contracts:
        body_node = _body_node_from_condition(condition=contract.condition)

        if body_node is None:
            remaining_contracts.append(contract)
            continue

        if isinstance(body_node, ast.Compare):
            inferred = _infer_min_max_from_node(
                condition=contract.condition, node=body_node, arg_name=arg_name
            )

            if inferred is not None:
                # We need to constrain min and max values.
                # Hence we use ``max`` for min and ``min`` for max, respectively.
                # This might be a bit counter-intuitive at the first sight.

                if inferred.min_value is not None:
                    if min_value is None:
                        min_value = inferred.min_value
                        min_inclusive = inferred.min_inclusive
                    else:
                        if inferred.min_value > min_value:
                            min_value = inferred.min_value
                            min_inclusive = inferred.min_inclusive
                        else:
                            # Current min is already the correct lower bound.
                            pass

                if inferred.max_value is not None:
                    if max_value is None:
                        max_value = inferred.max_value
                        max_inclusive = inferred.max_inclusive
                    else:
                        if inferred.max_value < max_value:
                            max_value = inferred.max_value
                            max_inclusive = inferred.max_inclusive
                        else:
                            # Current max is already the correct upper bound.
                            pass

                # We can not have exclusive bounds on Fractions and Decimals,
                # so we need to leave the contracts in place to filter out these edge cases.
                if (
                    (inferred.max_value is not None and not inferred.max_inclusive)
                    or (inferred.min_value is not None and not inferred.min_inclusive)
                ) and type_hint in [fractions.Fraction, decimal.Decimal]:
                    remaining_contracts.append(contract)
            else:
                remaining_contracts.append(contract)
        else:
            remaining_contracts.append(contract)

    return (
        _InferredMinMax(
            min_value=min_value,
            max_value=max_value,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
        ),
        remaining_contracts,
    )


def _make_strategy_with_min_max_for_type(
    a_type: Type[T], inferred: _InferredMinMax
) -> hypothesis.strategies.SearchStrategy[T]:
    if a_type == int:
        # hypothesis.strategies.integers is always inclusive so we have to cut off the boundaries
        # a bit if they are exclusive.
        min_value = inferred.min_value
        if min_value is not None and not inferred.min_inclusive:
            min_value += 1

        max_value = inferred.max_value
        if max_value is not None and not inferred.max_inclusive:
            max_value -= 1

        if min_value is not None and min_value == max_value:
            strategy = hypothesis.strategies.just(min_value)  # type: hypothesis.strategies.SearchStrategy[Any]
        else:
            strategy = hypothesis.strategies.integers(
                min_value=min_value,
                max_value=max_value,
            )

    elif a_type == float:
        strategy = hypothesis.strategies.floats(
            min_value=inferred.min_value,
            max_value=inferred.max_value,
            exclude_min=inferred.min_value is not None and not inferred.min_inclusive,
            exclude_max=inferred.max_value is not None and not inferred.max_inclusive,
        )

    elif a_type == fractions.Fraction:
        # Fractions must also include their contracts as we can not bound them
        # by filtering! This needs to be a special case.
        strategy = hypothesis.strategies.fractions(
            min_value=inferred.min_value, max_value=inferred.max_value
        )

    elif a_type == decimal.Decimal:
        # Decimals must also include their contracts as we can not bound them
        # by filtering! This needs to be a special case.
        strategy = hypothesis.strategies.decimals(
            min_value=inferred.min_value, max_value=inferred.max_value
        )

    elif a_type == datetime.date:
        min_value = datetime.date.min
        if inferred.min_value is not None:
            if inferred.min_inclusive:
                min_value = inferred.min_value
            else:
                min_value = inferred.min_value + datetime.timedelta(days=1)

        max_value = datetime.date.max
        if inferred.max_value is not None:
            if inferred.max_inclusive:
                max_value = inferred.max_value
            else:
                max_value = inferred.max_value - datetime.timedelta(days=1)

        strategy = hypothesis.strategies.dates(min_value=min_value, max_value=max_value)

    elif a_type == datetime.datetime:
        min_value = datetime.datetime.min
        if inferred.min_value is not None:
            if inferred.min_inclusive:
                min_value = inferred.min_value
            else:
                min_value = inferred.min_value + datetime.timedelta(microseconds=1)

        max_value = datetime.datetime.max
        if inferred.max_value is not None:
            if inferred.max_inclusive:
                max_value = inferred.max_value
            else:
                max_value = inferred.max_value - datetime.timedelta(microseconds=1)

        strategy = hypothesis.strategies.datetimes(
            min_value=min_value, max_value=max_value
        )

    elif a_type == datetime.time:
        min_value = datetime.time.min
        if inferred.min_value is not None:
            min_value = inferred.min_value

            if not inferred.min_inclusive:
                if min_value == datetime.time.max:
                    raise ValueError(
                        f"The inferred exclusive lower bound for the time is equal "
                        f"datetime.time.max ({datetime.time.max}) "
                        f"so we can not compute the next greater time."
                    )

                min_value = (
                    datetime.datetime.combine(
                        datetime.date(2000, 1, 1), min_value, min_value.tzinfo
                    )
                    + datetime.timedelta(microseconds=1)
                ).time()

        max_value = datetime.time.max
        if inferred.max_value is not None:
            max_value = inferred.max_value

            if not inferred.max_inclusive:
                if max_value == datetime.time.min:
                    raise ValueError(
                        f"The inferred exclusive upper bound for the time is equal "
                        f"datetime.time.min ({datetime.time.min}) "
                        f"so we can not compute the previous less-than time."
                    )

                max_value = (
                    datetime.datetime.combine(
                        datetime.date(2000, 1, 1), max_value, max_value.tzinfo
                    )
                    - datetime.timedelta(microseconds=1)
                ).time()

        strategy = hypothesis.strategies.times(min_value=min_value, max_value=max_value)

    elif a_type == datetime.timedelta:
        min_value = datetime.timedelta.min
        if inferred.min_value is not None:
            if inferred.min_inclusive:
                min_value = inferred.min_value
            else:
                min_value = inferred.min_value + datetime.timedelta(microseconds=1)

        max_value = datetime.timedelta.max
        if inferred.max_value is not None:
            if inferred.max_inclusive:
                max_value = inferred.max_value
            else:
                max_value = inferred.max_value - datetime.timedelta(microseconds=1)

        strategy = hypothesis.strategies.timedeltas(
            min_value=min_value, max_value=max_value
        )

    else:
        raise AssertionError(f"Unexpected type hint: {a_type}")

    return strategy


# We need to compile a dummy pattern so that we can compare addresses of re.Pattern.match functions.
_DUMMY_RE = re.compile(r"something")


class PatternInCondition(Generic[AnyStr]):
    """Represent a pattern that defines the contract condition."""

    def __init__(self, pattern: Pattern[AnyStr], fullmatch: bool) -> None:
        """Initialize with the given values."""
        self.pattern: Pattern[AnyStr] = pattern
        self.fullmatch = fullmatch


def _infer_regexp_from_condition(
    arg_name: str, condition: Callable[..., Any]
) -> Optional[PatternInCondition[AnyStr]]:
    """Try to infer the regular expression pattern from a precondition."""
    body_node = _body_node_from_condition(condition=condition)
    if body_node is None:
        return None

    if not isinstance(body_node, ast.Call):
        return None

    if not _no_name_in_descendants(root=body_node.func, name=arg_name):
        return None

    if not isinstance(body_node.func, ast.Attribute):
        return None

    if body_node.func.attr == "match":
        fullmatch = False
    elif body_node.func.attr == "fullmatch":
        fullmatch = True
    else:
        # This is neither ``re.match`` nor ``re.fullmatch`` function.
        return None

    callee, recomputed = _recompute(condition=condition, node=body_node.func.value)
    if not recomputed:
        return None

    if callee == re:
        # Match "re.match(r'Some pattern', s, *args, *kwargs)
        if (
            len(body_node.args) >= 2
            and _no_name_in_descendants(root=body_node.args[0], name=arg_name)
            and isinstance(body_node.args[1], ast.Name)
            and body_node.args[1].id == arg_name
            and not any(
                _no_name_in_descendants(root=arg, name=arg_name)
                for arg in body_node.args[2:]
            )
        ):
            # Recompute the pattern
            args = []  # type: List[Any]
            for arg in [body_node.args[0]] + body_node.args[2:]:
                value, recomputed = _recompute(condition=condition, node=arg)
                if not recomputed:
                    return None

                args.append(value)

            kwargs = {}  # type: Dict[str, Any]
            for keyword in body_node.keywords:
                value, recomputed = _recompute(condition=condition, node=keyword.value)
                if not recomputed:
                    return None

                assert keyword.arg is not None, (
                    f"Unexpected missing arg for a keyword: {ast.dump(keyword)}"
                )
                kwargs[keyword.arg] = value

            pattern = re.compile(*args, **kwargs)
            return PatternInCondition(pattern=pattern, fullmatch=fullmatch)

    elif isinstance(callee, Pattern):
        return PatternInCondition(pattern=callee, fullmatch=fullmatch)

    return None


@icontract.require(
    lambda arg_name, contracts: all(
        len(contract.condition_args) == 1 and contract.condition_args[0] == arg_name
        for contract in contracts
    ),
    "All contracts are single-argument contracts related to this argument.",
)
def _infer_str_strategy_from_preconditions(
    arg_name: str, contracts: List[icontract._types.Contract]
) -> Tuple[
    Optional[hypothesis.strategies.SearchStrategy[AnyStr]],
    List[icontract._types.Contract],
]:
    """
    Try to match code patterns on AST of the preconditions contracts and infer the string strategy.

    Return (strategy if possible, remaining contracts).
    """
    found_idx = -1  # Index of the contract that defines the pattern, -1 if not found
    pattern_in_condition = None  # type: Optional[PatternInCondition[AnyStr]]

    for i, contract in enumerate(contracts):
        pattern_in_condition = _infer_regexp_from_condition(
            arg_name=arg_name, condition=contract.condition
        )
        if pattern_in_condition is not None:
            found_idx = i
            break

    if found_idx == -1:
        return None, contracts.copy()

    assert pattern_in_condition is not None
    return (
        hypothesis.strategies.from_regex(
            regex=pattern_in_condition.pattern, fullmatch=pattern_in_condition.fullmatch
        ),
        contracts[:found_idx] + contracts[found_idx + 1 :],
    )


@icontract.require(
    lambda arg_name, contracts: (
        contracts is None
        or all(
            len(contract.condition_args) == 1 and contract.condition_args[0] == arg_name
            for contract in contracts
        )
    )
)
def _infer_strategy_for_argument(
    arg_name: str, type_hint: Any, contracts: Optional[List[icontract._types.Contract]]
) -> hypothesis.strategies.SearchStrategy[Any]:
    """Infer the initial strategy for the argument."""
    if contracts is None:
        return hypothesis.strategies.from_type(type_hint)

    strategy = None  # type: Optional[hypothesis.strategies.SearchStrategy[Any]]
    remaining_contracts = contracts

    if type_hint in [
        int,
        float,
        fractions.Fraction,
        decimal.Decimal,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
    ]:
        inferred, remaining_contracts = _infer_min_max_from_preconditions(
            arg_name=arg_name, type_hint=type_hint, contracts=contracts
        )

        if (
            inferred.min_value is not None
            and inferred.max_value is not None
            and inferred.min_value > inferred.max_value
        ):
            raise ValueError(
                f"The min and max values inferred for the argument {arg_name} could not be satisfied: "
                f"inferred min is {inferred.min_value}, inferred max is {inferred.max_value}. Are your preconditions correct?"
            )

        strategy = _make_strategy_with_min_max_for_type(
            a_type=type_hint, inferred=inferred
        )

    if strategy is None and type_hint == str:
        strategy, remaining_contracts = _infer_str_strategy_from_preconditions(
            arg_name=arg_name, contracts=contracts
        )

    if strategy is None:
        strategy = hypothesis.strategies.from_type(type_hint)
        remaining_contracts = contracts

    assert strategy is not None

    for contract in remaining_contracts:
        # Since ``hypothesis.internal.reflection.extract_lambda_source`` is quite
        # limited (as of 2021-04-01), we need to replace the representation of
        # the condition with our own source code.
        #
        # For example, a multi-line condition such as ``lambda lst: all(...)`` will be
        # rendered as ``lambda lst: <unknown>``.
        lambda_inspection = _inspect_condition(contract.condition)

        # As this attribute is non-invasive, we polute the original condition.
        if lambda_inspection is not None and not hasattr(
            contract.condition, "__icontract_hypothesis_source_code__"
        ):
            condition_repr = lambda_inspection.atok.get_text(lambda_inspection.node)
            contract.condition.__icontract_hypothesis_source_code__ = condition_repr

        strategy = strategy.filter(contract.condition)

    return strategy


def _condition_repr(
    contract: icontract._types.Contract,
    inspection: Optional[icontract._represent.ConditionLambdaInspection],
) -> str:
    """Describe the dictionary adapter without changing executable code."""
    condition_name = getattr(contract.condition, "__name__", "")
    if not condition_name.isidentifier() or keyword.iskeyword(condition_name):
        condition_name = "_condition"

    used_names = {condition_name}
    if inspection is not None:
        used_names.update(
            node.id
            for node in ast.walk(inspection.node)
            if isinstance(node, ast.Name) and node.id not in contract.condition_arg_set
        )

    parameter = "d"
    while parameter in used_names:
        parameter = "_" + parameter

    values = {
        name: parameter if name == "_KWARGS" else f"{parameter}[{name!r}]"
        for name in contract.condition_args
    }
    arguments = ", ".join(f"{name}={value}" for name, value in values.items())
    if inspection is None:
        return f"lambda {parameter}: {condition_name}({arguments})"

    node = inspection.node
    source = inspection.atok.get_text(node)
    call_repr = f"lambda {parameter}: ({source})({arguments})"

    # A textual substitution cannot safely distinguish bindings in nested scopes.
    scoped_nodes = (
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.NamedExpr,
    )
    if any(isinstance(child, scoped_nodes) for child in ast.walk(node.body)):
        return call_repr

    start, end = inspection.atok.get_text_range(node.body)
    text = inspection.atok.text[start:end]
    replacements = []  # type: List[Tuple[int, int, str]]
    for child in ast.walk(node.body):
        if isinstance(child, ast.Name) and child.id in values:
            left, right = inspection.atok.get_text_range(child)
            if right <= left:
                return call_repr
            replacements.append((left - start, right - start, values[child.id]))

    if not replacements:
        # Preserve the original representation for conditions without arguments.
        return source if not contract.condition_args else call_repr

    # Apply changes from right to left to preserve the original offsets.
    for left, right, value in sorted(replacements, reverse=True):
        text = text[:left] + value + text[right:]

    return f"lambda {parameter}: {text}"


@icontract.ensure(
    lambda result: hasattr(result, "__icontract_hypothesis_source_code__"),
    "The attribute is available (we depend on monkey-patching Hypothesis reflection function)",
)
@icontract.ensure(
    lambda result: result.__name__ == "<lambda>",
    "Condition-as-filter-function is lambda "
    "(even in cases where it calls a function as it is rewritten)",
)
def _rewrite_condition_as_filter_on_kwargs(
    contract: icontract._types.Contract,
) -> Callable[..., Any]:
    """Adapt a condition to accept a dictionary of function arguments."""
    if "_ARGS" in contract.condition_arg_set:
        raise NotImplementedError(
            "The handling of the special argument _ARGS is not currently handled "
            "by icontract-hypothesis. Please create a new issue on GitHub if you need "
            "this feature: https://github.com/mristin/icontract-hypothesis/issues"
        )

    condition = contract.condition
    names = tuple(contract.condition_args)
    predicate = lambda d: condition(
        **{name: d if name == "_KWARGS" else d[name] for name in names}
    )
    inspection = _inspect_condition(condition)
    predicate.__icontract_hypothesis_source_code__ = _condition_repr(  # type: ignore
        contract, inspection
    )
    return predicate


def _infer_strategy_from_conjunction(
    type_hints: Mapping[str, Any],
    conjunction: List[icontract._types.Contract],
    self_instance: Optional[Any],
) -> hypothesis.strategies.SearchStrategy[Any]:
    """
    Infer the strategy that satisfies the given conjunction of contracts.

    If ``self_instance`` has been specified, it is included as 'self' in the initial
    fixed dictionary of strategies.
    """
    ##
    # Group single-argument contracts by the argument,
    # keep the list of the zero or multi-argument contracts
    ##

    single_argument_contracts: MutableMapping[str, List[icontract._types.Contract]] = {}

    non_single_argument_contracts = []  # type: List[icontract._types.Contract]

    for contract in conjunction:
        if len(contract.condition_args) == 1 and contract.condition_args[0] not in (
            "_ARGS",
            "_KWARGS",
        ):
            arg_name = contract.condition_args[0]

            if arg_name not in single_argument_contracts:
                single_argument_contracts[arg_name] = [contract]
            else:
                single_argument_contracts[arg_name].append(contract)
        else:
            non_single_argument_contracts.append(contract)

    if self_instance is not None and "self" in type_hints:
        raise AssertionError(
            f"Unexpected 'self' in ``type_hints`` {type_hints!r} "
            f"while ``self_instance`` is given as {self_instance}. "
            f"We expected 'self' to be removed from type hints in this case."
        )

    mapping = {
        arg_name: _infer_strategy_for_argument(
            arg_name=arg_name,
            type_hint=type_hint,
            contracts=single_argument_contracts.get(arg_name, None),
        )
        for arg_name, type_hint in type_hints.items()
    }

    if self_instance is not None and "self" in mapping:
        raise AssertionError(
            f"Unexpected 'self' in mapping of initial strategies {mapping!r} "
            f"while ``self_instance`` is given as {self_instance}."
        )

    if self_instance is not None:
        # The contracts on ``self`` will be handled in a special way, see below.
        mapping["self"] = hypothesis.strategies.just(self_instance)

    strategy = hypothesis.strategies.fixed_dictionaries(mapping)

    # We need to chain contracts on ``self`` *after* the initial dictionary as Hypothesis
    # optimizes away ``.filter`` on ``just`` strategy in representation (though the filtering
    # is actually performed).
    #
    # See https://github.com/HypothesisWorks/hypothesis/pull/2688 (code) and
    # https://github.com/HypothesisWorks/hypothesis/issues/2036 (issue) and
    # https://github.com/HypothesisWorks/hypothesis/issues/2964 (issue).
    if self_instance is not None and "self" in single_argument_contracts:
        for contract in single_argument_contracts["self"]:
            strategy = strategy.filter(
                _rewrite_condition_as_filter_on_kwargs(contract=contract)
            )

    for contract in non_single_argument_contracts:
        strategy = strategy.filter(
            _rewrite_condition_as_filter_on_kwargs(contract=contract)
        )

    return strategy


def _create_strategy_only_from_type_hints(
    type_hints: Mapping[str, Any], self_instance: Optional[Any]
) -> hypothesis.strategies.SearchStrategy[Any]:
    """
    Create a strategy based only on type hints.

    For example, this is the strategy that you can infer if there are no preconditions.

    If ``self_instance`` is specified, it is included as a single-example strategy for 'self'.
    """
    mapping = {
        arg_name: hypothesis.strategies.from_type(arg_type)
        for arg_name, arg_type in type_hints.items()
    }

    if self_instance is not None and "self" in mapping:
        raise AssertionError(
            f"Unexpected 'self' in strategy mapping {mapping!r} "
            f"while self_instance has been specified: {self_instance}"
        )

    if self_instance is not None:
        mapping["self"] = hypothesis.strategies.just(self_instance)

    return hypothesis.strategies.fixed_dictionaries(mapping)


def infer_strategy(
    func: CallableT,
    localns: Optional[Dict[str, Any]] = None,
    globalns: Optional[Dict[str, Any]] = None,
) -> hypothesis.strategies.SearchStrategy[Any]:
    r"""
    Infer the search strategy of the arguments for the given function.

    You need to supply ``localns`` and ``globalns``, the local namespace and the global namespace
    of the function, respectively, in case there are forward references which can not be
    automatically resolved. For example, if you have nested classes and functions.
    See PEP 563 and the bug reports related to ``typing.get_type_hints`` for more details.

    Apart from the internal usage, this function is mainly meant for manual inspection
    of the inferred strategies.

    Here is an example how you can debug what strategies will be used to test ``some_func``:

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... def some_func(x: int) -> None:
    ...    ...

    >>> strategy = icontract_hypothesis.infer_strategy(some_func)
    >>> str(strategy)
    "fixed_dictionaries({'x': integers(min_value=1)})"
    """
    parameters = inspect.signature(func).parameters
    if len(parameters) == 0 and not hasattr(func, "__self__"):
        return hypothesis.strategies.just({})

    for name, parameter in parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise NotImplementedError(
                "Inferring strategy of a function with positional-only arguments is not "
                "currently implemented in icontract-hypothesis. "
                "Please create an issue on GitHub if you need this feature: "
                "https://github.com/mristin/icontract-hypothesis/issues"
            )

    type_hints = typing.get_type_hints(func, localns=localns, globalns=globalns)
    if "return" in type_hints:
        del type_hints["return"]

    parameter_set = set(parameters.keys())

    # We need to take special care when handling the bound methods whose contracts involve 'self'.
    if hasattr(func, "__self__"):
        assert "self" not in parameter_set, (
            f"Unexpected ``self`` in the parameter set when ``__self__`` was set "
            f"on the function {func}; the parameter set was: {parameter_set}"
        )

        if "self" in type_hints:
            del type_hints["self"]

    if "self" in parameter_set and func.__name__ == "__init__":
        # The self instance does not exist yet in the __init__ so we can not generate it and
        # can safely remove it from the parameters.
        parameter_set.remove("self")

        if "self" in type_hints:
            del type_hints["self"]

    # If the class specifies its own ``__new__`` we have to remove the first parameter corresponding
    # to the class from the list of parameters. Python will supply it by convention.
    #
    # However, in case of ``object.__new__``, there is no ``cls`` as the first argument!
    #
    elif (
        func.__name__ == "__new__" and func != object.__new__  # pylint: disable=comparison-with-callable
    ):
        if len(parameters.keys()) > 0:
            # Remove the first parameter which points to the class.
            #
            # The dunder ``__new__`` is special as we do not want to generate the first argument
            # corresponding to the class.
            cls_parameter = next(k for k in parameters)

            parameter_set.remove(cls_parameter)

            if cls_parameter in type_hints:
                del type_hints[cls_parameter]

    # This might be an unbound instance method.
    # There is no way in Python 3 how we can directly figure out the class of an unbound
    # method, so we try to hack our way through using ``__qualname__``.
    #
    # See https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3
    elif (
        not hasattr(func, "__self__")
        and "self" in parameters
        and "self" not in type_hints
        and inspect.isfunction(func)
        and ".<locals>." not in func.__qualname__
    ):
        parts = func.__qualname__.rsplit(".", 1)
        if len(parts) >= 2:
            maybe_cls = getattr(inspect.getmodule(func), parts[0], None)
            if inspect.isclass(maybe_cls):
                type_hints["self"] = maybe_cls

    typed_args = set(type_hints)

    for name, parameter in parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            # Ignore variable keyword and positional arguments in further analysis
            parameter_set.remove(name)

            if name in type_hints:
                typed_args.remove(name)
                del type_hints[name]

    if typed_args != parameter_set:
        raise TypeError(
            f"No search strategy could be inferred for the function: {func}; "
            f"the following arguments are missing the type annotations: {list(parameter_set.difference(typed_args))};\n\n"
            f"sorted typed_args was {sorted(typed_args)}, sorted parameter_set was {sorted(parameter_set)}"
        )

    checker = icontract._checkers.find_checker(func)

    if checker is None:
        return _create_strategy_only_from_type_hints(
            type_hints=type_hints, self_instance=getattr(func, "__self__", None)
        )

    maybe_preconditions = getattr(checker, "__preconditions__", None)

    preconditions = None  # type: Optional[List[List[icontract._types.Contract]]]
    if maybe_preconditions is not None:
        assert isinstance(maybe_preconditions, list)
        assert all(isinstance(conjunction, list) for conjunction in maybe_preconditions)
        assert all(
            isinstance(contract, icontract._types.Contract)
            for conjunction in maybe_preconditions
            for contract in conjunction
        )

        preconditions = cast(List[List[icontract._types.Contract]], maybe_preconditions)

    if preconditions is None or len(preconditions) == 0:
        return _create_strategy_only_from_type_hints(
            type_hints=type_hints, self_instance=getattr(func, "__self__", None)
        )

    strategies = [
        _infer_strategy_from_conjunction(
            type_hints=type_hints,
            conjunction=conjunction,
            self_instance=getattr(func, "__self__", None),
        )
        for conjunction in preconditions
    ]

    if len(strategies) == 1:
        return strategies[0]

    return hypothesis.strategies.one_of(strategies)


def test_with_inferred_strategy(
    func: CallableT,
    localns: Optional[Dict[str, Any]] = None,
    globalns: Optional[Dict[str, Any]] = None,
) -> None:
    r"""
    Use type hints and contracts to infer the search strategy and test the function.

    You need to supply ``localns`` and ``globalns``, the local namespace and the global namespace
    of the function, respectively, in case there are forward references which can not be
    automatically resolved. For example, if you have nested classes and functions.
    See PEP 563 and the bug reports related to ``typing.get_type_hints`` for more details.

    Here is an example test case which tests a function ``some_func``:

    >>> import unittest
    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.require(lambda x: x < 100)
    ... @icontract.require(lambda y: 0 < y < 100)
    ... def some_func(x: int, y: int) -> None:
    ...     ...

    >>> class TestSomething(unittest.TestCase):
    ...     def test_that_it_works(self) -> None:
    ...         icontract_hypothesis.test_with_inferred_strategy(some_func)

    >>> TestSomething().test_that_it_works()
    """
    # for pydocstyle

    # Since ``self`` is already included in the strategy, we have to unbind the ``func``.
    # Please see ``infer_strategy`` function and how it deals with ``self_instance``.
    func_to_execute = func
    if hasattr(func, "__self__"):
        func_to_execute = func.__func__  # type: ignore

    strategy = infer_strategy(func=func, localns=localns, globalns=globalns)

    def execute(kwargs: Dict[str, Any]) -> None:
        func_to_execute(**kwargs)

    wrapped = hypothesis.given(strategy)(execute)
    wrapped()


def _register_with_hypothesis(cls: Type[T]) -> None:
    """
    Register ``cls`` with Hypothesis based on the inferred strategy.

    The registration is necessary so that the preconditions on the __init__ are propagated
    in ``hypothesis.strategies.builds``.
    """
    # We should not register abstract classes as this will mislead Hypothesis to instantiate
    # them. However, if a class provides a ``__new__``, we can instantiate it even if it is
    # abstract!
    # pylint: disable=comparison-with-callable
    if inspect.isabstract(cls) and cls.__new__ == object.__new__:
        return

    if cls in hypothesis.strategies._internal.types._global_type_lookup:
        # Do not re-register
        return

    init = cls.__init__

    if inspect.isfunction(init):
        # If there is is no checker nor pre-conditions on ``__init__``, we should simply delegate
        # the case to Hypothesis and not register the strategy for this type.

        checker = icontract._checkers.find_checker(init)
        if checker is None:
            return

        maybe_preconditions = getattr(checker, "__preconditions__", None)
        if maybe_preconditions is None or len(maybe_preconditions) == 0:
            return

        # We have to add the ``a_type`` itself to a local namespace for forward references.
        #
        # This is needed if we register a class through the ``icontract.DBCMeta``
        # meta-class where it references itself. For example, a node in a linked list.

        strategy = infer_strategy(init, localns={cls.__name__: cls})

    elif isinstance(init, icontract._checkers._SLOT_WRAPPER_TYPE):
        # We have to distinguish this special case which is used by named tuples and
        # possibly other optimized data structures.
        # In those cases, we have to infer the strategy based on __new__ instead of __init__.
        new = cls.__new__
        assert new is not None, (
            f"Expected __new__ in {cls} if __init__ is a slot wrapper."
        )

        # We have to add the ``a_type`` itself to a local namespace for forward references.
        # This is usually the case for the return value of ``__new__``.
        #
        # In particular, the class is not available in the module while we are
        # registering it through the ``icontract.DBCMeta`` meta-class as its loading is still
        # in progress.

        strategy = infer_strategy(new, localns={cls.__name__: cls})
    else:
        raise AssertionError(
            f"Expected __init__ to be either a function or a slot wrapper, but got: {type(init)}"
        )

    pack_repr = f"lambda d: {cls.__name__}(**d)"

    pack = lambda d: cls(**d)  # type: ignore
    pack.__icontract_hypothesis_source_code__ = pack_repr  # type: ignore

    strategy = strategy.map(pack=pack)
    hypothesis.strategies.register_type_strategy(
        custom_type=cls,
        strategy=lambda _: strategy,
    )


def _hook_into_icontract_and_hypothesis() -> None:
    """
    Redirect ``icontract._metaclass._register_for_hypothesis``.

    All the classes previously registered by icontract are now re-registered
    by ``_register_with_hypothesis``.
    """
    if not hasattr(icontract._metaclass, "_CONTRACT_CLASSES"):
        return  # already hooked in

    icontract._metaclass._register_for_hypothesis = _register_with_hypothesis

    for cls in icontract._metaclass._CONTRACT_CLASSES:
        _register_with_hypothesis(cls)

    # Delete in order to fail fast
    del icontract._metaclass._CONTRACT_CLASSES

    # Monkey-patch lambda source so that we do not have to introduce
    # strategy classes just for this functionality.
    # See https://github.com/HypothesisWorks/hypothesis/issues/2713
    reflection = hypothesis.internal.reflection
    if hasattr(reflection, "lambda_sources"):
        source_module = reflection.lambda_sources
        source_name = "lambda_description"
    else:
        source_module = reflection
        source_name = "extract_lambda_source"

    upstream_extract_lambda_source = getattr(source_module, source_name)

    def describe_lambda(condition: Callable[..., Any]) -> str:
        return getattr(condition, "__icontract_hypothesis_source_code__", None) or (
            upstream_extract_lambda_source(condition)
        )

    setattr(source_module, source_name, describe_lambda)


_hook_into_icontract_and_hypothesis()
