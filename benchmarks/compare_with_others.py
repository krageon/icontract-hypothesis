#!/usr/bin/env python3
"""Benchmark icontract against deal when used together with hypothesis."""

import os
import sys
import timeit

import deal
import dpcontracts
import hypothesis
import hypothesis.extra.dpcontracts
import hypothesis.strategies
import icontract
import tabulate

import icontract_hypothesis


def benchmark_icontract_assume_preconditions(arg_count: int = 1) -> None:
    """Benchmark the Hypothesis testing with icontract and rejection sampling."""
    count = 0

    if arg_count == 1:

        @icontract.require(lambda a: a > 0)
        def some_func(a: int) -> None:
            nonlocal count
            count += 1

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        @hypothesis.settings(
            suppress_health_check=(hypothesis.HealthCheck.filter_too_much,)
        )
        @hypothesis.given(a=hypothesis.strategies.integers())
        def execute(a: int) -> None:
            assume_preconditions(a)
            some_func(a)

    elif arg_count == 2:

        @icontract.require(lambda a: a > 0)
        @icontract.require(lambda b: b > 0)
        def some_func(a: int, b: int) -> None:
            nonlocal count
            count += 1

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        @hypothesis.settings(
            suppress_health_check=(hypothesis.HealthCheck.filter_too_much,)
        )
        @hypothesis.given(
            a=hypothesis.strategies.integers(), b=hypothesis.strategies.integers()
        )
        def execute(a: int, b: int) -> None:
            assume_preconditions(a=a, b=b)
            some_func(a, b)

    elif arg_count == 3:

        @icontract.require(lambda a: a > 0)
        @icontract.require(lambda b: b > 0)
        @icontract.require(lambda c: c > 0)
        def some_func(a: int, b: int, c: int) -> None:
            nonlocal count
            count += 1

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        @hypothesis.settings(
            suppress_health_check=(hypothesis.HealthCheck.filter_too_much,)
        )
        @hypothesis.given(
            a=hypothesis.strategies.integers(),
            b=hypothesis.strategies.integers(),
            c=hypothesis.strategies.integers(),
        )
        def execute(a: int, b: int, c: int) -> None:
            assume_preconditions(a=a, b=b, c=c)
            some_func(a, b, c)

    else:
        raise NotImplementedError(f"arg_count {arg_count}")

    execute()

    # Assert the count of function executions for fair tests
    assert count == 100


def benchmark_icontract_inferred_strategy(arg_count: int = 1) -> None:
    """Benchmark the Hypothesis testing with icontract and inferred search strategies."""
    count = 0

    if arg_count == 1:

        @icontract.require(lambda a: a > 0)
        def some_func(a: int) -> None:
            nonlocal count
            count += 1

    elif arg_count == 2:

        @icontract.require(lambda a: a > 0)
        @icontract.require(lambda b: b > 0)
        def some_func(a: int, b: int) -> None:
            nonlocal count
            count += 1

    elif arg_count == 3:

        @icontract.require(lambda a: a > 0)
        @icontract.require(lambda b: b > 0)
        @icontract.require(lambda c: c > 0)
        def some_func(a: int, b: int, c: int) -> None:
            nonlocal count
            count += 1

    else:
        raise NotImplementedError(f"arg_count {arg_count}")

    icontract_hypothesis.test_with_inferred_strategy(some_func)

    # Assert the count of function executions for fair tests
    assert count == 100


def benchmark_dpcontracts(arg_count: int = 1) -> None:
    """Benchmark the Hypothesis testing with dpcontracts."""
    count = 0

    if arg_count == 1:

        @dpcontracts.require("some dummy contract", lambda args: args.a > 0)
        def some_func(a: int) -> None:
            nonlocal count
            count += 1

        @hypothesis.settings(
            suppress_health_check=(hypothesis.HealthCheck.filter_too_much,)
        )
        @hypothesis.given(a=hypothesis.strategies.integers())
        def execute(a: int) -> None:
            hypothesis.extra.dpcontracts.fulfill(some_func)(a)

    elif arg_count == 2:

        @dpcontracts.require("some dummy contract", lambda args: args.a > 0)
        @dpcontracts.require("some dummy contract", lambda args: args.b > 0)
        def some_func(a: int, b: int) -> None:
            nonlocal count
            count += 1

        @hypothesis.settings(
            suppress_health_check=(hypothesis.HealthCheck.filter_too_much,)
        )
        @hypothesis.given(
            a=hypothesis.strategies.integers(), b=hypothesis.strategies.integers()
        )
        def execute(a: int, b: int) -> None:
            hypothesis.extra.dpcontracts.fulfill(some_func)(a, b)

    elif arg_count == 3:

        @dpcontracts.require("some dummy contract", lambda args: args.a > 0)
        @dpcontracts.require("some dummy contract", lambda args: args.b > 0)
        @dpcontracts.require("some dummy contract", lambda args: args.c > 0)
        def some_func(a: int, b: int, c: int) -> None:
            nonlocal count
            count += 1

        @hypothesis.settings(
            suppress_health_check=(hypothesis.HealthCheck.filter_too_much,)
        )
        @hypothesis.given(
            a=hypothesis.strategies.integers(),
            b=hypothesis.strategies.integers(),
            c=hypothesis.strategies.integers(),
        )
        def execute(a: int, b: int, c: int) -> None:
            hypothesis.extra.dpcontracts.fulfill(some_func)(a, b, c)

    else:
        raise NotImplementedError(f"arg_count {arg_count}")

    execute()

    # Assert the count of function executions for fair tests
    assert count == 100


def benchmark_deal(arg_count: int = 1) -> None:
    """Benchmark the Hypothesis testing with deal."""
    count = 0

    if arg_count == 1:

        @deal.pre(lambda _: _.a > 0)
        def some_func(a: int) -> None:
            nonlocal count
            count += 1

        for case in deal.cases(some_func, count=100):
            case()

    elif arg_count == 2:

        @deal.pre(lambda _: _.a > 0)
        @deal.pre(lambda _: _.b > 0)
        def some_func(a: int, b: int) -> None:
            nonlocal count
            count += 1

        for case in deal.cases(some_func, count=100):
            case()

    elif arg_count == 3:

        @deal.pre(lambda _: _.a > 0)
        @deal.pre(lambda _: _.b > 0)
        @deal.pre(lambda _: _.c > 0)
        def some_func(a: int, b: int, c: int) -> None:
            nonlocal count
            count += 1

        for case in deal.cases(some_func, count=100):
            case()
    else:
        raise NotImplementedError(f"arg_count {arg_count}")

    assert count == 100


def writeln_utf8(text: str = "") -> None:
    """
    Write the text to STDOUT using UTF-8 encoding followed by a new-line character.

    We can not use ``print()`` as we can not rely on the correct encoding in Windows.
    See: https://stackoverflow.com/questions/31469707/changing-the-locale-preferred-encoding-in-python-3-in-windows
    """
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(os.linesep.encode("utf-8"))


def measure_functions() -> None:
    # yapf: disable
    funcs = [
        'benchmark_icontract_inferred_strategy',
        'benchmark_icontract_assume_preconditions',
        'benchmark_dpcontracts',
        'benchmark_deal',
    ]
    # yapf: enable

    durations = [0.0] * len(funcs)

    number = 10
    for arg_count in [1, 2, 3]:
        for i, func in enumerate(funcs):
            duration = timeit.timeit(
                f"{func}(arg_count={arg_count})",
                setup=f"from __main__ import {func}",
                number=number,
            )

            durations[i] = duration

        table = []  # type: List[List[str]]

        for func, duration in zip(funcs, durations):
            # yapf: disable
            table.append([
                f'`{func}`',
                f'{duration:.2f} s',
                f'{duration * 1000 / number:.2f} ms',
                f'{duration * 100 / durations[0]:.0f}%'
            ])
            # yapf: enable

        # yapf: disable
        table_str = tabulate.tabulate(
            table,
            headers=['Case', 'Total time', 'Time per run', 'Relative time per run'],
            colalign=('left', 'right', 'right', 'right'),
            tablefmt='rst')
        # yapf: enable

        writeln_utf8()
        writeln_utf8(f"Argument count: {arg_count}")
        writeln_utf8()
        writeln_utf8(table_str)


if __name__ == "__main__":
    writeln_utf8("Benchmarking Hypothesis testing:")
    writeln_utf8("")
    measure_functions()
