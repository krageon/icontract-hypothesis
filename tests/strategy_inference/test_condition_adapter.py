"""Test condition adapters independently of strategy representations."""

import ast
import unittest
from unittest import mock

import hypothesis
import icontract
import icontract._types

import icontract_hypothesis as ih

GLOBAL_LIMIT = 0


def _adapter(func):
    return ih._rewrite_condition_as_filter_on_kwargs(func.__preconditions__[0][0])


def _source(predicate):
    return predicate.__icontract_hypothesis_source_code__


def _check_inferred_strategy(func):
    @hypothesis.given(ih.infer_strategy(func))
    @hypothesis.settings(max_examples=10)
    def check(data):
        func(**data)

    check()


class TestConditionAdapter(unittest.TestCase):
    def test_source_and_bounds(self):
        @icontract.require(lambda x, y: x < y)
        def func(x: int, y: int):
            pass

        predicate = _adapter(func)
        self.assertEqual("<lambda>", predicate.__name__)
        self.assertEqual("lambda d: d['x'] < d['y']", _source(predicate))
        self.assertTrue(predicate({"x": 1, "y": 2}))
        self.assertFalse(predicate({"x": 2, "y": 1}))

        @icontract.require(lambda x: x > 0)
        def positive(x: int):
            pass

        condition = positive.__preconditions__[0][0].condition
        self.assertIsInstance(ih._body_node_from_condition(condition), ast.Compare)
        self.assertIn("integers(min_value=1)", str(ih.infer_strategy(positive)))

    def test_source_free_lambdas(self):
        for expression in ("lambda x: x > 0", "lambda x, y: x < y"):
            condition = eval(compile(expression, "<source-free>", "eval"))
            self.assertIsNone(ih._inspect_condition(condition))
            self.assertIsNone(ih._body_node_from_condition(condition))
            predicate = ih._rewrite_condition_as_filter_on_kwargs(
                icontract._types.Contract(condition)
            )
            self.assertTrue(predicate({"x": 1, "y": 2}))
            self.assertIn("_condition(", _source(predicate))

            def func(x: int, y: int):
                pass

            _check_inferred_strategy(icontract.require(condition)(func))

    def test_named_condition_and_defaults(self):
        limit = 0

        def condition(x, y=2):
            return limit < x < y

        predicate = ih._rewrite_condition_as_filter_on_kwargs(
            icontract._types.Contract(condition)
        )
        self.assertTrue(predicate({"x": 1, "y": 2}))
        limit = 10
        self.assertFalse(predicate({"x": 1, "y": 2}))
        self.assertEqual(2, condition.__defaults__[0])

    def test_live_closure_and_globals(self):
        limit = 0

        @icontract.require(lambda x, y: GLOBAL_LIMIT + limit < x < y)
        def func(x, y):
            pass

        predicate = _adapter(func)
        self.assertTrue(predicate({"x": 1, "y": 2}))
        limit = 10
        self.assertFalse(predicate({"x": 1, "y": 2}))
        limit = 0
        with mock.patch(__name__ + ".GLOBAL_LIMIT", 10):
            self.assertFalse(predicate({"x": 1, "y": 2}))

    def test_kwargs_identity(self):
        data = {"x": 1, "extra": True}

        def condition(x, _KWARGS):
            return x == 1 and _KWARGS is data

        predicate = ih._rewrite_condition_as_filter_on_kwargs(
            icontract._types.Contract(condition)
        )
        self.assertTrue(predicate(data))

    def test_callable_object(self):
        class Condition:
            def __call__(self, x, y):
                return x < y

        condition = Condition()
        self.assertIsNone(ih._inspect_condition(condition))
        predicate = ih._rewrite_condition_as_filter_on_kwargs(
            icontract._types.Contract(condition)
        )
        self.assertTrue(predicate({"x": 1, "y": 2}))
        self.assertFalse(predicate({"x": 2, "y": 1}))

    def test_args_remains_unsupported(self):
        condition = lambda _ARGS: True
        with self.assertRaises(NotImplementedError):
            ih._rewrite_condition_as_filter_on_kwargs(
                icontract._types.Contract(condition)
            )

    def test_parameter_name_collisions(self):
        @icontract.require(lambda d, e: d < e)
        def local(d, e):
            pass

        self.assertEqual("lambda d: d['d'] < d['e']", _source(_adapter(local)))

        d = 2

        @icontract.require(lambda x, y: x + d < y)
        def closed(x, y):
            pass

        predicate = _adapter(closed)
        self.assertEqual("lambda _d: _d['x'] + d < _d['y']", _source(predicate))
        self.assertTrue(eval(_source(predicate), {"d": d})({"x": 1, "y": 4}))

    def test_nested_scopes_and_assignment(self):
        @icontract.require(lambda xs, x: all(x > 0 for x in xs))
        def comprehension(xs, x):
            pass

        @icontract.require(lambda x, y: (lambda x: x > 0)(y))
        def nested(x, y):
            pass

        @icontract.require(lambda x, y: (z := x + y) > 0)
        def assignment(x, y):
            pass

        cases = (
            (comprehension, {"xs": [1, 2], "x": -100}),
            (nested, {"x": -100, "y": 2}),
            (assignment, {"x": 1, "y": 2}),
        )
        for func, data in cases:
            with self.subTest(func=func.__name__):
                predicate = _adapter(func)
                self.assertTrue(predicate(data))
                self.assertTrue(eval(_source(predicate))(data))

    def test_zero_arguments(self):
        @icontract.require(lambda: True)
        def func(x):
            pass

        predicate = _adapter(func)
        self.assertTrue(predicate({"x": 1}))
        self.assertEqual("lambda: True", _source(predicate))

    def test_unavailable_token_range(self):
        @icontract.require(lambda x, y: x < y)
        def func(x, y):
            pass

        contract = func.__preconditions__[0][0]
        inspection = ih._inspect_condition(contract.condition)
        original = inspection.atok.get_text_range

        def get_text_range(node, padded=True):
            return (0, 0) if isinstance(node, ast.Name) else original(node, padded)

        with mock.patch.object(inspection.atok, "get_text_range", get_text_range):
            text = ih._condition_repr(contract, inspection)
        self.assertTrue(eval(text)({"x": 1, "y": 2}))

    def test_inspection_errors(self):
        condition = lambda x: x > 0
        for error in (OSError, TypeError, SyntaxError, ValueError):
            with mock.patch.object(
                icontract._represent, "inspect_lambda_condition", side_effect=error
            ):
                self.assertIsNone(ih._inspect_condition(condition))

        with mock.patch.object(
            icontract._represent, "inspect_lambda_condition", side_effect=RuntimeError
        ), self.assertRaises(RuntimeError):
            ih._inspect_condition(condition)


if __name__ == "__main__":
    unittest.main()
