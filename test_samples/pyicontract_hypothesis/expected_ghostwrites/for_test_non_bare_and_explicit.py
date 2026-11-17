"""Test test_samples.pyicontract_hypothesis.sample_module with inferred Hypothesis strategies."""

import unittest

from hypothesis import given

from hypothesis.strategies import fixed_dictionaries, integers

import icontract_hypothesis

import test_samples.pyicontract_hypothesis.sample_module


class TestWithInferredStrategies(unittest.TestCase):
    """Test all functions from test_samples.pyicontract_hypothesis.sample_module with inferred Hypothesis strategies."""

    def test_some_func(self) -> None:
        @given(
            x=integers(min_value=1)
        )
        def execute(**kwargs) -> None:
            test_samples.pyicontract_hypothesis.sample_module.some_func(**kwargs)

        execute()

    def test_another_func(self) -> None:
        icontract_hypothesis.test_with_inferred_strategy(
            test_samples.pyicontract_hypothesis.sample_module.another_func)

    def test_yet_another_func(self) -> None:
        @given(
            fixed_dictionaries({'x': integers(), 'y': integers()}).filter(lambda d: d['x'] < d['y'])
        )
        def execute(kwargs) -> None:
            test_samples.pyicontract_hypothesis.sample_module.yet_another_func(**kwargs)

        execute()


if __name__ == '__main__':
    unittest.main()
