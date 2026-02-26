"""Test test_samples.pyicontract_hypothesis.well_formatted_with_two_arguments with inferred Hypothesis strategies."""

import unittest

from hypothesis import given


class TestWithInferredStrategies(unittest.TestCase):
    """Test all functions from test_samples.pyicontract_hypothesis.well_formatted_with_two_arguments with inferred Hypothesis strategies."""

    def test_add(self) -> None:
        @given(
            a=integers(),
            b=integers()
        )
        def execute(**kwargs) -> None:
            tests.test_samples.pyicontract_hypothesis.well_formatted_with_two_arguments.add(**kwargs)

        execute()


if __name__ == '__main__':
    unittest.main()
