"""
Test Python 3.7-specific features.

For example, we could not test ``test_strategy_inference_on_forward_declarations`` in Python 3.6 as
the multiple inheritance of ``icontract.DBC`` and ``Sequance[int]`` caused a meta-class conflict.
"""
