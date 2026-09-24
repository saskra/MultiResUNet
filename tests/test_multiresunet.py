"""Minimal smoke tests for the TensorFlow MultiResUnet model.

Checks that the default call still reproduces the original single-channel model
exactly, and that the optional n_classes argument produces the requested number
of output channels. Run directly with: python tests/test_multiresunet.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tensorflow'))

from MultiResUNet import MultiResUnet

DEFAULT_PARAM_COUNT = 7262750


def check_default_matches_original():
    """Default call (no n_classes) must reproduce the original single-channel model."""
    model = MultiResUnet(128, 128, 3)
    assert model.output_shape == (None, 128, 128, 1), f'unexpected output shape: {model.output_shape}'
    assert model.count_params() == DEFAULT_PARAM_COUNT, f'unexpected param count: {model.count_params()}'


def check_explicit_n_classes_one_matches_default():
    """n_classes=1 given explicitly must be identical to the implicit default."""
    model = MultiResUnet(128, 128, 3, n_classes=1)
    assert model.output_shape == (None, 128, 128, 1)
    assert model.count_params() == DEFAULT_PARAM_COUNT


def check_multi_class_output():
    """n_classes > 1 must produce that many output channels."""
    model = MultiResUnet(128, 128, 3, n_classes=5)
    assert model.output_shape == (None, 128, 128, 5)


def main():
    """Run all smoke checks and report success."""
    check_default_matches_original()
    check_explicit_n_classes_one_matches_default()
    check_multi_class_output()
    print('All MultiResUnet smoke checks passed.')


if __name__ == '__main__':
    main()
