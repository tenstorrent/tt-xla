# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    config.addinivalue_line("markers", 'slow: marks tests as slow (deselect with -m "not slow")')
    config.addinivalue_line("markers", 'run_in_pp: marks tests to run in pipeline (select with -m "run_in_pp")')


def pytest_addoption(parser):
    # softmax_model
    parser.addoption(
        "--softmax_model",
        action="store",
        default="ModelFromAnotherOp",
        help="The name of the model type.",
    )
    # softmax_input_shape_dim
    parser.addoption(
        "--softmax_input_shape_dim",
        action="store",
        default=((1, 4), 1),
        help="Shape of the tensor and dim parameter for the operation.",
    )
    # softmax_dev_data_format
    parser.addoption(
        "--df",
        action="store",
        default=((1, 4), 1),
        help="Dev data format for the operation.",
    )
    # softmax_math_fidelity
    parser.addoption(
        "--mf",
        action="store",
        default=((1, 4), 1),
        help="Math Fidelity for the operation.",
    )


def pytest_generate_tests(metafunc):

    option_model = metafunc.config.option.softmax_model
    if "softmax_model" in metafunc.fixturenames and option_model is not None:
        metafunc.parametrize("softmax_model", [option_model])

    option_shape_dim = metafunc.config.option.softmax_input_shape_dim
    if "softmax_input_shape_dim" in metafunc.fixturenames and option_shape_dim is not None:
        input_shape_dim = eval(option_shape_dim) if isinstance(option_shape_dim, str) else option_shape_dim
        metafunc.parametrize("softmax_input_shape_dim", [input_shape_dim])

    option_df = metafunc.config.option.df
    if "df" in metafunc.fixturenames and option_df is not None:
        metafunc.parametrize("df", [option_df])

    option_mf = metafunc.config.option.mf
    if "mf" in metafunc.fixturenames and option_mf is not None:
        metafunc.parametrize("mf", [option_mf])
