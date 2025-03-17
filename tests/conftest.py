# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_configure(config: pytest.Config):
    """
    Registers custom pytest marker `record_test_properties(key1=val1, key2=val2, ...)`.

    Allowed keys are:
        - Every test:
            - `category`: utils.Category

        - Op tests:
            - `jax_op_name`: name of the operation in jax, e.g. `jax.numpy.exp`
            - `shlo_op_name`: name of the matching stablehlo operation

        - Model tests:
            - `model_name`: name of the model under test
            - 'model_group': utils.ModelGroup
            - `run_mode`: infra.RunMode
            - `bringup_status`: utils.BringupStatus
            - `pcc`: float
            - `atol`: float

    These are used to tag the function under test with properties which will be dumped
    to the final XML test report. These reports get picked up by other CI workflows and
    are used to display state of tests on a dashboard.
    """
    config.addinivalue_line(
        "markers",
        "record_test_properties(key_value_pairs): Record custom properties for the test",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runner",
        action="store",
    )

def record_test_properties_validate_keys(keys: dict):
        valid_keys = [
            "category",
            "jax_op_name",
            "shlo_op_name",
            "model_name",
            "model_group",
            "run_mode",
            "bringup_status",
            "pcc",
            "atol",
        ]

        # Check that only valid keys are used.
        if not all(key in valid_keys for key in keys):
            raise KeyError(
                f"Invalid keys found in 'record_test_properties' marker: {', '.join(keys)}. "
                f"Allowed keys are: {', '.join(valid_keys)}"
            )

def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to process the custom marker and attach recorder properties to the test.
    """

    runner = config.getoption("--runner")

    for item in items:
        # Add some test metadata in a 'tags' dictionary.
        tags = {"test_name": item.originalname, "specific_test_case": item.name, "runner": runner}

        # Look for the custom marker.
        properties_marker = item.get_closest_marker(name="record_test_properties")

        # Utils flags helping handling model tests properly.
        is_model_test = False
        model_group = None

        if properties_marker:
            # Extract the key-value pairs passed to the marker.
            properties: dict = properties_marker.kwargs

            # Validate that only allowed keys are used.
            record_test_properties_validate_keys(properties.keys())

            # Check if test contains marker ex. "nightly"
            is_model_test = item.get_closest_marker(name="model_test") is not None
            is_nightly_test = item.get_closest_marker(name="nightly") is not None

            # Turn all properties to strings.
            for k, v in properties.items():
                properties[k] = str(v)

            # Hydrate tags depending on test mark.
            if is_model_test:
                tags['mark'] = 'model_test'
                model_group = properties.get("model_group")

            if is_nightly_test:
                tags['mark'] = 'nightly'

            # Tag them.
            for key, value in properties.items():
                # Skip model_group, we don't need it in tags, we will insert it separately.
                if key != "model_group":
                    tags[key] = value

        # Attach metadata and tags dictionary as a single property.
        item.user_properties.extend([("tags", tags), ("owner", "tt-xla")])
        if is_model_test:
            # Add model group independently of tags dict.
            item.user_properties.append(("group", model_group))
