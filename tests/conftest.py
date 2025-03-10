# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_configure(config: pytest.Config):
    """
    Registers custom pytest marker `record_test_properties(key1=val1, key2=val2, ...)`.

    Allowed keys are ["test_category", "jax_op_name", "op_name", "model_name"].
        - `test_category`: one of ["op_test", "graph_test", "model_test", "multichip_test", "other"]
        - `jax_op_name`: name of the operation in jax, e.g. `jax.numpy.exp`
        - `shlo_op_name`: name of the matching stablehlo operation
        - `model_name`: name of the model under test (if recorded from a model test, or op
          under test comes from some model and we want to note that in the report)
        - 'model_group': one of ["priority", "generality"]
        - `run_mode`: one of ["inference", "training"]. Only exists for model tests.

    These are used to tag the function under test with properties which will be dumped
    to the final XML test report. These reports get picked up by other CI workflows and
    are used to display state of tests on a dashboard.
    """
    config.addinivalue_line(
        "markers",
        "record_test_properties(key_value_pairs): Record custom properties for the test",
    )


def pytest_collection_modifyitems(items):
    """
    Pytest hook to process the custom marker and attach recorder properties to the test.
    """

    def validate_keys(keys):
        valid_keys = [
            "test_category",
            "jax_op_name",
            "shlo_op_name",
            "model_name",
            "model_group",
            "run_mode",
        ]

        if not all(key in valid_keys for key in keys):
            raise KeyError(
                f"Invalid keys found in 'record_test_properties' marker: {', '.join(keys)}. "
                f"Allowed keys are: {', '.join(valid_keys)}"
            )

        if "model_name" in keys and "model_group" not in keys:
            raise KeyError(f"Model tests must have `model_group` property {keys}.")

    for item in items:
        # Add some test metadata in a 'tags' dictionary.
        tags = {"test_name": item.originalname, "specific_test_case": item.name}

        # Look for the custom marker.
        properties_marker = item.get_closest_marker(name="record_test_properties")

        # Specific model test handling.
        is_model_test = False
        model_group = None

        if properties_marker:
            # Extract the key-value pairs passed to the marker.
            properties: dict = properties_marker.kwargs
            # Validate that only allowed keys are used.
            validate_keys(properties.keys())

            is_model_test = properties.get("test_category", None) == "model_test"
            if is_model_test:
                model_group = properties.get("model_group", None)

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
