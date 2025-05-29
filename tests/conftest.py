# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from .utils import ModelInfo


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
            - `model_info`: utils.ModelInfo
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


def pytest_collection_modifyitems(items):
    """
    Pytest hook to process the custom marker and attach recorder properties to the test.
    """

    def validate_keys(keys: dict, tagged_as_model_test: bool):
        valid_keys = [
            "category",
            "jax_op_name",
            "shlo_op_name",
            "model_info",
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

        # If model test, check all necessary properties are provided.
        if tagged_as_model_test:
            mandatory_model_properties = ["model_info", "run_mode", "bringup_status"]

            if not all(
                model_property in keys for model_property in mandatory_model_properties
            ):
                raise KeyError(
                    f"Model tests must have following properties: "
                    f"{mandatory_model_properties}."
                )

    for item in items:
        # Add some test metadata in a 'tags' dictionary.
        tags = {"test_name": item.originalname, "specific_test_case": item.name}

        # Look for the custom marker.
        properties_marker = item.get_closest_marker(name="record_test_properties")

        # Utils flags helping handling model tests properly.
        tagged_as_model_test = False

        if properties_marker:
            # Extract the key-value pairs passed to the marker.
            properties: dict = properties_marker.kwargs

            # Check if the test is marked using the "model_test" marker.
            tagged_as_model_test = (
                item.get_closest_marker(name="model_test") is not None
            )
            # Validate that only allowed keys are used.
            validate_keys(properties.keys(), tagged_as_model_test)

            # Put all properties in tags.
            for key, value in properties.items():
                if key == "model_info":
                    model_info: ModelInfo = value
                    tags["model_name"] = model_info.name
                    tags["model_info"] = model_info.to_report_dict()

                    if tagged_as_model_test:
                        tags["group"] = str(model_info.group)
                        # Add model group independently of tags dict also.
                        item.user_properties.append(("group", str(model_info.group)))
                else:
                    tags[key] = str(value)

        # Attach tags dictionary as a single property. Also set owner.
        item.user_properties.extend([("tags", tags), ("owner", "tt-xla")])
