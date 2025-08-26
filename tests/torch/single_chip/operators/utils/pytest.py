# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# pytest utilities

from .datatypes import DataFormat, MathFidelity
import pytest
import re

import _pytest
import _pytest.reports

from _pytest.mark import ParameterSet
from typing import List, Tuple


class PyTestUtils:
    @classmethod
    def get_xfail_reason(cls, item: _pytest.python.Function) -> str:
        """Get xfail reason from pytest item

        Args:
            item (_pytest.python.Function): Pytest item

        Returns:
            str: Xfail reason
        """
        xfail_marker = item.get_closest_marker("xfail")

        if xfail_marker:
            xfail_reason = xfail_marker.kwargs.get("reason", "No reason provided")
            return xfail_reason

        return None

    @classmethod
    def remove_colors(cls, text: str) -> str:
        # Remove colors from text
        text = re.sub(r"#x1B\[\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[1A", "", text)
        text = re.sub(r"\[1B", "", text)
        text = re.sub(r"\[2K", "", text)

        return text


class PytestParamsUtils:
    def __init__(self):
        self.test_list_fields = ""
        self.test_list = []

    @staticmethod
    def get_basic_shapes(*shape_dim: int, microbatch: bool = None) -> List[Tuple]:
        """
        Get basic shapes for single operator testing.

        This method generates a list of basic shapes (tuples of integers) that can be used for testing single operators.
        The shapes are generated based on the provided dimensions and an optional microbatch flag.

        Args:
            shape_dim (int): Variable length argument list of integers representing the dimensions for the shapes.
            microbatch (bool, optional): A flag indicating whether to include microbatch shapes. Defaults to None.

        Returns:
            List[Tuple[int, ...]]: A list of tuples, where each tuple represents a shape with the specified dimensions.
        """

        # fmt: off
         # 2-dimensional shape, microbatch_size = 1:
        dim2_no_microbatch = [
            (1, 4),                 #00      # 3.1 Full tensor (i.e. full expected shape)
            (1, 17),                #01      # 3.1 Full tensor (i.e. full expected shape)
            (1, 23),                #02      # 3.2 Tensor reduce on one or more dims to 1
            (1, 1),                 #03      # 3.2 Tensor reduce on one or more dims to 1
            (1, 100),               #04      # 4.3 Very large (thousands, 10s of thousands)
            (1, 500),               #05      # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000),              #06      # 4.4 Extreme ratios between height/width
            (1, 1920),              #07      # 4.4 Extreme ratios between height/width
            (1, 10000),             #08      # 4.4 Extreme ratios between height/width
            (1, 64),                #09      # 4.1 Divisible by 32
            (1, 96),                #10      # 4.1 Divisible by 32
            (1, 41),                #11      # 4.2 Prime numbers
            (1, 3),                 #12      # 4.2 Prime numbers
        ]

        # 2-dimensional shape, microbatch_size > 1:
        dim2_microbatch = [
            (3, 4),                 #13      # 3.1 Full tensor (i.e. full expected shape)
            (45, 17),               #14      # 3.1 Full tensor (i.e. full expected shape)
            (64, 1),                #15      # 3.2 Tensor reduce on one or more dims to 1
            (100, 100),             #16      # 4.3 Very large (thousands, 10s of thousands)
            (1000, 100),            #17      # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000),             #18      # 4.4 Extreme ratios between height/width
            (9920, 1),              #19      # 4.4 Extreme ratios between height/width
            (10000, 1),             #20      # 4.4 Extreme ratios between height/width
            (32, 64),               #21      # 4.1 Divisible by 32
            (160, 96),              #22      # 4.1 Divisible by 32
            (17, 41),               #23      # 4.2 Prime numbers
            (89, 3),                #24      # 4.2 Prime numbers
        ]

        # 3-dimensional shape, microbatch_size = 1:
        dim3_no_microbatch = [
            (1, 3, 4),              #25     # 3.1 Full tensor (i.e. full expected shape)
            (1, 45, 17),            #26     # 3.1 Full tensor (i.e. full expected shape)
            (1, 1, 23),             #27     # 3.2 Tensor reduce on one or more dims to 1
            (1, 64, 1),             #28     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100),          #29     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1000, 100),         #30     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000),          #31     # 4.4 Extreme ratios between height/width
            (1, 9920, 1),           #32     # 4.4 Extreme ratios between height/width
            (1, 10000, 1),          #33     # 4.4 Extreme ratios between height/width
            (1, 32, 64),            #34     # 4.1 Divisible by 32
            (1, 160, 96),           #35     # 4.1 Divisible by 32
            (1, 17, 41),            #36     # 4.2 Prime numbers
            (1, 89, 3),             #37     # 4.2 Prime numbers
        ]

        # 3-dimensional shape, microbatch_size > 1:
        dim3_microbatch = [
            (2, 3, 4),              #38     # 3.1 Full tensor (i.e. full expected shape)
            (11, 45, 17),           #39     # 3.1 Full tensor (i.e. full expected shape)
            (11, 1, 23),            #40     # 3.2 Tensor reduce on one or more dims to 1
            (11, 64, 1),            #41     # 3.2 Tensor reduce on one or more dims to 1
            (100, 100, 100),        #42     # 4.3 Very large (thousands, 10s of thousands)
            (10, 1000, 100),        #43     # 4.3 Very large (thousands, 10s of thousands)
            (10, 10000, 1),         #44     # 4.4 Extreme ratios between height/width
            (32, 32, 64),           #45     # 4.1 Divisible by 32
            (64, 160, 96),          #46     # 4.1 Divisible by 32
            (11, 17, 41),           #47     # 4.2 Prime numbers
            (13, 89, 3),            #48     # 4.2 Prime numbers
        ]

        # 4-dimensional shape, microbatch_size = 1:
        dim4_no_microbatch = [
            (1, 2, 3, 4),           #49     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 45, 17),        #50     # 3.1 Full tensor (i.e. full expected shape)
            (1, 11, 1, 23),         #51     # 3.2 Tensor reduce on one or more dims to 1
            (1, 11, 64, 1),         #52     # 3.2 Tensor reduce on one or more dims to 1
            (1, 100, 100, 100),     #53     # 4.3 Very large (thousands, 10s of thousands)
            (1, 10, 1000, 100),     #54     # 4.3 Very large (thousands, 10s of thousands)
            (1, 1, 10, 1000),       #55     # 4.4 Extreme ratios between height/width
            (1, 1, 9920, 1),        #56     # 4.4 Extreme ratios between height/width
            (1, 10, 10000, 1),      #57     # 4.4 Extreme ratios between height/width
            (1, 32, 32, 64),        #58     # 4.1 Divisible by 32
            (1, 64, 160, 96),       #59     # 4.1 Divisible by 32
            (1, 11, 17, 41),        #60     # 4.2 Prime numbers
            (1, 13, 89, 3),         #61     # 4.2 Prime numbers
        ]

        # 4-dimensional shape, microbatch_size > 1:
        dim4_microbatch = [
            (3, 11, 45, 17),        #62     # 3.1 Full tensor (i.e. full expected shape)
            (2, 2, 3, 4),           #63     # 3.1 Full tensor (i.e. full expected shape)
            (4, 11, 1, 23),         #64     # 3.2 Tensor reduce on one or more dims to 1
            (5, 11, 64, 1),         #65     # 3.2 Tensor reduce on one or more dims to 1
            (6, 100, 100, 100),     #66     # 4.3 Very large (thousands, 10s of thousands)
            (7, 10, 1000, 100),     #67     # 4.3 Very large (thousands, 10s of thousands)
            (8, 1, 10, 1000),       #68     # 4.4 Extreme ratios between height/width
            (9, 1, 9920, 1),        #69     # 4.4 Extreme ratios between height/width
            (10, 10, 10000, 1),     #70     # 4.4 Extreme ratios between height/width
            (11, 32, 32, 64),       #71     # 4.1 Divisible by 32
            (12, 64, 160, 96),      #72     # 4.1 Divisible by 32
            (13, 11, 17, 41),       #73     # 4.2 Prime numbers
            (14, 13, 89, 3),        #74     # 4.2 Prime numbers
        ]
        # fmt: on

        result = []

        if 2 in shape_dim:
            if microbatch is False:
                result = [*result, *dim2_no_microbatch]
            elif microbatch is True:
                result = [*result, *dim2_microbatch]
            else:
                result = [*result, *dim2_no_microbatch, *dim2_microbatch]
        if 3 in shape_dim:
            if microbatch is False:
                result = [*result, *dim3_no_microbatch]
            elif microbatch is True:
                result = [*result, *dim3_microbatch]
            else:
                result = [*result, *dim3_no_microbatch, *dim3_microbatch]
        if 4 in shape_dim:
            if microbatch is False:
                result = [*result, *dim4_no_microbatch]
            elif microbatch is True:
                result = [*result, *dim4_microbatch]
            else:
                result = [*result, *dim4_no_microbatch, *dim4_microbatch]

        return result

    @staticmethod
    def get_shape_params(*shape_dim: int, microbatch: bool = None, id_name: str = "shape") -> List[pytest.param]:
        """
        Generate pytest parameters for given shape dimensions.
        Args:
            shape_dim (int): Variable length argument list of shape dimensions.
            microbatch (bool, optional): A flag indicating whether to include microbatch shapes. Defaults to None.
        Returns:
            List[pytest.param]: A list of pytest parameters generated from the given shape dimensions.
        """

        return PytestParamsUtils.create_pytest_params(
            PytestParamsUtils.get_basic_shapes(*shape_dim, microbatch=microbatch), id_name=id_name
        )

    @staticmethod
    def create_pytest_params(input_list: list, id_name, mark=()) -> List[pytest.param]:
        """
        Create a list of pytest parameters from the input list.

        Args:
            input_list (list): The list of items to convert into pytest parameters.
            id_name (str): The name to use for the id of each pytest parameter.
            mark (tuple, optional): Marks to apply to each pytest parameter. Defaults to ().

        Returns:
            list[pytest.param]: A list of pytest parameters.
        """

        params = []
        for item in input_list:
            if isinstance(item, type):
                id = item.__name__
            elif isinstance(item, DataFormat) or isinstance(item, MathFidelity):
                id = item.name
            else:
                id = str(item)
            params.append(pytest.param(item, id=f"{id_name}={id}", marks=mark))
        return params

    @staticmethod
    def join_two_params_lists(input_list_1: list[pytest.param], input_list_2: list[pytest.param]) -> List[pytest.param]:
        """
        Joins two lists of pytest parameters into a single list.
        This function takes two lists of pytest parameters and combines each element
        from the first list with each element from the second list. The resulting list
        contains pytest parameters with combined values and marks.
        Args:
            input_list_1 (list[pytest.param]): The first list of pytest parameters.
            input_list_2 (list[pytest.param]): The second list of pytest parameters.
        Returns:
            list[pytest.param]: A list of pytest parameters with combined values and marks.
        """

        result_list = []
        for item_1 in input_list_1:
            for item_2 in input_list_2:
                marks = [*item_1.marks, *item_2.marks]
                result_list.append(
                    pytest.param(*item_1.values, *item_2.values, marks=marks, id=f"{item_1.id}-{item_2.id}")
                )
        return result_list

    @staticmethod
    def join_multiple_params_list(*input_list: list[pytest.param]):
        """
        Joins multiple lists of pytest.param objects into a single list.
        All params from input lists will be joined together.
        Args:
            *input_list (list[pytest.param]): Variable number of lists containing pytest.param objects.
        Returns:
            list[pytest.param]: A single list containing all the pytest.param objects from the input lists.
        """

        # join all input lists:
        result_list = input_list[0]
        for i in range(1, len(input_list)):
            if isinstance(input_list[i], list):
                result_list = PytestParamsUtils.join_two_params_lists(result_list, input_list[i])
            else:
                result_list = PytestParamsUtils.join_two_params_lists(result_list, [input_list[i]])
        return result_list

    def set_list_fields(self, fields: str):
        """
        Sets the list fields for the test object.
        Args:
            fields (str): A string representing the fields to be set.
        Returns:
            self: Returns the instance of the object to allow method chaining.
        """

        self.test_list_fields = fields
        return self

    def generate_test_params_list(self, *input_list: list[pytest.param]):
        """
        Generates a list of test parameters by joining multiple lists of pytest parameters.
        It also sets the test_list_fields attribute to a string containing the names of the fields in the test list.
        Args:
            *input_list (list[pytest.param]): Variable number of lists containing pytest parameters.
        Returns:
            Self: The instance of the class.
        """
        # set test_list_fields:
        for ll in input_list:
            # Extract the id based on the type of ll
            id = ll[0].id if isinstance(ll, list) else ll.id

            # Replace '=' with '-' in the id
            id = id.replace("=", "-")

            # Append the first part of the id to test_list_fields
            self.test_list_fields += f"{id.split('-')[0]}, "

        # join all input lists:
        result_list = input_list[0]
        for i in range(1, len(input_list)):
            if isinstance(input_list[i], list):
                result_list = PytestParamsUtils.join_two_params_lists(result_list, input_list[i])
            else:
                result_list = PytestParamsUtils.join_two_params_lists(result_list, [input_list[i]])
        self.test_list.extend(result_list)
        return self

    @staticmethod
    def __contains_subtuple_with_wildcards_and_options(tuple1, tuple2, wildcard=None) -> bool:
        """
        Check if `tuple2` is a subtuple of `tuple1` with support for wildcards and options.
        Args:
            tuple1 (tuple): The main tuple to search within.
            tuple2 (tuple): The subtuple to search for.
            wildcard (optional): A value that acts as a wildcard, matching any element in `tuple1`.
        Returns:
            bool: True if `tuple2` is a subtuple of `tuple1` considering wildcards and options, False otherwise.
        Notes:
            - If `tuple2` is longer than `tuple1`, the function returns False immediately.
            - Elements in `tuple2` can be lists, tuples, or sets of possible values.
            - If an element in `tuple2` is a list, tuple, or set containing the wildcard, it will match any corresponding element in `tuple1`.
        """

        # Check if tuple2 is longer than tuple1, if so, return False immediately
        if len(tuple2) > len(tuple1):
            return False

        # Iterate over tuple1 with a sliding window of size len(tuple2)
        for i in range(len(tuple1) - len(tuple2) + 1):
            match = True
            for j in range(len(tuple2)):
                elem2 = tuple2[j]
                elem1 = tuple1[i + j]

                # Check if it's a wildcard
                if elem2 is wildcard:
                    continue  # Skip comparison

                # Check if elem2 is a list/tuple/set of possible values
                if isinstance(elem2, (list, tuple, set)):
                    if wildcard in elem2:
                        if not (
                            elem1 in elem2
                            or elem1 == elem2
                            or PytestParamsUtils.__contains_subtuple_with_wildcards_and_options(elem1, elem2)
                        ):
                            match = False
                            break
                        elif len(elem1) != len(elem2):
                            match = False
                            break
                    elif not (elem1 in elem2 or elem1 == elem2):
                        match = False
                        break

                # Check for direct comparison
                elif elem1 != elem2:
                    match = False
                    break

            if match:
                return True
        return False

    def extend_shape_params_with_marks(self, *shapes_to_extend: Tuple[Tuple, str]):
        """
        Extends the specified parameter set with specified marks in self.test_list.
        This method iterates over the provided param sets and their corresponding marks,
        and for each param set, if found in self.test_list (considering wildcards) it updates
        the marks for that parameter.
        \nIf None is provided as a mark, the existing marks are removed.
        Args:
            *shapes_to_extend (Tuple[Tuple, str]): A variable number of tuples where each
                tuple contains a params (as a tuple) and a mark (as a string).
        Returns:
            Self: The instance of the class.
        """

        parameters = self.test_list
        for shape, mark in shapes_to_extend:
            for param in parameters:
                if PytestParamsUtils.__contains_subtuple_with_wildcards_and_options(param.values, shape):
                    index = parameters.index(param)
                    current_marks = list(param.marks)
                    if mark is None:
                        current_marks = []
                    else:
                        current_marks.append(mark)
                    parameters.remove(param)
                    values = param.values
                    param_to_insert = pytest.param(*values, marks=current_marks, id=param.id)
                    parameters.insert(index, param_to_insert)
        return self

    def add_test_params(self, *input_list: list[pytest.param]):
        """
        Adds the provided list of pytest parameters to the test list.
        Args:
            *input_list (list[pytest.param]): Variable number of lists containing pytest parameters.
        Returns:
            Self: The instance of the class.
        """

        self.test_list.extend(input_list)
        return self

    @staticmethod
    def get_default_df_param(id_name: str = "dev_data_format") -> pytest.param:
        """
        Returns a pytest parameter with a default data format.
        This function creates and returns a pytest parameter with the default
        data format set to `DataFormat.Float16_b`. The parameter is
        identified with the id "dev-data-format=Float16_b".
        Returns:
            pytest.param: A pytest parameter with the default data format.
        """

        return pytest.param(DataFormat.Float16_b, id=f"{id_name}=Float16_b")

    @staticmethod
    def get_default_mf_param(id_name: str = "math_fidelity") -> pytest.param:
        """
        Returns a pytest parameter with a default MathFidelity value.
        This function creates and returns a pytest parameter with the
        MathFidelity value set to HiFi4. The parameter is also given an
        identifier "math-fidelity=HiFi4" for easier identification in test
        outputs.
        Returns:
            pytest.param: A pytest parameter with MathFidelity.HiFi4 and
                          an identifier "math-fidelity=HiFi4".
        """

        return pytest.param(MathFidelity.HiFi4, id=f"{id_name}=HiFi4")

    def add_mf_test_params_list(self, *input_list: list[pytest.param]):
        """
        Adds multiple pytest parameter lists to the test list.
        This method extends the current test list by joining multiple pytest parameter lists
        provided as input along with default parameters for data format and math fidelity.
        Args:
            *input_list (list[pytest.param]): Variable length argument list of pytest parameter lists.
        Returns:
            self: The instance of the class to allow method chaining.
        """

        self.test_list.extend(
            PytestParamsUtils.join_multiple_params_list(
                *input_list,
                PytestParamsUtils.get_default_df_param(id_name="dev_data_format"),
                PytestParamsUtils.create_pytest_params(
                    PytestParamsUtils.get_forge_math_fidelities(), id_name="math_fidelity"
                ),
            )
        )
        return self

    def add_df_test_params_list(self, *input_list: list[pytest.param]):
        """
        Adds a list of pytest parameters to the test list, excluding the Float16_b data format.
        This method extends the test list with a combination of input parameters and additional
        parameters generated for different data formats and math fidelity.
        Args:
            *input_list (list[pytest.param]): Variable length list of pytest parameters to be added.
        Returns:
            self: The instance of the class with the updated test list.
        """

        forge_data_formats_to_test = PytestParamsUtils.get_forge_data_formats()
        forge_data_formats_to_test.remove(
            DataFormat.Float16_b
        )  # remove duplicate - this will be added via add_mf_test_params_list
        self.test_list.extend(
            PytestParamsUtils.join_multiple_params_list(
                *input_list,
                PytestParamsUtils.create_pytest_params(forge_data_formats_to_test, id_name="dev_data_format"),
                PytestParamsUtils.get_default_mf_param(id_name="math_fidelity"),
            )
        )
        return self

    @staticmethod
    def get_forge_data_formats():
        """
        Returns a list of available data formats in the forge module.
        The data formats include various bit floating point formats,
        integer formats, and raw unsigned integer formats.
        Returns:
            list: A list of DataFormat enums representing different data formats.
        """

        return [
            DataFormat.Bfp2,
            DataFormat.Bfp2_b,
            DataFormat.Bfp4,
            DataFormat.Bfp4_b,
            DataFormat.Bfp8,
            DataFormat.Bfp8_b,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int8,
            DataFormat.Lf8,
            DataFormat.RawUInt16,
            DataFormat.RawUInt32,
            DataFormat.RawUInt8,
            DataFormat.UInt16,
        ]

    @staticmethod
    def get_forge_math_fidelities():
        """
        Retrieve a list of different math fidelity levels from the forge module.
        Returns:
            list: A list containing various math fidelity levels.
        """

        return [
            MathFidelity.LoFi,
            MathFidelity.HiFi2,
            MathFidelity.HiFi3,
            MathFidelity.HiFi4,
        ]

    @classmethod
    def strip_param_set(cls, value):
        if isinstance(value, ParameterSet):
            value = value[0][0]
        return value

    @classmethod
    def strip_param_sets(cls, values):
        return [cls.strip_param_set(value) for value in values]
