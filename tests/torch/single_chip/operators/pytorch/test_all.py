# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# TestSuite and query configuration for all pytorch operators


import os
from ..utils.datatypes import DataFormat
from ..utils.datatypes import MathFidelity
import textwrap
import json

from loguru import logger
from tabulate import tabulate
from typing import List, Tuple, Optional, Generator, Callable
from dataclasses import dataclass

from ..utils import InputSource
from ..utils import TestVector
from ..utils import TestCollection
from ..utils import TestCollectionCommon
from ..utils import TestCollectionTorch
from ..utils import TestQuery
from ..utils import TestSuite
from ..utils import TestPlanScanner
from ..utils import TestPlanUtils
from ..utils import FailingReasons
from ..utils import TestSweepsFeatures


@dataclass
class RunQueryParams:
    """Run query parameters"""

    test_id: Optional[str]
    id_files: Optional[List[str]]
    id_files_ignore: Optional[List[str]]
    operators: Optional[List[str]]
    exclude_operators: bool
    filters: Optional[List[str]]
    input_sources: Optional[List[InputSource]]
    input_shapes: Optional[List[Tuple[int, ...]]]
    dev_data_formats: Optional[List[DataFormat]]
    math_fidelities: Optional[List[MathFidelity]]
    kwargs: Optional[List[dict]]
    failing_reasons: Optional[List[FailingReasons]]
    skip_reasons: Optional[List[FailingReasons]]
    random_seed: int
    sample: Optional[float]
    range: Optional[tuple[int, int]]

    def get_operators(self) -> Optional[List[str]]:
        """Get operators list if exclude_operators is False, otherwise return None"""
        return self.operators if not self.exclude_operators else None

    def get_exclude_operators(self) -> Optional[List[str]]:
        """Get exclude_operators list if exclude_operators is True, otherwise return None"""
        return self.operators if self.exclude_operators else None

    @classmethod
    def _get_env_list(cls, name: str, default: str = "") -> Optional[List[str]]:
        values = os.getenv(name, default).strip().split(",")
        values = [value for value in values if value]
        if len(values) == 0:
            values = None
        return values

    @classmethod
    def from_env(cls):
        test_id = os.getenv("TEST_ID", None)
        id_files = cls._get_env_list("ID_FILES")
        id_files_ignore = cls._get_env_list("ID_FILES_IGNORE", "ids/blacklist.txt")
        operators = cls._get_env_list("OPERATORS")
        if operators and operators[0].startswith("-"):
            exclude_operators = True
            # Remove exclude - mark from the first operator
            operators[0] = operators[0][1:]
        else:
            exclude_operators = False
        filters = cls._get_env_list("FILTERS")
        input_sources = cls._get_env_list("INPUT_SOURCES")
        if input_sources:
            input_sources = [getattr(InputSource, input_source) for input_source in input_sources]
        input_shapes = os.getenv("INPUT_SHAPES", None)
        if input_shapes:
            input_shapes = eval(input_shapes)
        dev_data_formats = cls._get_env_list("DEV_DATA_FORMATS")
        if dev_data_formats:
            dev_data_formats = [
                TestPlanUtils.dev_data_format_from_str(dev_data_format) for dev_data_format in dev_data_formats
            ]
        math_fidelities = cls._get_env_list("MATH_FIDELITIES")
        if math_fidelities:
            math_fidelities = [TestPlanUtils.math_fidelity_from_str(math_fidelity) for math_fidelity in math_fidelities]
        kwargs = os.getenv("KWARGS", None)
        if kwargs:
            kwargs = eval(kwargs)
        failing_reasons = cls._get_env_list("FAILING_REASONS")
        if failing_reasons:
            failing_reasons = [getattr(FailingReasons, failing_reason) for failing_reason in failing_reasons]
        skip_reasons = cls._get_env_list("SKIP_REASONS")
        if skip_reasons:
            skip_reasons = [getattr(FailingReasons, skip_reason) for skip_reason in skip_reasons]
        random_seed = int(os.getenv("RANDOM_SEED", 0))
        sample = os.getenv("SAMPLE", None)
        sample = float(sample) if sample else None
        range = cls._get_env_list("RANGE")
        if range:
            range = [int(i) for i in range]
            if len(range) == 1:
                range = 0, range[0]
            else:
                range = range[0], range[1]

        # Construct query parameters
        query_params = cls(
            test_id=test_id,
            id_files=id_files,
            id_files_ignore=id_files_ignore,
            operators=operators,
            exclude_operators=exclude_operators,
            filters=filters,
            input_sources=input_sources,
            input_shapes=input_shapes,
            dev_data_formats=dev_data_formats,
            math_fidelities=math_fidelities,
            kwargs=kwargs,
            failing_reasons=failing_reasons,
            skip_reasons=skip_reasons,
            random_seed=random_seed,
            sample=sample,
            range=range,
        )

        logger.info(f"Query parameters: {query_params}")

        if query_params.test_id and query_params.id_files:
            raise ValueError("TEST_ID and ID_FILES cannot be used together")

        return query_params


query_params = RunQueryParams.from_env()


class TestVerification:
    """Helper class for performing test verification. It allows running tests in dry-run mode."""

    @classmethod
    def verify(cls, test_vector: TestVector, test_device):
        if TestSweepsFeatures.params.dry_run:
            # pytest.skip("Dry run")
            return
        test_vector.verify(test_device)


class TestParamsData:
    """
    Helper class for providing data for test parameters.
    This is a parameter manager that collects quering criterias from environment variables to determine the tests that should run. It can helper filtering test collection and filtering lambdas.
    """

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    base_dir = os.path.dirname(os.path.abspath(__file__))


class TestCollectionData:
    """Helper test collections"""

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    # Test collections for query criterias from environment variables
    filtered = TestCollection(
        operators=query_params.get_operators(),
        input_sources=query_params.input_sources,
        input_shapes=query_params.input_shapes,
        dev_data_formats=query_params.dev_data_formats,
        math_fidelities=query_params.math_fidelities,
        kwargs=query_params.kwargs,
    )

    excluded = TestCollection(
        operators=query_params.get_exclude_operators(),
    )

    # All available test vectors
    all = TestCollection(
        # operators=None,  # All available operators
        operators=filtered.operators,  # Operators selected by filter
    )

    # Quick test collection for faster testing consisting of a subset of input shapes and data formats
    quick = TestCollection(
        # 2 examples for each dimension and microbatch size
        input_shapes=[]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (2,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (2,) and shape[0] != 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (3,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (3,) and shape[0] != 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (4,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (4,) and shape[0] != 1][:2],
        # one example for float and int data formats
        dev_data_formats=[
            None,
            DataFormat.Float16_b,
            DataFormat.Int8,
        ],
    )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    @staticmethod
    def _load_test_ids_from_files(id_files: List[str]) -> Generator[str, None, None]:
        return TestPlanUtils.load_test_ids_from_files(TestParamsData.base_dir, id_files) if id_files else None

    test_ids_list = _load_test_ids_from_files(query_params.id_files)

    single_ids_list = [query_params.test_id] if query_params.test_id else None

    ignore_list = (
        list(_load_test_ids_from_files(query_params.id_files_ignore)) if query_params.id_files_ignore else None
    )


class TestSuiteData:

    __test__ = False  # Avoid collecting TestSuiteData as a pytest test

    # Explicitly ignore test plans
    ignore_test_plan_paths = [
        # "test.operators.pytorch.test_",
    ]

    @staticmethod
    def _filter_by_operators(test_suite: TestSuite, operators: List[str]) -> TestSuite:
        """Filter test plans based on operator list to speed up test filtering"""
        if operators is None:
            return test_suite
        else:
            test_plans = [
                test_plan
                for test_plan in test_suite.test_plans
                if len(list(set(test_plan.operators) & set(operators))) > 0
            ]
            return TestSuite(test_plans)

    all = TestPlanScanner.build_test_suite(
        scan_file=__file__, scan_package=__package__, ignore_test_plan_paths=ignore_test_plan_paths
    )

    filtered = _filter_by_operators(all, TestCollectionData.all.operators)


class QueryMethods:
    @classmethod
    def _remove_failing_result(cls, query: TestQuery) -> Generator[TestVector, None, None]:
        for test_vector in query.test_vectors:
            test_vector.failing_result = None
            yield test_vector

    @classmethod
    def _remove_xfail_reason(cls, query: TestQuery) -> Generator[TestVector, None, None]:
        for test_vector in query.test_vectors:
            if test_vector.failing_result is not None:
                test_vector.failing_result.failing_reason = None
            yield test_vector

    @classmethod
    def remove_failing_result(cls, query: TestQuery) -> TestQuery:
        return TestQuery(cls._remove_failing_result(query))

    @classmethod
    def remove_xfail_reason(cls, query: TestQuery) -> TestQuery:
        return TestQuery(cls._remove_xfail_reason(query))


class QueryLambdas:

    SAMPLE = lambda query: query.sample(query_params.sample, query_params.random_seed)

    RANGE = lambda query: query.range(*query_params.range)

    UNIQUE_KWARGS = lambda query: query.group_limit(["operator", "kwargs"], 1)

    CALCULATE_FAILING_RESULT = lambda query: query.calculate_failing_result()
    REMOVE_FAILING_RESULT = lambda query: QueryMethods.remove_failing_result(query)
    REMOVE_XFAIL_REASON = lambda query: QueryMethods.remove_xfail_reason(query)

    LOG = lambda query: query.log()

    SAMPLE5 = lambda query: query.sample(5, query_params.random_seed)
    SAMPLE10 = lambda query: query.sample(10, query_params.random_seed)
    SAMPLE20 = lambda query: query.sample(20, query_params.random_seed)


class VectorLambdas:
    """Helper lambdas for filtering test vectors"""

    QUICK = lambda test_vector: test_vector in TestCollectionData.quick
    FILTERED = lambda test_vector: test_vector in TestCollectionData.filtered
    EXCLUDED = lambda test_vector: test_vector not in TestCollectionData.excluded

    SKIP_IGNORE_LIST = lambda test_vector: test_vector.get_id() not in TestIdsData.ignore_list

    SINGLE_SHAPE = lambda test_vector: test_vector.input_shape in TestCollectionCommon.single.input_shapes

    SHAPES_2D = lambda test_vector: len(test_vector.input_shape) == 2
    SHAPES_3D = lambda test_vector: len(test_vector.input_shape) == 3
    SHAPES_4D = lambda test_vector: len(test_vector.input_shape) == 4

    MICROBATCH_SIZE_ONE = lambda test_vector: test_vector.input_shape[0] == 1
    MICROBATCH_SIZE_MULTI = lambda test_vector: test_vector.input_shape[0] > 1

    FAILING = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason is not None
    )
    SKIPED = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.skip_reason is not None
    )
    SKIPED_FATAL = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.skip_reason == FailingReasons.FATAL_ERROR
    )
    NOT_FAILING = (
        lambda test_vector: test_vector.failing_result is None or test_vector.failing_result.failing_reason is None
    )
    NOT_SKIPED = (
        lambda test_vector: test_vector.failing_result is None or test_vector.failing_result.skip_reason is None
    )

    HAS_DATA_FORMAT = lambda test_vector: test_vector.dev_data_format is not None
    NO_DATA_FORMAT = lambda test_vector: test_vector.dev_data_format is None

    NOT_IMPLEMENTED = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason
        in (
            FailingReasons.NOT_IMPLEMENTED_ATEN,
            FailingReasons.LOWERING_UNSUPPORTED_OPERATION,
            FailingReasons.UNSUPPORTED_OP_TYPES,
        )
    )
    DATA_MISMATCH = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason == FailingReasons.DATA_MISMATCH
    )
    UNSUPPORTED_DATA_FORMAT = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason == FailingReasons.UNSUPPORTED_DATA_FORMAT
    )

    FAILING_REASONS = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason in query_params.failing_reasons
    )
    SKIP_REASONS = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.skip_reason in query_params.skip_reasons
    )


class TestQueries:

    __test__ = False  # Avoid collecting TestQueries as a pytest test

    @classmethod
    def _filter_tests_ids_by_operators(cls, test_ids: List[str]) -> List[str]:
        """Keep only test ids that contain any of the operators from filtered collection"""
        operators = TestCollectionData.filtered.operators
        if operators is None:
            return test_ids
        operators = [f"{operator}-" for operator in operators]
        test_ids = [test_id for test_id in test_ids if any([operator in test_id for operator in operators])]
        return test_ids

    @classmethod
    def _build_filter_lambdas(cls, filters: List[str]) -> Generator[Callable[[TestQuery], TestQuery], None, None]:
        """
        Builds a list of lambdas for filtering test vectors based on environment variables.
        The lambdas are built based on the following environment variables:
        - FILTERS: List of lambdas defined in VectorLambdas to filter
        - FAILING_REASONS: List of failing reasons to filter
        - SKIP_REASONS: List of skip reasons to filter
        """
        for filter in filters:
            if hasattr(VectorLambdas, filter):
                vector_filter = getattr(VectorLambdas, filter)
                yield lambda test_query: test_query.filter(vector_filter)
            elif hasattr(QueryLambdas, filter):
                query_filter = getattr(QueryLambdas, filter)
                yield lambda test_query: query_filter(test_query)
            else:
                raise ValueError(f"Filter {filter} not found in VectorLambdas or QueryLambdas")

    @classmethod
    def is_empty_collection(cls, collection: TestCollection) -> bool:
        """Check if the collection is empty"""
        logger.trace(f"Checking if collection is empty: {collection}")
        if collection.operators:
            return False
        if collection.input_sources:
            return False
        if collection.input_shapes:
            return False
        if collection.dev_data_formats:
            return False
        if collection.math_fidelities:
            return False
        if collection.kwargs:
            return False
        return True

    @classmethod
    def _get_filters_names(cls) -> Generator[str, None, None]:
        if not cls.is_empty_collection(TestCollectionData.filtered):
            yield "FILTERED"

        if not cls.is_empty_collection(TestCollectionData.excluded):
            yield "EXCLUDED"

        yield "CALCULATE_FAILING_RESULT"

        if query_params.failing_reasons:
            yield "FAILING_REASONS"

        if query_params.skip_reasons:
            yield "SKIP_REASONS"

        # Include selected filters from VectorLambdas and QueryLambdas
        if query_params.filters:
            yield from query_params.filters

        if query_params.sample is not None:
            yield "SAMPLE"

        if query_params.id_files_ignore is not None:
            yield "SKIP_IGNORE_LIST"

        if query_params.range is not None:
            yield "RANGE"

    @classmethod
    def query_source(cls) -> TestQuery:
        if query_params.test_id and query_params.id_files:
            raise ValueError("TEST_ID and ID_FILES cannot be used together")

        test_suite = TestSuiteData.filtered

        if TestIdsData.test_ids_list:
            logger.info("Using test ids from file")
            test_ids = cls._filter_tests_ids_by_operators(TestIdsData.test_ids_list)
            test_vectors = test_suite.load_test_vectors_from_id_list(test_ids)
        elif TestIdsData.single_ids_list:
            logger.info("Using single test id")
            test_ids = cls._filter_tests_ids_by_operators(TestIdsData.single_ids_list)
            test_vectors = test_suite.load_test_vectors_from_id_list(test_ids)
        else:
            logger.info("Using all test vectors")
            test_vectors = test_suite.generate()

        query = TestQuery(test_vectors)

        return query

    @classmethod
    def query_filter(cls, query: TestQuery) -> TestQuery:
        filters = list(cls._get_filters_names())
        logger.info(f"Filters: {filters}")
        for filter_lambda in cls._build_filter_lambdas(filters):
            query = filter_lambda(query)

        return query


class InfoUtils:
    @classmethod
    def print_query_params(cls, max_width=80):
        print("Query parameters:")
        cls.print_query_values(max_width)
        print("Query examples:")
        cls.print_query_examples(max_width)
        print("Configuration parameters:")
        cls.print_configuration_params(max_width)
        print("Configuration examples:")
        cls.print_configuration_examples(max_width)

    @classmethod
    def export(cls, file_name: str):
        test_vectors = TestQueries.query_filter(TestQueries.query_source()).test_vectors
        logger.info(f"Exporting test vectors to {file_name}")

        test_vectors = [
            {
                "id": test_vector.get_id(),
                "operator": test_vector.operator,
                "input_source": test_vector.input_source.name if test_vector.input_source is not None else None,
                "input_shape": f"{test_vector.input_shape}",
                "dev_data_format": TestPlanUtils.dev_data_format_to_str(test_vector.dev_data_format),
                "math_fidelity": test_vector.math_fidelity.name if test_vector.math_fidelity is not None else None,
                "kwargs": f"{test_vector.kwargs}",
                "failing_reason": test_vector.failing_result.failing_reason.name
                if test_vector.failing_result and test_vector.failing_result.failing_reason
                else None,
                "skip_reason": test_vector.failing_result.skip_reason.name
                if test_vector.failing_result and test_vector.failing_result.skip_reason
                else None,
                "status": "skipped"
                if test_vector.failing_result and test_vector.failing_result.skip_reason
                else "xfailed"
                if test_vector.failing_result and test_vector.failing_result.failing_reason
                else "passed",
            }
            for test_vector in test_vectors
        ]

        with open(file_name, "w") as file:
            json.dump(test_vectors, file, indent=4)

    @classmethod
    def print_query_values(cls, max_width=80):

        operators = [key for key in TestSuiteData.all.indices]
        operators = sorted(operators)
        operators = ", ".join(operators)

        filters = []
        filters += [key for key, value in VectorLambdas.__dict__.items() if not key.startswith("__")]
        filters += [key for key, value in QueryLambdas.__dict__.items() if not key.startswith("__")]
        filters = [filter for filter in filters if filter]
        filters = ", ".join(filters)

        input_sources = [f"{input_source.name}" for input_source in TestCollectionCommon.all.input_sources]
        input_sources = ", ".join(input_sources)

        input_shapes = [f"{input_shape}" for input_shape in TestCollectionCommon.all.input_shapes]
        input_shapes = ", ".join(input_shapes)

        all_dev_data_formats = []
        all_dev_data_formats += TestCollectionCommon.all.dev_data_formats
        all_dev_data_formats += TestCollectionTorch.all.dev_data_formats
        dev_data_formats = [
            f"{TestPlanUtils.dev_data_format_to_str(dev_data_format)}" for dev_data_format in all_dev_data_formats
        ]
        dev_data_formats = ", ".join(dev_data_formats)

        math_fidelities = [f"{math_fidelity.name}" for math_fidelity in TestCollectionCommon.all.math_fidelities]
        math_fidelities = ", ".join(math_fidelities)

        failing_reasons = [failing_reason.name for failing_reason in FailingReasons]
        failing_reasons = ", ".join(failing_reasons)

        parameters = [
            {"name": "OPERATORS", "description": f"List of operators. Supported values: {operators}"},
            {"name": "FILTERS", "description": f"List of lambda filters. Supported values: {filters}"},
            {"name": "INPUT_SOURCES", "description": f"List of input sources. Supported values: {input_sources}"},
            {"name": "INPUT_SHAPES", "description": f"List of input shapes. Supported values: {input_shapes}"},
            {
                "name": "DEV_DATA_FORMATS",
                "description": f"List of dev data formats. Supported values: {dev_data_formats}",
            },
            {"name": "MATH_FIDELITIES", "description": f"List of math fidelities. Supported values: {math_fidelities}"},
            {
                "name": "KWARGS",
                "description": "List of kwargs dictionaries. Kwarg is a mandatory or optional attribute of an operator. See operator documentation for each operator or use filter `UNIQUE_KWARGS` to find examples.",
            },
            {"name": "FAILING_REASONS", "description": f"List of failing reasons. Supported values: {failing_reasons}"},
            {"name": "SKIP_REASONS", "description": "Same as FAILING_REASONS"},
            {"name": "RANDOM_SEED", "description": "Seed for random number generator"},
            {"name": "SAMPLE", "description": "Percentage of results to sample"},
            {"name": "RANGE", "description": "Limit number of results"},
            {"name": "TEST_ID", "description": "Id of a test containing test parameters"},
            {"name": "ID_FILES", "description": "Paths to files containing test ids instead of tests from test plan"},
            {"name": "ID_FILES_IGNORE", "description": "Paths to files containing test ids to be ignored"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Supported values"])

    @classmethod
    def print_query_examples(cls, max_width=80):

        parameters = [
            {"name": "OPERATORS", "description": "export OPERATORS=add"},
            {"name": "OPERATORS", "description": "export OPERATORS=add,div"},
            {"name": "FILTERS", "description": "export FILTERS=HAS_DATA_FORMAT,QUICK"},
            {"name": "INPUT_SOURCES", "description": "export INPUT_SOURCES=FROM_HOST,FROM_DRAM_QUEUE"},
            {"name": "INPUT_SHAPES", "description": 'export INPUT_SHAPES="[(3, 4), (45, 17)]"'},
            {"name": "DEV_DATA_FORMATS", "description": "export DEV_DATA_FORMATS=Float16_b,Int8"},
            {"name": "MATH_FIDELITIES", "description": "export MATH_FIDELITIES=HiFi4,HiFi3"},
            {
                "name": "KWARGS",
                "description": "export KWARGS=\"[{'rounding_mode': 'trunc'},{'rounding_mode': 'floor'}]\"",
            },
            {"name": "FAILING_REASONS", "description": "export FAILING_REASONS=DATA_MISMATCH,UNSUPPORTED_DATA_FORMAT"},
            {"name": "FAILING_REASONS", "description": "export FAILING_REASONS=NOT_IMPLEMENTED"},
            {"name": "SKIP_REASONS", "description": "export SKIP_REASONS=FATAL_ERROR"},
            {"name": "RANGE", "description": "export RANGE=5"},
            {"name": "RANDOM_SEED", "description": "export RANDOM_SEED=42"},
            {"name": "SAMPLE", "description": "export SAMPLE=20"},
            {"name": "RANGE", "description": "export RANGE=10,20"},
            {"name": "TEST_ID", "description": "export TEST_ID='ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4'"},
            {"name": "ID_FILES", "description": "export ID_FILES='/path/to/test_ids.log'"},
            {"name": "ID_FILES_IGNORE", "description": "export ID_FILES_IGNORE='ids/blacklist.txt,ids/ignore.txt'"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Examples"])

    @classmethod
    def print_configuration_params(cls, max_width=80):

        parameters = [
            {
                "name": "SKIP_FORGE_VERIFICATION",
                "description": f"Skip Forge model verification including model compiling and inference",
                "default": "false",
            },
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Description", "Default"])

    @classmethod
    def print_configuration_examples(cls, max_width=80):

        parameters = [
            {"name": "SKIP_FORGE_VERIFICATION", "description": "export SKIP_FORGE_VERIFICATION=true"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Examples"])

    @classmethod
    def print_formatted_parameters(cls, parameters, max_width=80, headers=["Parameter", "Description"]):
        for param in parameters:
            param["description"] = "\n".join(textwrap.wrap(param["description"], width=max_width))

        table_data = [[param["name"], param["description"]] for param in parameters]

        print(tabulate(table_data, headers, tablefmt="grid"))
