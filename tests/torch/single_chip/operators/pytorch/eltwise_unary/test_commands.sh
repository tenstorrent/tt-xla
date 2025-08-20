# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Commands for running unary operators tests

# Run implemented unary operators tests:
OPERATORS=\
"relu",\
"sqrt",\
"reciprocal",\
"sigmoid",\
"abs",\
"cos",\
"exp",\
"neg",\
"rsqrt",\
"sin",\
"square",\
"pow",\
"clamp",\
"log",\
"log1p" \
pytest -svv forge/test/operators/pytorch/test_all.py::test_query -rap

# Run not-implemented unary operators tests:
OPERATORS=\
"acos",\
"arccos",\
"acosh",\
"arccosh",\
"angle",\
"asin",\
"arcsin",\
"asinh",\
"arcsinh",\
"atan",\
"arctan",\
"atanh",\
"arctanh",\
"bitwise_not",\
"ceil",\
"conj_physical",\
"cosh",\
"deg2rad",\
"digamma",\
"erf",\
"erfc",\
"erfinv",\
"exp2",\
"expm1",\
"fix",\
"floor",\
"frac",\
"lgamma",\
"log10",\
"log2",\
"logit",\
"i0",\
"isnan",\
"nan_to_num",\
"positive",\
"rad2deg",\
"round",\
"sign",\
"sgn",\
"signbit",\
"sinc",\
"sinh",\
"tan",\
"tanh",\
"trunc" \
pytest -svv forge/test/operators/pytorch/test_all.py::test_query -rap

# Run all unary operators tests:
OPERATORS=\
"relu",\
"sqrt",\
"reciprocal",\
"sigmoid",\
"abs",\
"cos",\
"exp",\
"neg",\
"rsqrt",\
"sin",\
"square",\
"acos",\
"arccos",\
"acosh",\
"arccosh",\
"angle",\
"asin",\
"arcsin",\
"asinh",\
"arcsinh",\
"atan",\
"arctan",\
"atanh",\
"arctanh",\
"bitwise_not",\
"ceil",\
"clamp",\
"clip",\
"conj_physical",\
"cosh",\
"deg2rad",\
"digamma",\
"erf",\
"erfc",\
"erfinv",\
"exp2",\
"expm1",\
"fix",\
"floor",\
"frac",\
"lgamma",\
"log",\
"log10",\
"log1p",\
"log2",\
"logit",\
"i0",\
"isnan",\
"nan_to_num",\
"positive",\
"pow",\
"rad2deg",\
"round",\
"sign",\
"sgn",\
"signbit",\
"sinc",\
"sinh",\
"tan",\
"tanh",\
"trunc",\
"clamp" \
pytest -svv forge/test/operators/pytorch/test_all.py::test_query -rap
