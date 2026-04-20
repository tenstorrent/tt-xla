# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for multi-host experimental tests.

NOTE: TT_DISTRIBUTED_* (and related) environment variables are set by CI / the test
harness (e.g. eval $(python scripts/multihost_topology.py ...)) before pytest runs,
not by pytest fixtures during collection.
"""
