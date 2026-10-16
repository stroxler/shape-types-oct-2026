# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only overlay stubs are not Buck Python library targets.
# @lint-ignore-every AUTODEPS2

"""Calls do not establish a host-grid dependency or a memory-ordering proof."""

def gdc_wait() -> None: ...
def gdc_launch_dependents() -> None: ...
