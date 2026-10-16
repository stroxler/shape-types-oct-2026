# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any, overload

from shape_extensions import Int, IntVar

def next_power_of_2(value: int) -> int: ...
def cdiv[Length: IntVar, Block: IntVar](
    value: Int[Length], block: Int[Block]
) -> int: ...
@overload
def jit[F: Callable[..., Any]](fn: F) -> F: ...
@overload
def jit[F: Callable[..., Any]](
    *, launch_metadata: Callable[..., object]
) -> Callable[[F], F]: ...
def autotune[F: Callable[..., Any]](
    *,
    configs: list[object],
    key: list[str],
    prune_configs_by: dict[str, object] | None = None,
) -> Callable[[F], F]: ...
