"""Shape expressions for logical offset grids and their address steps."""

from shape_extensions import Int, IntTuple, dsl, type_shape_dsl_function

@type_shape_dsl_function
def insert_extent(tile: IntTuple, axis: int) -> IntTuple:
    if axis < 0 or axis > len(tile):
        return dsl.Invalid("offset axis out of range")
    return dsl.concat(tile[:axis], dsl.concat(dsl.IntTuple((1,)), tile[axis:]))

@type_shape_dsl_function
def insert_step(tile: IntTuple, steps: IntTuple, axis: int) -> IntTuple:
    if len(tile) != len(steps):
        return dsl.Invalid("offset extents and address steps must have equal rank")
    if axis < 0 or axis > len(steps):
        return dsl.Invalid("offset axis out of range")
    return dsl.concat(steps[:axis], dsl.concat(dsl.IntTuple((0,)), steps[axis:]))

@type_shape_dsl_function
def bounds_target(tile: IntTuple, steps: IntTuple, bound: Int) -> IntTuple:
    if len(tile) != len(steps):
        return dsl.Invalid("offset extents and steps must have equal rank")
    # Only one unit-step logical axis establishes an allocation bound.
    if any(step != 0 and step != 1 for step in steps):
        return dsl.Invalid("expected unit-step logical indices")
    if not any(step == 1 for step in steps):
        return dsl.Invalid("expected one varying index axis")
    if any(
        steps[i] == 1 and any(steps[j] == 1 for j in range(i + 1, len(steps)))
        for i in range(len(steps))
    ):
        return dsl.Invalid("expected one varying index axis")
    return dsl.IntTuple((bound if step == 1 else 1 for step in steps))

@type_shape_dsl_function
def scale_steps(tile: IntTuple, steps: IntTuple, scale: Int) -> IntTuple:
    if len(tile) != len(steps):
        return dsl.Invalid("offset extents and address steps must have equal rank")
    # A generator retains each symbolic step and its position for known-rank tuples.
    return dsl.IntTuple((step * scale for step in steps))

@type_shape_dsl_function
def add_steps(
    left_tile: IntTuple,
    left_steps: IntTuple,
    right_tile: IntTuple,
    right_steps: IntTuple,
) -> IntTuple:
    if len(left_tile) != len(left_steps) or len(right_tile) != len(right_steps):
        return dsl.Invalid("offset extents and steps must have equal rank")
    if len(left_tile) != len(right_tile):
        return dsl.Invalid("offset grids must have equal rank")
    # Singleton axes introduced by indexing carry zero steps before broadcast.
    return dsl.IntTuple(
        (left_steps[i] + right_steps[i] for i in range(len(left_steps)))
    )
