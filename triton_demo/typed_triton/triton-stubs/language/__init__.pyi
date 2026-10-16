# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Static-only semantic roles for a tile and the allocation it accesses."""

from collections.abc import Iterator
from typing import Literal, overload

from shape_extensions import (
    Int,
    IntListLiteral,
    IntTuple,
    IntTupleOrList,
    IntVar,
    broadcast,
)

from typed_triton._shapes import (
    add_steps,
    bounds_target,
    insert_extent,
    insert_step,
    scale_steps,
)
from typed_triton.tlt import InOutPointer, InPointer, OutPointer

from . import extra, math

class tensor[Tile: IntTuple]:
    dtype: object
    shape: tuple[int, ...]
    def __neg__(self) -> tensor[Tile]: ...
    @property
    def T[Rows: IntVar, Cols: IntVar](
        self: tensor[[Rows, Cols]],
    ) -> tensor[[Cols, Rows]]: ...
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> tensor[insert_extent(Tile, 1)]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> tensor[insert_extent(Tile, 0)]: ...
    @overload
    def __add__[Other: IntTuple](
        self, other: tensor[Other]
    ) -> tensor[broadcast(Tile, Other)]: ...
    @overload
    def __add__(self, other: float) -> tensor[Tile]: ...
    def __iadd__(self, other: tensor[Tile]) -> tensor[Tile]: ...
    def __rsub__(self, other: float) -> tensor[Tile]: ...
    @overload
    def __sub__(self, other: float) -> tensor[Tile]: ...
    @overload
    def __sub__[Other: IntTuple](
        self, other: tensor[Other]
    ) -> tensor[broadcast(Tile, Other)]: ...
    @overload
    def __truediv__[Other: IntTuple](
        self, other: tensor[Other]
    ) -> tensor[broadcast(Tile, Other)]: ...
    @overload
    def __truediv__(self, other: float) -> tensor[Tile]: ...
    def __rtruediv__(self, other: float) -> tensor[Tile]: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __mul__[Other: IntTuple](
        self, other: tensor[Other]
    ) -> tensor[broadcast(Tile, Other)]: ...
    @overload
    def __mul__(self, other: float) -> tensor[Tile]: ...
    def __rmul__(self, other: float) -> tensor[Tile]: ...
    def __ge__(self, other: int) -> LogicalMask[Tile]: ...
    def __gt__(self, other: float) -> LogicalMask[Tile]: ...
    def to(self, dtype: object) -> tensor[Tile]: ...
    @overload
    def reshape[Rows: IntVar, K: IntVar](
        self: tensor[[1, Rows, K, 2, 256]],
        rows: Int[Rows],
        k: Int[K],
        d32: Literal[32],
        d4: Literal[4],
        d4b: Literal[4],
    ) -> tensor[[Rows, K, 32, 4, 4]]: ...
    @overload
    def reshape[Rows: IntVar, K: IntVar](
        self: tensor[[Rows, 4, 32, K, 4]],
        rows: Int[Rows * 128],
        k: Int[K * 4],
    ) -> tensor[[Rows * 128, K * 4]]: ...
    def trans[Rows: IntVar, K: IntVar](
        self: tensor[[Rows, K, 32, 4, 4]],
        a: Literal[0],
        b: Literal[3],
        c: Literal[2],
        d: Literal[1],
        e: Literal[4],
    ) -> tensor[[Rows, 4, 32, K, 4]]: ...

class LogicalMask[Tile: IntTuple](tensor[Tile]):
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> LogicalMask[insert_extent(Tile, 1)]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> LogicalMask[insert_extent(Tile, 0)]: ...
    def __and__(self, other: LogicalMask[Tile]) -> LogicalMask[Tile]: ...
    def __or__(self, other: LogicalMask[Tile]) -> LogicalMask[Tile]: ...
    def __invert__(self) -> LogicalMask[Tile]: ...

class tensor_descriptor[
    Target: IntTuple,
    Strides: IntTuple,
    Block: IntTuple,
    Access: str,
]:
    @overload
    def load[BR: IntVar, BC: IntVar](
        self: tensor_descriptor[Target, Strides, [BR, BC], Literal["read"]],
        offsets: list[int | BlockStart[BR]],
    ) -> tensor[[BR, BC]]: ...
    @overload
    def load[BR: IntVar, BC: IntVar](
        self: tensor_descriptor[Target, Strides, [BR, BC], Literal["read_write"]],
        offsets: list[int | BlockStart[BR]],
    ) -> tensor[[BR, BC]]: ...
    @overload
    def load[BR: IntVar, BK: IntVar](
        self: tensor_descriptor[Target, Strides, [1, BR, BK, 2, 256], Literal["read"]],
        offsets: list[int | BlockStart[BR]],
    ) -> tensor[[1, BR, BK, 2, 256]]: ...
    @overload
    def load[BR: IntVar, BK: IntVar](
        self: tensor_descriptor[
            Target, Strides, [1, BR, BK, 2, 256], Literal["read_write"]
        ],
        offsets: list[int | BlockStart[BR]],
    ) -> tensor[[1, BR, BK, 2, 256]]: ...
    @overload
    def load(
        self: tensor_descriptor[Target, Strides, Block, Literal["read"]],
        offsets: list[int],
    ) -> tensor[Block]: ...
    @overload
    def load(
        self: tensor_descriptor[Target, Strides, Block, Literal["read_write"]],
        offsets: list[int],
    ) -> tensor[Block]: ...
    @overload
    def store[BR: IntVar, BC: IntVar](
        self: tensor_descriptor[Target, Strides, [BR, BC], Literal["write"]],
        offsets: list[int | BlockStart[BR] | BlockStart[BC]],
        value: tensor[[BR, BC]],
    ) -> None: ...
    @overload
    def store[BR: IntVar, BC: IntVar](
        self: tensor_descriptor[Target, Strides, [BR, BC], Literal["read_write"]],
        offsets: list[int | BlockStart[BR] | BlockStart[BC]],
        value: tensor[[BR, BC]],
    ) -> None: ...
    @overload
    def store(
        self: tensor_descriptor[Target, Strides, Block, Literal["write"]],
        offsets: list[int],
        value: tensor[Block],
    ) -> None: ...
    @overload
    def store(
        self: tensor_descriptor[Target, Strides, Block, Literal["read_write"]],
        offsets: list[int],
        value: tensor[Block],
    ) -> None: ...

# Packing is an allocation layout: the logical shape, physical byte shape,
# and logical block are all needed to check addressing and loads. Shuffled
# scales use logical row bounds even though storage has Rows // 32 rows; their
# physical bounds must still be established by the host validator.
class PackedPointer[
    Logical: IntTuple,
    Storage: IntTuple,
    Strides: IntTuple,
    Block: IntTuple,
    Layout: str,
]:
    @overload
    def __add__[
        Rows: IntVar,
        PackedK: IntVar,
        RS: IntVar,
        KS: IntVar,
        BR: IntVar,
        BK: IntVar,
    ](
        self: PackedPointer[
            [Rows, PackedK],
            [Rows // 32, 2 * PackedK],
            [RS, KS],
            [BR, BK],
            Literal["shuffled_scale"],
        ],
        row: BoundedAxisAddress[Rows, [BR // 32], RS, Literal["wrapped"], Literal[0]],
    ) -> PackedTilePointers[
        [Rows, PackedK],
        [Rows // 32, 2 * PackedK],
        [RS, KS],
        [BR, BK],
        Literal["scale_row"],
    ]: ...
    @overload
    def __add__[
        Rows: IntVar,
        PackedK: IntVar,
        RS: IntVar,
        KS: IntVar,
        BR: IntVar,
        BK: IntVar,
    ](
        self: PackedPointer[
            [Rows, PackedK],
            [Rows, PackedK // 2],
            [RS, KS],
            [BR, BK],
            Literal["packed_last"],
        ],
        address: BoundedAddress[
            Rows, [BR, BK // 2], [RS, KS], Literal["wrapped"], Literal[0]
        ],
    ) -> PackedTilePointers[
        [Rows, PackedK],
        [Rows, PackedK // 2],
        [RS, KS],
        [BR, BK],
        Literal["packed_last"],
    ]: ...
    @overload
    def __add__[
        PackedK: IntVar,
        Cols: IntVar,
        KS: IntVar,
        CS: IntVar,
        BC: IntVar,
        BK: IntVar,
    ](
        self: PackedPointer[
            [PackedK, Cols],
            [PackedK // 2, Cols],
            [KS, CS],
            [BC, BK],
            Literal["packed_first"],
        ],
        address: BoundedAddress[
            Cols, [BK // 2, BC], [KS, CS], Literal["wrapped"], Literal[1]
        ],
    ) -> PackedTilePointers[
        [PackedK, Cols],
        [PackedK // 2, Cols],
        [KS, CS],
        [BC, BK],
        Literal["packed_first"],
    ]: ...

class PackedTilePointers[
    Logical: IntTuple,
    Storage: IntTuple,
    Strides: IntTuple,
    Block: IntTuple,
    Layout: str,
]:
    @overload
    def __add__[
        Rows: IntVar,
        PackedK: IntVar,
        RS: IntVar,
        KS: IntVar,
        BR: IntVar,
        BK: IntVar,
    ](
        self: PackedTilePointers[
            [Rows, PackedK],
            [Rows // 32, 2 * PackedK],
            [RS, KS],
            [BR, BK],
            Literal["scale_row"],
        ],
        k: ColumnAddress[BK // 32 * 32, KS],
    ) -> PackedTilePointers[
        [Rows, PackedK],
        [Rows // 32, 2 * PackedK],
        [RS, KS],
        [BR, BK],
        Literal["shuffled_scale"],
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        PackedK: IntVar,
        RS: IntVar,
        KS: IntVar,
        BR: IntVar,
        BK: IntVar,
    ](
        self: PackedTilePointers[
            [Rows, PackedK],
            [Rows // 32, 2 * PackedK],
            [RS, KS],
            [BR, BK],
            Literal["shuffled_scale"],
        ],
        k: Int[BK * KS],
    ) -> PackedTilePointers[
        [Rows, PackedK],
        [Rows // 32, 2 * PackedK],
        [RS, KS],
        [BR, BK],
        Literal["shuffled_scale"],
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        PackedK: IntVar,
        RS: IntVar,
        KS: IntVar,
        BR: IntVar,
        BK: IntVar,
    ](
        self: PackedTilePointers[
            [Rows, PackedK],
            [Rows, PackedK // 2],
            [RS, KS],
            [BR, BK],
            Literal["packed_last"],
        ],
        k: Int[(BK // 2) * KS],
    ) -> PackedTilePointers[
        [Rows, PackedK],
        [Rows, PackedK // 2],
        [RS, KS],
        [BR, BK],
        Literal["packed_last"],
    ]: ...
    @overload
    def __iadd__[
        PackedK: IntVar,
        Cols: IntVar,
        KS: IntVar,
        CS: IntVar,
        BC: IntVar,
        BK: IntVar,
    ](
        self: PackedTilePointers[
            [PackedK, Cols],
            [PackedK // 2, Cols],
            [KS, CS],
            [BC, BK],
            Literal["packed_first"],
        ],
        k: Int[(BK // 2) * KS],
    ) -> PackedTilePointers[
        [PackedK, Cols],
        [PackedK // 2, Cols],
        [KS, CS],
        [BC, BK],
        Literal["packed_first"],
    ]: ...

class PackedScaleTile[Logical: IntTuple, Physical: IntTuple, Stage: str]:
    @overload
    def reshape[BR: IntVar, BK: IntVar](
        self: PackedScaleTile[[BR, BK], [BR // 32, BK], Literal["loaded"]],
        row_groups: Int[BR // 32],
        k_groups: int,
        d2: Literal[2],
        d32: Literal[32],
        d4: Literal[4],
        d1: Literal[1],
    ) -> PackedScaleTile[[BR, BK], [BR // 32, int, 2, 32, 4, 1], Literal["mfma32"]]: ...
    @overload
    def reshape[BR: IntVar, BK: IntVar](
        self: PackedScaleTile[[BR, BK], [BR // 32, BK], Literal["loaded"]],
        row_groups: Int[BR // 32],
        k_groups: int,
        d4: Literal[4],
        d16: Literal[16],
        d2a: Literal[2],
        d2b: Literal[2],
        d1: Literal[1],
    ) -> PackedScaleTile[
        [BR, BK], [BR // 32, int, 4, 16, 2, 2, 1], Literal["mfma16"]
    ]: ...
    @overload
    def permute[BR: IntVar, BK: IntVar](
        self: PackedScaleTile[
            [BR, BK], [BR // 32, int, 2, 32, 4, 1], Literal["mfma32"]
        ],
        a: Literal[0],
        b: Literal[3],
        c: Literal[1],
        d: Literal[4],
        e: Literal[2],
        f: Literal[5],
    ) -> PackedScaleTile[
        [BR, BK], [BR // 32, 32, int, 4, 2, 1], Literal["reordered32"]
    ]: ...
    @overload
    def permute[BR: IntVar, BK: IntVar](
        self: PackedScaleTile[
            [BR, BK], [BR // 32, int, 4, 16, 2, 2, 1], Literal["mfma16"]
        ],
        a: Literal[0],
        b: Literal[5],
        c: Literal[3],
        d: Literal[1],
        e: Literal[4],
        f: Literal[2],
        g: Literal[6],
    ) -> PackedScaleTile[
        [BR, BK], [BR // 32, 2, 16, int, 2, 4, 1], Literal["reordered16"]
    ]: ...
    @overload
    def reshape[BR: IntVar, BK: IntVar, Scale: IntVar](
        self: PackedScaleTile[
            [BR, BK], [BR // 32, 32, int, 4, 2, 1], Literal["reordered32"]
        ],
        rows: Int[BR],
        scales: Int[Scale],
    ) -> tensor[[BR, Scale]]: ...
    @overload
    def reshape[BR: IntVar, BK: IntVar, Scale: IntVar](
        self: PackedScaleTile[
            [BR, BK], [BR // 32, 2, 16, int, 2, 4, 1], Literal["reordered16"]
        ],
        rows: Int[BR],
        scales: Int[Scale],
    ) -> tensor[[BR, Scale]]: ...

class Divisor[Factor: IntVar](int): ...

class AxisStride[Groups: IntVar, Step: IntVar, IndexKind: str]:
    @overload
    def __mul__(
        self: AxisStride[Groups, Step, Literal["remainder"]],
        index: DivmodIndex[Groups, Literal["remainder"]],
    ) -> AxisAddress[Groups, Step, Literal["remainder"]]: ...
    @overload
    def __mul__(
        self: AxisStride[Groups, Step, Literal["quotient"]],
        index: DivmodIndex[Groups, Literal["quotient"]],
    ) -> AxisAddress[Groups, Step, Literal["quotient"]]: ...

class AxisAddress[Groups: IntVar, Step: IntVar, IndexKind: str]:
    def __add__[OtherStep: IntVar](
        self: AxisAddress[Groups, Step, Literal["remainder"]],
        quotient: AxisAddress[Groups, OtherStep, Literal["quotient"]],
    ) -> CombinedAddress[Groups, [OtherStep, Step]]: ...

class CombinedAddress[Groups: IntVar, Steps: IntTuple]:
    def to(self, dtype: object) -> CombinedAddress[Groups, Steps]: ...

# The parent shape records which host allocation supplied a selected view.
# It does not prove that program-id arithmetic chooses an in-bounds location.
class SelectedInPointer[
    Parent: IntTuple,
    ParentStrides: IntTuple,
    View: IntTuple,
    ViewStrides: IntTuple,
    Selection: str,
]:
    @overload
    def __add__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: SelectedInPointer[
            Parent, ParentStrides, [Tokens, Dim], [TS, FS], Literal["grouped"]
        ],
        addresses: ColumnAddress[BM, TS],
    ) -> InTilePointers[
        [Tokens, Dim], [TS, FS], [1, BM], Literal["selected_unchecked_axis_1"]
    ]: ...
    @overload
    def __add__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: SelectedInPointer[
            Parent, ParentStrides, [Tokens, Dim], [TS, FS], Literal["grouped"]
        ],
        addresses: RowAddress[BM, TS],
    ) -> InTilePointers[
        [Tokens, Dim], [TS, FS], [BM, 1], Literal["selected_unchecked_axis_0"]
    ]: ...
    @overload
    def __add__[Tokens: IntVar, BM: IntVar, Origin: str, GridAxis: int](
        self: SelectedInPointer[Parent, ParentStrides, [Tokens], [1], Literal["row"]],
        offsets: Offsets[[BM], [1], Origin, GridAxis],
    ) -> InTilePointers[[Tokens], [1], [BM], Literal["selected_unchecked"]]: ...

class SelectedOutPointer[
    Parent: IntTuple,
    ParentStrides: IntTuple,
    View: IntTuple,
    ViewStrides: IntTuple,
    Selection: str,
]:
    @overload
    def __add__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: SelectedOutPointer[
            Parent, ParentStrides, [Tokens, Dim], [TS, FS], Literal["grouped"]
        ],
        addresses: RowAddress[BM, TS],
    ) -> OutTilePointers[
        [Tokens, Dim], [TS, FS], [BM, 1], Literal["selected_unchecked_axis_0"]
    ]: ...
    @overload
    def __add__[Tokens: IntVar, BM: IntVar, Origin: str, GridAxis: int](
        self: SelectedOutPointer[
            Parent, ParentStrides, [Tokens], [1], Literal["flat_row"]
        ],
        offsets: Offsets[[BM], [1], Origin, GridAxis],
    ) -> OutTilePointers[[Tokens], [1], [BM], Literal["selected_unchecked"]]: ...
    @overload
    def __add__[Tokens: IntVar, BM: IntVar](
        self: SelectedOutPointer[
            Parent, ParentStrides, [Tokens], [1], Literal["head_row"]
        ],
        offsets: Offsets[[BM], [1], Literal["program"], Literal[0]],
    ) -> OutTilePointers[[Tokens], [1], [BM], Literal["selected_unchecked"]]: ...

@overload
def make_tensor_descriptor[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
    BC: IntVar,
](
    ptr: InPointer[[Rows, Cols], [Cols, 1]],
    shape: IntListLiteral[[Rows, Cols]],
    strides: IntListLiteral[[Cols, 1]],
    block_shape: IntListLiteral[[BR, BC]],
) -> tensor_descriptor[[Rows, Cols], [Cols, 1], [BR, BC], Literal["read"]]: ...
@overload
def make_tensor_descriptor[
    Rows: IntVar,
    Cols: IntVar,
    BR: IntVar,
](
    ptr: OutPointer[[Rows, Cols], [Cols, 1]],
    shape: IntListLiteral[[Rows, Cols]],
    strides: IntListLiteral[[Cols, 1]],
    block_shape: IntListLiteral[[BR, int]],
) -> tensor_descriptor[[Rows, Cols], [Cols, 1], [BR, int], Literal["write"]]: ...
@overload
def make_tensor_descriptor[
    Rows: IntVar,
    Cols: IntVar,
    Stride: IntVar,
    BR: IntVar,
    BC: IntVar,
](
    ptr: InOutPointer[[Rows, Cols], [Stride, 1]],
    shape: list[int],
    strides: list[int],
    block_shape: list[int],
) -> tensor_descriptor[[Rows, Cols], [Stride, 1], [BR, BC], Literal["read_write"]]: ...
@overload
def make_tensor_descriptor[BR: IntVar, BC: IntVar](
    ptr: IndirectInPointer,
    shape: list[int],
    strides: list[int],
    block_shape: IntListLiteral[[BR, BC]],
) -> tensor_descriptor[[int, int], [int, 1], [BR, BC], Literal["read"]]: ...
@overload
def make_tensor_descriptor[BR: IntVar, BC: IntVar](
    ptr: IndirectOutPointer,
    shape: list[int],
    strides: list[int],
    block_shape: IntListLiteral[[BR, BC]],
) -> tensor_descriptor[[int, int], [int, 1], [BR, BC], Literal["write"]]: ...

class ProgramId[GridAxis: int = Literal[-1]]:
    @overload
    def __mul__[Stride: IntVar](
        self: ProgramId[Literal[1]], block: SplitStride[Stride]
    ) -> SplitAddress[Stride]: ...
    @overload
    def __mul__[Block: IntVar](
        self, block: Int[Block]
    ) -> TileStart[[Block], GridAxis]: ...
    def __add__(self, other: int) -> int: ...
    @overload
    def __floordiv__[Groups: IntVar](
        self, other: Divisor[Groups]
    ) -> DivmodIndex[Groups, Literal["quotient"]]: ...
    @overload
    def __floordiv__(self, other: int) -> int: ...
    def __sub__(self, other: int) -> int: ...
    def __ge__(self, other: int) -> bool: ...
    def __lt__(self, other: int) -> bool: ...
    @overload
    def __mod__[Groups: IntVar](
        self, other: Divisor[Groups]
    ) -> DivmodIndex[Groups, Literal["remainder"]]: ...
    @overload
    def __mod__(self, other: int) -> int: ...

class DivmodIndex[Divisor: IntVar, Kind: str]:
    def __radd__(self, other: int) -> int: ...
    def __mul__[Step: IntVar](self, n: Int[Step]) -> BlockStart[Step]: ...
    def __mod__(self, other: int) -> int: ...
    def __floordiv__(self, other: int) -> int: ...

class BlockStart[Step: IntVar]:
    def __radd__(self, other: int) -> int: ...
    def __add__[Tile: IntTuple](
        self, offsets: Offsets[Tile, [1]]
    ) -> Offsets[Tile, [1], Literal["shifted"]]: ...

class LockArrayPointer[Groups: IntVar, Capacity: IntVar]:
    def __add__(
        self, index: DivmodIndex[Groups, Literal["remainder"]]
    ) -> LockSlotPointer[Groups]: ...

class LockSlotPointer[Groups: IntVar]:
    @overload
    def __add__(self, span: Divisor[Groups]) -> CountPointer[Groups]: ...
    @overload
    def __add__(self, span: Int[Groups]) -> CountPointer[Groups]: ...

class CountPointer[Groups: IntVar]: ...

class TileStart[Tile: IntTuple, GridAxis: int = Literal[-1]](int):
    def to(self, dtype: object) -> TileStart[Tile, GridAxis]: ...
    def __rsub__(self, other: int) -> int: ...
    def __radd__(self, other: int) -> int: ...
    def __mul__[HeadDim: IntVar, Tokens: IntVar](
        self: TileStart[[HeadDim], GridAxis], extent: Int[Tokens]
    ) -> ScaledTileStart[HeadDim, Tokens]: ...
    @overload
    def __add__(self, offset: int) -> int: ...
    @overload
    def __add__[Rows: IntVar](
        self, offsets: RowAxisOffsets[Rows]
    ) -> RowAxisOffsets[Rows]: ...
    @overload
    def __add__[Cols: IntVar](
        self, offsets: ColumnAxisOffsets[Cols]
    ) -> ColumnAxisOffsets[Cols]: ...
    @overload
    def __add__(
        self, offsets: Offsets[Tile, [1], Literal["local"], Literal[-1]]
    ) -> Offsets[Tile, [1], Literal["program"], GridAxis]: ...

class ScaledTileStart[HeadDim: IntVar, Tokens: IntVar]:
    def __iadd__(self, step: Int[HeadDim]) -> ScaledTileStart[HeadDim, Tokens]: ...
    # A loop-indexed offset loses its block/step relationship; the resulting
    # tile width and allocation mask are checked separately at load/store.
    def __add__(self, loop_offset: int) -> int: ...

class SplitAddress[Stride: IntVar]: ...

class SplitStride[Stride: IntVar](int):
    def __rmul__(self, index: int) -> SplitAddress[Stride]: ...

class Offsets[
    Tile: IntTuple,
    Steps: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    def to(self, dtype: object) -> Offsets[Tile, Steps, Origin, GridAxis]: ...
    @overload
    def __add__(self, value: int) -> tensor[Tile]: ...
    @overload
    def __add__[Other: IntTuple, OtherSteps: IntTuple](
        self: Offsets[Tile, Steps, Literal["local"], Literal[-1]],
        other: Offsets[Other, OtherSteps, Literal["local"], Literal[-1]],
    ) -> Offsets[
        broadcast(Tile, Other),
        add_steps(Tile, Steps, Other, OtherSteps),
        Literal["indexed"],
    ]: ...
    @overload
    def __add__[Block: IntVar, Stride: IntVar, Dim: IntVar, Cols: IntVar, CS: IntVar](
        self: RowAddress[Block, Stride],
        other: BoundedAxisAddress[Dim, [Cols], CS, Literal["wrapped"], Literal[1]],
    ) -> BoundedAddress[
        Dim, [Block, Cols], [Stride, CS], Literal["wrapped"], Literal[1]
    ]: ...
    @overload
    def __add__[Block: IntVar, Stride: IntVar, Dim: IntVar, Cols: IntVar, CS: IntVar](
        self: RowAddress[Block, Stride],
        other: BoundedAxisAddress[Dim, [Cols], CS, Literal["clamped"], Literal[1]],
    ) -> BoundedAddress[
        Dim, [Block, Cols], [Stride, CS], Literal["clamped"], Literal[1]
    ]: ...
    def __ge__[Other: IntTuple, OtherSteps: IntTuple](
        self: Offsets[Tile, Steps, Literal["local"], Literal[-1]],
        other: Offsets[Other, OtherSteps, Literal["local"], Literal[-1]],
    ) -> LogicalMask[broadcast(Tile, Other)]: ...
    def __lt__[N: IntVar](
        self, bound: Int[N]
    ) -> Mask[bounds_target(Tile, Steps, Int[N]), Tile, Origin, GridAxis]: ...
    @overload
    def __radd__[Block: IntVar, Step: IntVar](
        self: RowAxisOffsets[Block, Step], start: int
    ) -> RowAxisOffsets[Block, Step]: ...
    @overload
    def __radd__[Block: IntVar, Step: IntVar](
        self: ColumnAxisOffsets[Block, Step], start: int
    ) -> ColumnAxisOffsets[Block, Step]: ...
    @overload
    def __radd__(self, start: int) -> Offsets[Tile, Steps, Literal["shifted"]]: ...
    def __mod__[N: IntVar](self, bound: Int[N]) -> WrappedOffsets[N, Tile]: ...
    def __mul__[Scale: IntVar](
        self, stride: Int[Scale]
    ) -> Offsets[Tile, scale_steps(Tile, Steps, Int[Scale]), Origin, GridAxis]: ...
    def __rmul__[Scale: IntVar](
        self, stride: Int[Scale]
    ) -> Offsets[Tile, scale_steps(Tile, Steps, Int[Scale]), Origin, GridAxis]: ...
    @overload
    def __getitem__[Block: IntVar, Step: IntVar](
        self: Offsets[[Block], [Step], Origin, GridAxis], index: tuple[slice, None]
    ) -> RowAxisOffsets[Block, Step]: ...
    @overload
    def __getitem__[Block: IntVar, Step: IntVar](
        self: Offsets[[Block], [Step], Origin, GridAxis], index: tuple[None, slice]
    ) -> ColumnAxisOffsets[Block, Step]: ...
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> Offsets[
        insert_extent(Tile, 1),
        insert_step(Tile, Steps, 1),
        Origin,
        GridAxis,
    ]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> Offsets[
        insert_extent(Tile, 0),
        insert_step(Tile, Steps, 0),
        Origin,
        GridAxis,
    ]: ...

class BoundedOffsets[Dim: IntVar, Tile: IntTuple, Transform: str]:
    @overload
    def __getitem__(
        self, index: tuple[slice, None]
    ) -> BoundedAxisOffsets[Dim, Tile, Transform, Literal[0]]: ...
    @overload
    def __getitem__(
        self, index: tuple[None, slice]
    ) -> BoundedAxisOffsets[Dim, Tile, Transform, Literal[1]]: ...

type WrappedOffsets[Dim: IntVar, Tile: IntTuple] = BoundedOffsets[
    Dim, Tile, Literal["wrapped"]
]
type ClampedOffsets[Dim: IntVar, Tile: IntTuple] = BoundedOffsets[
    Dim, Tile, Literal["clamped"]
]

type RowAxisOffsets[Block: IntVar, Stride: IntVar = 1] = Offsets[
    [Block, 1], [Stride, 0]
]
type ColumnAxisOffsets[Block: IntVar, Stride: IntVar = 1] = Offsets[
    [1, Block], [0, Stride]
]
type RowAddress[Block: IntVar, Stride: IntVar] = Offsets[[Block, 1], [Stride, 0]]
type ColumnAddress[Block: IntVar, Stride: IntVar] = Offsets[[1, Block], [0, Stride]]

# Wrapping and clamping are distinct bounds operations on the same axis view.
class BoundedAxisOffsets[Dim: IntVar, Tile: IntTuple, Transform: str, Axis: int]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> BoundedAxisAddress[Dim, Tile, Stride, Transform, Axis]: ...

# A device pointer table contains addresses, not the allocations' shapes.
# The element strides of each indirection are only known when the kernel loads
# its leading dimensions, so the target extents and row stride remain gradual.
class PointerTable[Groups: IntVar, Access: str]:
    def __add__(self, index: int) -> PointerTableSlot[Groups, Access]: ...

class PointerTableSlot[Groups: IntVar, Access: str]: ...

class DeviceAddress[Access: str]:
    @overload
    def to(
        self: DeviceAddress[Literal["read"]], dtype: object
    ) -> IndirectInPointer: ...
    @overload
    def to(
        self: DeviceAddress[Literal["write"]], dtype: object
    ) -> IndirectOutPointer: ...

class PackedIntTable[Groups: IntVar, Fields: IntVar]:
    def __add__(self, index: int) -> PackedIntSlot[Groups, Fields]: ...

class PackedIntSlot[Groups: IntVar, Fields: IntVar]:
    def __add__(self, field: Literal[1, 2]) -> PackedIntSlot[Groups, Fields]: ...

class IndirectInPointer:
    def __add__[BM: IntVar, Stride: IntVar](
        self, offsets: RowAddress[BM, Stride]
    ) -> InTilePointers[
        [int, int], [Stride, 1], [BM, 1], Literal["indirect_axis_0"]
    ]: ...

class IndirectOutPointer:
    def __add__[BM: IntVar, Stride: IntVar](
        self, offsets: RowAddress[BM, Stride]
    ) -> OutTilePointers[
        [int, int], [Stride, 1], [BM, 1], Literal["indirect_axis_0"]
    ]: ...

class BoundedAxisAddress[
    Dim: IntVar,
    Tile: IntTuple,
    Stride: IntVar,
    Transform: str,
    Axis: int,
]:
    @overload
    def __add__[Block: IntVar, Cols: IntVar, ColumnStride: IntVar](
        self: BoundedAxisAddress[Dim, [Block], Stride, Literal["wrapped"], Literal[0]],
        other: ColumnAddress[Cols, ColumnStride],
    ) -> BoundedAddress[
        Dim, [Block, Cols], [Stride, ColumnStride], Literal["wrapped"], Literal[0]
    ]: ...
    @overload
    def __add__[Block: IntVar, Cols: IntVar, ColumnStride: IntVar](
        self: BoundedAxisAddress[Dim, [Block], Stride, Literal["clamped"], Literal[0]],
        other: ColumnAddress[Cols, ColumnStride],
    ) -> BoundedAddress[
        Dim, [Block, Cols], [Stride, ColumnStride], Literal["clamped"], Literal[0]
    ]: ...

class BoundedAddress[
    Dim: IntVar,
    Tile: IntTuple,
    Steps: IntTuple,
    Transform: str,
    Axis: int,
]: ...

class PointerDType:
    element_ty: object

class Mask[
    Target: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    def __getitem__[Rows: IntVar, BR: IntVar](
        self: Mask[[Rows], [BR]], index: tuple[slice, None]
    ) -> Mask[[Rows, 1], [BR, 1]]: ...
    def __and__[OtherTarget: IntTuple, OtherTile: IntTuple](
        self: Mask[Target, Tile, Literal["local"], Literal[-1]],
        other: Mask[OtherTarget, OtherTile, Literal["local"], Literal[-1]],
    ) -> Mask[broadcast(Target, OtherTarget), broadcast(Tile, OtherTile)]: ...

class InScalarPointer[Target: IntTuple]: ...
class OutScalarPointer[Target: IntTuple]: ...

class InOutTilePointers[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InOutTilePointers[
            [Rows, Cols], [RowStride, ColStride], [TileRows, 1], Literal["axis_0"]
        ],
        columns: ColumnAddress[TileCols, ColStride],
    ) -> InOutTilePointers[
        [Rows, Cols], [RowStride, ColStride], [TileRows, TileCols], Literal["indexed"]
    ]: ...

class InTilePointers[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    @overload
    def __add__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: InTilePointers[
            [Tokens, Dim], [TS, FS], [1, BM], Literal["selected_unchecked_axis_1"]
        ],
        features: RowAddress[Dim, FS],
    ) -> InTilePointers[
        [Tokens, Dim], [TS, FS], [Dim, BM], Literal["selected_unchecked_1"]
    ]: ...
    @overload
    def __add__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: InTilePointers[
            [Tokens, Dim], [TS, FS], [BM, 1], Literal["selected_unchecked_axis_0"]
        ],
        features: ColumnAddress[Dim, FS],
    ) -> InTilePointers[
        [Tokens, Dim], [TS, FS], [BM, Dim], Literal["selected_unchecked_0"]
    ]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, BM: IntVar](
        self: InTilePointers[
            [Rows, Cols], [Cols, 1], [BM, 1], Literal["unchecked_axis_0"]
        ],
        columns: ColumnAxisOffsets[Cols],
    ) -> InTilePointers[[Rows, Cols], [Cols, 1], [BM, Cols], Literal["unchecked"]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, RS: IntVar, BM: IntVar, BN: IntVar](
        self: InTilePointers[
            [Rows, Cols], [RS, 1], [BM, 1], Literal["indirect_axis_0"]
        ],
        columns: ColumnAxisOffsets[BN],
    ) -> InTilePointers[[Rows, Cols], [RS, 1], [BM, BN], Literal["indirect"]]: ...
    @overload
    def __iadd__[Rows: IntVar, Cols: IntVar, RS: IntVar, BM: IntVar, BN: IntVar](
        self: InTilePointers[[Rows, Cols], [RS, 1], [BM, BN], Literal["indirect"]],
        step: int,
    ) -> InTilePointers[[Rows, Cols], [RS, 1], [BM, BN], Literal["indirect"]]: ...
    @overload
    def __iadd__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: InTilePointers[
            [Tokens, Dim], [TS, FS], [Dim, BM], Literal["selected_unchecked_1"]
        ],
        step: Int[BM * TS],
    ) -> InTilePointers[
        [Tokens, Dim], [TS, FS], [Dim, BM], Literal["selected_unchecked_1"]
    ]: ...
    @overload
    def __iadd__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: InTilePointers[
            [Tokens, Dim], [TS, FS], [BM, Dim], Literal["selected_unchecked_0"]
        ],
        step: Int[BM * TS],
    ) -> InTilePointers[
        [Tokens, Dim], [TS, FS], [BM, Dim], Literal["selected_unchecked_0"]
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InTilePointers[
            [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_0"], GridAxis
        ],
        step: Int[TileCols * CS],
    ) -> InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_0"], GridAxis
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InTilePointers[
            [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_1"], GridAxis
        ],
        step: Int[TileRows * RS],
    ) -> InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_1"], GridAxis
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InTilePointers[
            [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["clamped_0"], GridAxis
        ],
        step: Int[TileCols * CS],
    ) -> InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["clamped_0"], GridAxis
    ]: ...
    @overload
    def __iadd__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InTilePointers[
            [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["clamped_1"], GridAxis
        ],
        step: Int[TileRows * RS],
    ) -> InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["clamped_1"], GridAxis
    ]: ...

class OutTilePointers[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str = Literal["local"],
    GridAxis: int = Literal[-1],
]:
    @overload
    def __add__[Tokens: IntVar, Dim: IntVar, TS: IntVar, FS: IntVar, BM: IntVar](
        self: OutTilePointers[
            [Tokens, Dim], [TS, FS], [BM, 1], Literal["selected_unchecked_axis_0"]
        ],
        features: ColumnAddress[Dim, FS],
    ) -> OutTilePointers[
        [Tokens, Dim], [TS, FS], [BM, Dim], Literal["selected_unchecked_0"]
    ]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, RS: IntVar, BM: IntVar, BN: IntVar](
        self: OutTilePointers[
            [Rows, Cols], [RS, 1], [BM, 1], Literal["indirect_axis_0"]
        ],
        columns: ColumnAxisOffsets[BN],
    ) -> OutTilePointers[[Rows, Cols], [RS, 1], [BM, BN], Literal["indirect"]]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: OutTilePointers[[Rows, Cols], [RS, CS], [TileRows, 1], Literal["axis_0"]],
        address: ColumnAddress[TileCols, CS],
    ) -> OutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ]: ...

class AxisOffset[Extent: IntVar, Stride: IntVar, Axis: int]: ...

class AxisIndex[Extent: IntVar, Axis: int]:
    def __mul__[Stride: IntVar](
        self, stride: Int[Stride]
    ) -> AxisOffset[Extent, Stride, Axis]: ...

class AxisRange[Extent: IntVar, Axis: int]:
    def __iter__(self) -> Iterator[AxisIndex[Extent, Axis]]: ...

@overload
def program_id(axis: Literal[2]) -> ProgramId[Literal[2]]: ...
@overload
def program_id(axis: Literal[0]) -> ProgramId[Literal[0]]: ...
@overload
def program_id(axis: Literal[1]) -> ProgramId[Literal[1]]: ...
@overload
def program_id(axis: int) -> ProgramId: ...
def num_programs(axis: int) -> ProgramCount: ...

class ProgramCount(int): ...

@overload
def range[Rows: IntVar, GridAxis: int](
    start: ProgramId[GridAxis], stop: Int[Rows], step: ProgramCount, *, num_stages: int
) -> AxisRange[Rows, Literal[0]]: ...
@overload
def range(start: Literal[0], stop: int) -> Iterator[int]: ...
@overload
def range[Steps: IntVar](
    start: Literal[0], stop: Int[Steps], *, num_stages: Literal[1, 2]
) -> Iterator[int]: ...
@overload
def range(
    start: int | TileStart[[int]],
    stop: int | TileStart[[int]],
    step: int,
    *,
    warp_specialize: bool,
) -> Iterator[int]: ...
@overload
def range(stop: int, *, warp_specialize: bool) -> Iterator[int]: ...
@overload
def range(start: Literal[0], stop: int, *, warp_specialize: bool) -> Iterator[int]: ...
@overload
def range[Cols: IntVar, Block: IntVar](
    start: Literal[0], stop: Int[Cols], step: Int[Block], *, multi_cta: bool
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis], stop: int, step: Int[SMs], *, flatten: Literal[True]
) -> Iterator[int]: ...
@overload
def range(
    start: int, stop: int, *, num_stages: int, disallow_acc_multi_buffer: bool
) -> Iterator[int]: ...
@overload
def range(
    start: int,
    stop: int,
    step: int,
    *,
    warp_specialize: bool,
    disallow_acc_multi_buffer: Literal[True],
    data_partition_factor: Literal[2],
) -> Iterator[int]: ...
@overload
def range(
    start: int,
    stop: int,
    step: int,
    *,
    warp_specialize: bool,
    merge_epilogue: Literal[True],
    merge_correction: Literal[True],
    data_partition_factor: Literal[2],
) -> Iterator[int]: ...
@overload
def range(
    start: Literal[0],
    stop: int,
    *,
    warp_specialize: bool,
    merge_epilogue: Literal[True],
    merge_correction: Literal[True],
    data_partition_factor: Literal[2],
) -> Iterator[int]: ...
@overload
def range(
    start: int,
    stop: int,
    step: int,
    *,
    warp_specialize: bool,
    disallow_acc_multi_buffer: bool,
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis],
    stop: int,
    step: Int[SMs],
    *,
    flatten: Literal[True],
    warp_specialize: bool,
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis],
    stop: int,
    step: Int[SMs],
    *,
    flatten: bool,
    warp_specialize: bool,
) -> Iterator[int]: ...
@overload
def range[SMs: IntVar, GridAxis: int](
    start: ProgramId[GridAxis],
    stop: int,
    step: Int[SMs],
    *,
    flatten: bool,
    warp_specialize: Literal[True],
    disallow_acc_multi_buffer: Literal[True],
    separate_epilogue_store: Literal[True],
) -> Iterator[int]: ...
@overload
def arange[Block: IntVar](
    start: Literal[0], end: Int[Block]
) -> Offsets[[Block], [1]]: ...
@overload
def arange[Start: IntVar, End: IntVar](
    start: Int[Start], end: Int[End]
) -> Offsets[[End - Start], [1]]: ...
def cdiv(value: int, block: int) -> int: ...
def assume(predicate: bool) -> None: ...
def static_assert(predicate: bool) -> None: ...
@overload
def multiple_of(value: int, block: int) -> int: ...
@overload
def multiple_of[Tile: IntTuple](value: TileStart[Tile], block: int) -> int: ...
@overload
def multiple_of[Dim: IntVar, Tile: IntTuple](
    value: BoundedOffsets[Dim, Tile, Literal["clamped"]], block: int
) -> BoundedOffsets[Dim, Tile, Literal["clamped"]]: ...
def max_contiguous[Dim: IntVar, Tile: IntTuple](
    value: BoundedOffsets[Dim, Tile, Literal["clamped"]], block: int
) -> BoundedOffsets[Dim, Tile, Literal["clamped"]]: ...
@overload
def multiple_of[Target: IntTuple, Strides: IntTuple, Tile: IntTuple](
    value: InTilePointers[Target, Strides, Tile, Literal["indirect"]],
    block: list[int],
) -> InTilePointers[Target, Strides, Tile, Literal["indirect"]]: ...
def pointer_type(dtype: object) -> object: ...
def zeros[Shape: IntTuple](
    shape: IntTupleOrList[Shape], *, dtype: object
) -> tensor[Shape]: ...
def full[Shape: IntTuple](
    shape: IntTupleOrList[Shape], value: int, *, dtype: object
) -> tensor[Shape]: ...
def reshape[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]],
    shape: tuple[Int[Rows], Literal[2], Int[Cols // 2]],
) -> tensor[[Rows, 2, Cols // 2]]: ...
def permute[Rows: IntVar, Half: IntVar](
    value: tensor[[Rows, 2, Half]], order: tuple[Literal[0], Literal[2], Literal[1]]
) -> tensor[[Rows, Half, 2]]: ...
def split[Rows: IntVar, Half: IntVar](
    value: tensor[[Rows, Half, 2]],
) -> tuple[tensor[[Rows, Half]], tensor[[Rows, Half]]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]]
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], *, input_precision: Literal["ieee", "tf32"]
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], acc: tensor[[M, N]]
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], *, attrs: object | None
) -> tensor[[M, N]]: ...
@overload
def dot[M: IntVar, K: IntVar, N: IntVar](
    a: tensor[[M, K]], b: tensor[[K, N]], acc: tensor[[M, N]], *, attrs: object | None
) -> tensor[[M, N]]: ...
def trans[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]],
) -> tensor[[Cols, Rows]]: ...
@overload
def dot_scaled[M: IntVar, N: IntVar, KA: IntVar, KB: IntVar, ScaleK: IntVar](
    a: tensor[[M, KA]],
    scale_a: tensor[[M, ScaleK]],
    format_a: str,
    b: tensor[[KB, N]],
    scale_b: tensor[[N, ScaleK]],
    format_b: str,
) -> tensor[[M, N]]: ...
@overload
def dot_scaled[
    M: IntVar,
    N: IntVar,
    KA: IntVar,
    KB: IntVar,
    ScaleK: IntVar,
](
    a: tensor[[M, KA]],
    scale_a: tensor[[M, ScaleK]],
    format_a: str,
    b: tensor[[KB, N]],
    scale_b: tensor[[N, ScaleK]],
    format_b: str,
    acc: tensor[[M, N]],
) -> tensor[[M, N]]: ...
@overload
def where[Tile: IntTuple](
    condition: LogicalMask[Tile], x: tensor[Tile], y: tensor[Tile]
) -> tensor[Tile]: ...
@overload
def where[Tile: IntTuple](
    condition: LogicalMask[Tile], x: tensor[Tile], y: float
) -> tensor[Tile]: ...

# Loaded boolean tiles have gradual dtype until pointers track element dtype.
@overload
def where[Tile: IntTuple](
    condition: tensor[Tile], x: tensor[Tile], y: float
) -> tensor[Tile]: ...
@overload
def where[Target: IntTuple, Tile: IntTuple, Origin: str](
    condition: Mask[Target, Tile, Origin], x: tensor[Tile], y: float
) -> tensor[Tile]: ...
@overload
def where[Dim: IntVar, Tile: IntTuple](
    condition: Mask[[Dim], Tile], x: Offsets[Tile, [1]], y: Literal[0]
) -> ClampedOffsets[Dim, Tile]: ...
@overload
def where[Dim: IntVar, Tile: IntTuple, Stride: IntVar, Origin: str, GridAxis: int](
    condition: Mask[[Dim], Tile, Origin, GridAxis],
    x: Offsets[Tile, [Stride], Origin, GridAxis],
    y: Literal[0],
) -> ClampedOffsets[Dim, Tile]: ...
@overload
def where[Tile: IntTuple](
    condition: LogicalMask[Tile], x: float, y: float
) -> tensor[Tile]: ...
def rand[Tile: IntTuple, Origin: str, GridAxis: int](
    seed: int, offsets: Offsets[Tile, [1], Origin, GridAxis]
) -> tensor[Tile]: ...
def sqrt[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...
def atomic_cas[Groups: IntVar](
    ptr: LockSlotPointer[Groups], old: int, new: int
) -> int: ...
@overload
def atomic_xchg[Groups: IntVar](ptr: LockSlotPointer[Groups], new: int) -> None: ...
@overload
def atomic_xchg[Groups: IntVar](ptr: CountPointer[Groups], new: int) -> None: ...
def debug_barrier() -> None: ...

float16: object
float32: object
bfloat16: object
float8e5: object
float8e4nv: object
int64: object
constexpr = int

def maximum[Tile: IntTuple](a: tensor[Tile], b: tensor[Tile]) -> tensor[Tile]: ...
@overload
def load[Target: IntTuple](ptr: InScalarPointer[Target]) -> tensor[[]]: ...
@overload
def load[Groups: IntVar, Access: str](
    ptr: PointerTableSlot[Groups, Access],
) -> DeviceAddress[Access]: ...
@overload
def load[Groups: IntVar, Fields: IntVar](ptr: PackedIntSlot[Groups, Fields]) -> int: ...
@overload
def load[Rows: IntVar, Cols: IntVar, RS: IntVar](
    ptr: InTilePointers[[int, int], [RS, 1], [Rows, Cols], Literal["indirect"]],
) -> tensor[[Rows, Cols]]: ...
@overload
def load[M: IntVar, K: IntVar, BM: IntVar, BK: IntVar, KS: IntVar](
    ptr: PackedTilePointers[
        [M, K], [M, K // 2], [int, KS], [BM, BK], Literal["packed_last"]
    ],
    *,
    cache_modifier: None = None,
) -> tensor[[BM, BK // 2]]: ...
@overload
def load[N: IntVar, K: IntVar, BN: IntVar, BK: IntVar, KS: IntVar](
    ptr: PackedTilePointers[
        [K, N], [K // 2, N], [KS, int], [BN, BK], Literal["packed_first"]
    ],
    *,
    cache_modifier: None = None,
) -> tensor[[BK // 2, BN]]: ...
@overload
def load[R: IntVar, K: IntVar, S: IntVar, BM: IntVar, BK: IntVar](
    ptr: PackedTilePointers[
        [R, K], [R // 32, 2 * K], [int, S], [BM, BK], Literal["shuffled_scale"]
    ],
) -> PackedScaleTile[[BM, BK], [BM // 32, BK], Literal["loaded"]]: ...
@overload
def load[Rows: IntVar, Cols: IntVar, BM: IntVar](
    ptrs: InTilePointers[[Rows, Cols], [Cols, 1], [BM, Cols], Literal["unchecked"]],
) -> tensor[[BM, Cols]]: ...
@overload
def load[Tokens: IntVar, Dim: IntVar, Stride: IntVar, BM: IntVar](
    ptrs: InTilePointers[
        [Tokens, Dim], [Stride, int], [Dim, BM], Literal["selected_unchecked_1"]
    ],
) -> tensor[[Dim, BM]]: ...
@overload
def load[Tokens: IntVar, Dim: IntVar, Stride: IntVar, BM: IntVar](
    ptrs: InTilePointers[
        [Tokens, Dim], [Stride, int], [BM, Dim], Literal["selected_unchecked_0"]
    ],
) -> tensor[[BM, Dim]]: ...
@overload
def load[Tokens: IntVar, BM: IntVar](
    ptrs: InTilePointers[[Tokens], [1], [BM], Literal["selected_unchecked"]],
) -> tensor[[BM]]: ...
@overload
def load[Groups: IntVar](ptr: CountPointer[Groups]) -> tensor[[]]: ...
@overload
def load[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str,
    GridAxis: int,
](
    ptrs: InTilePointers[Target, Strides, Tile, Origin, GridAxis],
    mask: Mask[Target, Tile, Origin, GridAxis],
) -> tensor[Tile]: ...
@overload
def load[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str,
    GridAxis: int,
](
    ptrs: InTilePointers[Target, Strides, Tile, Origin, GridAxis],
    mask: Mask[Target, Tile, Origin, GridAxis],
    *,
    other: float,
) -> tensor[Tile]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_0"]
    ],
    mask: Mask[[1, Cols], [1, TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["wrapped_1"]
    ],
    mask: Mask[[Rows, 1], [TileRows, 1]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[Cols: IntVar, BN: IntVar](
    ptrs: InTilePointers[[Cols], [1], [1, BN], Literal["column_axis"]],
    mask: Mask[[1, Cols], [1, BN]],
) -> tensor[[1, BN]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    BR: IntVar,
    BC: IntVar,
](
    ptrs: InTilePointers[[Rows, Cols], [RS, CS], [BR, BC], Literal["indexed"]],
    mask: Mask[[Rows, Cols], [BR, BC]],
    *,
    other: float,
) -> tensor[[BR, BC]]: ...
@overload
def load[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str,
    GridAxis: int,
](
    ptrs: InOutTilePointers[Target, Strides, Tile, Origin, GridAxis],
    mask: Mask[Target, Tile, Origin, GridAxis],
) -> tensor[Tile]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["clamped_0"]
    ],
    mask: Mask[[1, Cols], [1, TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: InTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["clamped_1"]
    ],
    mask: Mask[[Rows, 1], [TileRows, 1]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def load[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: InOutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ],
    mask: Mask[[Rows, Cols], [TileRows, TileCols]],
    *,
    other: float,
) -> tensor[[TileRows, TileCols]]: ...
@overload
def store[Target: IntTuple](
    ptr: OutScalarPointer[Target], value: tensor[[]]
) -> None: ...
@overload
def store[Tokens: IntVar, BM: IntVar](
    ptrs: OutTilePointers[[Tokens], [1], [BM], Literal["selected_unchecked"]],
    value: tensor[[BM]],
) -> None: ...
@overload
def store[Tokens: IntVar, Dim: IntVar, BM: IntVar](
    ptrs: OutTilePointers[
        [Tokens, Dim], [int, int], [BM, Dim], Literal["selected_unchecked_0"]
    ],
    value: tensor[[BM, Dim]],
) -> None: ...
@overload
def store[Rows: IntVar, Cols: IntVar, RS: IntVar](
    ptr: OutTilePointers[[int, int], [RS, 1], [Rows, Cols], Literal["indirect"]],
    value: tensor[[Rows, Cols]],
) -> None: ...
@overload
def store[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str,
    GridAxis: int,
](
    ptrs: InOutTilePointers[Target, Strides, Tile, Origin, GridAxis],
    value: tensor[Tile],
    mask: Mask[Target, Tile, Origin, GridAxis],
) -> None: ...
@overload
def store[
    Target: IntTuple,
    Strides: IntTuple,
    Tile: IntTuple,
    Origin: str,
    GridAxis: int,
](
    ptrs: OutTilePointers[Target, Strides, Tile, Origin, GridAxis],
    value: tensor[Tile],
    mask: Mask[Target, Tile, Origin, GridAxis],
) -> None: ...
@overload
def store[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: OutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ],
    value: tensor[[TileRows, TileCols]],
    mask: Mask[[Rows, Cols], [TileRows, TileCols]],
    cache_modifier: Literal[".wt"] | None = None,
) -> None: ...
@overload
def store[
    Rows: IntVar,
    Cols: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
    RS: IntVar,
    CS: IntVar,
](
    ptrs: InOutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ],
    value: tensor[[TileRows, TileCols]],
    mask: Mask[[Rows, Cols], [TileRows, TileCols]],
) -> None: ...
def atomic_add[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    TileRows: IntVar,
    TileCols: IntVar,
](
    ptrs: OutTilePointers[
        [Rows, Cols], [RS, CS], [TileRows, TileCols], Literal["indexed"]
    ],
    value: tensor[[TileRows, TileCols]],
    mask: Mask[[Rows, Cols], [TileRows, TileCols]],
) -> tensor[[TileRows, TileCols]]: ...
@overload
def max[Block: IntVar](value: tensor[[Block]], axis: Literal[0]) -> tensor[[]]: ...
@overload
def max[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]], axis: Literal[1]
) -> tensor[[Rows]]: ...

class ReductionOrdering:
    UNORDERED: ReductionOrdering
    INNER_TREE: ReductionOrdering

@overload
def sum[Block: IntVar](value: tensor[[Block]], axis: Literal[0]) -> tensor[[]]: ...
@overload
def sum[Block: IntVar](
    value: tensor[[Block]],
    axis: Literal[0],
    *,
    reduction_ordering: ReductionOrdering,
) -> tensor[[]]: ...
@overload
def sum[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]], axis: Literal[0]
) -> tensor[[Cols]]: ...
@overload
def sum[Rows: IntVar, Cols: IntVar](
    value: tensor[[Rows, Cols]], axis: Literal[1]
) -> tensor[[Rows]]: ...
def exp[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...
