"""Type-level Triton pointer operations for host tensor view markers."""

from typing import Literal, overload

import triton.language as tl
from shape_extensions import IntTuple, IntVar

class InPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RS: IntVar,
        CS: IntVar,
        BR: IntVar,
        BC: IntVar,
    ](
        self: InPointer[[Rows, Cols], [RS, CS]],
        address: tl.Offsets[[BR, BC], [RS, CS], Literal["indexed"]],
    ) -> tl.InTilePointers[[Rows, Cols], [RS, CS], [BR, BC], Literal["indexed"]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, BM: IntVar](
        self: InPointer[[Rows, Cols], [Cols, 1]],
        address: tl.RowAddress[BM, Cols],
    ) -> tl.InTilePointers[
        [Rows, Cols], [Cols, 1], [BM, 1], Literal["unchecked_axis_0"]
    ]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, RS: IntVar, CS: IntVar, BR: IntVar](
        self: InPointer[[Rows, Cols], [RS, CS]],
        address: tl.BoundedAxisAddress[Rows, [BR], RS, Literal["wrapped"], Literal[0]],
    ) -> SelectedInputRowPointer[Rows, Cols, RS, CS, BR, Literal["wrapped_0"]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, RS: IntVar, CS: IntVar, BR: IntVar](
        self: InPointer[[Rows, Cols], [RS, CS]],
        address: tl.RowAddress[BR, RS],
    ) -> SelectedInputRowPointer[Rows, Cols, RS, CS, BR, Literal["wrapped_1"]]: ...
    @overload
    def __add__[M: IntVar, K: IntVar, AM: IntVar, AK: IntVar, BM: IntVar, BK: IntVar](
        self: InPointer[[M, K], [AM, AK]],
        address: tl.BoundedAddress[
            M, [BM, BK], [AM, AK], Literal["clamped"], Literal[0]
        ],
    ) -> tl.InTilePointers[[M, K], [AM, AK], [BM, BK], Literal["clamped_0"]]: ...
    @overload
    def __add__[
        K: IntVar,
        N: IntVar,
        BK: IntVar,
        BN: IntVar,
        TileK: IntVar,
        TileN: IntVar,
    ](
        self: InPointer[[K, N], [BK, BN]],
        address: tl.BoundedAddress[
            N, [TileK, TileN], [BK, BN], Literal["clamped"], Literal[1]
        ],
    ) -> tl.InTilePointers[[K, N], [BK, BN], [TileK, TileN], Literal["clamped_1"]]: ...
    @overload
    def __add__[
        Batch: IntVar,
        Heads: IntVar,
        Tokens: IntVar,
        Dim: IntVar,
        StrideZ: IntVar,
        StrideH: IntVar,
        TokenStride: IntVar,
        FeatureStride: IntVar,
    ](
        self: InPointer[
            [Batch, Heads, Tokens, Dim],
            [StrideZ, StrideH, TokenStride, FeatureStride],
        ],
        offset: tl.CombinedAddress[Heads, [StrideZ, StrideH]],
    ) -> tl.SelectedInPointer[
        [Batch, Heads, Tokens, Dim],
        [StrideZ, StrideH, TokenStride, FeatureStride],
        [Tokens, Dim],
        [TokenStride, FeatureStride],
        Literal["grouped"],
    ]: ...
    @overload
    def __add__[Batch: IntVar, Heads: IntVar, Tokens: IntVar, GridAxis: int](
        self: InPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
        offset: tl.TileStart[[Tokens], GridAxis],
    ) -> tl.SelectedInPointer[
        [Batch, Heads, Tokens],
        [int, Tokens, 1],
        [Tokens],
        [1],
        Literal["row"],
    ]: ...
    @overload
    def __add__[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Dim: IntVar](
        self: InPointer[[Batch, Heads, Tokens, Dim], [int, int, Dim, 1]],
        start: tl.ScaledTileStart[Dim, Tokens],
    ) -> InPointer[[Tokens, Dim], [Dim, 1]]: ...
    @overload
    def __add__[Length: IntVar, GridAxis: int](
        self: InPointer[[Length], [1]], row: tl.ProgramId[GridAxis]
    ) -> tl.InScalarPointer[[Length]]: ...
    @overload
    def __add__[
        Length: IntVar,
        Stride: IntVar,
        Tile: IntTuple,
        Origin: str,
        GridAxis: int,
    ](
        self: InPointer[[Length], [Stride]],
        offsets: tl.Offsets[Tile, [Stride], Origin, GridAxis],
    ) -> tl.InTilePointers[[Length], [Stride], Tile, Origin, GridAxis]: ...
    @overload
    def __add__[Cols: IntVar, BC: IntVar](
        self: InPointer[[Cols], [1]], offsets: tl.ColumnAxisOffsets[BC, 1]
    ) -> tl.InTilePointers[[Cols], [1], [1, BC], Literal["column_axis"]]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        Stride: IntVar,
        ColumnStride: IntVar,
        GridAxis: int,
    ](
        self: InPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride], GridAxis],
    ) -> InPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: InPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.AxisOffset[Rows, Stride, Literal[0]],
    ) -> InPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InPointer[[Rows, Cols], [RowStride, ColStride]],
        address: tl.BoundedAddress[
            Rows,
            [TileRows, TileCols],
            [RowStride, ColStride],
            Literal["wrapped"],
            Literal[0],
        ],
    ) -> tl.InTilePointers[
        [Rows, Cols],
        [RowStride, ColStride],
        [TileRows, TileCols],
        Literal["wrapped_0"],
    ]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
        TileCols: IntVar,
    ](
        self: InPointer[[Rows, Cols], [RowStride, ColStride]],
        address: tl.BoundedAddress[
            Cols,
            [TileRows, TileCols],
            [RowStride, ColStride],
            Literal["wrapped"],
            Literal[1],
        ],
    ) -> tl.InTilePointers[
        [Rows, Cols],
        [RowStride, ColStride],
        [TileRows, TileCols],
        Literal["wrapped_1"],
    ]: ...

class SelectedInputRowPointer[
    Rows: IntVar,
    Cols: IntVar,
    RS: IntVar,
    CS: IntVar,
    BR: IntVar,
    Origin: str,
]:
    @overload
    def __add__[BC: IntVar](
        self: SelectedInputRowPointer[Rows, Cols, RS, CS, BR, Literal["wrapped_0"]],
        address: tl.ColumnAddress[BC, CS],
    ) -> tl.InTilePointers[[Rows, Cols], [RS, CS], [BR, BC], Literal["wrapped_0"]]: ...
    @overload
    def __add__[BC: IntVar](
        self: SelectedInputRowPointer[Rows, Cols, RS, CS, BR, Literal["wrapped_1"]],
        address: tl.BoundedAxisAddress[Cols, [BC], CS, Literal["wrapped"], Literal[1]],
    ) -> tl.InTilePointers[[Rows, Cols], [RS, CS], [BR, BC], Literal["wrapped_1"]]: ...

class OutPointer[Target: IntTuple, Strides: IntTuple]:
    dtype: tl.PointerDType
    type: tl.PointerDType
    @overload
    def __add__[M: IntVar, N: IntVar, CM: IntVar, CN: IntVar, BM: IntVar, BN: IntVar](
        self: OutPointer[[M, N], [CM, CN]],
        address: tl.Offsets[[BM, BN], [CM, CN], Literal["indexed"]],
    ) -> tl.OutTilePointers[[M, N], [CM, CN], [BM, BN], Literal["indexed"]]: ...
    @overload
    def __add__[
        Batch: IntVar,
        Heads: IntVar,
        Tokens: IntVar,
        Dim: IntVar,
        StrideZ: IntVar,
        StrideH: IntVar,
        TokenStride: IntVar,
        FeatureStride: IntVar,
    ](
        self: OutPointer[
            [Batch, Heads, Tokens, Dim],
            [StrideZ, StrideH, TokenStride, FeatureStride],
        ],
        offset: tl.CombinedAddress[Heads, [StrideZ, StrideH]],
    ) -> tl.SelectedOutPointer[
        [Batch, Heads, Tokens, Dim],
        [StrideZ, StrideH, TokenStride, FeatureStride],
        [Tokens, Dim],
        [TokenStride, FeatureStride],
        Literal["grouped"],
    ]: ...
    @overload
    def __add__[Batch: IntVar, Heads: IntVar, Tokens: IntVar, Axis: int](
        self: OutPointer[[Batch, Heads, Tokens], [int, Tokens, 1]],
        start: tl.TileStart[[Tokens], Axis],
    ) -> tl.SelectedOutPointer[
        [Batch, Heads, Tokens],
        [int, Tokens, 1],
        [Tokens],
        [1],
        Literal["head_row"],
    ]: ...
    @overload
    def __add__[Rows: IntVar, Tokens: IntVar, Axis: int](
        self: OutPointer[[Rows, Tokens], [Tokens, 1]],
        start: tl.TileStart[[Tokens], Axis],
    ) -> tl.SelectedOutPointer[
        [Rows, Tokens], [Tokens, 1], [Tokens], [1], Literal["flat_row"]
    ]: ...
    @overload
    def __add__[Length: IntVar, GridAxis: int](
        self: OutPointer[[Length], [1]], row: tl.ProgramId[GridAxis]
    ) -> tl.OutScalarPointer[[Length]]: ...
    @overload
    def __add__[
        Length: IntVar,
        Stride: IntVar,
        Tile: IntTuple,
        Origin: str,
        GridAxis: int,
    ](
        self: OutPointer[[Length], [Stride]],
        offsets: tl.Offsets[Tile, [Stride], Origin, GridAxis],
    ) -> tl.OutTilePointers[[Length], [Stride], Tile, Origin, GridAxis]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        Stride: IntVar,
        ColumnStride: IntVar,
        GridAxis: int,
    ](
        self: OutPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.TileStart[[Stride], GridAxis],
    ) -> OutPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[Rows: IntVar, Cols: IntVar, Stride: IntVar, ColumnStride: IntVar](
        self: OutPointer[[Rows, Cols], [Stride, ColumnStride]],
        offset: tl.AxisOffset[Rows, Stride, Literal[0]],
    ) -> OutPointer[[Cols], [ColumnStride]]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
    ](
        self: OutPointer[[Rows, Cols], [RowStride, ColStride]],
        address: tl.RowAddress[TileRows, RowStride],
    ) -> tl.OutTilePointers[
        [Rows, Cols], [RowStride, ColStride], [TileRows, 1], Literal["axis_0"]
    ]: ...

class InOutPointer[Target: IntTuple, Strides: IntTuple]:
    @overload
    def __add__[
        Parts: IntVar,
        Rows: IntVar,
        Cols: IntVar,
        SplitStride: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
    ](
        self: InOutPointer[[Parts, Rows, Cols], [SplitStride, RowStride, ColStride]],
        offset: tl.TileStart[[SplitStride]] | tl.SplitAddress[SplitStride],
    ) -> InOutPointer[[Rows, Cols], [RowStride, ColStride]]: ...
    @overload
    def __add__[
        Rows: IntVar,
        Cols: IntVar,
        RowStride: IntVar,
        ColStride: IntVar,
        TileRows: IntVar,
    ](
        self: InOutPointer[[Rows, Cols], [RowStride, ColStride]],
        offsets: tl.RowAddress[TileRows, RowStride],
    ) -> tl.InOutTilePointers[
        [Rows, Cols], [RowStride, ColStride], [TileRows, 1], Literal["axis_0"]
    ]: ...
    @overload
    def __add__[Groups: IntVar, Cols: IntVar](
        self: InOutPointer[[Groups, Cols], [Cols, 1]],
        start: tl.BlockStart[Cols],
    ) -> InOutPointer[[Cols], [1]]: ...
    @overload
    def __add__[Cols: IntVar, Tile: IntTuple, Origin: str, GridAxis: int](
        self: InOutPointer[[Cols], [1]],
        offsets: tl.Offsets[Tile, [1], Origin, GridAxis],
    ) -> tl.InOutTilePointers[[Cols], [1], Tile, Origin, GridAxis]: ...
