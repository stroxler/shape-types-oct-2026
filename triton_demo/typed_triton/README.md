# Typed Triton library

`@typed_triton.jit` translates static-only semantic annotations into Triton's
runtime signature and source. It strips both parameter and return annotations
from the source Triton compiles, while retaining compile-time `constexpr`
markers. Its static `SemanticKernel` adapter keeps the
decorated function signature for grid-indexed launches; a JIT function is
still the actual runtime object.
`constexpr[T]` is a shape-aware alias imported from `typed_triton`, not a
monkey-patch of `tl.constexpr`. The adapter presents it to Triton as the
ordinary `tl.constexpr` annotation. With runtime-checkable pointer annotations,
including postponed `tlt.InPointer`, `tlt.OutPointer`, and `tlt.InOutPointer`
annotations, a pre-run
hook checks actual Torch shapes, strides, and matching scalar arguments for
both JIT and interpreter launches. It also checks `Int[...]` inside
`constexpr[...]` and permits unannotated parameters. Postponed annotations
recognize direct imports/aliases of those marker classes as well as their
`tlt.*` spelling, but not arbitrary indirect type aliases. Stub-only
pointer forms that cannot be evaluated at runtime do not register this
hook. The hook does not check devices, dtypes, grid sizes, or launch-only
options. Writable pointer arguments reject overlapping strides within
each tensor, but separate tensor arguments may still alias. `tlt.py` provides
evaluable pointer markers, with their arithmetic specified in `tlt.pyi`.
`host_tensor.py` is a separate Torch-side layout view: `as_host_tensor(x)`
checks a vector's unit element stride or, by default, a matrix's unit inner
stride, preserves the original Torch object, and marks a matrix's row stride
as unknown `int`. An explicit host-view type permits independently checked
matrix strides, including transposed inputs. `torch_views.py` converts checked
host views to Triton pointer views and returns observed dimensions and strides.
The pre-run hook validates symbolic relationships not retained statically at
the launch (see the Triton examples README).

The TMA overlay uses `tl.tensor_descriptor[Allocation, Strides, Block, Access]`
for descriptor-backed block transfers of any rank. Read-only and write-only
capabilities distinguish descriptor inputs from outputs, and the block shape
determines loaded values and permissible stores. Kernel parameters for
host-constructed descriptors currently describe geometry without validating
their host construction; `tl.make_tensor_descriptor` checks declared shapes
and strides for the pointer forms supported by this prototype.
Symbolic block-start descriptor offsets are checked against the corresponding block
dimension for the modeled two-axis data/output and five-axis scale layouts.
Plain integer offsets remain gradual; the current list-based API does not
statically associate each list position with an allocation axis.
The `make_tensor_descriptor` read-write pointer overload currently accepts
ordinary `list[int]` shape, stride, and block arguments, then returns a block
shape with unconstrained symbolic extents. This does not prove that those
lists describe the pointer or the returned block. `IntListLiteral` captures
direct literals but not lists forwarded through the attention descriptor
helpers. Tightening this constructor requires an interface that retains
block extents across those helpers, such as typed scalar witnesses, as well
as tests rejecting mismatched shapes, strides, and blocks.

The vector-add example supplies its own explicit launch metadata, including
the output dtype. Its checked host function verifies matching device, length,
and layout but leaves input dtypes unconstrained; Triton specializes them
when launching the kernel.
The `triton-stubs/` overlay is local to v1 and can diverge from v0. Its
ordinary Triton stubs were copied from v0; Gluon stubs are out of scope here.
The first executable usage is in `../triton_examples/`.
The 1D pointer-array overloads check that logical offsets were scaled by the
pointer's element stride, and their load/store masks must use the same logical
allocation length, tile width, offset-origin category, and known launch grid
axis. `Mask[Target, Tile, Origin, GridAxis]` also represents two-dimensional
bounds. A row mask uses `[Rows, 1]` and `[TileRows, 1]`; a column mask uses
`[1, Cols]` and `[1, TileCols]`. Combining them with `&` produces the full
`Mask[[Rows, Cols], [TileRows, TileCols]]`. Comparison with a bound derives
the masked axis from the unit-step axis of `Offsets[Tile, Steps]`; inserted
zero steps mark the other axes. The same rule accepts a single varying axis
at any known rank and rejects non-unit, absent, or multiple index steps.
Mask conjunction and logical comparison of offset grids broadcast at any
rank. Rank-two load/store overloads
require the corresponding allocation bounds and tile widths; overloads for
partial-axis loads accept only the correct row or column orientation. This
checks dimensional compatibility, not that the mask predicate guards the
precise pointer address. Distinct grid axes are checked separately from matrix
address orientation. The untagged axis (`-1`) from `program_id(axis: int)` or
adding arbitrary scalar indices to offsets can match another untagged axis;
that match supplies **no** grid-axis guarantee. Branches joining axis-zero
and axis-one offsets retain a union that is rejected against an axis-zero-only
mask or pointer, in both `tl.load` and `tl.store`. A bare `ProgramId` can
shift a unit-stride vector pointer but not a strided vector or matrix pointer.
Mask singleton-axis insertion currently accepts only untagged one-dimensional
masks: indexing a `Mask` with a nondefault `Origin` or `GridAxis` is rejected,
not silently stripped of its tags. A tag-preserving one-dimensional signature
or a rank-generic signature admits the wrong-grid masked loads probed in
`../triton_examples/test_vector_add.py`; insertion needs to preserve grid-axis
rejection elsewhere before it can be generalized.
`Offsets[Tile, Steps, Origin, GridAxis]` records a tuple of tile extents and
a tuple of element-address contributions, one per axis. For example,
`tl.arange(0, B)` has `Offsets[[B], [1]]`; inserting a singleton axis with
`[:, None]` gives the row-oriented `Offsets[[B, 1], [1, 0]]`, and multiplying
by a symbolic stride `S` gives address steps `[S, 0]`. Generic insertion of a
singleton axis is computed by the `_shapes.pyi` DSL for higher-rank offsets;
the row and column offset aliases are just 2D instances of `Offsets`. Triton
is not limited to rank two: `Offsets` handles known-rank insertion, stride
scaling, and equal-rank broadcast addition of compatible grids. A scalar
shift of a broadcasted row or column preserves its singleton axis. The shape
DSL checks that tile and step tuples have matching ranks when computing their
steps; annotating mismatched tuples without performing an operation is not
itself rejected. The sum of two offset grids has an `indexed` origin; final
pointer addition checks its steps against the allocation strides. This
removes the need for separate grouped, strided, or column-major offset
classes. An offset grid of unknown rank remains gradual.
`WrappedOffsets` and `ClampedOffsets` distinguish modulo from substituting
zero for an out-of-range value. They share a bound-and-transform
representation; the types do not prove that a clamped input is nonnegative.
`BoundedAxisOffsets` retains which singleton axis was inserted after either
transformation; it is not a third kind of bounds operation. `BoundedAddress`
retains an address-step tuple of any rank.
`BoundedAxisAddress` instead has one stride because it describes one varying
axis before it is combined with another address. Its combination with another
axis, and subsequent pointer additions, still use rank-two signatures: the
type checker must compare the bounded dimension and each address step with
the corresponding allocation axis. A rank-generic result-shape DSL alone
cannot check those argument-side equalities. The rank-two overloads continue
to reject mismatched bounds and strides in the matrix-multiplication fixtures.
These signatures retain one bounded axis; an address with independently
wrapped and zero-substituted axes needs per-axis transformation metadata.
Pointer additions still use stride-matching row and column overloads. A
shape-DSL check in the result type does not reject addition with a different
symbolic stride, and Pyrefly does not allow DSL calls in parameter annotations
where that constraint would have to be enforced. The `selected + wrong_row`
and `rows + wrong_column` probes in
`../triton_examples/test_split_k_matmul.py` exercise these rejections. A
rank-generic pointer addition needs a parameter-side stride constraint or a
nominal representation of the address steps before these overloads can go.
Literal `program_id` axes and reduction axes retain their overloads: replacing
them with `Flag` parameters either loses the known program axis or accepts
dynamic reduction axes that the present signatures reject. Shape DSL rules
calculate result dimensions, but cannot replace these parameter-side checks
or the nominal access and provenance distinctions on pointers and masks.
Top-level `reshape` and `permute`, and `tensor.reshape` and `tensor.trans`,
retain their input-shape and literal-order constraints. A result-shape DSL can
compute permutations and check concrete element counts, but cannot enforce a
symbolic reshape product or a rank-dependent order in a parameter annotation.
Packed-scale reshapes and permutations additionally track physical layout;
`../triton_examples/test_block_scaled_matmul.py` rejects an incorrect order
and an invalid reshape stage. The final scale reshape retains its supplied
extent, rather than claiming it equals the logical K extent when the kernel's
intermediate scale-group count is only typed `int`. Host validation checks
the packed allocation and scale-array shapes; a general product-checked
reshape remains a separate type-system experiment.
For rank-two matrix pointers, column offsets require a prior row selection:
`x_row = X + row * stride` produces a rank-one pointer. Rebinding annotated
parameters with `+=` would retain their declared rank, so these examples use
fresh local names. This records a stride-matched shift but does not prove
that the selected row remains in bounds. `InOutPointer` is a generic
read/write allocation pointer used for grouped scratch and split-K scratch;
selecting a split-K slice lowers its rank from three to two, while indexed
row/column offsets produce `InOutTilePointers`, an array of addresses. Grouped
scratch selects a lower-rank pointer before constructing its address array.
The lock/count allocation retains
a separate `LockArrayPointer` role so arbitrary read/write vectors cannot be
passed to atomic lock operations. Its length is checked as `2 * Groups` at
runtime; selecting one lock yields `LockSlotPointer`, then its count address
has a distinct role. The stubs do not prove predicate
implication or track masked-lane validity through tensors.
Seeded dropout also preserves the tile width when `tl.rand` receives shifted
unit-stride offsets, so `tl.where` and `tl.store` check its value-tile shape.

Shape-aware matrix kernels use evaluable generic `tlt.InPointer` and `tlt.OutPointer`
markers with two shape and two stride parameters. The launch hook checks
these against runtime tensors and scalar arguments. Generic pointer overloads
accept the 2D addresses and return generic `InTilePointers` and
`OutTilePointers`, retaining allocation shape, strides, tile shape, and a
wrapped-axis tag. Single-axis address expressions still use row/column-specific
subclasses, which remain a migration target. Masks use the generic
shape-and-bounds type. Indirect grouped GEMM uses `PointerTable[Groups, Access]`
for arrays of device addresses and `PackedIntTable[Groups, Fields]` for flat
metadata. Loading a table entry produces a generic indirect input or output
pointer. Since each problem's dimensions and row stride come from device
memory, they remain gradual inside the kernel, while the pointer-array tile
shapes and input/output access directions remain checked.

`launch_layout.py` checks one-dimensional grid size against a checked 1D or 2D
output view and its tile shape. Its `TiledOutputLayout` carries those symbolic
shapes; `launch` validates the named dimension, block-size, and optional
metadata arguments actually passed to the kernel. Vector add and grouped
matmul use it to bind lengths/dimensions, tile sizes, and grouping metadata to
their host output and launch configuration. Softmax instead uses
`GridStrideOutputLayout`, which binds output dimensions and block width while
allowing an occupancy-dependent number of programs up to the row count.
The mapping from `tl.program_id(0)`
to each output tile is still a kernel-side contract, not a verified property
of the layout.
