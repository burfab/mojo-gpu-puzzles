from sys import sizeof, argv
from testing import assert_equal
from gpu.host import DeviceContext

from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async
from layout.tensor_builder import LayoutTensorBuild as tb

# ANCHOR: naive_matmul
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor



alias TPB = 3
alias SIZE = 2
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, TPB)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


fn naive_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 6 lines)
    if row < size and col < size:
        s : output.element_type = 0
        @parameter
        for i in range(size):
            s += (a[row, i] * b[i,col])
        output[row, col] = s


# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
fn single_block_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    local_row = thread_idx.y
    local_col = thread_idx.x
    # FILL ME IN (roughly 12 lines)
    shared_a = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    shared_b = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    if row < size and col < size:
        shared_a[local_row, local_col] = a[row, col]
        shared_b[local_row, local_col] = b[row, col]
    else:
        shared_a[local_row, local_col] = 0
        shared_b[local_row, local_col] = 0
    barrier()

    s : output.element_type = 0
    @parameter
    for i in range(TPB):
        s += (shared_a[row, i] * shared_b[i,col])
    if row < size and col < size:
        output[row, col] = s


# ANCHOR_END: single_block_matmul

# ANCHOR: matmul_tiled
alias SIZE_TILED = 8
alias BLOCKS_PER_GRID_TILED = (3, 3)  # each block convers 3x3 elements
alias THREADS_PER_BLOCK_TILED = (TPB, TPB)
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED)


fn matmul_tiled_manual_tiling[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + thread_idx.y
    tiled_col = block_idx.x * TPB + thread_idx.x
    # FILL ME IN (roughly 20 lines)
    shared_a = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    shared_b = tb[dtype]().row_major[TPB, TPB]().shared().alloc()


    s : output.element_type = 0
    
    @parameter
    for stride in range((size+TPB-1) // TPB):
        load_col = local_col + stride * TPB
        if tiled_row < size and load_col < size:
            shared_a[local_row, local_col] = a[tiled_row, load_col]
        else: 
            shared_a[local_row, local_col] = 0

        load_row = local_row + stride * TPB
        if load_row < size and tiled_col < size:
            shared_b[local_row, local_col] = b[load_row, tiled_col]
        else: 
            shared_b[local_row, local_col] = 0

        barrier()
        @parameter
        for i in range(TPB):
            s += shared_a[local_row, i] * shared_b[i, local_col]

        barrier()

    if tiled_row < size and tiled_col < size:
        output[tiled_row, tiled_col] = s
    
fn matmul_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + thread_idx.y
    tiled_col = block_idx.x * TPB + thread_idx.x
    # FILL ME IN (roughly 20 lines)
    shared_a = tb[dtype]().row_major[TPB, TPB]().shared().alloc().fill(0)
    shared_b = tb[dtype]().row_major[TPB, TPB]().shared().alloc().fill(0)

    tile_o = output.tile[TPB, TPB](block_idx.y, block_idx.x)


    alias load_a_layout = Layout.row_major(1, TPB)  # Coalesced loading
    alias load_b_layout = Layout.row_major(1, TPB)  # Coalesced loading 
    
    s : output.element_type = 0

    @parameter
    for stride in range((size+TPB-1) // TPB):
        tile_a = a.tile[TPB, TPB](block_idx.y, stride)
        tile_b = b.tile[TPB, TPB](stride, block_idx.x)

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async[thread_layout=load_a_layout](shared_a, tile_a)
        copy_dram_to_sram_async[thread_layout=load_b_layout](shared_b, tile_b)

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        @parameter
        for i in range(TPB):
            s += shared_a[local_row, i] * shared_b[i, local_col]

        barrier()

    if tiled_row < size and tiled_col < size:
        tile_o[local_row, local_col] = s



# ANCHOR_END: matmul_tiled


def main():
    with DeviceContext() as ctx:
        if len(argv()) != 2 or argv()[1] not in [
            "--naive",
            "--single-block",
            "--tiled",
        ]:
            raise Error(
                "Expected one argument: '--naive', '--single-block', or"
                " '--tiled'"
            )
        size = SIZE_TILED if argv()[1] == "--tiled" else SIZE
        out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp1 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp2 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            size * size
        ).enqueue_fill(0)
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    val = row * size + col
                    # row major: placing elements row by row
                    inp1_host[row * size + col] = val
                    inp2_host[row * size + col] = Float32(2.0) * val

            # inp1 @ inp2.T
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        out_tensor = LayoutTensor[mut=False, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr())

        if argv()[1] == "--naive":
            ctx.enqueue_function[naive_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--single-block":
            ctx.enqueue_function[single_block_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif argv()[1] == "--tiled":
            # Need to update the layout of the tensors to the tiled layout
            out_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                out.unsafe_ptr()
            )
            a_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp1.unsafe_ptr()
            )
            b_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp2.unsafe_ptr()
            )

            ctx.enqueue_function[matmul_tiled[layout_tiled, SIZE_TILED]](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,
                block_dim=THREADS_PER_BLOCK_TILED,
            )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for col in range(size):
                for row in range(size):
                    assert_equal(
                        out_host[col * size + row], expected[col * size + row]
                    )
