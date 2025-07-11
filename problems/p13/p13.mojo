from sys import sizeof
from testing import assert_almost_equal
from gpu.host import DeviceContext

# ANCHOR: axis_sum
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb


alias TPB = 32
alias BATCH = 65_535
alias SIZE = 32
alias BLOCKS_PER_GRID = (1, BATCH)
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias in_layout = Layout.row_major(BATCH, SIZE)
alias out_layout = Layout.row_major(BATCH, 1)


fn axis_sum[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    batch = block_idx.y
    # FILL ME IN (roughly 15 lines)
    shared_a = tb[dtype]().row_major[TPB]().shared().alloc()
    if local_i < size and batch < BATCH: shared_a[local_i] = a[batch, local_i]
    else: shared_a[local_i] = 0

    fn determine_number_of_steps(var tpb:Int) -> Int:
        var cnt : Int = 0
        while tpb:
            tpb = tpb // 2
            cnt+=1
        return cnt


    alias STEPS = determine_number_of_steps(TPB)
    step = 1
    @parameter
    for _ in range(STEPS):
        barrier()
        if local_i - step >= 0:
            shared_a[local_i] = shared_a[local_i] + shared_a[local_i - step]
        step *= 2
    if local_i == TPB - 1 and batch < BATCH:
        output[batch, 0] = shared_a[local_i]


# ANCHOR_END: axis_sum


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](BATCH).enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](BATCH * SIZE).enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    inp_host[row * SIZE + col] = row * SIZE + col

        out_tensor = LayoutTensor[mut=False, dtype, out_layout](
            out.unsafe_ptr()
        )
        inp_tensor = LayoutTensor[mut=False, dtype, in_layout](inp.unsafe_ptr())

        ctx.enqueue_function[axis_sum[in_layout, out_layout]](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](BATCH).enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    expected[row] += inp_host[row * SIZE + col]

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out)
            print("expected:", expected)
            for i in range(BATCH):
                assert_almost_equal(out_host[i], expected[i])
