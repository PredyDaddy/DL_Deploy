#include <cuda_runtime.h>
#include <stdio.h>

__global__ void demo_kernel()
{
    // 这个案例是去每一个grid里面的第一个block, 第一个block的第一个线程输出信息
    // 因为block是grid的索引，thread是block的索引
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("Run kernel. blockIdx = %d,%d,%d  threadIdx = %d,%d,%d\n",
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

void launch(int* grids, int* blocks)
{
    dim3 gird_dims(grids[0], grids[1], grids[2]);
    dim3 blocks_dims(blocks[0], blocks[1], blocks[2]);
    demo_kernel<<<gird_dims, blocks_dims, 0, nullptr>>>();
}