#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void sum_kernel(float* array, int n, float* output){
/*
array: input_deivce
n: 需要规约求和的数, 也是数组array的长度
output: output_device
*/

    int position = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用 extern声明外部的动态大小共享内存，由启动核函数的第三个参数指定
    extern __shared__ float cache[]; // 这个cache 的大小为 block_size * sizeof(float)
    int block_size = blockDim.x;
    int lane       = threadIdx.x;
    float value    = 0;

    // 判断当前线程处理的元素是否在数组内部，是的就会给value 
    if(position < n)
        value = array[position];

    for(int i = block_size / 2; i > 0; i /= 2){ // 如何理解reduce sum 参考图片：figure/1.reduce_sum.jpg
        cache[lane] = value;
        __syncthreads();  // 等待block内的所有线程储存完毕
        if(lane < i) value += cache[lane + i];
        __syncthreads();  // 等待block内的所有线程读取完毕
    }

    if(lane == 0){
        printf("block %d value = %f\n", blockIdx.x, value);
        atomicAdd(output, value); // 由于可能动用了多个block，所以汇总结果的时候需要用atomicAdd。（注意这里的value仅仅是一个block的threads reduce sum 后的结果）
    }
}

void launch_reduce_sum(float* array, int n, float* output){
/*
array: input_deivce
n: 需要规约求和的数
output: output_device
*/
    const int nthreads = 512;
    // 如果n小于512线程就启动n个线程, 这个case启动101个线程
    int block_size = n < nthreads ? n : nthreads;
    // 通用的一种grid_Size的写法
    int grid_size = (n + block_size - 1) / block_size;

    // 这里要求block_size必须是2的幂次
    float block_sqrt = log2(block_size);
    printf("old block_size = %d, block_sqrt = %.2f\n", block_size, block_sqrt);

    // 
    block_sqrt = ceil(block_sqrt);
    block_size = pow(2, block_sqrt);
    printf("计算过后 new block size = %d, block_sqart = %2.f\n", block_size, block_sqrt);

    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
    sum_kernel<<<grid_size, block_size, block_size * sizeof(float), nullptr>>>( // 这里 
        array, n, output
    ); // 这里要开辟 block_size * sizeof(float) 这么大的共享内存
}