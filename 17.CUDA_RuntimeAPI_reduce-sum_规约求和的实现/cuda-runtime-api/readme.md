# 1. 知识点
1. reduce思路，规约求和
    1. 将数组划分为n个块block。每个块大小为b，b设置为2的幂次方
    2. 分配b个大小的共享内存，记为float cache[b];
    3. 对每个块的数据进行规约求和：
        1. 定义plus = b / 2
        2. value = array[position], tx = threadIdx.x
        3. 将当前块内的每一个线程的value载入到cache中，即cache[tx] = value
        4. 对与tx < plus的线程，计算value += array[tx + plus]
        5. 定义plus = plus / 2，循环到3步骤
    4. 将每一个块的求和结果累加到output上（只需要考虑tx=0的线程就是总和的结果了）
2. __syncthreads，同步所有block内的线程，即block内的所有线程都执行到这一行后再并行往下执行
3. atomicAdd，原子加法，返回的是旧值
4. 规约求和我看了这篇帖子 https://blog.csdn.net/shandianfengfan/article/details/120407846
5. 

# 2. main.cpp文件详细注释版本
**整体代码流程如下: **
这段代码是一个CUDA的例子程序，用于计算一个数组中所有元素的和，并将结果与预期值（通过CPU计算得出）进行比较。整体流程如下

1. 定义数组和变量, 定义了一个包含101个元素的数组input_host和一个float类型的变量ground_truth，用于存储CPU计算得出的预期结果。
2. 初始化数组,通过循环对数组进行初始化，并同时计算CPU预期结果。
3. 分配内存并将数组数据传输到设备, 使用CUDA API中的cudaMalloc()函数分配一个设备上的数组input_device，并使用cudaMemcpy()函数将主机上的数据input_host复制到设备上的数组中。
4. 分配内存并初始化设备上的输出变量, 分配内存并初始化设备上的输出变量, 定义一个变量output_host，用于存储从设备中读取的输出结果，同时分配一个设备上的变量output_device，并使用cudaMemset()函数将其初始化为0。
5. 调用核函数计算数组元素的和, 调用launch_reduce_sum()函数，该函数启动了一个CUDA核函数，在核函数中对数组元素进行求和操作，并将结果存储在设备上的output_device中。这里还调用了cudaPeekAtLastError()函数，用于检查是否出现了运行时错误。
6. 将设备上的结果复制回主机端，并计算误差,将设备上的结果复制回主机端，并计算误差
7. 释放内存

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void launch_reduce_sum(float* input_array, int input_size, float* output);

int main(){
    // 定义数组和变量
    // 定义了一个包含101个元素的数组input_host和一个float类型的变量ground_truth，
    // ground_truth用于存储CPU计算得出的预期结果。
    const int n = 101;
    float* input_host = new float[n];
    float *input_device = nullptr;
    float ground_truth = 0; 

    // 初始化数组
    // 通过循环对数组进行初始化，并同时计算CPU预期结果
    for(int i = 0; i < 101; i++){
    input_host[i] = i;
    ground_truth += i;
    }
    printf("ground_truth = %f\n", ground_truth);

    // 分配内存并将数组传输到设备上
    checkRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    checkRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float), cudaMemcpyHostToDevice));

    // 分配内存并初始化设备上的输出变量
    // 因为这里是求和所以只用一个标量就可以了
    float output_host = 0;
    float* output_device = nullptr;
    checkRuntime(cudaMalloc(&output_device, sizeof(float)));
    // 用cudaMemset()将其初始化为0
    checkRuntime(cudaMemset(output_device, 0, sizeof(float)));

    // 调用launch_reduce_sum()函数，该函数启动了一个CUDA核函数，
    // 在核函数中对数组元素进行求和操作，并将结果存储在设备上的output_device中。
    // 这里还调用了cudaPeekAtLastError()函数，用于检查是否出现了运行时错误。
    launch_reduce_sum(input_device, n, output_device);
    checkRuntime(cudaPeekAtLastError());

    // 将结果副指挥主机端，并且计算误差
    checkRuntime(cudaMemcpy(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    // 对比， FLT_EPSILON是float类型数据的最小精度

    printf("output_host = %f, ground_truth = %f\n", output_host, ground_truth);
    if(fabs(output_host - ground_truth) <= __FLT_EPSILON__){
        printf("结果正确.\n");
    }else{
        printf("结果错误.\n");
    }
    // 释放内存
    cudaFree(input_device);
    cudaFree(output_device);

    delete [] input_host;
    printf("done\n");
    return 0;
}
```
# 3. kernel.cu文件解读
这里的代码是为了将 block_size 转换成 2 的幂次方大小。在规约求和中，为了保证每个线程的计算和数据在共享内存中的存储都是规整的，需要把 block_size 转换成 2 的幂次方大小。ceil()函数会将 block_sqrt 向上取整，并将结果转换成 int 类型，这个结果表示共享内存的大小（单位：float），即 block_size * sizeof(float)。pow()函数会根据共享内存的大小计算出最小的 2 的幂次方，作为新的 block_size 值。这样，共享内存的大小就能满足规约求和的需求，每个线程的数据也能在共享内存中占用一个整数倍的空间，方便计算。

在这个case中我们的目的是去求101个数字相加，所以明显小于512($2^9$),这里不能像之前直接启动101个线程就可以了，需要对启动的线程数进行
```cpp
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
```
```
old block_size = 101, block_sqrt = 6.66
计算过后 new block size = 128, block_sqart =  7
```
# 4. 核函数内部
```cpp
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

```
**讲解前三次的for循环**
当 i = block_size / 2 = 64 时，cache[lane] 和 value 初始值都是从输入数组 array 中读取的数据，cache[lane] 存储了线程读取到的元素值，value 存储当前线程处理的元素值。

第一轮循环开始前，假设线程 0 的 lane = 0，那么它会将输入数据存储到 cache[0] 中，即 cache[0] = value = array[0]。线程 1 的 lane = 1，会将 array[1] 存储到 cache[1] 中，即 cache[1] = value = array[1]。以此类推，所有线程都将自己读取到的元素存储到 cache 数组对应的位置上。

第一次循环结束后，线程 0 判断 lane < i，即 0 < 64，所以会将 value 加上 cache[0 + 64] 的值（cache[64]），即 value += cache[64]。此时 cache 数组的值不会变化。

第二轮循环开始前，i 除以 2 变成 32。线程 0 会将当前的 value 值存储到 cache[0] 中，即 cache[0] = value。线程 1 会将 value 存储到 cache[1] 中，即 cache[1] = value。其他线程以此类推。

第二次循环结束后，线程 0 判断 lane < i，即 0 < 32，所以会将 value 加上 cache[0 + 32] 的值（cache[32]），即 value += cache[32]。此时 cache 数组的值不会变化。

第三轮循环开始前，i 除以 2 变成 16。线程 0 会将当前的 value 值存储到 cache[0] 中，即 cache[0] = value。线程 1 会将 value 存储到 cache[1] 中，即 cache[1] = value。其他线程以此类推。

第三次循环结束后，线程 0 判断 lane < i，即 0 < 16，所以会将 value 加上 cache[0 + 16] 的值（cache[16]），即 value += cache[16]。此时 cache 数组的值不会变化。

**以此类推，一共是循环$log2(128) = 7$**次。

atomicAdd 是 CUDA 提供的一个原子操作函数，用于在多个线程同时更新同一个变量时保证操作的原子性，防止出现数据竞争的情况。

原子操作是指在执行的过程中不能被中断的操作，这种操作不会被线程调度机制打断。在多线程并发执行的情况下，如果多个线程同时修改同一块内存，可能会出现数据不一致的情况。而原子操作可以保证在多线程并发执行的情况下，对同一块内存的读写操作是原子性的，即每次只会有一个线程在读写该内存，避免了数据不一致的问题。在CUDA中，原子操作的函数是以atomic开头的一系列函数，如atomicAdd、atomicCAS等。

在这段代码中，由于多个线程都会在 if(lane == 0) 的条件满足时对 output 进行累加操作，因此需要使用 atomicAdd 来保证对 output 的操作是原子的，防止不同线程对 output 进行写操作时发生冲突导致数据不正确。

