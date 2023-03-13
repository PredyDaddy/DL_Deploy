```cpp
// CUDA运行时头文件
#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <string.h>

using namespace std;

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

__global__ void add_vector(const float* a, const float* b, float* c, int count){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= count) return;
    c[index] = a[index] + b[index];
}

__global__ void mul_vector(const float* a, const float* b, float* c, int count){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= count) return;
    c[index] = a[index] * b[index];
}

cudaStream_t stream1, stream2;
float *a, *b, *c1, *c2;
const int num_element = 100000;
const size_t bytes = sizeof(float) * num_element;
const int blocks = 512;
const int grids = (num_element + blocks - 1) / blocks;
const int ntry = 1000;

// 多个流异步
void async(){

    cudaEvent_t event_start1, event_stop1;
    cudaEvent_t event_start2, event_stop2;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));
    checkRuntime(cudaEventCreate(&event_start2));
    checkRuntime(cudaEventCreate(&event_stop2));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    checkRuntime(cudaEventRecord(event_stop1, stream1));
    
    checkRuntime(cudaEventRecord(event_start2, stream2));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream2>>>(a, b, c2, num_element);
    checkRuntime(cudaEventRecord(event_stop2, stream2));

    checkRuntime(cudaStreamSynchronize(stream1));
    checkRuntime(cudaStreamSynchronize(stream2));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1, time2;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    checkRuntime(cudaEventElapsedTime(&time2, event_start2, event_stop2));
    printf("async: time1 = %.2f ms, time2 = %.2f ms, count = %.2f ms\n", time1, time2, toc - tic);
}

// 单个流串行
void sync(){

    cudaEvent_t event_start1, event_stop1;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);
    checkRuntime(cudaEventRecord(event_stop1, stream1));

    checkRuntime(cudaStreamSynchronize(stream1));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    printf("sync: time1 = %.2f ms, count = %.2f ms\n", time1, toc - tic);
}

// 多个流之间并行
void multi_stream_async(){

    // 这个案例主要实现多个流之间互相等待，使用event控制实现
    // 存在step1  ->  step2 \ 
    //                      ->  step3   ->  step4
    //               stepa / 
    //
    // 这个案例中，存在流程1：step1 -> step2的流程
    //           存在流程2：stepa
    //           存在流程3：step3 -> step4，step3要求step2与stepa作为输入
    // 此时，可以让流程1使用stream1，流程2使用stream2，而流程3继续使用stream1，仅仅在stream1中加入等待（event的等待）

    // step1 = add_vector
    // step2 = mul_vector
    // step3 = add_vector
    // step4 = mul_vector
    // stepa = add_vector
    #define step1 add_vector
    #define step2 mul_vector
    #define step3 add_vector
    #define step4 mul_vector
    #define stepa add_vector

    cudaEvent_t event_async;
    checkRuntime(cudaEventCreate(&event_async));

    // stream1的执行流程
    step1<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    step2<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);

    // 等待event_async有事件
    checkRuntime(cudaStreamWaitEvent(stream1, event_async));
    step3<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);
    step4<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);

    // stream2的执行流程
    stepa<<<grids, blocks, 0, stream2>>>(a, b, c2, num_element);
    
    // 为event_async触发事件，通知cudaStreamWaitEvent函数可以继续了
    checkRuntime(cudaEventRecord(event_async, stream2));
    checkRuntime(cudaStreamSynchronize(stream1));

    printf("multi_stream_async done.\n");
}

int main(){

    // 本程序实现两个核函数的并行，通过多个流实现
    
    checkRuntime(cudaStreamCreate(&stream1));
    checkRuntime(cudaStreamCreate(&stream2));

    checkRuntime(cudaMalloc(&a, bytes));
    checkRuntime(cudaMalloc(&b, bytes));
    checkRuntime(cudaMalloc(&c1, bytes));
    checkRuntime(cudaMalloc(&c2, bytes));

    // 演示多流之间的异步执行
    async();

    // 演示单个流内的同步执行
    sync();

    // 演示多个流之间互相等待的操作
    multi_stream_async();
    return 0;
}
```
# 2. 单个流串行
```cpp
void sync(){

    cudaEvent_t event_start1, event_stop1;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);
    checkRuntime(cudaEventRecord(event_stop1, stream1));

    checkRuntime(cudaStreamSynchronize(stream1));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    printf("sync: time1 = %.2f ms, count = %.2f ms\n", time1, toc - tic);
}
```

```
cuda count time: 12.26, cpp count time: 12.28
```
这个函数演示了单个流中的同步执行，具体解释如下：

cudaEvent_t 是 CUDA Runtime API 中的一个结构体，定义在 cuda_runtime_api.h 中。它用于表示一个 CUDA 事件对象，用于记录 GPU 上某个时间点的状态。

CUDA 事件可以用于两种目的：

记录一个时间点（如开始时间点或结束时间点）。
记录一个时间间隔（即时间差）。
通常情况下，CUDA 事件被用于在主机和设备之间进行同步，或在设备内部进行同步。例如，可以在主机代码中调用 cudaEventRecord() 来记录一个事件，然后在设备代码中使用 cudaStreamWaitEvent() 等待该事件，以确保某些设备操作发生在之前记录的事件之后。又或者，可以在设备代码中记录两个事件，然后在主机代码中使用 cudaEventElapsedTime() 计算它们之间的时间差。

首先创建两个事件 event_start1 和 event_stop1，用于记录同步执行的时间；

使用 cudaEventRecord 将 event_start1 记录在 stream1 中，表示从这个时间点开始，将会执行在 stream1 中的操作；

使用 for 循环调用 add_vector 核函数，在 stream1 中执行 ntry 次，计算向量 a 和 b 的加和，存储在向量 c1 和 c2 中；

使用 cudaEventRecord 将 event_stop1 记录在 stream1 中，表示到达这个时间点，stream1 中的操作都已经完成；

使用 cudaStreamSynchronize 等待 stream1 中的所有操作执行完毕；

计算同步执行的时间 time1，并输出时间和整个操作的时间。

可以看到，这个函数中只使用了一个流，因此 add_vector 的计算是按照顺序执行的，不能充分发挥 GPU 的并行计算能力。因此，这个函数的计算时间会比异步执行的 async 函数要长

这段代码中使用了两种方法来计算代码执行的时间。

第一种方法是使用了C++标准库中的chrono库来计算代码执行的起始时间和终止时间，通过计算时间差得到代码执行的时间，这个方法在计算异步执行时比较方便，因为我们需要分别记录多个异步操作的起始时间和终止时间。

第二种方法是使用了CUDA提供的API cudaEventElapsedTime，这个API可以计算CUDA事件的时间差，用于计算CUDA事件执行的时间。在这个例子中，我们使用了这个API来计算在单个流上串行执行的时间。

# 3. 向量相加相乘的kernel function
```cpp
__global__ void add_vector(const float* a, const float* b, float* c, int count){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= count) return;
    c[index] = a[index] + b[index];
}

__global__ void mul_vector(const float* a, const float* b, float* c, int count){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= count) return;
    c[index] = a[index] * b[index];
}
```
count 是用来限制线程不要访问到超出数组的地址，因为数组的长度在我们开辟的时候就已经定义好了
```cpp
checkRuntime(cudaMalloc(&a, bytes)); 
```
count是num_element, byte是**num_element * sizeof(float)**, 超出地址会访问到虚拟地址

# 4. 多个流的异步
```cpp
void async(){

    cudaEvent_t event_start1, event_stop1;
    cudaEvent_t event_start2, event_stop2;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));
    checkRuntime(cudaEventCreate(&event_start2));
    checkRuntime(cudaEventCreate(&event_stop2));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    checkRuntime(cudaEventRecord(event_stop1, stream1));
    
    checkRuntime(cudaEventRecord(event_start2, stream2));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream2>>>(a, b, c2, num_element);
    checkRuntime(cudaEventRecord(event_stop2, stream2));

    checkRuntime(cudaStreamSynchronize(stream1));
    checkRuntime(cudaStreamSynchronize(stream2));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1, time2;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    checkRuntime(cudaEventElapsedTime(&time2, event_start2, event_stop2));
    printf("async: time1 = %.2f ms, time2 = %.2f ms, count = %.2f ms\n", time1, time2, toc - tic);
}
```

```
async: time1 = 6.97 ms, time2 = 6.94 ms, count = 9.32 ms
```
输出的内容中包含了在两个流上异步执行的两个内核函数的时间，分别为time1和time2，它们的值应该是相当接近的。同时，输出中还包含了整个函数执行的总时间count，可以看出相比于同步执行的情况，异步执行使得程序的总执行时间更短，效率更高。

# 5. 多个流之间互相等待的操作
```cpp
// 这个案例主要实现多个流之间互相等待，使用event控制实现
    // 存在step1  ->  step2 \ 
    //                      ->  step3   ->  step4
    //               stepa / 
    //
    // 这个案例中，存在流程1：step1 -> step2的流程
    //           存在流程2：stepa
    //           存在流程3：step3 -> step4，step3要求step2与stepa作为输入
    // 此时，可以让流程1使用stream1，流程2使用stream2，而流程3继续使用stream1，仅仅在stream1中加入等待（event的等待）

    // step1 = add_vector
    // step2 = mul_vector
    // step3 = add_vector
    // step4 = mul_vector
    // stepa = add_vector
    #define step1 add_vector
    #define step2 mul_vector
    #define step3 add_vector
    #define step4 mul_vector
    #define stepa add_vector

    cudaEvent_t event_async;
    checkRuntime(cudaEventCreate(&event_async));

    // stream1的执行流程
    step1<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    step2<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);

    // 等待event_async有事件
    checkRuntime(cudaStreamWaitEvent(stream1, event_async));
    step3<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);
    step4<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);

    // stream2的执行流程
    stepa<<<grids, blocks, 0, stream2>>>(a, b, c2, num_element);
    
    // 为event_async触发事件，通知cudaStreamWaitEvent函数可以继续了
    checkRuntime(cudaEventRecord(event_async, stream2));
    checkRuntime(cudaStreamSynchronize(stream1));

    printf("multi_stream_async done.\n");
```
具体流程如下：

在stream1中先执行step1，然后执行step2，这两个步骤是串行执行的；

在stream1中调用cudaStreamWaitEvent函数等待event_async事件，此时流程3（step3和step4）还不能开始执行；

在stream2中执行stepa，此时stepa和之前的步骤是并行执行的；

在stream2中调用cudaEventRecord函数触发event_async事件，通知stream1可以开始执行流程3；

在stream1中执行step3和step4，这两个步骤是串行执行的；

在stream1中调用cudaStreamSynchronize函数等待所有在该流中的操作执行完毕，程序结束。

总结起来，这个多流程的示例展示了如何使用事件来控制不同流之间的顺序和同步，从而实现流程之间的依赖关系和并行执行。