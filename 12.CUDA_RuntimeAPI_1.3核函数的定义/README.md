# 1. 核函数的
核函数是cuda编程的关键

通过xxx.cu创建一个cudac程序文件，并把cu交给nvcc编译，才能识别cuda语法

__global__表示为核函数，由host调用。__device__表示为设备函数，由device调用

__host__表示为主机函数，由host调用。__shared__表示变量为共享变量

host调用核函数：function<<<gridDim, blockDim, sharedMemorySize, stream>>>(args…);

Stream: 流   gridDim blockDim 告诉function启动多少个线程

只有__global__修饰的函数才可以用<<<>>>的方式调用

调用核函数是传值的，不能传引用，可以传递类、结构体等，核函数可以是模板返回值必须是void

核函数的执行，是异步的，也就是立即返回的

线程layout主要用到blockDim、gridDim

核函数内访问线程索引主要用到threadIdx、blockIdx、blockDim、gridDim这些内置变量

# 2. main.cpp文件
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

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

void test_print(const float* pdata, int ndata);

int main(){

    // 定义device指针和host指针
    float* parray_host = nullptr;
    float *parray_device = nullptr;
    int narray = 10;
    int array_bytes = sizeof(float) * narray;

    // 开辟GPU的内存，cudaMalloc返回的指针指向GPU
    checkRuntime(cudaMalloc(&parray_device, array_bytes));

    // 开辟主机内存
    parray_host = new float[narray];

    // 往主机内存放进10个数字
    for (int i = 0; i < narray; i++){
        parray_host[i] = i;
    }

    // 把主机的内存复制上去
    checkRuntime(cudaMemcpy(parray_device, parray_host, array_bytes,cudaMemcpyHostToDevice));
    
    // 把在GPU的东西打印出来
    test_print(parray_device, narray);

    checkRuntime(cudaDeviceSynchronize());

    // 释放device内存, 释放host内存
    checkRuntime(cudaFree(parray_device));
    delete[] parray_host;
    return 0;
}
```
这段代码首先声明了两个指针变量 parray_host 和 parray_device 以及一个整数变量 narray，它们都将在之后的代码中被用到。

然后，它为 parray_host 在主机上分配了 narray 个 float 类型的空间，这些空间将被用于存储输入数据。之后，它使用 cudaMalloc 函数在设备上为 parray_device 分配了与主机端相同的空间大小，也即 narray 个 float 类型的空间，这些空间将被用于存储设备端的数据。

接下来，使用一个 for 循环，将主机端的输入数据逐个赋值给 parray_host。

接着，使用 cudaMemcpy 函数，将主机端的 parray_host 数据异步地复制到设备端的 parray_device 数据空间。

随后，它调用了 test_print 函数，将 parray_device 和 narray 作为参数传入该函数。test_print 函数用于打印出设备端的数据内容。

最后，它使用 cudaDeviceSynchronize 函数来同步设备和主机，等待设备端的计算完成，防止程序过早退出，以便后面释放设备和主机上分配的空间，避免内存泄漏。然后，释放了 parray_device 和 parray_host 分配的空间，程序结束。

# 3.kernel.cu文件
```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// 全部的函数 exp std::io 英伟达GPU都封装好了
__device__ __host__ float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

__global__ void test_print_kernel(const float* pdata, int ndata){

    // 内置变量 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /*    dims                 indexs
        gridDim.z    1        blockIdx.z      0
        gridDim.y    1        blockIdx.y      0
        gridDim.x    1        blockIdx.x      0
        blockDim.z   1        threadIdx.z     0
        blockDim.y   1        threadIdx.y     0
        blockDim.x   10        threadIdx.x    0-9

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    float y = sigmoid(0.5f);
    printf("Element[%d] = %f, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", idx, pdata[idx], threadIdx.x, blockIdx.x, blockDim.x);
}

// 这个__host__写不写都是一样的，他就是一个设备函数
__host__ void test_print(const float* pdata, int ndata){

    float y = sigmoid(0.5f);
    // <<<gridDim, blockDim, bytes_of_shared_memory, stream>>>
    dim3 gridDim;
    dim3 blockDim;
    // 总线程数
    int nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;

    // 为什么补nthreads = 10; ?
    // 我们会遇到多维度的问题，
    // 这些是两个Dim的极限了
    // gridDim(21亿, 65536, 65536) 
    // blockDim(1024, 64, 64) blockDim.x * blockDim.y * blockDim.z <= 1024;

    // nullptr这里是默认流，想要异步操作就放个stream
    test_print_kernel<<<dim3(1), dim3(ndata), 0, nullptr>>>(pdata, ndata);

    // test_print_kernel<<<1, ndata, 0, nullptr>>>(pdata, ndata);

    // 在核函数执行结束后，通过cudaPeekAtLastError获取得到的代码，来知道是否出现错误
    // cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
    // cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
    // 而cudaPeekAtLastError是获取当前错误，但是再一次执行 cudaPeekAtLastError 或者 cudaGetLastError 拿到的还是那个错
    // cuda的错误会传递，如果这里出错了，不移除。那么后续的任意api的返回值都会是这个错误，都会失败
    cudaError_t code = cudaPeekAtLastError();
    if(code != cudaSuccess){    
        const char* err_name    = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);   
    }
}
```
NVCC 编译器其实封装好了全部的C++的特性，在此之上还封装了CUDA C++的特性

__device__ 让设备调用 __host__让主机调用

如果想让函数被主机和设备调用，前面加上修饰词 __device__ __host__

为什么这个地方不用nthreads = 10; 而是这种操作方式 
因为我们其实是会遇到多维度的问题

核函数的nullptr是默认流，想要异步操作就要用stream

test_print_kernel<<<dim3(1), dim3(20), 0, nullptr>>>(pdata, ndata);

dim3 是 CUDA 提供的表示三维向量的类型，它有 3 个成员变量 x、y、z 分别表示向量的三个分量。

dim3(1) 表示只有一个线程块，即在 x 方向上只有一个线程块；dim3(20) 表示在每个线程块中有 20 个线程，即在 x 方向上有 20 个线程。

因此，这个核函数总共有 1 * 20 = 20 个线程。


