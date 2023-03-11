#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// 全部的函数 exp std::io 英伟达GPU都封装好了
__device__ __host__ float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

__global__ void test_print_kernel(const float *pdata, int ndata)
{

    // 内置变量
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /*    dims                 indexs
        gridDim.z    1        blockIdx.z      0
        gridDim.y    1        blockIdx.y      0
        gridDim.x    1        blockIdx.x      0
        blockDim.z   1        threadIdx.z     0
        blockDim.y   1        threadIdx.y     0
        blockDim.x   10       threadIdx.x    0-9

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
__host__ void test_print(const float *pdata, int ndata)
{

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
    test_print_kernel<<<dim3(1), dim3(20), 0, nullptr>>>(pdata, ndata);

    // test_print_kernel<<<1, ndata, 0, nullptr>>>(pdata, ndata);

    // 在核函数执行结束后，通过cudaPeekAtLastError获取得到的代码，来知道是否出现错误
    // cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
    // cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
    // 而cudaPeekAtLastError是获取当前错误，但是再一次执行 cudaPeekAtLastError 或者 cudaGetLastError 拿到的还是那个错
    // cuda的错误会传递，如果这里出错了，不移除。那么后续的任意api的返回值都会是这个错误，都会失败
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);
    }
}