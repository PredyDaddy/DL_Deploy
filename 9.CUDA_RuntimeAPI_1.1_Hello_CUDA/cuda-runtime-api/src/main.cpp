# include <cuda_runtime.h> // 使用RunTime API
#include <cuda.h>          // 使用CUDA Driver API
#include <stdio.h>         // 使用printf
#include <string.h>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
{
    if (code != cudaSuccess)   // if (code !=0)
    {
        const char* err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d %s failed. \n", file, line, op);
        printf("错误code: %s\n", err_name);
        printf("错误message: \n",err_message);
        return false;
    }
    return true;
}

int main()
{
    // 这里还是CUDA Driver API
    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);
    printf("Current context = %p, 当前无context\n ", context);

    // 查看device数量
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("当前一共有%d台设备\n", device_count);

    // SetDevice自动创建context，自动cuDevicePrimaryCtxRetain(device, &context);
    int device_id = 0;
    printf("set current device to : %d, 这个API依赖CUcontext, 触发创建并设置\n", device_id);
    checkRuntime(cudaSetDevice(device_id));

    cuCtxGetCurrent(&context);
    printf("SetDevice after, Current context = %p, 获取当前context\n", context);

    // 查看当前device
    int current_device = 0;
    checkRuntime(cudaGetDevice(&current_device));
    printf("current_device = %d\n", current_device);
    return 0;
}

