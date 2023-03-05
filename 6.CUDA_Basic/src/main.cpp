#include <cuda.h>  // CUDA 驱动的头文件
#include <stdio.h> // 因为要使用printf
#include <string>
int main()
{
    CUresult code = cuInit(0);
    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_message = nullptr;
        cuGetErrorName(code, &err_message);      // 也可以直接获取错误代码的字符串
        cuGetErrorString(code, &err_message);    // 获取错误代码的字符串描述
    }

    // 获取CUDA驱动版本
    int driver_version = 0;
    auto code1 = cuDriverGetVersion(&driver_version);
    printf("CUDA Driver version is %d\n", driver_version);
    printf("cuDriverGetVersion code is: %d\n", code1);

    // 获取当前设备信息
    char device_name[100];      // char 数组
    CUdevice device = 0;        // typedef int CUdevice;   
    auto code2 = cuDeviceGetName(device_name, sizeof(device_name), device);
    printf("device: %d is %s\n", device, device_name);
    printf("cuDeviceGetName code is: %d\n", code2);
    return 0;
}