#include <cuda.h> // CUDA驱动头文件cuda.h

#include <stdio.h> // 使用printf
#include <string.h>

#define checkDriver(op) Mycheck((op), #op, __FILE__, __LINE__)

bool Mycheck(CUresult code, const char* op, const char* file, int line)
{   
    if (code != CUresult::CUDA_SUCCESS) // 等同于 if(code != 0)
    {
        const char* err_name = nullptr;
        const char *err_message = nullptr;
        // 修改err_name, error_message指针，指向错误信息，报错的字符串的首地址
        cuGetErrorName(code, &err_name);      
        cuGetErrorString(code, &err_message);
        printf("%s, %d %s 失败\n", file, line, op);
        printf("错误名字: %s\n", err_name);
        printf("错误信息: %s\n", err_message);
        return false;
    }

    return true;
}

int main()
{   
    // 检查cuda driver的初始化
    checkDriver(cuInit(0));
    

    // 测试当前CUDA版本
    int driver_version = 0;
    if (!checkDriver(cuDriverGetVersion(&driver_version)) ) // if (false)
    {
        return -1;
    }
    printf("当前驱动版本是: %d\n", driver_version);

    // 测试当前设备信息
    char device_name[100];
    int device = 0;  // CUdevice device = 0

    if(!checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device)))
    {
        return -1;
    };
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}