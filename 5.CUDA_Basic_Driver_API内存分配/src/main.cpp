#include <cuda.h>  // CUDA驱动头文件
#include <stdio.h> // 使用printf
#include <string.h>

#define MycheckDriver(op) MyCheck((op), #op, __FILE__, __LINE__)

bool MyCheck(CUresult code, const char* op, const char* file, int line)
{
    if (code != CUresult::CUDA_SUCCESS) 
    {
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s: %d %s \n", file, line, op);
        printf("错误code:  %s \n错误信息: is %s\n", err_name, err_message);
        return false;
    }
    return true;
}

int main()
{
    // 1. 检查cuda driver的初始化
    MycheckDriver(cuInit(0));

    // 2. 创建上下文
    CUcontext context = nullptr;
    CUdevice device = 0; // int device = 0;
    cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
    printf("context = %p\n", context);

    // 3. 输入device prt向设备要一个100 byte的线性内存，并返回设备指针
    // CUdeviceptr指向设备内存的指针类型
    CUdeviceptr device_memory_pointer = 0;
    cuMemAlloc(&device_memory_pointer, 100);
    printf("device_memory_pointer = %p\n", device_memory_pointer);

    // 4. 在主机上分配100个字节的锁页内存，即专门供设备访问的主机内存
    float* host_page_locked_memory = nullptr;
    cuMemAllocHost((void**)&host_page_locked_memory, 100);
    printf("host_page_locked_memory: ", host_page_locked_memory);

    // 5. 向page-locked memory 里放数据（仍在CPU上），可以让GPU可快速读取
    host_page_locked_memory[0] = 123;
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);
    
    // 6.使用cuMemsetD32函数将锁页内存中的值设置为新的值，
    // 该函数用于将内存中的数据初始化为指定值。
    float new_value = 555;
    cuMemsetD32((CUdeviceptr)host_page_locked_memory, *(int*)&new_value, 1);
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);

    // 释放锁页内存
    cuMemFreeHost((host_page_locked_memory));
    return 0;
}