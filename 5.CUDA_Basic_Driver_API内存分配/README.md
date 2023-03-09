# 本小节思路
- **将主机内存分配到锁页内存，可以使得GPU在需要访问主机内存时能够更快地访问，提高程序的性能。这是因为锁页内存的特殊性质使得GPU能够直接访问其所在的物理内存地址，避免了主机和设备之间的数据传输。**

# 1. 整体代码
```cpp
// CUDA驱动头文件cuda.h
#include <cuda.h>

#include <stdio.h>
#include <string.h>

#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){

    if(code != CUresult::CUDA_SUCCESS){    
        const char* err_name = nullptr;    
        const char* err_message = nullptr;  
        cuGetErrorName(code, &err_name);    
        cuGetErrorString(code, &err_message);   
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

int main(){

    // 检查cuda driver的初始化
    checkDriver(cuInit(0));

    // 创建上下文
    CUcontext context = nullptr;
    CUdevice device = 0;
    checkDriver(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
    printf("context = %p\n", context);

    // 输入device prt向设备要一个100 byte的线性内存，并返回地址
    CUdeviceptr device_memory_pointer = 0;
    checkDriver(cuMemAlloc(&device_memory_pointer, 100)); // 注意这是指向device的pointer, 
    printf("device_memory_pointer = %p\n", device_memory_pointer);

    // 输入二级指针向host要一个100 byte的锁页内存，专供设备访问。参考 2.cuMemAllocHost.jpg 讲解视频：https://v.douyin.com/NrYL5KB/
    float* host_page_locked_memory = nullptr;
    checkDriver(cuMemAllocHost((void**)&host_page_locked_memory, 100));
    printf("host_page_locked_memory = %p\n", host_page_locked_memory);

    // 向page-locked memory 里放数据（仍在CPU上），可以让GPU可快速读取
    host_page_locked_memory[0] = 123;
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);
    /* 
        记住这一点
        host page locked memory 声明的时候为float*型，可以直接转换为device ptr，这才可以送给cuda核函数（利用DMA(Direct Memory Access)技术）
        初始化内存的值: cuMemsetD32 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N )
        初始化值必须是无符号整型，因此需要将new_value进行数据转换：
        但不能直接写为:(int)value，必须写为*(int*)&new_value, 我们来分解一下这条语句的作用：
        1. &new_value获取float new_value的地址
        (int*)将地址从float * 转换为int*以避免64位架构上的精度损失
        *(int*)取消引用地址，最后获取引用的int值
     */
    
    float new_value = 555;
    checkDriver(cuMemsetD32((CUdeviceptr)host_page_locked_memory, *(int*)&new_value, 1)); //??? cuMemset用来干嘛？
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);

    // 释放内存
    checkDriver(cuMemFreeHost(host_page_locked_memory));
    return 0;
}
```

# 2. CUdeviceptr 复习数据类型
```cpp
// 3. 输入device prt向设备要一个100 byte的线性内存，并返回设备指针
// CUdeviceptr指向设备内存的指针类型
CUdeviceptr device_memory_pointer = 0;
cuMemAlloc(&device_memory_pointer, 100);
printf("device_memory_pointer = %p\n", device_memory_pointer);
```
- 这段在GPU分配出来的内存可以储存 100 / 4 = 25 个元素，其中每个元素为 4 字节，因为是分配给 GPU 使用的线性内存，所以数据类型为 unsigned int 或 float。
```cpp 
typedef unsigned long long CUdeviceptr;
```
- 有符号整型可以表示负数、零和正数。在C语言中，有符号整型可以使用 signed 关键字来定义，如果没有指定类型，默认是 int 类型。

- 无符号整型只能表示零和正数，不能表示负数。在C语言中，无符号整型可以使用 unsigned 关键字来定义，如果没有指定类型，默认是 unsigned int 类型。

- 在本例中，unsigned long long 是一种无符号的整型数据类型，它可以存储更大的非负整数，因为它使用所有的位来存储值，不需要用一位来表示符号。
- **CUdeviceptr被定义为unsigned long long类型，但实际上它是一个指向设备内存的指针类型。在CUDA程序中，CUdeviceptr通常用于表示设备上的内存地址。所以初始化的时候是用0不是空指针**
- CUdeviceptr device_memory_pointer = 0;
- unsigned long long device_memory_pointer = 0;

# 3. 锁页内存
- 专门供设备访问的主机内存
- 锁页内存的特性也保证了这些内存不会被操作系统或者其他进程回收或者交换到磁盘上，从而保证了数据的安全性和一致性。GPU可以通过DMA（Direct Memory Access）直接从锁页内存中读取数据，而CPU则不能直接访问锁页内存，需要使用GPU作为中介进行访问
- 代码
```cpp
float* host_page_locked_memory = nullptr;
cuMemAllocHost((void**)&host_page_locked_memory, 100);
printf("host_page_locked_memory: ", host_page_locked_memory);
```
- 这段代码是在主机上要了100字节，可存储100/sizeof(float) = 100 / 4 = 25个元素
- 在这段代码中，host_page_locked_memory 被定义为 float* 类型是因为它将用于在主机上存储一段连续的浮点数数据。如果将其定义为 void* 类型，则无法直接进行浮点数数据的访问和操作。
- (void**)是一个二级指针，它的作用是将一个 void* 类型的指针的地址传递给函数。在这个例子中，cuMemAllocHost 函数期望的参数类型是 (void**) 类型的指针，因为该函数需要为其分配的内存返回指向该内存的指针。由于 host_page_locked_memory 的类型是 float*，因此需要将其地址强制转换为 (void**) 类型的指针，以便将其传递给 cuMemAllocHost 函数。在函数调用结束后，host_page_locked_memory 将包含指向分配的内存的指针。
- host_page_locked_memory是一个float*类型的指针。如果直接将&host_page_locked_memory传递给cuMemAllocHost函数，类型不匹配，编译器会报错。因此，需要使用强制类型转换将&host_page_locked_memory的类型从float**转换为void**。这样，cuMemAllocHost函数才能正确地接收这个参数。
- **通过cuMemAllocHost分配100字节内存后，host_page_locked_memory从一个空指针变成了一个指向100字节内存的指针。**
- 为什么锁页内存GPU可以快速读取？
- **锁页内存是指在操作系统中锁定的内存页，可以被CPU和GPU共享访问。在将锁页内存传输到GPU时，由于已经锁定了内存页，因此GPU不需要等待内存传输完成，即可开始访问内存数据，这样可以显著减少CPU与GPU之间的数据传输延迟。因此在需要频繁地进行CPU和GPU之间的数据传输时，使用锁页内存可以提高数据传输的效率和速度。**

# 4. cuMemset
1. 代码
```cpp
float new_value = 555;
cuMemsetD32((CUdeviceptr)host_page_locked_memory, *(int*)&new_value, 1);
printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);
```
2. (CUdeviceptr)host_page_locked_memory  
- 把主机上的锁页内存地址 host_page_locked_memory 转换成设备上的地址，即将 host_page_locked_memory 强制转换为 CUdeviceptr 类型。这个转换是为了让 cuMemsetD32 函数能够使用主机上的锁页内存来进行设备内存的设置。
- 把主机上的锁页内存地址 host_page_locked_memory 转换成设备上的地址，即将 host_page_locked_memory 强制转换为 CUdeviceptr 类型。这个转换是为了让 cuMemsetD32 函数能够使用主机上的锁页内存来进行设备内存的设置。

3. *(int*)&new_value
- (int*)将地址从float * 转换为int*以避免64位架构上的精度损失
- *(int*)取消引用地址，最后获取引用的int值

# 5. 释放锁页内存
1. cuMemFreeHost((host_page_locked_memory)); 
