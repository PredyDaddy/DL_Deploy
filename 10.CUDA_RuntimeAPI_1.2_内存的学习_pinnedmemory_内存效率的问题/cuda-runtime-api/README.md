# 1. 前言
1. 主要理解pinned memory、global memory、shared memory即可

# 2. 主机内存
1. 主机内存很多名字: CPU内存，pinned内存，host memory，这些都是储存在内存条上的
2. Pageable Memory(可分页内存) + Page lock Memory(页锁定内存) 共同组成内存
3. 你可以理解为Page lock memory是vip房间，锁定给你一个人用。而Pageable memory是普通房间，在酒店房间不够的时候，选择性的把你的房间腾出来给其他人交换用，这就可以容纳更多人了。造成房间很多的假象，代价是性能降低
# 3. 页锁定内存 (pinned memory/Page lock Memory)
1. pinned memory具有锁定性，是稳定不会被交换的
2. pageable memory没有锁定特性，对于第三方设备（比如GPU），去访问时，因为无法感知内存是否被交换，可能得不到正确的数据（每次去房间找，说不准你的房间被人交换了）
3. pageable memory的性能比pinned memory差，很可能降低你程序的优先级然后把内存交换给别人用
4. pageable memory策略能使用内存假象，实际8GB但是可以使用15GB，提高程序运行数量（不是速度）
5. pinned memory太多，会导致操作系统整体性能降低（程序运行数量减少），8GB就只能用8GB。注意不是你的应用程序性能降低，这一点一般都是废话，不用当回事
6. GPU可以直接访问pinned memory而不能访问pageable memory（因为第二条）


# 4. 内存总结:
1. GPU可以直接访问pinned memory，称之为（DMA Direct Memory Access）
2. 对于GPU访问而言，距离计算单元越近，效率越高，所以PinnedMemory<GlobalMemory<SharedMemory
3. 代码中，由new、malloc分配的，是pageable memory，由cudaMallocHost分配的是PinnedMemory，由cudaMalloc分配的是GlobalMemory
4. 尽量多用PinnedMemory储存host数据，或者显式处理Host到Device时，用PinnedMemory做缓存，都是提高性能的关键

# 5. 案例代码
```cpp
// CUDA运行时头文件
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

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

int main(){

    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    // 分配global memory
    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); // pointer to device

    // 分配pageable memory
    float* memory_host = new float[100];
    memory_host[2] = 520.25;
    checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice)); // 返回的地址是开辟的device地址，存放在memory_device

    // 分配pinned memory page locked memory
    float* memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float))); // 返回的地址是被开辟的pin memory的地址，存放在memory_page_locked
    checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost)); // 

    printf("%f\n", memory_page_locked[2]);
    checkRuntime(cudaFreeHost(memory_page_locked));
    delete [] memory_host;
    checkRuntime(cudaFree(memory_device)); 

    return 0;
}
```

# 6. 案例代码分段解析
```cpp
int device_id = 0;
cudaSetDevice(device_id); // 如果不写device_id = 0
```
是由于set device函数是“第一个执行的需要context的函数”，所以他会执行cuDevicePrimaryCtxRetain。

如果不指定设备ID，则默认使用设备ID为0的设备。在调用其他CUDA API函数之前，通常需要先调用cudaSetDevice来设置当前线程要使用的GPU设备。

```cpp
float *memory_device = nullptr;
checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); // pointer to device
```
分配global memory需要使用cudaMalloc() 这里使用cudaMalloc()函数在GPU上分配一块100个float类型元素的内存，返回一个指向设备内存的指针memory_device。

```cpp
// 分配pageable memory
float* memory_host = new float[100];
memory_host[2] = 520.25;
checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice)); // 返回的地址是开辟的device地址，存放在memory_device
```
**分配pageable memory(主机上的内存)**， 使用new运算符在主机上分配一块100个float类型元素的内存，返回一个指向主机内存的指针memory_host。使用new运算符在主机上分配一块100个float类型元素的内存，返回一个指向主机内存的指针memory_host。

```cpp
// 分配pinned memory page locked memory
float* memory_page_locked = nullptr;
checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float))); // 返回的地址是被开辟的pin memory的地址，存放在memory_page_locked
checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost)); 
```
**分配pinned memory page locked memory**使用cudaMallocHost()函数在主机上分配一块100个float类型元素的锁页内存(pinned memory),返回一个指向锁页内存的指针memory_page_locked。
将memory_device中的数据复制到memory_page_locked中，使用cudaMemcpy()函数，该函数将memory_device的数据从设备复制到主机的锁页内存。

```cpp
printf("%f\n", memory_page_locked[2]);
checkRuntime(cudaFreeHost(memory_page_locked));
delete [] memory_host;
checkRuntime(cudaFree(memory_device)); 
```
输出memory_page_locked[2]的值，即设备内存的第三个元素的值（数组下标从0开始）。

使用cudaFreeHost()函数释放锁页内存。

使用delete[]运算符释放主机内存。

使用cudaFree()函数释放设备内存。