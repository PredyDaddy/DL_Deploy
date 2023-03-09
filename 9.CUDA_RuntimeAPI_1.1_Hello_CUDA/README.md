![在这里插入图片描述](https://img-blog.csdnimg.cn/c9bc4416c65d4e20a028793b54d8ac2d.png)
# 1. CUDA Driver API和CUDA Runtime API
CUDA Driver API和CUDA Runtime API都是用于访问GPU的API。它们之间的区别在于它们的功能和使用方法不同。

CUDA Driver API是一个底层的API，它提供了对GPU硬件的底层访问，以及GPU硬件的直接控制。使用Driver API需要编写更多的底层代码，例如手动管理GPU内存分配、执行GPU kernel等。它对于需要更细粒度控制GPU的应用程序非常有用。

而CUDA Runtime API则是一个更高层次的API，它提供了对GPU硬件的更简单的访问和控制。CUDA Runtime API隐藏了大部分底层细节，例如内存管理、调度和线程同步等。这使得开发人员可以更容易地开发出GPU加速的应用程序。

总之，CUDA Driver API是一个更底层的API，提供了更大的灵活性和控制力，但需要编写更多的底层代码。而CUDA Runtime API则提供了更高层次的抽象，使得开发人员更容易地编写GPU加速的应用程序，但是在一些场景下可能会有一些性能瓶颈。开发人员需要根据具体的需求来选择使用哪种API。

# 2. 两种API的区别
1. 对于runtimeAPI，与driver最大区别是懒加载
2. 即，第一个runtime API调用时，会进行cuInit初始化，避免驱动api的初始化窘境
3. 即，第一个需要context的API调用时，会进行context关联并创建context和设置当前context，调用cuDevicePrimaryCtxRetain实现
4. 绝大部分api需要context，例如查询当前显卡名称、参数、内存分配、释放等
5. CUDA Runtime是封装了CUDA Driver的高级别更友好的API
6. 使用cuDevicePrimaryCtxRetain为每个设备设置context，不再手工管理context，并且不提供直接管理context的API（可Driver API管理，通常不需要）
7. 可以更友好的执行核函数，.cpp可以与.cu文件无缝对接
8. 对应cuda_runtime.h和libcudart.so
9. runtime api随cuda toolkit发布
10. 主要知识点是核函数的使用、线程束布局、内存模型、流的使用
11. 主要实现归约求和、仿射变换、矩阵乘法、模型后处理，就可以解决绝大部分问题

# 3. 第一个CUDA RunTime API 程序Hello CUDA
```cpp

// CUDA运行时头文件
#include <cuda_runtime.h>

// CUDA驱动头文件
#include <cuda.h>
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

    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);
    printf("Current context = %p，当前无context\n", context);

    // cuda runtime是以cuda为基准开发的运行时库
    // cuda runtime所使用的CUcontext是基于cuDevicePrimaryCtxRetain函数获取的
    // 即，cuDevicePrimaryCtxRetain会为每个设备关联一个context，通过cuDevicePrimaryCtxRetain函数可以获取到
    // 而context初始化的时机是懒加载模式，即当你调用一个runtime api时，会触发创建动作
    // 也因此，避免了cu驱动级别的init和destroy操作。使得api的调用更加容易
    int device_count = 0;
    checkRuntime(cudaGetDeviceCount(&device_count));
    printf("device_count = %d\n", device_count);

    // 取而代之，是使用setdevice来控制当前上下文，当你要使用不同设备时
    // 使用不同的device id
    // 注意，context是线程内作用的，其他线程不相关的, 一个线程一个context stack
    int device_id = 0;
    printf("set current device to : %d，这个API依赖CUcontext，触发创建并设置\n", device_id);
    checkRuntime(cudaSetDevice(device_id));

    // 注意，是由于set device函数是“第一个执行的需要context的函数”，所以他会执行cuDevicePrimaryCtxRetain
    // 并设置当前context，这一切都是默认执行的。注意：cudaGetDeviceCount是一个不需要context的函数
    // 你可以认为绝大部分runtime api都是需要context的，所以第一个执行的cuda runtime函数，会创建context并设置上下文
    cuCtxGetCurrent(&context);
    printf("SetDevice after, Current context = %p，获取当前context\n", context);

    int current_device = 0;
    checkRuntime(cudaGetDevice(&current_device));
    printf("current_device = %d\n", current_device);
    return 0;
}
```

# 4. 分解这个案例
```cpp
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
```
这里跟之前没有区别，唯一改变的就是CUresult改成了cudaError_t 这个一样，也是枚举类，0是成功的
```cpp
// 查看device数量
int device_count = 0;
cudaGetDeviceCount(&device_count);
printf("当前一共有%d台设备\n", device_count);

cuCtxGetCurrent(&context);
printf("SetDevice after, Current context = %p, 获取当前context\n", context);
```
CUDA RunTime API是不需要手动管理context, context在CUDA编程中很重要的，管理了CUDA操作的状态信息，表示CUDA操作在哪个设备上执行，包括分配的内存，执行的CUDA线程。

Driver API会需要使用cuCtxCreate函数创建和set来管理，而RuntimeAPI不需要 

cuda runtime所使用的CUcontext是基于cuDevicePrimaryCtxRetain函数获取的
即，cuDevicePrimaryCtxRetain会为每个设备关联一个context，通过cuDevicePrimaryCtxRetain函数可以获取到

因为是懒加载模式，所以调用API的时候自动创建，所以这里也不用cuInit，也不用destory释放内存，使得API调用更加简洁

注意，是由于set device函数是“第一个执行的需要context的函数”，所以他会执行cuDevicePrimaryCtxRetain并设置当前context，这一切都是默认执行的。注意：cudaGetDeviceCount是一个不需要context的函数

你可以认为绝大部分runtime api都是需要context的，所以第一个执行的cuda runtime函数，会创建context并设置上下文

这个案例里的context依然是在Driver API，只不过为了演示拿出来而已
