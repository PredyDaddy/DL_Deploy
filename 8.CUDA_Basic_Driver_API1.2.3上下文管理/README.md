# 1. CUcontext上下文管理
1. context是一种上下文，关联对GPU的所有操作
2. context与一块显卡关联，一个显卡可以被多个context关联
3. 每个线程都有一个栈结构储存context，栈顶是当前使用的context，对应有push、pop函数操作context的栈，所有api都以当前context为操作目标
4. 试想一下，如果执行任何操作你都需要传递一个device决定送到哪个设备执行，得多麻烦

# 2. 从两段代码看context的效率
1. 不带context的版本
```cpp
cuMalloc(device, &ptr, 100); 
cuFree(device);
cuMemcpy(device, dst, src, 100);
```
- 理解这段代码
- cuMalloc(device, &ptr, 100);：在CUDA设备上分配100字节的内存空间，并将其地址存储在指针变量ptr中。

- cuFree(device);：释放在CUDA设备上分配的内存空间。

- cuMemcpy(device, dst, src, 100);：将CUDA设备上源地址src处的100字节数据复制到目的地址dst处，其中device表示目标设备。
- 
2. 带context的版本
```cpp
cuCreateContext(device, &context);
cuPushCurrent(context);
cuMalloc(&ptr, 100);
cuFree(ptr);
cuMemcpy(dst, drc, 100);
cuPopCurrent(context);
```

4. **context只是为了方便管理设备的一种手段**
5. **用栈的结构只是为了方便管理更多的设备，使用栈不会出现被重置的现象**
6. 使用栈管理context可以方便地跟踪和管理多个context的创建和销毁。在一个程序中可能会使用多个context，如果不进行管理，可能会出现内存泄漏、资源浪费等问题。使用栈可以很方便地实现先进后出的context管理方式，而不需要手动跟踪和释放每一个context。同时，栈还具有自动管理内存的特性，当一个context出栈时，其对应的内存也会被自动释放，避免了手动释放内存的麻烦。

# 3. 还是复杂，继续简化
1. 基本上高频的使用是一个线程固定访问一个显卡，只使用一个context，所以Create, push, pop这一套流程看起来就很复杂了
2. 因此推出了cuDevicePrimaryCtxRetain，为设备关联主context，分配、释放、设置、栈都不用你管
3. primaryContext：给我设备id，给你context并设置好，此时一个显卡对应一个primary context
4. 不同线程，只要设备id一样，primary context就一样。context是线程安全的
5. 上面代码再简化
```cpp
cuDevicePrimaryCtxRetain(device, &context);
cuMalloc(&ptr, 100);
cuFree(ptr);
cuMemcpy(dst, src, 100);
```
6. 学习这个是因为runtimeAPI自动使用cuDevicePrimaryCtxRetain

# 4. 代码
```cpp
// CUDA驱动头文件cuda.h
#include <cuda.h>   // include <> 和 "" 的区别    
#include <stdio.h>  // include <> : 标准库文件 
#include <string.h> // include "" : 自定义文件  详细情况请查看 readme.md -> 5

#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){
    if(code != CUresult::CUDA_SUCCESS){    // 如果 成功获取CUDA情况下的返回值 与我们给定的值(0)不相等， 即条件成立， 返回值为flase
        const char* err_name = nullptr;    // 定义了一个字符串常量的空指针
        const char* err_message = nullptr;  
        cuGetErrorName(code, &err_name);    
        cuGetErrorString(code, &err_message);   
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message); //打印错误信息
        return false;
    }
    return true;
}

int main(){

    // 检查cuda driver的初始化
    checkDriver(cuInit(0));

    // 为设备创建上下文
    CUcontext ctxA = nullptr;                                   // CUcontext 其实是 struct CUctx_st*（是一个指向结构体CUctx_st的指针）
    CUcontext ctxB = nullptr;
    CUdevice device = 0;
    checkDriver(cuCtxCreate(&ctxA, CU_CTX_SCHED_AUTO, device)); // 这一步相当于告知要某一块设备上的某块地方创建 ctxA 管理数据。输入参数 参考 https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__CTX_g65dc0012348bc84810e2103a40d8e2cf.html
    checkDriver(cuCtxCreate(&ctxB, CU_CTX_SCHED_AUTO, device)); // 参考 1.ctx-stack.jpg
    printf("ctxA = %p\n", ctxA);
    printf("ctxB = %p\n", ctxB);
    /* 
        contexts 栈：
            ctxB -- top <--- current_context
            ctxA 
            ...
     */

    // 获取当前上下文信息
    CUcontext current_context = nullptr;
    checkDriver(cuCtxGetCurrent(&current_context));             // 这个时候current_context 就是上面创建的context
    printf("current_context = %p\n", current_context);

    // 可以使用上下文堆栈对设备管理多个上下文
    // 压入当前context
    checkDriver(cuCtxPushCurrent(ctxA));                        // 将这个 ctxA 压入CPU调用的thread上。专门用一个thread以栈的方式来管理多个contexts的切换
    checkDriver(cuCtxGetCurrent(&current_context));             // 获取current_context (即栈顶的context)
    printf("after pushing, current_context = %p\n", current_context);
    /* 
        contexts 栈：
            ctxA -- top <--- current_context
            ctxB
            ...
    */
    

    // 弹出当前context
    CUcontext popped_ctx = nullptr;
    checkDriver(cuCtxPopCurrent(&popped_ctx));                   // 将当前的context pop掉，并用popped_ctx承接它pop出来的context
    checkDriver(cuCtxGetCurrent(&current_context));              // 获取current_context(栈顶的)
    printf("after poping, popped_ctx = %p\n", popped_ctx);       // 弹出的是ctxA
    printf("after poping, current_context = %p\n", current_context); // current_context是ctxB

    checkDriver(cuCtxDestroy(ctxA));
    checkDriver(cuCtxDestroy(ctxB));

    // 更推荐使用cuDevicePrimaryCtxRetain获取与设备关联的context
    // 注意这个重点，以后的runtime也是基于此, 自动为设备只关联一个context
    checkDriver(cuDevicePrimaryCtxRetain(&ctxA, device));       // 在 device 上指定一个新地址对ctxA进行管理
    printf("ctxA = %p\n", ctxA);
    checkDriver(cuDevicePrimaryCtxRelease(device));
    return 0;
}
```

# 5. 整体代码流程解析
1. 首先检查CUDA driver是否初始化成功；

2. 使用cuCtxCreate函数为设备创建两个上下文，分别命名为ctxA和ctxB，并打印它们的地址；

3. 使用cuCtxGetCurrent函数获取当前上下文信息并打印其地址；

4. 使用cuCtxPushCurrent函数将ctxA压入CPU调用的线程上，成为当前上下文，并使用cuCtxGetCurrent函数获取当前上下文信息并打印其地址；

5. 使用cuCtxPopCurrent函数弹出当前上下文（即ctxA），并使用cuCtxGetCurrent函数获取当前上下文信息并打印其地址；

6. 使用cuCtxDestroy函数分别销毁ctxA和ctxB；

7. 使用cuDevicePrimaryCtxRetain函数在指定设备上分配一个新地址给ctxA进行管理，然后打印ctxA的地址；

8. 最后使用cuDevicePrimaryCtxRelease函数释放该设备与ctxA的关联。这个是做对比的，看出来自动管理的优势