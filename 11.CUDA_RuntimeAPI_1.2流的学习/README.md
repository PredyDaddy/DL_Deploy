# 1. 流的定义
**流（Stream）是一个基于上下文（Context）的任务管道抽象**，是一组由GPU依次执行的CUDA操作序列，其中每个操作可能会使用或产生数据。在一个上下文中可以创建多个流，每个流都拥有自己的任务队列和执行状态。通过在不同的流上执行不同的CUDA操作，可以使得这些操作能够异步地并行执行，提高了CUDA程序的性能。

默认情况下，**每个线程都有自己的默认流，可以使用nullptr来代表默认流**。在默认流上执行的CUDA操作会被添加到默认流的任务队列中，然后在GPU上异步执行。如果您创建了多个流，则需要使用流的句柄来将CUDA操作添加到特定的流中。

# 2. 同步和异步
## 2.1 同步
**女朋友(context)发出指令(任务队列)后就什么事情都不能干了，还要等我们这些工具人返回消息，身为一个合格的工具人，你们觉得这样可以吗？身为工具人之王，我们当然不能这样**
![在这里插入图片描述](https://img-blog.csdnimg.cn/cfa41c77f1014657925d349d8af96b13.png)
## 2.2 异步
**这个案例是一个流，也就是一个男朋友，当然，漂亮的女生是应该被更好的对待，例如说多个男朋友**

**女朋友难道要等我们工具人买回来全部东西吗？ 当然不用了，女朋友可以不停的给我们发指令，她想吃苹果了，突然她又想吃个西瓜，也给我们发一下，突然她又想喝奶茶了，再喊我们去给她买奶茶，然后一次性叫我们拿回来**

**如果女朋友想知道买的奶茶的信息，她完全可以给我们发个消息，到了奶茶店给她拍一下今天有什么特别新品，等等等等。**

**是女朋友也可以新建一个流，这个流就是我的好兄弟，工具人二号**

**最重要的是，我们都不是她的男朋友，我只是觉得我是她的男朋友， 也就是nullptr, 默认流，但是很可能我是异步的任务队列，只是为了异步的执行操作，提高context的管理而已**

**活没干完之前，舔狗不许回家**
![在这里插入图片描述](https://img-blog.csdnimg.cn/00913f38189742b0b1d54cc0765528c5.png)
# 3. 正常的解释
![在这里插入图片描述](https://img-blog.csdnimg.cn/bb46ff3449cc40c0b9c1a4830d9adf36.png)


# 4. 代码案例
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

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    // 在GPU上开辟空间
    float* memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));

    // 在CPU上开辟空间并且放数据进去，将数据复制到GPU
    float* memory_host = new float[100];
    memory_host[2] = 520.25;
    checkRuntime(cudaMemcpyAsync(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice, stream)); // 异步复制操作，主线程不需要等待复制结束才继续

    // 在CPU上开辟pin memory,并将GPU上的数据复制回来 
    float* memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost, stream)); // 异步复制操作，主线程不需要等待复制结束才继续
    printf("%f\n", memory_page_locked[2]);
    checkRuntime(cudaStreamSynchronize(stream));
    
    printf("%f\n", memory_page_locked[2]);
    
    // 释放内存
    checkRuntime(cudaFreeHost(memory_page_locked));
    checkRuntime(cudaFree(memory_device));
    checkRuntime(cudaStreamDestroy(stream));
    delete [] memory_host;
    return 0;
}
```