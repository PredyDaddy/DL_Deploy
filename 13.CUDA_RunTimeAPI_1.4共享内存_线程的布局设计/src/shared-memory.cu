#include <cuda_runtime.h>
#include <stdio.h>


//////////////////////demo1 //////////////////////////
/*
demo1 主要为了展示查看静态和动态共享变量的地址
静态可以指定内存大小
 */

// 静态共享内存, 可以指定大小，定义多了，指针也会随之增加
// 静态共享内存是可以在kernel函数外面定义的，而共享内存是不可以的
const size_t static_shared_memory_num_element = 6 * 1024; // 6kb
__shared__ char static_shared_memory[static_shared_memory_num_element];
__shared__ char static_shared_memory2[2];

// 定义动态共享内存, 这个也能够在kernel函数外部进行的
// 而且还没有办法定义大小，这个大小的调用的第三个参数
// 无论定义多少个，地址都是一样
extern __shared__ char dynaminc_shared_memory[];
extern __shared__ char dynaminc_shared_memory1[];


__global__ void demo1_kernel()
{   
    // 打印静态共享内存的指针，这里指针式不同的
    printf("static_shared_memory = %p\n", static_shared_memory);
    printf("static_shared_memory2 = %p\n", static_shared_memory2);

    // 要使用这个global的目的就是为了分配共享内存的大小，普通函数不行
    printf("第1个定义的动态共享内存指针: %p\n", dynaminc_shared_memory);
    printf("第2个定义的动态共享内存指针: %p\n", dynaminc_shared_memory1);
}

/////////////////////demo2//////////////////////////////////
/*
demo2 主要是为了演示的是如何给 共享变量进行赋值
这个经典的使用场景，在第0个线程赋予共享变量，然后其他的线程也可以去使用这个定义的共享变量
 */
// 定义共享变量，但是不能给初始值，必须由线程或者其他方式赋值
__shared__ int shared_value1;
__shared__ int shared_value2;
__global__ void demo2_kernel()
{   
    // 两个block的第一个threadIdx.x 都是0
    if (threadIdx.x ==0)
    {   
        // 0号block的共享变量
        if (blockIdx.x == 0)
        {
            shared_value1 = 111;
            shared_value2 = 222;
        }
        else // 1号block的共享变量
        {
            shared_value1 = 333;
            shared_value2 = 444;
        }
    }

    // 等待block所有线程执行到这一步
    __syncthreads();

    printf("%d.%d. shared_value1 = %d[%p], shared_value2 = %d[%p]\n",
           blockIdx.x, threadIdx.x,
           shared_value1, &shared_value1,
           shared_value2, &shared_value2);
}

void launch()
{   
    // 启动一个block, 一个线程， 分配12个字节的内存，是用默认流nullptr
    // 12这个参数是用来指定动态共享内存的
    demo1_kernel<<<1, 1, 12, nullptr>>>();

    // 2个block, 每个block5个线程，10个线程被划分成两块
    demo2_kernel<<<2, 5, 0, nullptr>>>();
}
