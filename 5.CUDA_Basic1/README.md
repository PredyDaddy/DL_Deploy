


# 1. GPU 架构
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/a67db898bb6242589ddeff59313e05af.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f8e00d29e8fa416dabe193dee18b9c17.png)

1. 这篇博客讲的很好了
https://blog.csdn.net/asasasaababab/article/details/80447254
2. CUDA Core: 表示在GPU设备上执行的核心数量,表示在GPU设备上执行的核心数量
3. CUDA SM: Streaming Multiprocessor, SM是一个独立的处理器单元，具有自己的流处理器和寄存器文件，可以同时执行多个线程。在一个GPU中可能会有多个SM，每个SM可以同时执行许多线程。
4. CUDA SMP: SMP是一种硬件特性，允许在同一时刻在一个SM上执行多个并发内核。SMP的存在可以提高GPU的并行处理能力，使得GPU能够处理更多的任务。
5. **博主以GP 100为案例这个GPU上有68块SM 每个SM有2个SMP，每个SMP有32个CUDA core(一个warp size),  也是最小的的调度单位是32core一起调度**
6. 每个SMP有16个DP Unit (Double Precision) 双精度的
7. 双精度（double-precision）是一种数据类型，它使用64位（8字节）来存储一个数值。它可以表示比单精度（float）类型更大的数值范围和更高的精度。在CUDA中，DP通常指的是双精度浮点数运算，即使用64位浮点数进行计算。由于双精度浮点数需要更多的存储空间和更复杂的运算过程，因此在计算密集型的任务中可能会对性能造成一定的影响。
8. SM的单精度处理能力与双精度处理能力的比例通常在2:1到4:1之间。

# 2. GPU内存架构
![在这里插入图片描述](https://img-blog.csdnimg.cn/76e33f7c652a4467a8fea035d7d17286.png)
1. 寄存器（Register）是计算机内部一种非常快速的存储设备，它通常用来暂时存放计算机正在运算的数据和指令，以及存储函数调用时需要保存的状态信息。寄存器是计算机体系结构中最快的存储设备，由于它们位于CPU内部，因此可以在一个CPU周期内完成数据的读写操作，因此被广泛应用于CPU的优化和加速。**速度快但是内存小，CPU直接在寄存器上面跑**
2. L1缓存是位于每个流多处理器（SM）内部的缓存，它存储了最近使用的内存数据和指令，以供下一次访问时快速获取。L1缓存速度非常快，但它的大小通常较小。
3. L2缓存是位于整个GPU芯片上的缓存，它存储了未被L1缓存所缓存的数据。L2缓存的速度比L1慢，但它的大小通常比L1大很多。L2缓存对于一些大型内存操作比如矩阵乘法和卷积操作是非常有用的，可以显著提高内存访问的效率。
4. SMEM指的是Shared Memory，也叫做共享内存。Shared Memory是一种高速的、低延迟的内存，位于GPU的每个Streaming Multiprocessor（SM）内部，是一个供多个线程共享的内存空间。相比于全局内存，Shared Memory的读写速度更快，因为它在同一个SM内部，可以直接通过寄存器和数据缓存访问。
5. **简单点理解: 离SM越近越快，不同的缓存速度不一样**

# 3. CPU对比GPU
1. GPU控制单元很少，GPU该干计算的活，不是拿来干逻辑的活(if else)
2. 比较适合拿来搞并行比较高的东西
3. 
![在这里插入图片描述](https://img-blog.csdnimg.cn/b91a93ca1b424834a0b05e137a539a6d.png)
# 4. NVIDIA 不同结构
![请添加图片描述](https://img-blog.csdnimg.cn/826f201010bb4670bea9d6240b4239fb.png)

# 5. 第一个CUDA小程序
1. 把我自己写好的模板下载下来git clone https://github.com/PredyDaddy/Makefile_Template.git
2. 