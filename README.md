# 目录
# 1. CUDA安装
1. CUDA安装
2. cudnn安装
3. 在vscode中配置自己task, launch, C/C++扩展
4. 测试自己的配置，提供案例代码

# 2. .vscode文件夹下面的文件
1.  下次直接复制就好了

# 3. g++: 从编译器的角度编译代码
1. g++
1.1 g++ 介绍
1.2 g++编译的过程
1.3 预处理
1.4 汇编
1.5 编译
1.6 链接
2. C++编译链接 / 编译时和运行时
2.1 C++编译链接流程图
2.2 C++声明和实现的区别
3. C++编译过程
4. C++链接过程

# 4. 从零开始搭建自己的CUDA编程Makefile模板
学习使用maikefile建立自己的工程模板
1. Makefile初探
2. makefile基本语法(够用就行)
3. 代码域: 变量定义域，依赖项定义域，command(shell语法)
4. 基础语法
5. 写一个小demo
6. 逐行解释这个小demo
7. 往下走
8. 完整学习整个makefile文件
9. 定义源码路径
10. 定义名称参数
11. 定义头文件，库文件和链接目标
12. 定义编译选项
13. 合并选项
14. 把合并后的选项给到编译器选项
15. 定义cpp cuda编译方式
16. 在workspace下编译出可执行文件
17. 定义伪标签， 作为指令
18. 完整的makefile文件
# 5. CUDA编程的基础Grid Block 

# 6. CUDA Driver Api
1. 为什么学习TensorRT需要学习CUDA编程
2. 先验知识:
3. 为什么要学习Driver-API?
4. CUDA驱动API-初始化: cuInit
4. cuInit案例完整代码
5. CuInit案例解析
- 字符串的复习
- CUresult
- CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);
- CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);
- 不带Cuinit的整体
- 带Cuinit的整体结构

# 7. CUDA 驱动API, 检查功能
1. 案例(Cuda-Driver API 1.2)
2. 完善check功能(Cuda-Driver API 1.3)
3. 完善版本的整体解读
4. 通过这个案例复习宏定义
5. 完整自己理解注释版
# 8. CUDA 驱动API, 检查功能 CUcontext上下文管理
1. CUcontext上下文管理
2. 从两段代码看context的效率
3. 还是复杂，继续简化 cuDevicePrimaryCtxRetain
4. 代码
5. 整体代码流程解析

# 9. CUDA 驱动API, 内存分配
1. 思路
2. 整体代码
3. CUDdeviceptr以及复习数据结构
4. 锁页内存
5. cuMemset
6. 释放锁页内存

# 10. CUDA RunTime API 2.1 Hello CUDA
1. CUDA Driver API和CUDA Runtime API
2. 两种API的区别
3. 第一个CUDA RunTime API 程序Hello CUDA
4. 分解这个案例

# 11. CUDA RunTime API 2.2 Stream
1. 流的定义
2. 同步和异步
3. 注意的地方
4. 代码案例

# 12. CUDA RunTime API 2.2 kernel function
1. 核函数的定义
2. main.cpp文件及详细解释
3. kernel.cu文件及其详细解释

# 13. CUDA RunTime API 2.3 shared_memeory, shared_value
1. 共享内存的引入
2. CU文件
3. cpp文件
4. 代码解析解析
5. demo1(静态共享内存和动态共享内存)
6. demo2(共享变量)

# 14. CUDA_Run_Time_API_thread_layout线程布局
1. 知识点
2. 图解知识点
3. main.cpp文件
4. cu文件
5. 代码拆解
6. cu文件解读

# 15.CUDA_Run_Time_API_parallel_多流并行，以及多流之间互相同步等待的操作方式
1. 整体代码
2. 单个流串行
3. 向量相加相乘的kernel function
4. 多个流的异步
5. 多个流之间互相等待的操作