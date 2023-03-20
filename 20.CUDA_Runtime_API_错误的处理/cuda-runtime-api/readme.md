# 知识点
1. 若cuda核函数出错，由于他是异步的，立即执行cudaPeekAtLastError只会拿到对输入参数校验是否正确的状态，而不会拿到核函数是否执行正确的状态
2. 因此需要等待核函数执行完毕后，才真的知道当前核函数是否出错，一般通过设备同步或者流同步进行等待
3. 错误分为可恢复和不可恢复两种：
    - 可恢复：
        - 参数配置错误等，例如block越界（一般最大值是1024），shared memory大小超出范围（一般是48KB）
        - 通过cudaGetLastError可以获取错误代码，同时把当前状态恢复为success
        - 该错误在调用核函数后可以立即通过cudaGetLastError/cudaPeekAtLastError拿到
        - 该错误在下一个函数调用的时候会覆盖
    - 不可恢复：
        - 核函数执行错误，例如访问越界等等异常
        - 该错误则会传递到之后的所有cuda操作上
        - 错误状态通常需要等到核函数执行完毕才能够拿到，也就是有可能在后续的任何流程中突然异常（因为是异步的）

# 可恢复错误代码
```cpp

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void func(float* ptr){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(pos == 999){
        ptr[999] = 5;
    }
}

int main(){

    float* ptr = nullptr;
    cudaMalloc(&ptr, sizeof(float) * 1000);  // 为指针分配内存

    // 因为核函数是异步的，因此不会立即检查到他是否存在异常
    // func<<<100, 10>>>(ptr);
    func<<<100, 1050>>>(ptr);
    auto code1 = cudaPeekAtLastError();
    cout << cudaGetErrorString(code1) << endl;

    // 对当前设备的核函数进行同步，等待执行完毕，可以发现过程是否存在异常
    auto code2 = cudaDeviceSynchronize();
    cout << cudaGetErrorString(code2) << endl;

    // 异常会一直存在，以至于后续的函数都会失败
    float* new_ptr = nullptr;
    auto code3 = cudaMalloc(&new_ptr, 100);
    cout << cudaGetErrorString(code3) << endl;
    return 0;
}
```
**输出:**
```
invalid configuration argument
no error
no error
```

# 不可回复错误代码: 一般是指针, 除0这种错误
cudaDeviceSynchronize()函数会同步等待设备完成之前的所有任务并检查是否有错误发生，如果有错误会返回相应的错误码。所以使用cudaDeviceSynchronize()函数可以及时发现是否有错误发生，进而打印出错误信息。
```cpp

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void func(float* ptr){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(pos == 999){
        ptr[999] = 5;
    }
}

int main(){

    float* ptr = nullptr;
    // cudaMalloc(&ptr, sizeof(float) * 1000);  // 为指针分配内存

    // 因为核函数是异步的，因此不会立即检查到他是否存在异常
    func<<<100, 10>>>(ptr);
    // func<<<100, 1050>>>(ptr);
    auto code1 = cudaPeekAtLastError();
    cout << cudaGetErrorString(code1) << endl;

    // 对当前设备的核函数进行同步，等待执行完毕，可以发现过程是否存在异常
    auto code2 = cudaDeviceSynchronize();
    cout << cudaGetErrorString(code2) << endl;

    // 异常会一直存在，以至于后续的函数都会失败
    float* new_ptr = nullptr;
    auto code3 = cudaMalloc(&new_ptr, 100);
    cout << cudaGetErrorString(code3) << endl;
    return 0;
}
```
**输出**
```
no error
an illegal memory access was encountered
an illegal memory access was encountered
```