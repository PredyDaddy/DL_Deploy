# 1. 知识点
1. nthreads的取值，不能大于block能取值的最大值。一般可以直接给512、256，性能就是比较不错的
    - (input_size + block_size - 1) / block_size;是向上取整
2. 对于一维数组时，采用只定义layout的x维度，若处理的是二维，则可以考虑定义x、y维度，例如处理的是图像
3. 关于把数据视作一维时，索引的计算
    - 以下是通用的计算公式
    ```python
    Pseudo code:
    position = 0
    for i in range(6):
        position *= dims[i]
        position += indexs[i]
    ```
    - 例如当只使用x维度时，实际上dims = [1, 1, gd, 1, 1, bd]，indexs = [0, 0, bi, 0, 0, ti]
        - 因为0和1的存在，上面的循环则可以简化为：idx = threadIdx.x + blockIdx.x * blockDim.x
        - 即：idx = ti + bi * bd



# 2. main.cpp文件
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

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

void vector_add(const float* a, const float* b, float* c, int ndata);

int main(){

    const int size = 3;
    float vector_a[size] = {2, 3, 2};
    float vector_b[size] = {5, 3, 3};
    float vector_c[size] = {0};

    float* vector_a_device = nullptr;
    float* vector_b_device = nullptr;
    float* vector_c_device = nullptr;
    checkRuntime(cudaMalloc(&vector_a_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_b_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_c_device, size * sizeof(float)));

    checkRuntime(cudaMemcpy(vector_a_device, vector_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(vector_b_device, vector_b, size * sizeof(float), cudaMemcpyHostToDevice));

    vector_add(vector_a_device, vector_b_device, vector_c_device, size);
    checkRuntime(cudaMemcpy(vector_c, vector_c_device, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < size; ++i){
        printf("vector_c[%d] = %f\n", i, vector_c[i]);
    }

    checkRuntime(cudaFree(vector_a_device));
    checkRuntime(cudaFree(vector_b_device));
    checkRuntime(cudaFree(vector_c_device));
    return 0;
}
```
先定义三个数组: a, b, c 再用cudaMalloc()在GPU上开辟三个内存，在GPU上让a + b 并且让结果存储进c上，再把c的内存从GPU上放到Host上输出

# 3. 案例.cu文件
```cpp
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int ndata){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= ndata) return;
    /*    dims                 indexs
        gridDim.z            blockIdx.z
        gridDim.y            blockIdx.y
        gridDim.x            blockIdx.x
        blockDim.z           threadIdx.z
        blockDim.y           threadIdx.y
        blockDim.x           threadIdx.x

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    c[idx] = a[idx] + b[idx];
}

void vector_add(const float* a, const float* b, float* c, int ndata){

    const int nthreads = 512;
    int block_size = ndata < nthreads ? ndata : nthreads;  // 如果ndata < nthreads 那block_size = ndata就够了
    int grid_size = (ndata + block_size - 1) / block_size; // 其含义是我需要多少个blocks可以处理完所有的任务
    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
    vector_add_kernel<<<grid_size, block_size, 0, nullptr>>>(a, b, c, ndata);

    // 在核函数执行结束后，通过cudaPeekAtLastError获取得到的代码，来知道是否出现错误
    // cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
    // cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
    // 而cudaPeekAtLastError是获取当前错误，但是再一次执行cudaPeekAtLastError或者cudaGetLastErro拿到的还是那个错
    cudaError_t code = cudaPeekAtLastError();
    if(code != cudaSuccess){    
        const char* err_name    = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);   
    }
}
```

两个注意的点

1. 像这个案例他就三个数相加，其实启动三个线程就足够了，但是一般block给的是512， 256，所以要设定一下，如果数组的长度小于256/512, 就直接用数组的长度的线程数就好。这里就是3个线程

2. 如果线程索引大于了数组的长度就直接返回了，不然就访问了不知道在哪里的内存了  