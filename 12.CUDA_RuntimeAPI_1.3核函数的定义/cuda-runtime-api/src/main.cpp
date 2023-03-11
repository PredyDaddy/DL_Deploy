
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

void test_print(const float* pdata, int ndata);

int main(){

    // 定义device指针和host指针
    float* parray_host = nullptr;
    float *parray_device = nullptr;
    int narray = 10;
    int array_bytes = sizeof(float) * narray;

    // 开辟GPU的内存，cudaMalloc返回的指针指向GPU
    checkRuntime(cudaMalloc(&parray_device, array_bytes));

    // 开辟主机内存
    parray_host = new float[narray];

    // 往主机内存放进10个数字
    for (int i = 0; i < narray; i++){
        parray_host[i] = i;
    }

    // 把主机的内存复制上去
    checkRuntime(cudaMemcpy(parray_device, parray_host, array_bytes,cudaMemcpyHostToDevice));
    
    // 把在GPU的东西打印出来
    test_print(parray_device, narray);

    checkRuntime(cudaDeviceSynchronize());

    // 释放device内存, 释放host内存
    checkRuntime(cudaFree(parray_device));
    delete[] parray_host;
    return 0;
}