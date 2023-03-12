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

void launch(int* grids, int* blocks);

void search_demo()
{
    // 定义一个结构体用来储存设备信息
    // 返回一个指向0号设备的指针, 如果写1号设备但是没有，checkRuntime会报错
    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0)); 

    // 查询maxGrid的数量，也是看每一个维度能放多少个block
    printf("prop.maxGridSize = %d, %d, %d\n", prop.maxGridSize[0], 
    prop.maxGridSize[1], prop.maxGridSize[2]);

    // 查询每一个block不同维度的最大线程数，看能放多少个线程
    printf("prop.maxThreadsDim = %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    
    // 查询warp size
    printf("prop.warpSize = %d\n", prop.warpSize);

    printf("prop.maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
}

int main(){
    search_demo(); // 查询设备信息，可以通过查询设备信息了解到

    // 布局的demo, 定义布局
    int grids[] = {1, 2, 3}; // girdDim.x, gridDim.y, gridDim.z
    int blocks[] = {1024, 1, 1}; // blockDim.x, blockDim.y, blockDim.z
    launch(grids, blocks);   // grids表示的是有几个大格子，blocks表示的是每个大格子里面有多少个小格子
    checkRuntime(cudaPeekAtLastError());   // 获取错误 code 但不清楚error
    checkRuntime(cudaDeviceSynchronize()); // 进行同步，这句话以上的代码全部可以异步操作
    return 0;
}
