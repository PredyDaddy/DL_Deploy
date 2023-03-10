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

int main()
{
    int device_id = 0;
    cudaSetDevice(device_id); // 如果不写device_id = 0

    // 分配global memory，也就是GPU上的显存
    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); 

    // 分配pageable memory, 这个是在host上，没有API，用new delete管理
    float *memory_host = new float[100];
    memory_host[2] = 1000;
    checkRuntime(cudaMemcpy(memory_device, memory_host, 100 * sizeof(float), cudaMemcpyHostToDevice));

    // 分配pinned memory 
    float *memory_page_locked = nullptr;
    cudaMallocHost(&memory_page_locked, 100 * sizeof(float));
    cudaMemcpy(memory_page_locked, memory_device, 100 * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证是否真的成功的把pageable locked memory 的改变，传给了设备deivce, 再从设备传给了Host
    // 这里查看Host
    printf("主机上的页锁定内存: %f\n", memory_page_locked[2]);

    // 这个会报错的由于memory_device是在GPU上分配的显存，不能直接访问其地址来读取或写入数据
    // printf("显卡上的memory device内存: %f\n", memory_device[2]);
    


    // 释放内存, 不同内存的释放
    delete[] memory_host;
    cudaFreeHost(memory_page_locked);
    cudaFree(memory_device);
    return 0;
}