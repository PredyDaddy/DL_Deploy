
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))
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

void launch_reduce_sum(float* input_array, int input_size, float* output);

int main(){
    // 定义数组和变量
    // 定义了一个包含101个元素的数组input_host和一个float类型的变量ground_truth，
    // ground_truth用于存储CPU计算得出的预期结果。
    const int n = 101;
    float* input_host = new float[n];
    float *input_device = nullptr;
    float ground_truth = 0; 

    // 初始化数组
    // 通过循环对数组进行初始化，并同时计算CPU预期结果
    for(int i = 0; i < 101; i++){
    input_host[i] = i;
    ground_truth += i;
    }
    printf("ground_truth = %f\n", ground_truth);

    // 分配内存并将数组传输到设备上
    checkRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    checkRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float), cudaMemcpyHostToDevice));

    // 分配内存并初始化设备上的输出变量
    // 因为这里是求和所以只用一个标量就可以了
    float output_host = 0;
    float* output_device = nullptr;
    checkRuntime(cudaMalloc(&output_device, sizeof(float)));
    // 用cudaMemset()将其初始化为0
    checkRuntime(cudaMemset(output_device, 0, sizeof(float)));

    // 调用launch_reduce_sum()函数，该函数启动了一个CUDA核函数，
    // 在核函数中对数组元素进行求和操作，并将结果存储在设备上的output_device中。
    // 这里还调用了cudaPeekAtLastError()函数，用于检查是否出现了运行时错误。
    launch_reduce_sum(input_device, n, output_device);
    checkRuntime(cudaPeekAtLastError());

    // 将结果副指挥主机端，并且计算误差
    checkRuntime(cudaMemcpy(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    // 对比， FLT_EPSILON是float类型数据的最小精度

    printf("output_host = %f, ground_truth = %f\n", output_host, ground_truth);
    if(fabs(output_host - ground_truth) <= __FLT_EPSILON__){
        printf("结果正确.\n");
    }else{
        printf("结果错误.\n");
    }
    // 释放内存
    cudaFree(input_device);
    cudaFree(output_device);

    delete [] input_host;
    printf("done\n");
    return 0;
}