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

    // 定义三个vector 
    const int size = 3;
    float vector_a[size] = {1, 2, 3};
    float vector_b[size] = {5, 3, 3};
    float vector_c[size] = {0};

     for(int i = 0; i < size; ++i){
        printf("vector_c[%d] = %f\n", i, vector_c[i]);
    }
// 定义三个device vector, 后面cudaMalloc开辟内存
float *vector_a_device = nullptr;
float *vector_b_device = nullptr;
float *vector_c_device = nullptr;

// cudaMalloc()开辟内存
checkRuntime(cudaMalloc(&vector_a_device, size * sizeof(float)));
checkRuntime(cudaMalloc(&vector_b_device, size * sizeof(float)));
checkRuntime(cudaMalloc(&vector_c_device, size * sizeof(float)));

// 把Host上的a, b内存用Memcpy的方式复制金device
checkRuntime(cudaMemcpy(vector_a_device, vector_a, size * sizeof(float), cudaMemcpyHostToDevice));
checkRuntime(cudaMemcpy(vector_b_device, vector_b, size * sizeof(float), cudaMemcpyHostToDevice));

// 在GPU 上 vector a + vector b， 把数据放在vector c上， 这三个都在device上 
vector_add(vector_a_device, vector_b_device, vector_c_device, size);

// 把GPU上的内存复制到主机上，用于打印
checkRuntime(cudaMemcpy(vector_c, vector_c_device, size * sizeof(float), cudaMemcpyDeviceToHost));
 for(int i = 0; i < size; ++i){
        printf("vector_c[%d] = %f\n", i, vector_c[i]);
    }

return 0;
}