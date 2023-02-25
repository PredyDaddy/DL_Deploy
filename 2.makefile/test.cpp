#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int a = 2, b = 3, c;
    int *d_c;

    cudaMalloc((void **)&d_c, sizeof(int));

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << a << " + " << b << " = " << c << std::endl;

    cudaFree(d_c);
    
    return 0;
}