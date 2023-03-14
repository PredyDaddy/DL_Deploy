#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

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

void warp_affine_bilinear( // 声明
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value
);

Mat warpaffine_to_center_align(const Mat& image, const Size& size){
    // 创建一个Mat类型的output, 大小为size，类型为CV_8UC3
    Mat output(size, CV_8UC3);

    // 定义CPU上的指针
    // uint8_t是一种无符号8位整数类型，可以表示0到255之间的整数。在OpenCV中，像素值通常以8位无符号整数类型存储，
    // 因此使用uint8_t类型可以确保正确的数据类型匹配和内存使用，同时提供了更好的可读性
    uint8_t *psrc_device = nullptr;
    uint8_t *pdst_device = nullptr;

    // 在GPU上开辟内存的大小
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3; 

    // cudaMalloc开辟内存
    checkRuntime(cudaMalloc(&psrc_device, src_size));
    checkRuntime(cudaMalloc(&pdst_device, dst_size));

    // 把数据搬运到GPU上面去
    // image.data指的是image的首地址
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size,cudaMemcpyHostToDevice));

    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost));
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}

int main(){ 
        Mat image = imread("yq.jpg");

    // 参数类型                              Mat       Size
    Mat output = warpaffine_to_center_align(image, Size(640, 640));
    imwrite("output.jpg", output);
    printf("Done. save to output.jpg\n");
    return 0;
}