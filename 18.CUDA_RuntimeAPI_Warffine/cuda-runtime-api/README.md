
# 1. 仿射变换
warpAffine是一种二维仿射变换技术，可以将图像从一种形式转换为另一种形式。它是OpenCV图像处理库中的一个函数，可用于对图像进行平移、旋转、缩放和剪切等操作。

仿射变换可以通过线性变换来描述，可以用一个2x3的矩阵来表示。在图像处理中，这个矩阵通常称为变换矩阵，其中包含了平移、旋转、缩放和剪切的参数。

通过warpAffine函数，我们可以将这个变换矩阵应用到输入图像中的每个像素上，从而生成一个新的输出图像。这使得我们可以对图像进行各种几何变换，如旋转、缩放、翻转和扭曲等。

要使用warpAffine函数，您需要指定一个输入图像、一个输出图像和一个2x3的变换矩阵。该函数将变换矩阵应用于输入图像中的每个像素，并将结果存储在输出图像中。

总的来说，warpAffine函数是一种非常有用的图像处理工具，它可以帮助我们实现各种几何变换，以实现我们所需的图像效果。

**仓库里面有一个杜老师写好的ipynb文件，写了一个python版本的仿射变换，双线性插值**

# 2. 双线性插值
双线性插值是一种图像处理中常用的插值方法。在进行缩放、旋转、仿射变换等操作时，由于目标图像的像素点坐标通常不是整数，因此需要对像素进行插值计算，以得到目标图像中对应的像素值。双线性插值就是其中一种常用的插值方法。

双线性插值的原理是基于线性插值的基础上，对两个方向分别进行一次线性插值。具体地，假设我们要计算一个目标像素点的像素值，该像素点的坐标在源图像中为(x, y)，并且在该像素点的四个邻近像素点的坐标分别为(x1, y1)，(x1, y2)，(x2, y1)，(x2, y2)。那么双线性插值的计算方法如下：

1. 在x方向上，对x和x1之间以及x2和x之间的像素值进行线性插值，得到两个中间值f1和f2：
f1 = (x2 - x) / (x2 - x1) * I(y1, x1) + (x - x1) / (x2 - x1) * I(y1, x2)
f2 = (x2 - x) / (x2 - x1) * I(y2, x1) + (x - x1) / (x2 - x1) * I(y2, x2)

2. 在y方向上，对f1和f2进行线性插值，得到最终的像素值f：
f = (y2 - y) / (y2 - y1) * f1 + (y - y1) / (y2 - y1) * f2

其中I(y, x)表示源图像中坐标为(y, x)的像素值。

双线性插值的主要优点是计算简单、速度快，并且可以得到相对较好的插值效果，可以避免出现明显的锯齿和图像失真等问题。因此，它被广泛应用于图像处理中的各种操作，如缩放、旋转、仿射变换等。

# 3. 杜老师的原版代码main.cpp文件
```cpp
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
    /*
       建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            思路讲解：https://v.douyin.com/NhrNnVm/
            代码讲解: https://v.douyin.com/NhMv4nr/
    */        

    Mat output(size, CV_8UC3);
    uint8_t* psrc_device = nullptr;
    uint8_t* pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkRuntime(cudaMalloc(&psrc_device, src_size)); // 在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&pdst_device, dst_size));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上
    
    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}

int main(){ 
    /*
    若有疑问，可点击抖音短视频辅助讲解(建议1.5倍速观看) 
        https://v.douyin.com/NhMrb2A/
     */
    // int device_count = 1;
    // checkRuntime(cudaGetDeviceCount(&device_count));

    Mat image = imread("yq.jpg");
    Mat output = warpaffine_to_center_align(image, Size(640, 640));
    imwrite("output.jpg", output);
    printf("Done. save to output.jpg\n");
    return 0;
}
```

# 4. 杜老师的原版affine.cu
```cpp

#include <cuda_runtime.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define num_threads   512

typedef unsigned char uint8_t;

struct Size{
    int width = 0, height = 0;

    Size() = default;
    Size(int w, int h)
    :width(w), height(h){}
};

// 计算仿射变换矩阵
// 计算的矩阵是居中缩放
struct AffineMatrix{
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr5UdL/
     */

    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    // 这里其实是求解imat的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    void compute(const Size& from, const Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        // 这里取min的理由是
        // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
        // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
        // **
        float scale = min(scale_x, scale_y); // 缩放比例辅助视频讲解 https://v.douyin.com/NhrH8Gm/
        /**
        这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
        scale, 0, -scale * from.width * 0.5 + to.width * 0.5
        0, scale, -scale * from.height * 0.5 + to.height * 0.5
        
        这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
        例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
        S = [
        scale,     0,      0
        0,     scale,      0
        0,         0,      1
        ]
        
        P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
        P = [
        1,        0,      -scale * from.width * 0.5
        0,        1,      -scale * from.height * 0.5
        0,        0,                1
        ]

        T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
        T = [
        1,        0,      to.width * 0.5,
        0,        1,      to.height * 0.5,
        0,        0,            1
        ]

        通过将3个矩阵顺序乘起来，即可得到下面的表达式：
        M = [
        scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
        0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
        0,        0,                     1
        ]
        去掉第三行就得到opencv需要的输入2x3矩阵
        **/

        /* 
            + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
            参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 
            -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;

        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 
            -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){

    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix
){
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr4vTF/
     */
    
    // 2个1D的
    int dx = blockDim.x * blockIdx.x + threadIdx.x; 
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height)  return;        // 启动线程数可能超过图像大小, 超过就return

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    
    /*
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - 双线性理论讲解：https://v.douyin.com/NhrH2tb/
        - 代码代码：https://v.douyin.com/NhrBqpc/ 
     */
    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }else{
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;
}

void warp_affine_bilinear(
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhre7fV/
     */
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value
){
    // 2D layout 
    // 2个1D layout
    dim3 block_size(32, 32); // blocksize最大就是1024，这里用2d来看更好理解
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);        // 目标图像
    AffineMatrix affine;
    affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));        

    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine
    );
}
```




