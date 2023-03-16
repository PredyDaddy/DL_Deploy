#include <cuda_runtime.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define num_threads 512

typedef unsigned char uint8_t;

struct Size
{
    int width = 0, height = 0;

    Size() = default;
    Size(int w, int h)
        : width(w), height(h) {}
};

// 计算仿射变换矩阵
// 计算的矩阵是居中缩放
struct AffineMatrix
{
    /*
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr5UdL/
     */

    float i2d[6]; // image to dst(network), 2x3 matrix
    float d2i[6]; // dst to image, 2x3 matrix

    // 这里其实是求解imat的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
    void invertAffineTransform(float imat[6], float omat[6])
    {
        float i00 = imat[0];
        float i01 = imat[1];
        float i02 = imat[2];
        float i10 = imat[3];
        float i11 = imat[4];
        float i12 = imat[5];

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
        omat[0] = A11;
        omat[1] = A12;
        omat[2] = b1;
        omat[3] = A21;
        omat[4] = A22;
        omat[5] = b2;
    }

    // affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));
    void compute(const Size &from, const Size &to)
    {
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
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] =
            -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;

        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] =
            -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float *matrix, int x, int y, float *proj_x, float *proj_y)
{

    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix matrix)
{
    // 2个1D的  block维度 * block索引 * 线程所在维度的索引
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    // 如果启动的线程索引超过了图像, 直接return
    if (dx > dst_width || dy > dst_height)
    {
        return;
    }

    // 定义fill value, 因为这里的fill value是main函数传进来的
    float c0 = fill_value;
    float c1 = fill_value;
    float c2 = fill_value;

    // 定义src_x, src_y 用与affine_project返回的
    // 通过dx, dy 线程索引返回src_x, src_y的值
    // 这一步的操作就是获得src_x, src_y
    // matrix.d2i 是inverse mapping, 用于从目标图像的坐标反向计算出原图像上的坐标
    float src_x = 0;
    float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    // 这里开始进行双线性插值
    if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
    {
        /*
        out of range 超出边界了，这里注意的一点是x_low, y_low是可以取到-1
        [-1, src_width] 这个区间x是有取值的
        [-1, src_height] 这个区间的y是有取值的
        */
    }
    else{
        // 找到最近四个点的坐标
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};

        // 4 个数值都写出来
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;    // 上面算出来的是相对位置，这边直接减就可以了
        float hx = 1 - lx;
        
        // 定义4个面积, 用于双线性插值的计算
        float w1 = hy * hx;
        float w2 = hy * lx;
        float w3 = ly * hx;
        float w4 = ly * lx;

        // 先给4个点赋予上初始值
        uint8_t *v1 = const_values;
        uint8_t *v2 = const_values;
        uint8_t *v3 = const_values;
        uint8_t *v4 = const_values;

        // 这里计算v1, v2, v3, v4的地址
        if (y_low >= 0){
            if (x_low >= 0){
                // 这里的src_line_size 在从main.cpp传入进来的时候就已经 * 3 
                v1 = src + y_low * src_line_size + x_low * 3;
            }
            if (x_high < src_width){
                v2 = src + y_low * src_line_size + x_high * 3;
            }   
        }
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // 计算双线性插值
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    // 上面写完了经历了从目标图像找回原始图像坐标
    // 从原始图像坐标做了个双线性插值，计算了目标图像的色调
    // 直接去dst图像上修改，这里通过地址去修改
    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0; pdst[1] = c1; pdst[2] = c2; // RGB
    pdst[2] = c0; pdst[1] = c1; pdst[0] = c2; // BGR

}

void warp_affine_bilinear(
    /*
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhre7fV/
     */
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value)
{

    /*
    参数内容
    原图参数:
    *src: 原图指针
    src_line_size: 原图每行的字节数  image.cols * 3
    src_width: 原图宽               image.cols
    src_height: 原图高              image.rows

    目标图像参数:
    *dst: 目标图指针
    dst_line_size: 目标图每行字节数 size.width * 3
    dst_width: 目标图宽
    dst_height: 目标图高

    fill_value: 仿射变换完其他地方需要填充的东西
    */

    // 定义block size, 每一个block里面有32x32个线程块
    dim3 block_size(32, 32); // blocksize最大就是1024

    // 定义grid_Size, 向上取整确保线程格中包含足够的线程块来覆盖整个图像区域
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);

    // 计算仿射矩阵
    AffineMatrix affine;
    // Size构建两个结构体传入affine.compute()里面
    affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));

    // 启动核函数实现栓线性插值仿射变换
    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine);
}