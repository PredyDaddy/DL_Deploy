#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include "box.hpp"

using namespace std;
using namespace cv;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

// 二进制模式打开文件(ios::binary), 使用static std::vector<uint8_t>存储数据
// uint8_t是一个无符号8位整数类型。
// 使用const string &file作为函数的参数，可以指定文件的路径和名称
static std::vector<uint8_t> load_file(const string &file)
{
    // 创建一个输入文件流 in，用于读取文件。
    // ios::in | ios::binary 表示以输入模式和二进制模式打开文件
    ifstream in(file, ios::in | ios::binary);
    // 如果文件未成功打开，函数返回一个空的 std::vector<uint8_t>
    if (!in.is_open()){
        return {};
    }

    // 将文件流的读取位置设置到文件末尾，获取文件长度
    in.seekg(0, ios::end);  // 将文件流的读取位置设置到文件末尾，获取文件长度
    size_t length = in.tellg();  // 获取当前读取位置，即文件长度

    std::vector<uint8_t> data;  // 用于存储文件
    if (length > 0)
    {   
        in.seekg(0, ios::beg); // 先把文件流的位置放回一开始
        data.resize(length);   // 把data resize成文件的长度

        // in.read()从文件流in中读取数据进指定的内存缓存区
        // 内存缓存区首地址是data[0], 缓冲区大小是length
        // in.read()读取二进制文件时需要传入 char*, float*  放到main函数做
        in.read((char *)&data[0], length);
    }
    in.close(); // 关闭文件流
    return data;}


    /*
    这个代码流程如下:
    1. 从nx85的维度中把每个结果的left, top, right, bottom confidence取出来,然后放进vector<box>里面
    2. 对储存好的box执行NMS操作
    3. 这里需要知道用了两个if减少n的维度减少了计算量
    4. 并没有直接的去boxes里面删除框, 做了预分配, 给他们打上了标签, 合适的用emplace_back加上
    */
    // auto boxes = cpu_decode(ptr, nrows, ncols);
    vector<Box> cpu_decode(float *predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f)
    {
    // 创建一个boxes的结构体储存box
    // box 里面储存着左上角的(x, y) 右小角的(x, y) confidence label
    vector<Box> boxes; // 这里面就储存着一堆的box

    // 计算这里面的类别数, 输出的cols前面5列都是位置信息
    int num_classes = cols - 5; 

    /*
    这个for 循环是用来遍历结果输出data的, n x 85个维度是n个结果
    85 包含了其中85是cx, cy, width, height, objness, classification * 80
    
    */
    for (int i = 0; i < rows; i++)
    {
        float *pitem = predict + i * cols;
        // objness是预测出这个bounding box是否包含目标的概率
        float objness = pitem[4];
        if (objness < confidence_threshold){
            continue;
        }

        // 第6个才指向类别 可以理解为pclass[0] = pitem[5];
        // 但是pclass更加清晰地表明了这部分内存的含义
        float *pclass = pitem + 5; 
        // 找到那个类别, 类别几，想象成是第一个类别是 1
        int label = std::max_element(pclass, pclass + num_classes) - pclass;
        // 获取类别置信度的最大值
        float prob = pclass[label];

        // 计算置信度
        float confidence = prob * objness; 
        if (confidence < confidence_threshold){
            continue;
        }

        /*
        这里面的操作的目的是为了把前面过预测出来的结果(n x 85的结果)变成 n 个box储存在
        当前前面用了两层条件就是为了减少这个的操作过程，也就是减少n这个rows, 减小维度
        上面两个if() 满足就直接跳出循环, 这样可以减少下面的操作
        还是CPU计算的思维问题, 尽可能地减少计算很重要的
        上面两个If其实不做也可以，但是问题就是会增加很多的计算量
        */

        // 拿到前面4个参数, cx, cy, width, height 
        float cx = pitem[0];
        float cy = pitem[1];
        float width = pitem[2];
        float height = pitem[3];

        // 通过cx, cy, width, height  左上角 右下角的坐标
        float left = cx - width * 0.5;
        float top = cy - height * 0.5;
        float right = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        // 将left, top, right, bottom, confidence, float(label) 都储存进boxes里面
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
        // 现在开始对全部的box做NMS操作
        /*
        这句话是拿来对confidence进行排序的, 从boxes的开头到结尾
        这句话是lambda表达式, C++中lambda表达式用[]开始
        Box &a, Box &b表示lambda表达式的函数头
        return a.confidence > b.confidence; 如果满足,返回True
        这里用引用的目的是为了不去修改
        */
        std::sort(boxes.begin(), boxes.end(), [](Box &a, Box &b)
                  { return a.confidence > b.confidence; });

        // 定义一个标签用于判断是否删除这个框, 长度跟boxes等同
        // false保留True删除
        std::vector<bool> remove_flags(boxes.size());

        // 提前做了预分配, 用了这个性能会好很多
        std::vector<Box> box_result; 
        box_result.reserve(boxes.size());   

        // 定义一个lambda表达式计算iou
        auto iou = [](const Box &a, const Box &b)
        {   
            // 求交集, 所以需要左上角里面的点和右下角里面的点，这里用的是里面的点
            // 思考用max还是min，考虑清楚图像中, 左上角才是(0, 0)
            float cross_left = std::max(a.left, b.left);
            float cross_right = std::min(a.right, b.right);
            float cross_top = std::max(a.top, b.top);
            float cross_bottom = std::min(a.bottom, b.bottom);
            //计算出来corss area
            float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
            // 计算出并集, 这里是计算出两个面积相加再减去cross_area, 比较巧妙的实现
            float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
            
            // 没有交集的情况
            if (cross_area == 0 || union_area == 0)
            {
                return 0.0f;
            }

            // 如果有交并集
            return cross_area / union_area;
        };

        // 便利每一个box, 看下
        for (int i = 0; i < boxes.size(); ++i)
        {   
            // if True, 表示前面标记过了会被删除, 跳过
            if (remove_flags[i]){
                continue;
            }

            /*
            第一次循环找到最大的那个框，然后开始对比其他删掉跟他iou重合度大的
            第二次就是第一次删完之后iou最大的框，这个框跟第一个框iou不大所以没有被删除
            开始删掉框框然后继续往下走
            这里说的删除就是给框框打上True的标签
            */
            auto &ibox = boxes[i];
            box_result.emplace_back(ibox);
            for (int j = i + 1; j < boxes.size(); j++)
            {
                if (remove_flags[j]){
                    continue; // 被标记过跳出循环
                }

                // 这里判断框框两个条件: 删掉重合度大的还有类别是一样的
                auto &jbox = boxes[j];
                if (ibox.label == jbox.label){
                    // 判断NMS阈值
                    if (iou(ibox, jbox) >= nms_threshold){
                        remove_flags[j] = true;
                    }
                }
            }
        }

        return box_result;
    }

void decode_kernel_invoker(
    float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float nms_threshold, float *invert_affine_matrix, float *parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

vector<Box> gpu_decode(float *predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f)
{

    vector<Box> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float *predict_device = nullptr;
    float *output_device = nullptr;
    float *output_host = nullptr;
    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));

    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(
        predict_device, rows, cols - 5, confidence_threshold,
        nms_threshold, nullptr, output_device, max_objects, NUM_BOX_ELEMENT, stream);
    checkRuntime(cudaMemcpyAsync(output_host, output_device,
                                 sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    int num_boxes = min((int)output_host[0], max_objects);
    for (int i = 0; i < num_boxes; ++i)
    {
        float *ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if (keep_flag)
        {
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]);
        }
    }
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFreeHost(output_host));
    return box_result;
}

int main()
{
    // yolov5的输出tensor(n x 85)
   // 其中85是cx, cy, width, height, objness, classification * 80
   // 每一行都是预测结果, 列数是预测出来的信息

    // 加载一个二进制的文件
    auto data = load_file("predict.data");
    auto image = cv::imread("input-image.jpg");

    // 因为数据是以二进制存储在文件中的, 如果想对二进制文件进行访问，需要使用指针
    float *ptr = (float *)data.data();
    int nelem = data.size() / sizeof(float); // 计算data有多少个数据
    int ncols = 85;                          // cx, cy, width, height, objness, classification * 80
    int nrows = nelem / ncols;

    // 这里是用gpu_decode拿到框框
    // 这里的boxes是一个vector的数据类型
    auto boxes = cpu_decode(ptr, nrows, ncols);

    // 这里是把框框在图像上画出来
    // for (auto it = boxes.begin(); it != boxes.end(); ++it) 有点像这句话
    for (auto &box : boxes)
    {

        // image, 左上角坐标，右小角坐标, 线的颜色, 线的宽度
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom),
                      cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7),
                    0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }

    cv::imwrite("image-draw.jpg", image);
    return 0;
}