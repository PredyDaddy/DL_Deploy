
#include <cuda_runtime.h>

static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(
    float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float *invert_affine_matrix, float *parray, int max_objects, int NUM_BOX_ELEMENT)
{
    // 确保有足够的thread, 每一个thread处理一个bounding box
    // 如果threadId超过了bounding box的数量, 这样就不会进行后续处理, 每个预测框都敲好被处理了一次
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes){
        return;
    }

    /*
     predict是n x 85tensor输出的首地址
     pitem 就是每行的指针, pitem[0] - pitem[3] 是位置信息, pitem[4]是objness
    */
    float *pitem = predict + (num_classes + 5) * position;
    float objectness = pitem[4];
    if (objectness < confidence_threshold){
        return;
    }

    // 从这个元素开始都是confidence
    float *class_confidence = pitem + 5;
    // 这里是第一个condience, 取到数值
    float confidence = *class_confidence++;

    // for循环判断是哪个类别
    int label = 0;
    for (int i = 1; i < num_classes; i++, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {   
            // 如果大了, 就更新class_confidence
            confidence = *class_confidence;
            label = i; // 取到label
        }
    }

    /*
    上面的最后算出来的condifence是class_confidence只是条件概率
    当前bounding box的 confidence(置信度) =  objectness(物体概率) x class_confidence(条件概率)
    最后拿来计算置信度的confidence是最大的class_confidence
    */
    confidence *= objectness;
    if (confidence < confidence_threshold){
        return;
    }

    /*
    这里是恢复boudingbox的操作, 需要先取出来中心点(cx, cy), width, height
    */
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    // affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    // affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    
    /*
    atomicAdd()简介: 
    int atomicAdd(int* address, int val);
    这个函数执行的操作是将指定地址 address 处的值与 val 相加，并将结果写回 address 处。这个操作是原子性的，即不会受到并发写入的干扰，保证了数据的正确性。
    使用 atomicAdd 函数可以保证多个线程在对同一个内存地址进行写操作时，不会发生数据覆盖的问题。
    由于每个线程都会在输出中写入一个bounding box，因此需要使用原子操作确保每个线程写入的位置唯一
    */

    /*
    [count, box1, box2, box3]
    因为GPU解码是多线程的, 所以需要用count记录已经处理了多少个bounding box。
    CPU单线程不需要, GPU需要确保不会将一个检测框重复输出或者漏掉。
    atomicAdd -> count +=1 返回 old_count
    这里是对parray(output_device第一个值+1)
    */
    int index = atomicAdd(parray, 1);
    // 如果超过了1000, 这个线程就没必要处理后面的boxes
    if (index >= max_objects)  
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom)
{

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT)
{
/*
参数解析: 
bboxes：存储了所有待处理的检测框信息的一维数组；
max_objects：最大的输出检测框数量； 案例设置的是1000, 预计一张图不会超过1000个bounding box
threshold：用于判断两个检测框是否重叠的 IOU 阈值；
NUM_BOX_ELEMENT：每个检测框存储的元素个数
一般包含: left, top, right, bottom, confidence, class, keepflag
*/
    
    // 计算position, 超过count不用进行下面计算了
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = min((int)*bboxes, max_objects);
    if (position >= count){
        return; 
    }

    /*
    重叠度高, 并且类别相同，然后是condience小于另外一个, 就删掉他
    极端情况下会有误删, 如果测试cpu map的时候, 只能采用cpu nms
    日常推理的时候, 则可以使用这个NMS
    left, top, right, bottom, confidence, class, keepflag
    */
    
    // 这里计算出来当前的指针, 在bboxes上
    float *pcurrent = bboxes + 1  + position * NUM_BOX_ELEMENT; 
    // 便利每一个bbox
    for (int i = 0; i < count; ++i){
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        // NMS计算需要保证类别必须相同
        if (i == position || pcurrent[5] != pitem[5] ){
            continue;
        }
        
        // 判断置信度大小, 如果比pcurrent大，干掉pcurrent
        if (pitem[4] > pcurrent[4]){
            // 如果两个一样大，保留编号小的那个
            if (pitem[4] == pcurrent[4] && i < position){
                continue;
            }
                
            // 拿前面四个信息计算IOU
            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold){
                pcurrent[6] = 0;  // 这里pitem跟pcurrent重合度高而且达到阈值
                return;
            }
        }
    }
}

/*
decode_kernel_invoker(
    predict_device, rows, cols - 5, confidence_threshold,
    nms_threshold, nullptr, output_device, max_objects, NUM_BOX_ELEMENT, stream);
*/
void decode_kernel_invoker(
    float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float nms_threshold, float *invert_affine_matrix, float *parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream)
{
    /*
    参数解析:
    predict: 预测结果, 这个就是data, 未处理未过滤的predict
    num_bboxes: 在预测结果的（n x num_classes+ 5） tensor中, 多少行就是多少个box
    num_classes: 类别数量
    confidence_threshold: 置信度阈值
    nms_threshold: nms阈值
    invert_affine_matrix: 逆矩阵的指针
    parray: 输出结果数组
    max_objects: 最大数量框, 这边设置的是1000, 只是拿来确保有足够的内存
    NUM_BOX_ELEMENT: Box的element, left, top, right, bottom, confidence, class, keepflag 一共7个
    stream： 流
    */
    // 这里是确保有足够的线程去处理每一个box, 也就是每一个预测结果，所以用num_boxxes
    // 确保每个block的线程不超过512
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    /* 如果核函数有波浪线，没关系，他是正常的，你只是看不顺眼罢了 */
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold,
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT);

    // 这里是针对每张图的框，确保每个狂都能被线程处理
    // 同样确保每个block的线程不超过512
    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}
