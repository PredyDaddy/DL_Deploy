#ifndef BOX_HPP
#define BOX_HPP
/*
这个Box结构体定义了一个矩形框，包括左上角和右下角的坐标，置信度和类别标签。具体解释如下：

left: 矩形框左上角点的x坐标。
top: 矩形框左上角点的y坐标。
right: 矩形框右下角点的x坐标。
bottom: 矩形框右下角点的y坐标。
confidence: 矩形框的置信度，代表这个矩形框中存在物体的概率。
label: 矩形框对应的类别标签，通常是一个整数。
*/
struct Box{
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

#endif // BOX_HPP