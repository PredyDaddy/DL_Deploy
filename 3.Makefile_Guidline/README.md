
# 对作者的尊重
- 这是一篇学习笔记，repo来自(https://github.com/shouxieai/makefile_tutorial_project)
- 如果您觉得我这个笔记好，请去给原作者点赞
- 作者还有一个挺好的工程模板，(https://github.com/shouxieai/cpp-proj-template)
- 作者的知乎: https://zhuanlan.zhihu.com/p/396448133

# 1. g++
## 1.1 g++ 介绍
1. g++是GNU C++编译器的名称，是一种广泛使用的编程工具，用于将C++源代码编译为可执行的程序。g++是GNU Compiler Collection（GNU编译器集合）的一部分，由自由软件基金会（Free Software Foundation）开发和维护。

2. 与C++代码一起使用的g++工具可以在多种操作系统上运行，包括Linux、macOS和Windows。它支持多种C++标准（如C++98、C++11、C++14等）和编译选项，包括生成调试信息、优化代码、生成目标平台的可执行文件等。

3. g++还支持多种输入格式，包括纯C++源代码、预编译头文件、汇编代码等，可以输出各种目标文件格式，例如ELF格式、Mach-O格式、PE格式等。g++还支持各种编译器插件，包括语法检查、代码优化、代码转换等，可以使用这些插件来增强编译器的功能和性能。

## 1.2 g++编译的过程
1. 完整过程
```cpp
预处理：g++ -E main.cpp -o main.i
汇编：g++ -S main.i -o main.s
编译：g++ -c main.s -o main.o
链接：g++ main.o -o main.bin
```
2. g++ 允许跳过中间步骤
- 直接g++ main.cpp 
- ./a.out 执行可执行文件

## 1.3 预处理
**g++ -E main.cpp -o main.i**

1. main.cpp
```cpp
#include "test.hpp"

#define Title    "This is "
#define Add(a, b)   a + b

int main()
{
  const char *text = Title "Computer Vision";
  int sum = Add(5, 9);
  return 0;
}
```
2. test.hpp
```cpp
int add(int a, int b);
int mul(int a, int b);
```
3. **打开main.i发现预处理后头文件宏会被展开**
![在这里插入图片描述](https://img-blog.csdnimg.cn/3d11ad1b1276436495bad4d058dccd41.png)


## 1.4 汇编
**g++ -S main.i -o main.s**
1. 生成的main.s文件是C++源代码的汇编版本，包含汇编指令和数据，但还没有被编译成机器语言的二进制代码。该文件通常被称为汇编代码，是一个文本文件，可以用文本编辑器打开查看。

2. 通过生成main.s文件，你可以检查编译器是否正确地将C++源代码转换为汇编代码，并且可以进一步分析和调试汇编代码。如果需要，你可以将main.s文件汇编为目标文件或可执行文件，最终得到可以在计算机上运行的程序。

# 1.5 编译
1. **g++ -c main.s -o main.o**这里生成的就是二进制的文件
# 1.6 链接
1. **g++ main.o -o out.bin **
2. 把二进制文件链接为可执行文件
3. **readelf -d out.bin** 分析可执行文件

# 2. C++编译链接 / 编译时和运行时
## 2.1 C++编译链接流程图

![在这里插入图片描述](https://img-blog.csdnimg.cn/f62d1f127a6f44cf8e40c18d1a37db97.png)

## 2.2 C++声明和实现的区别
1. 声明
- 声明不关心参数名称是什么，也不关心返回值是什么，也就是说int add(int a, int b和add(int, int)是一样的

```cpp
int add(int a, int b);
int add1(int a, int b);
```
2. 声明和实现
```cpp
int add(int a, int b)
{
  return a + b;
}
```
# 3. C++编译过程
1. main.cpp和test.cpp区分开
2. main.cpp负责声明，test.cpp负责实现
3. 打开汇编文件，main.s并没有add的具体实现，只是把形参压到寄存器里面去，然后call一下
4. 如果编译test.cpp,打开test.s可以看到明显的差异，里面是有addl(add函数的具体实现的)
5. 总结: 
- C++中函数/符号/变量会被编码
- 函数编码关心: 函数名称/所在的命名空间,参数类型
- 函数编码不关心: 返回值类型
- 调用一个函数，会生成call 函数名称的代码，函数的具体过程不会搬过来
- 这里附上我的另外一篇笔记，学习函数调用的开销(https://blog.csdn.net/bobchen1017/article/details/128753315)

# 4. C++链接过程
1. 把main.o test.o 动态库，静态库链接在一起
2. 案例
```bash
g++ -c main.cpp -o main.o
g++ -c test.cpp -o test.o
g++ main.o test.o -o out.bin
./out.bin
```

# 5. C++