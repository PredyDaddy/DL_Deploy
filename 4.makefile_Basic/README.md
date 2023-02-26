# 学习使用maikefile建立自己的工程模板
- Makefile是一个经典的构建工具，使用它可以根据一系列规则构建程序。在编写Makefile的过程中，需要了解一些底层的细节，包括链接器，编译器等等。Makefile可以被配置成在所有的平台上都能够使用，非常适合轻量级项目和较小的项目。
- 而CMake则是一种构建工具的高级语言，可以自动生成适合各种平台和编译器的Makefile文件，同时也能生成Visual Studio等IDE所需要的工程文件。CMake使用的是一种高级的、面向目标的语言，可以实现更加复杂的构建任务。CMake的优点在于它可以跨平台地生成Makefile文件，同时具有很好的可读性和可维护性，非常适合大型的、复杂的项目。CMake的缺点在于它的学习曲线相对较陡，需要学习一些新的概念和语法。
- 总之，Makefile和CMake各自有其优缺点，在实际使用中需要根据项目的规模和需求选择适合的构建工具。
# 1. Makefile初探
## 1.1 写一个小demo
1. 新建一个main.cpp文件
```cpp
#include <iostream>
using namespace std;

int main()
{
  cout << "Hello CUDA" << endl;
  return 0;
}
```
2. 新建一个Makefile文件
```
cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

debug :
	@echo $(cpp_objs)
```
3. 终端执行 make debug 如果有问题就是有bug

## 1.2 逐行解释这个小demo
1. 定义变量 cpp_srcs，它的值是通过调用 find 命令在 src 目录下查找所有以 .cpp 结尾的文件，并将它们的路径存入 cpp_srcs 变量中。
```cpp
cpp_srcs := $(shell find src -name "*.cpp")
```
2. 定义变量 cpp_objs，它的值是将 cpp_srcs 变量中所有 .cpp 后缀的文件名都替换成以 .o 结尾的目标文件名，并将这些目标文件名保存到 cpp_objs 变量中。
```cpp
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
```
3. 将 cpp_objs 变量中的所有 src/ 替换为 objs/，并将替换后的值存储回 cpp_objs 变量。
```cpp
cpp_objs := $(subst src/,objs/,$(cpp_objs))
```
4. 创建一个名为 debug 的伪目标，它的命令是打印 cpp_objs 变量的值，@ 告诉 Make 不要显示命令行上的命令，只显示命令的输出。
```cpp
debug :
    @echo $(cpp_objs)
```

# 2. 往下走
```makefile
# 定义cpp源码路径，并转换为objs目录先的o文件
cpp_srcs := $(shell find src -name "*.cpp")    
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

# 定义cu源码路径，并转换为objs目录先的cuo文件
# 如果cpp文件和cu名字一样，把.o换成.cuo
cu_srcs := $(shell find src -name "*.cu")   # 全部src下的*.cu存入变量cu_srcs
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs)) # cu_srcs中全部.cu换成.o
cu_objs := $(subst src/,objs/,$(cu_objs))    # cu_objs src/换成objs/


# 定义名称参数
workspace := workspace 
binary := pro

# makefile中定义cpp的编译方式
# $@：代表规则中的目标文件(生成项)
# $<：代表规则中的第一个依赖文件
# $^：代表规则中的所有依赖文件，以空格分隔
# $?：代表规则中所有比目标文件更新的依赖文件，以空格分隔


# 定义cpp文件的编译方式
# @echo Compile $<     输出正在编译的源文件的名称
objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@echo Compile $<        
	@g++ -c $< -o $@ 

# 定义.cu文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $< 
	@nvcc -c $< -o $@

# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $^
	@g++ $^ -o $@

# 定义pro快捷编译指令，这里只发生编译，不执行
# 快捷指令就是make pro
pro : $(workspace)/$(binary)

# 定义指令并且执行的指令，并且执行目录切换到workspace下
run : pro
	@cd $(workspace) && ./$(binary)
	
debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)
```
这里报错了，因为缺少了cuda的库文件，下面是查看自己cuda版本和找到在哪里的指令，以我的CUDA11.7为例, 去到目录下看看有什么库文件，我们当前需要一个cudaruntime的头文件
```
nvcc --version
whereis cuda-11.7
cuda-11: /usr/local/cuda-11.7
cd /usr/local/cuda-11.7/lib64 
```
