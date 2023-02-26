# 学习使用maikefile建立自己的工程模板
- Makefile是一个经典的构建工具，使用它可以根据一系列规则构建程序。在编写Makefile的过程中，需要了解一些底层的细节，包括链接器，编译器等等。Makefile可以被配置成在所有的平台上都能够使用，非常适合轻量级项目和较小的项目。
- 而CMake则是一种构建工具的高级语言，可以自动生成适合各种平台和编译器的Makefile文件，同时也能生成Visual Studio等IDE所需要的工程文件。CMake使用的是一种高级的、面向目标的语言，可以实现更加复杂的构建任务。CMake的优点在于它可以跨平台地生成Makefile文件，同时具有很好的可读性和可维护性，非常适合大型的、复杂的项目。CMake的缺点在于它的学习曲线相对较陡，需要学习一些新的概念和语法。
- 总之，Makefile和CMake各自有其优缺点，在实际使用中需要根据项目的规模和需求选择适合的构建工具。
# 1. Makefile初探
## 1.1 makefile基本语法(够用就行)
1. Makefile主要解决的问题是，描述生成依赖关系，根据生成和依赖文件修改时间新旧决定是否执行command。(可以想象成是一个写好排版的高级g++和Linux指令的一套模板)
2. 重点: 描述依赖关系，command (生成文件的指令)

## 1.2 代码域: 变量定义域，依赖项定义域，command(shell语法)
```cpp
# 变量定义域 line 1 - line 7
# 定义变量var
var := folder 

# 生成项是main.o, 依赖项是main.cpp
# main.cpp的修改时间比main.o新, 就会触发下面的command g++ -c main.cpp -o main.o
main.o : main.cpp              # 依赖项定义域
    g++ -c main.cpp -o main.o  # command
```

## 1.3 基础语法
![在这里插入图片描述](https://img-blog.csdnimg.cn/433a3847faea48d5aff8a88a5ec78ff2.png) 

## 1.4 写一个小demo
1. 新建一个main.cpp文件和一个01kernel.cu文件
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

## 1.5 逐行解释这个小demo
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

# 3. 完整学习整个makefile文件
来看看最后要完成的工程目录
```bash
project/
  - src/
      - main.cpp
      - 01.kernel.cu
  - objs/
  - workspace/
  - Makefile
```
## 3.1. 定义源码路径
- 这里定义源码路径的作用是为了在后续的代码中可以方便地引用源码文件，并将其编译成目标文件。具体地，使用find命令查找src目录下的所有扩展名为.cpp和.cu的文件，然后将它们分别存储到cpp_srcs和cu_srcs变量中。接着，使用patsubst和subst函数将源码文件路径中的src/替换成objs/，并将它们分别存储到cpp_objs和cu_objs变量中，这样就得到了在objs目录下存储的目标文件列表。在后面的编译和链接过程中，这些目标文件(*.o, *.cuo)会被用来生成最终的可执行文件。
- objs 目录中放置的是编译生成的目标文件，对于这个 makefile 来说，所有的 .o 和 .cuo 文件都会被放到 objs 目录下。在链接生成可执行文件时，makefile 会从这些目标文件中找到需要的文件进行链接。
```makefile
# 定义cpp源码路径，并转换为objs目录先的o文件
cpp_srcs := $(shell find src -name "*.cpp")    
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

# 定义cu源码路径，并转换为objs目录先的cuo文件
# 如果cpp文件和cu名字一样，把.o换成.cuo
cu_srcs := $(shell find src -name "*.cu")    # 全部src下的*.cu存入变量cu_srcs
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs)) # cu_srcs中全部.cu换成.o
cu_objs := $(subst src/,objs/,$(cu_objs))    # cu_objs src/换成objs/
```

## 3.2 定义名称参数
1. 这里定义的名称参数 workspace 和 binary 是用来指定工作空间和生成的可执行文件名称的。在这个 Makefile 中，workspace 表示的是工作空间的目录名称，binary 表示生成的可执行文件的名称。这些参数可以在 Makefile 中的其它规则中使用，例如在链接时指定目标文件路径和生成的可执行文件名称等。
```cpp
workspace := workspace
binary := pro
```
## 3.3 定义头文件，库文件和链接目标
- 因为我们用的是libcudart.so这个库，cudart 是 CUDA runtime库，CUDA编程必须用到的
- 先整个定义了，后面用foreach一次性加进来
- 每个人的位置可能不一样，可以用whereis cuda-版本查找位置
- 编译的过程中需要查找头文件，库文件，include_path, library_path告诉编译器去哪里找头文件，库文件
- link_librarys是告诉链接器需要链接那些库文件
```makefile
# 定义头文件库文件和链接目标，后面用foreach一次性增加
include_paths := /usr/local/cuda-11.7/include
library_paths := /usr/local/cuda-11.7/lib64
link_librarys := cudart
```

## 3.4 定义编译选项
```makefile
# 定义编译选项
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11
cu_compile_flags := -m64 -g -O0 -std=c++11
```
```
-m64：表示编译器生成的代码是 64 位的。
-fPIC：表示编译器要生成位置独立的代码。
-g：表示编译器会在生成的目标文件中加入调试信息，方便进行调试。
-O0：表示关闭优化。
-std=c++11：表示采用 C++11 标准进行编译。
```

## 3.5 合并选项
- -L 指定链接时查找的目录
- -l 指定链接的目标名称，符合libname.so -lname 规则
- -I 指定编译时头文件查找目录
- run path 链接的时查找动态链接库文件的路径，让程序运行的时候，自动查找并加载动态链接库
- 
```
rpath         := $(foreach item,$(link_librarys),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))
```
## 3.6 把合并后的选项给到编译器选项
- cpp_compile_flags += $(include_paths): 将include_paths添加到cpp_compile_flags中，用于在编译C++源代码时指定头文件搜索路径。
- cu_compile_flags += $(include_paths): 将include_paths添加到cu_compile_flags中，用于在编译CUDA源代码时指定头文件搜索路径。
- link_flags := $(rpath) $(library_paths) $(link_librarys): 将rpath、library_paths、link_librarys合并成一个链接选项link_flags。rpath指定运行时库搜索路径，library_paths指定链接库搜索路径，link_librarys指定要链接的库文件名。
```
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        := $(rpath) $(library_paths) $(link_librarys)
```
## 3.7 定义cpp cuda编译方式
- 把.cpp .cu 编译成 .o .cuo的文件中，放在objs里面
- .cpp, .cu文件是依赖项，生成.o .cuo文件
```
# 定义cpp文件的编译方式
# @echo Compile $<     输出正在编译的源文件的名称
objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@echo Compile $<        
	@g++ -c $< -o $@ $(cpp_compile_flags)

# 定义.cu文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $< 
	@nvcc -c $< -o $@ $(cu_compile_flags)
```

## 3.8 在workspace下编译出可执行文件
- 上面.cpp, .cu文件编译出来的结果在这里就是依赖项
- -L./objs 表示告诉链接器在当前目录下寻找库文件，./objs 是指定的路径。实际上，./objs 是目标文件存储的路径，不是库文件存储的路径，这里写的是 -L./objs 只是为了指定链接时查找目录的路径。

```
# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $^
	@g++ $^ -o $@ $(link_flags) -L./objs
```
## 3.9定义伪标签， 作为指令
- 上面编译的可执行文件，是指令依赖项
- 执行可执行文件
```
# 定义pro快捷编译指令，这里只发生编译，不执行
# 快捷指令就是make pro
pro : $(workspace)/$(binary)

# 定义指令并且执行的指令，并且执行目录切换到workspace下
run : pro
	@cd $(workspace) && ./$(binary)
	
debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)

clean : 
	@rm -rf objs $(workspace)/$(binary)

# 指定伪标签，作为指令
.PHONY : clean debug run pro
```
## 3.10 完整的makefile文件
```
# 定义cpp源码路径，并转换为objs目录先的o文件
cpp_srcs := $(shell find src -name "*.cpp")    
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))
cpp_objs := $(subst src/,objs/,$(cpp_objs))

# 定义cu源码路径，并转换为objs目录先的cuo文件
# 如果cpp文件和cu名字一样，把.o换成.cuo
cu_srcs := $(shell find src -name "*.cu")    # 全部src下的*.cu存入变量cu_srcs
cu_objs := $(patsubst %.cu,%.cuo,$(cu_srcs)) # cu_srcs中全部.cu换成.o
cu_objs := $(subst src/,objs/,$(cu_objs))    # cu_objs src/换成objs/


# 定义名称参数
workspace := workspace
binary := pro

# 定义头文件库文件和链接目标，后面用foreach一次性增加
include_paths := /usr/local/cuda-11.7/include
library_paths := /usr/local/cuda-11.7/lib64
link_librarys := cudart

# 定义编译选项
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11
cu_compile_flags := -m64 -g -O0 -std=c++11

# 对头文件, 库文件，目标统一增加 -I,-L-l
rpath         := $(foreach item,$(link_librarys),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 合并选项
# 合并完选项后就可以给到编译方式里面去了
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        := $(rpath) $(library_paths) $(link_librarys)


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
	@g++ -c $< -o $@ $(cpp_compile_flags)

# 定义.cu文件的编译方式
objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Compile $< 
	@nvcc -c $< -o $@ $(cu_compile_flags)

# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $^
	@g++ $^ -o $@ $(link_flags) -L./objs


# 定义pro快捷编译指令，这里只发生编译，不执行
# 快捷指令就是make pro
pro : $(workspace)/$(binary)

# 定义指令并且执行的指令，并且执行目录切换到workspace下
run : pro
	@cd $(workspace) && ./$(binary)
	
debug :
	@echo $(cpp_objs)
	@echo $(cu_objs)

clean : 
	@rm -rf objs $(workspace)/$(binary)

# 指定伪标签，作为指令
.PHONY : clean debug run pro
```