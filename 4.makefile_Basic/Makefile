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