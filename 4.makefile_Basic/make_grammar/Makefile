# 变量定义域 line 1 - line 7
# 定义变量var
var := folder 

# 生成项是main.o, 依赖项是main.cpp
# main.cpp的修改时间比main.o新, 就会触发下面的command g++ -c main.cpp -o main.o
main.o : main.cpp              # 依赖项定义域
    g++ -c main.cpp -o main.o  # command