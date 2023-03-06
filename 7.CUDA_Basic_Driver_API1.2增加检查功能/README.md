# 1. 案例(Cuda-Driver API 1.2)
**也是官方的方法**
```cpp

// CUDA驱动头文件cuda.h
#include <cuda.h>

#include <stdio.h>
#include <string.h>

// 使用有参宏定义检查cuda driver是否被正常初始化, 并定位程序出错的文件名、行数和错误信息
// 宏定义中带do...while循环可保证程序的正确性
#define checkDriver(op)    \
    do{                    \
        auto code = (op);  \
        if(code != CUresult::CUDA_SUCCESS){     \
            const char* err_name = nullptr;     \
            const char* err_message = nullptr;  \
            cuGetErrorName(code, &err_name);    \
            cuGetErrorString(code, &err_message);   \
            printf("%s:%d  %s failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, #op, err_name, err_message);   \
            return -1;   \
        }                \
    }while(0)

int main(){

    //检查cuda driver的初始化。虽然不初始化或错误初始化某些API不会报错（不信你试试），但安全起见调用任何API前务必检查cuda driver初始化
    cuInit(2); // 正确的初始化应该给flag = 0
    checkDriver(cuInit(0));

    // 测试获取当前cuda驱动的版本
    int driver_version = 0;
    checkDriver(cuDriverGetVersion(&driver_version));
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100];
    CUdevice device = 0;
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}
```
**可读性较差，看希望推荐的版本**
# 2. 完善check功能(Cuda-Driver API 1.3)
```cpp

// CUDA驱动头文件cuda.h
#include <cuda.h>

#include <stdio.h>
#include <string.h>

// 很明显，这种代码封装方式，更加的便于使用
//宏定义 #define <宏名>（<参数表>） <宏体>
#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){

    if(code != CUresult::CUDA_SUCCESS){    
        const char* err_name = nullptr;    
        const char* err_message = nullptr;  
        cuGetErrorName(code, &err_name);    
        cuGetErrorString(code, &err_message);   
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

int main(){

    // 检查cuda driver的初始化
    // 实际调用的是__check_cuda_driver这个函数
    checkDriver(cuInit(0));

    // 测试获取当前cuda驱动的版本
    int driver_version = 0;
    if(!checkDriver(cuDriverGetVersion(&driver_version))){
        return -1;
    }
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100];
    CUdevice device = 0;
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}
```
# 3. 完善版本的整体解读
1. 宏定义checkDriver(op)，在宏体内部调用__check_cuda_driver((op), #op, __FILE__, __LINE__)，其中#op表示将op参数转化为字符串。
2. __check_cuda_driver函数的作用是检查CUDA driver的初始化是否成功。函数的参数包括：code表示待检查的返回值，op表示操作名称（即checkDriver宏定义的op参数），file表示文件名，line表示行号。函数首先检查code是否等于CUDA_SUCCESS，如果不是，则调用cuGetErrorName和cuGetErrorString函数获取错误代码的字符串描述，然后输出错误信息。最后返回false。
3. 在主函数main中，先调用checkDriver(cuInit(0))检查CUDA driver的初始化是否成功。如果成功，接着测试获取当前CUDA驱动的版本和当前设备信息。其中，获取驱动版本和设备名称的操作也通过调用checkDriver函数进行了检查。如果checkDriver返回false，则退出程序。
4. 通过这样的封装和检查，可以让CUDA驱动的调用更加安全和可靠。

# 4. 通过这个案例复宏定义
1. 宏定义和内联函数都可以避免函数调用的开销，提高程序的效率。不过它们的实现方式不同，宏定义是在预处理阶段对代码进行替换，而内联函数则是在编译阶段将函数的代码直接嵌入到调用处。此外，宏定义可以接受任意类型的参数，而内联函数的参数类型必须是编译时确定的。
```cpp
#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)
```
2. 宏定义 #define <宏名>（<参数表>） <宏体>
3. 上面这个宏中，宏名是checkDriver, 参数表是op,这里对应的是后面的CUDA API返回的CUresult code，宏体是 __check_cuda_driver((op), #op, FILE, LINE)。它是一个函数调用语句，调用了名为 __check_cuda_driver 的函数，并传递了 4 个参数：(op)、#op、__FILE__, __LINE__。
4. __FILE__ 和 __LINE__ 是 C/C++ 中的预定义宏，分别表示当前代码所在文件名和行号。将它们作为参数传递给 __check_cuda_driver 函数，可以在函数内部将调用该函数的代码位置信息打印出来，方便在调试时定位错误。
5. __check_cuda_driver 是作者自定义的代码不应使用这些标识符，以免与系统标识符产生冲突。可以自己换的
6. 所以可以写成下面这种简便的版本，这样更好理解什么是宏定义
```cpp
#include <cuda.h> // CUDA驱动头文件cuda.h

#include <stdio.h> // 使用printf
#include <string.h>

#define checkDriver(op) Mycheck((op), #op, __FILE__, __LINE__)

bool Mycheck(CUresult code, const char* op, const char* file, int line)
{   
    if (code != 0)
    {
        printf("Something went Wrong\n");
    }
    else
    {
        printf("Everything is fine\n");
    }
    
    
}

int main()
{   
    // 检查cuda driver的初始化
    checkDriver(cuInit(0)); 
    checkDriver(cuInit(1));
    return 0;
}
```
```
Everything is fine
Something went Wrong
```

# 5. 完整自己理解注释版
```cpp
#include <cuda.h> // CUDA驱动头文件cuda.h

#include <stdio.h> // 使用printf
#include <string.h>

#define checkDriver(op) Mycheck((op), #op, __FILE__, __LINE__)

bool Mycheck(CUresult code, const char* op, const char* file, int line)
{   
    if (code != CUresult::CUDA_SUCCESS) // 等同于 if(code != 0)
    {
        const char* err_name = nullptr;
        const char *err_message = nullptr;
        // 修改err_name, error_message指针，指向错误信息，报错的字符串的首地址
        cuGetErrorName(code, &err_name);      
        cuGetErrorString(code, &err_message);
        printf("%s, %d %s 失败\n", file, line, op);
        printf("错误名字: %s\n", err_name);
        printf("错误信息: %s\n", err_message);
        return false;
    }

    return true;
}

int main()
{   
    // 检查cuda driver的初始化
    checkDriver(cuInit(0));
    

    // 测试当前CUDA版本
    int driver_version = 0;
    if (!checkDriver(cuDriverGetVersion(&driver_version)) ) // if (false)
    {
        return -1;
    }
    printf("当前驱动版本是: %d\n", driver_version);

    // 测试当前设备信息
    char device_name[100];
    int device = 0;  // CUdevice device = 0

    if(!checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device)))
    {
        return -1;
    };
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}
```