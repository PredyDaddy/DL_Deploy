# 1. 安装CUDA
1. 查看Ubuntu操作系统的版本
```cpp
lsb_release -a
```
我的版本是
```cpp
Distributor ID: Ubuntu
Description:    Ubuntu 18.04.1 LTS
Release:        18.04
Codename:       bionic
```
2. 查看驱动支持的最高级的CUDA Toolkit版本，下面输出我最高支持11.7 CUDA Version
```
nvidia-smi
```
```
Fri Feb 24 15:48:08 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.43.01    Driver Version: 516.01       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:02:00.0 Off |                  N/A |
| N/A    0C    P0    N/A /  N/A |      0MiB /  4096MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
3. 去官网下载CUDA Toolkits(https://developer.nvidia.com/cuda-toolkit-archive)
4. 选择Ubuntu 18.04 runfile版本，然后使用提供指令
5. 不要安装驱动
6. 整体我参考了这两篇帖子
- https://blog.csdn.net/my__blog/article/details/125720601
- https://mp.weixin.qq.com/s/pBeAUqqIXLdn8ggv30c0hA
7. 下载CUDA
# 2. 在vscode的环境下试验CUDA
1. 之前Ubuntu用g++编译的，所以直接 g++ main.cpp 就可以了，现在换了一下，用nvcc编译代码，换成了CUDA编程
2. vscode下面的几个文件
- task.json
```
{
  "version": "2.0.0",
  "tasks": [
      {
          "label": "build",
          "type": "shell",
          "command": "/usr/local/cuda-11.7/bin/nvcc",
          "args": [
            "-I/home/user/cuda-11.7/include",
            "${file}",
            "-o",
            "${fileDirname}/${fileBasenameNoExtension}"
        ],        
          "group": {
              "kind": "build",
              "isDefault": true
          }
      }
  ]
}

```
- launch.json 
```
{
  "version": "0.2.0",
  "configurations": [
    
      {
          "name": "CUDA Program",
          "type": "cppdbg",
          "request": "launch",
          "program": "${fileDirname}/test",
          "args": [],
          "stopAtEntry": false,
          "cwd": "${workspaceFolder}",
          "environment": [],
          "externalConsole": true,
          "MIMode": "gdb",
          "miDebuggerPath": "/usr/bin/gdb",
          "preLaunchTask": "build"
      }
  ]
}

```
- c_cpp_properties.json
```
{
    "configurations": [
        {
            "name": "Linux",
            "compilerPath": "/usr/local/cuda/bin/nvcc",
            "intelliSenseMode": "linux-gcc-x64",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/cuda-11.7/include/**"
              ],
              
            "defines": [],
            "browse": {
                "path": [
                    "${workspaceFolder}",
                    "/usr/local/cuda/include/**"
                ],
                "limitSymbolsToIncludedHeaders": true,
                "databaseFilename": ""
            },
            "cStandard": "c17",
            "cppStandard": "c++14"
        }
    ],
    "version": 4
}
```
3. 写一个简单的CUDA编程代码，然后选择nvcc build, 不是nvcc gdb 然后跑通了会有一个可执行文件，这样就算是简单的配置好了
```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int a = 2, b = 3, c;
    int *d_c;

    cudaMalloc((void **)&d_c, sizeof(int));

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << a << " + " << b << " = " << c << std::endl;

    cudaFree(d_c);
    
    return 0;
}
```


