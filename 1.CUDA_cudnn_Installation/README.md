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
- https://blog.csdn.net/weixin_42760399/article/details/122579073
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
# 3. 安装Cudnn
1. 官网安装(https://developer.nvidia.com/rdp/cudnn-archive)
2. 把包放到文件夹下面
3. cd /home/predy/Github/DL_Deploy/1.Install
4. sudo dpkg -i cudnn-local-repo-ubuntu1804-8.6.0.163_1.0-1_amd64.deb
- 出现公钥问题: sudo apt-key add /var/cudnn-local-repo-ubuntu1804-8.6.0.163/7fa2af80.pub
5. sudo apt-get update (我出现的问题，有两个仓库的公钥无法验证)
- sudo cp /var/cudnn-local-repo-ubuntu1804-8.5.0.96/cudnn-local-7B49EDBC-keyring.gpg /usr/share/keyrings/
- sudo apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
- 再次运行 sudo apt-get update
6. sudo apt-get install libcudnn8-dev(自动安装)
7. 或者使用这个指定版本的安装sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.7 libcudnn8-dev=8.6.0.163-1+cuda11.7
- nvcc --version(查看版本)
8. 总结: 先用nvdia-smi查看当前所支持的最高CUDA Tookilt, 再去cudnn查看版本，安装对应的cudnn和CUDA，最后通过测试
9. 成功案例
```
predy@DESKTOP-IU6IR4M:~/DL_Deploy/1.CUDA_cudnn_Installation$ cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 0
```
