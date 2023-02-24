# 安装CUDA
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
2. 去这个网站选择合适的cuda进行安装(https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04)

2. 不同版本的选择
在 NVIDIA CUDA 下载页面中，Installer Type 可以根据您的需求选择以下两个选项之一：

- Network Installer：这个选项是一个在线安装程序，它将下载并安装 CUDA 工具包及其依赖项。使用这个选项，您需要保持您的计算机联网状态。此选项通常适用于需要在多台计算机上安装 CUDA 的用户。

- Base Installer：这个选项是一个离线安装程序，它包含完整的 CUDA 工具包及其依赖项。使用这个选项，您可以下载安装程序并将其复制到目标计算机，然后在没有互联网连接的情况下进行安装。此选项通常适用于需要在单台计算机上安装 CUDA 的用户。

- 根据您的需求选择适当的 Installer Type，并按照页面上的说明进行下载和安装。