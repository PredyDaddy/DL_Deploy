# 1. 为什么学习CUDA变成
- TensorRT是一个深度学习推理加速库，是由NVIDIA开发的。TensorRT可以用于加速训练好的深度学习模型在NVIDIA GPU上的推理，从而提高深度学习应用的响应速度和吞吐量。

- 由于TensorRT需要在GPU上进行计算，因此了解CUDA编程是非常必要的。CUDA是NVIDIA开发的并行计算平台和编程模型，可以让开发者利用GPU的并行计算能力加速应用程序的计算，包括深度学习应用程序。在使用TensorRT对深度学习模型进行推理时，需要使用CUDA编程技术来编写和优化TensorRT代码，以最大程度地发挥GPU的计算能力和TensorRT的性能优势。因此，学习CUDA编程对于学习和使用TensorRT是非常重要的。


# 2. 先验知识:
1. nvidia-smi: 显示显卡驱动版本和此驱动最高支持的CUDA驱动版本
```
(base) ubuntu@VM-16-12-ubuntu:~/Github/DL_Deploy/6.CUDA_Basic/cuda-driver-api$ nvidia-smi
Sat Mar  4 17:10:27 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:08.0 Off |                    0 |
| N/A   26C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
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
2. CUDA Version 


# 3. CUDA驱动API-初始化: cuInit
源码
```cpp
 /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    CUDA_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    CUDA_ERROR_INVALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    CUDA_ERROR_NOT_INITIALIZED                = 3,
```