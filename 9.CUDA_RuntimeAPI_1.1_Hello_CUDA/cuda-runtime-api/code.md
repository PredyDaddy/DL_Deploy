```cpp
    // 查看device数量
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("当前一共有%d台设备\n", device_count);
```