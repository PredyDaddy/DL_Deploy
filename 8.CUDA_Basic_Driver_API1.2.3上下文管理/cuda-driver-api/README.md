# 1. CUcontext上下文管理
1. context是一种上下文，关联对GPU的所有操作
2. context与一块显卡关联，一个显卡可以被多个context关联
3. 每个线程都有一个栈结构储存context，栈顶是当前使用的context，对应有push、pop函数操作context的栈，所有api都以当前context为操作目标
4. 试想一下，如果执行任何操作你都需要传递一个device决定送到哪个设备执行，得多麻烦

# 2. 从两段代码看context的效率
1. 不带context的版本
```cpp
cuMalloc(device, &ptr, 100); 
cuFree(device);
cuMemcpy(device, dst, src, 100);
```
- 理解这段代码
- cuMalloc(device, &ptr, 100);：在CUDA设备上分配100字节的内存空间，并将其地址存储在指针变量ptr中。

- cuFree(device);：释放在CUDA设备上分配的内存空间。

- cuMemcpy(device, dst, src, 100);：将CUDA设备上源地址src处的100字节数据复制到目的地址dst处，其中device表示目标设备。
- 
2. 带context的版本
```cpp
cuCreateContext(device, &context);
cuPushCurrent(context);
cuMalloc(&ptr, 100);
cuFree(ptr);
cuMemcpy(dst, drc, 100);
cuPopCurrent(context);
```

4. **context只是为了方便管理设备的一种手段**
5. **用栈的结构只是为了方便管理更多的设备，使用栈不会出现被重置的现象**
6. 使用栈管理context可以方便地跟踪和管理多个context的创建和销毁。在一个程序中可能会使用多个context，如果不进行管理，可能会出现内存泄漏、资源浪费等问题。使用栈可以很方便地实现先进后出的context管理方式，而不需要手动跟踪和释放每一个context。同时，栈还具有自动管理内存的特性，当一个context出栈时，其对应的内存也会被自动释放，避免了手动释放内存的麻烦。

# 3. 还是复杂，继续简化
1. 基本上高频的使用是一个线程固定访问一个显卡，只使用一个context，所以Create, push, pop这一套流程看起来就很复杂了
2. 推出了cuDevicePrimaryCtx
3. fkldsafbsdakfbkhsdabfhksdbakfldsbalkfjbdsajklfdslkjfbdsa
4. 