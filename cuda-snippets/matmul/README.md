# Matrix Multiple


对于 N = 1024 的情况，一行数据的存储占用是 1024 * 4B = 4KB。

当 CPU 读取内存中的数据时，缓存是一次缓存 64B，也称为cache line

所以当们按列读取矩阵 B 的时候，比如第`j`列的所有元素，其实是每行读取 64B，那么就是L1 （32K）全部用来缓存 B，那么读到到 (32K / 64 = 512)行的时候，最早的缓存就要被刷掉了。这就导致我们再读取 `j+1` 列时，就需要从新加载到 L1 cache里。

一行是：1024 * 4B / 64


```
build: matmul.cpp
	g++ -I../ -std=c++17 -O3 -fopenmp -mavx2 -o test_cache matmul.cpp
perf: test_cache
	perf record -e L1-dcache-load-misses ./test_cache
	perf report
clean:
	rm test_cache perf.data*
```