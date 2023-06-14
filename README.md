# Parallel hashtable - CUDA
## Tăiatu Iulian-Marius - 332CB

This project implements a `GPU-based hash table` called `GpuHashTable` using `CUDA` (Compute Unified Device Architecture). The `GpuHashTable` allows for efficient `insertion` and `retrieval` of key-value pairs using `GPU parallelism` and `auto-resizing` to maintain the `load factor` within a desired range.

## Table of contents
* [Introduction](#introduction)
* [Block-Size](#block-size)
* [Hash-Function](#hash-function)
* [Load-Factor](#load-factor)
* [Constructor](#constructor)
* [Destructor](#destructor)
* [Reshape](#reshape)
* [Insert-Batch](#insert-batch)
* [Reshape](#reshape)
* [Get-Batch](#get-batch)
* [Performance-Tests](#performance-tests)

## Introduction
The `hash table` is stored in `Unified Memory` which is accessible by both the `CPU` and the `GPU`. The hash table is implemented using `linear probing` for collision resolution. The hash table is `reshaped` when the `load factor` exceeds a certain threshold. The hash table is `thread-safe` and it uses `atomic operations` to ensure that the hash table is accessed by only one thread at a time.

```c
// Hash table entry (key-value pair)
struct TableEntry {
	uint32 key;
	uint32 value;
};

// Hash table fields
struct TableEntry *table;
uint32 used_buckets;
uint32 capacity;
```

## Block-Size
The function `estimateBlockSize()` is used to estimate the number of blocks and threads given the number of items. It retrieves the maximum number of threads per block supported by the GPU and calculates the number of blocks required to process all items `efficiently`.

## Hash-Function
The hash function used is `Jenkins` hash function. It is a non-cryptographic hash function which is used to map `arbitrary` length data to a `fixed-length` data. The hash function is `fast` and `efficient` and it is used to `minimize` the number of collisions.

```c
// Jenkins hash function
hash = ~hash + (hash << 15);
hash =  hash ^ (hash >> 12);
hash =  hash + (hash << 2);
hash =  hash ^ (hash >> 4);
hash = (hash + (hash << 3)) + (hash << 11);
hash =  hash ^ (hash >> 16);
```

## Load-Factor
The load factor is the ratio between the number of items and the number of buckets. The load factor is used to determine when the hash table needs to be `reshaped`. The load factor is calculated using the formula: `loadFactor = (float) used_buckets / (float) capacity`.

The load factor is limited to the following values:
```c
#define MIN_LOAD_FACTOR 0.5f
#define MAX_LOAD_FACTOR 0.8f
```

## Constructor
The constructor allocates `Unified Memory` for the hash table and initializes the `capacity` and `size` of the hash table. The constructor also initializes the hash table with `KEY_INVALID` values.

## Destructor
The destructor frees the memory allocated for the hash table.

## Insert-Batch
The `insertBatch()` function inserts a batch of key-value pairs into the hash table using GPU parallelism. It resizes the hash table if needed to maintain the load factor within the desired range. The function allocates memory on the GPU for the batch of keys and values, copies them from the host to the GPU, and invokes the `kernelInsert` kernel to perform the actual insertion.

The `kernelReshape()` kernel function is used by the reshape function to rehash the entries from the old hash table to the new hash table. It is executed on the GPU and processes each entry from the old hash table. The kernel calculates a new hash for each entry and finds the first empty bucket in the new hash table using `linear probing`. It atomically updates the bucket with the key and copies the value. The kernel continues searching for an empty bucket if the current bucket is already occupied. The kernel ensures thread safety using atomic compare-and-swap operations (`atomicCAS`)

It also uses a `count_updated_keys` variable to keep track of the number of keys that were updated, because succesive insertions of the same key will not increase the size of the hash table. Thus, we need to increment this variable in a thread-safe manner using atomic operations (`atomicAdd`).

## Reshape
The `reshape()` function performs a resize of the hash table based on the load factor. It resizes the hash table to the specified number of buckets (`numBucketsReshape`). The function allocates unified memory for the new hash table, initializes it with zero values, and reshapes the hash table by `rehashing` the entries from the old table to the new table using `linear probing`.

The `kernelReshape()` kernel function is used by the reshape function to rehash the entries from the old hash table to the new hash table. It is executed on the GPU and processes each entry from the old hash table. The kernel calculates a new hash for each entry and finds the first empty bucket in the new hash table using linear probing. It atomically updates the bucket with the key and copies the value. The kernel continues searching for an empty bucket if the current bucket is already occupied. The kernel ensures thread safety using atomic `compare-and-swap` operations (`atomicCAS`)

## Get-Batch
The `getBatch()` function retrieves a batch of values corresponding to a batch of keys from the hash table using GPU parallelism. The function allocates memory on the GPU for the batch of keys and retrieved values, copies them from the host to the GPU, and invokes the `kernelGet()` kernel to perform the actual retrieval.

The `kernelGet()` kernel function is used by the `getBatch() `function to retrieve values corresponding to a batch of keys from the hash table. It is executed on the GPU and processes each key from the batch. The kernel calculates the hash for each key and searches for the corresponding bucket using linear probing. It compares the key in the bucket with the requested key and copies the value if a match is found. The kernel continues searching if the current bucket does not contain the requested key.

## Performance-Tests

```sh
------- Test T6 START   ----------

HASH_BATCH_INSERT   count: 10000000         speed: 102M/sec         loadfactor: 50%
HASH_BATCH_INSERT   count: 10000000         speed: 73M/sec          loadfactor: 50%
HASH_BATCH_INSERT   count: 10000000         speed: 105M/sec         loadfactor: 75%
HASH_BATCH_INSERT   count: 10000000         speed: 48M/sec          loadfactor: 50%
HASH_BATCH_INSERT   count: 10000000         speed: 113M/sec         loadfactor: 62%
HASH_BATCH_INSERT   count: 10000000         speed: 83M/sec          loadfactor: 75%
HASH_BATCH_INSERT   count: 10000000         speed: 28M/sec          loadfactor: 50%
HASH_BATCH_INSERT   count: 10000000         speed: 111M/sec         loadfactor: 57%
HASH_BATCH_INSERT   count: 10000000         speed: 98M/sec          loadfactor: 64%
HASH_BATCH_INSERT   count: 10000000         speed: 81M/sec          loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 136M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 134M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 134M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 135M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 135M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 133M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 116M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 106M/sec         loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 94M/sec          loadfactor: 71%
HASH_BATCH_GET      count: 10000000         speed: 78M/sec          loadfactor: 71%

----------------------------------------------
AVG_INSERT: 84 M/sec,   AVG_GET: 120 M/sec,     MIN_SPEED_REQ: 50 M/sec
```

Based on the test `T6` results, the hash table can perform `84 M/sec` insertions and `120 M/sec` retrievals on average. The minimum speed requirement for this test is `50 M/sec`.

We can also see that the `loadFactor` is between `50%` and `75%` throughout the test. This means that the hash table is resized when the load factor exceeds `MAX_LOAD_FACTOR` (80%) and the hash table is reshaped when the load factor falls below `MIN_LOAD_FACTOR` (50%), as expected.


The following results are obtained using the `nvprof` tool for the test `T6`.
```sh
==2086944== Profiling application: ./gpu_hashtable 100000000 10 50
==2086944== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   31.43%  540.96ms        11  49.178ms  11.552us  85.710ms  kernelInsert(...)
                   25.34%  436.23ms        10  43.623ms  31.689ms  84.773ms  kernelGet(...)
                   24.86%  427.92ms        32  13.372ms     992ns  22.120ms  [CUDA memcpy HtoD]
                   12.99%  223.58ms        10  22.358ms  22.169ms  22.635ms  [CUDA memcpy DtoH]
                    5.37%  92.482ms         5  18.496ms  3.0080us  64.147ms  kernelReshape(...)
                    0.00%  10.400us         6  1.7330us  1.2800us  3.8400us  [CUDA memset]
      API calls:   37.28%  1.06996s        26  41.152ms  19.788us  85.720ms  cudaDeviceSynchronize
                   24.74%  710.10ms        17  41.771ms  245.13us  443.15ms  cudaMallocManaged
                   23.37%  670.74ms        42  15.970ms  9.9470us  23.449ms  cudaMemcpy
                   13.11%  376.44ms        59  6.3804ms  11.621us  122.42ms  cudaFree
                    0.46%  13.278ms         6  2.2130ms  224.84us  5.9884ms  cudaMemset
                    0.44%  12.548ms        42  298.76us  7.1990us  462.07us  cudaMalloc
                    0.29%  8.3749ms        26  322.11us  247.23us  512.91us  cudaGetDeviceProperties
                    0.19%  5.4780ms        26  210.69us  134.73us  401.87us  cudaLaunch
                    0.08%  2.3297ms         2  1.1649ms  1.1627ms  1.1670ms  cuDeviceTotalMem
                    0.03%  883.01us       188  4.6960us     351ns  182.88us  cuDeviceGetAttribute
                    0.00%  114.26us       182     627ns     197ns  2.0050us  cudaGetLastError
                    0.00%  104.70us         2  52.348us  44.098us  60.598us  cuDeviceGetName
                    0.00%  51.287us       136     377ns     179ns  1.9220us  cudaSetupArgument
                    0.00%  27.743us        26  1.0670us     493ns  3.0110us  cudaConfigureCall
                    0.00%  4.4380us         4  1.1090us     519ns  2.1720us  cuDeviceGet
                    0.00%  4.4080us         3  1.4690us     462ns  2.7700us  cuDeviceGetCount
```

From this, we can see that the `kernelInsert` and `kernelGet` kernels take up most of the time, which is a good indicator (the GPU is doing work most of the time). The `cudaMemcpy` (`HtoD` and `DtoH`) also take up a significant amount of time, which is expected since the GPU needs to copy data from the host to the device and vice versa, but this overhead cannot be avoided, because of the `RAM` allocation from `test_map.cu`.
