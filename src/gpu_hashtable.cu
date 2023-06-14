#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

////////////////////////////// BLOCK-SIZE //////////////////////////////
/**
 * Estimate the number of blocks and threads, given the number of items
 * This function interogates the GPU for the maximum number of threads per block
 * @param nBlocks: number of blocks
 * @param nThreads: number of threads
 * @param nItems: number of items
 * @ref: CUDA Device Properties (https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
*/
static void estimateBlockSize(size_t &nBlocks, size_t &nThreads, uint32 nItems) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	nThreads = prop.maxThreadsPerBlock;
	nBlocks = nItems / nThreads + (nItems % nThreads == 0 ? 0 : 1);
}

////////////////////////////// HASH-FUNCTION //////////////////////////////
/**
 * Hash function
 * @param key: key to be hashed
 * @return hash value: hash value of the key generated by the hash function
 * @ref: Jenkins hash function (https://gist.github.com/badboy/6267743#32-bit-mix-functions)
 */
static __device__ __host__ uint32_t hashFunc(int key) {
	uint32_t hash = (uint32_t) key;

	hash = ~hash + (hash << 15);
	hash = hash ^ (hash >> 12);
	hash = hash + (hash << 2);
	hash = hash ^ (hash >> 4);
	hash = (hash + (hash << 3)) + (hash << 11);
	hash = hash ^ (hash >> 16);

	return hash;
}

////////////////////////////// LOAD-FACTOR //////////////////////////////
/**
 * Calculate the load factor of the hash table
 * @param used_buckets: number of keys in the hash table
 * @param capacity: capacity of the hash table
*/
static float loadFactor(uint32 used_buckets, uint32 capacity) {
	return (float) used_buckets / (float) capacity;
}

////////////////////////////// INIT //////////////////////////////
/**
 * Function constructor GpuHashTable
 * Performs `init`
 * @param size: size of the hash table
 * @ref: CUDA Unified Memory (https://devblogs.nvidia.com/unified-memory-cuda-beginners/)
 */
GpuHashTable::GpuHashTable(int size) {
	// Allocate `Unified Memory` for the hash table
	glbGpuAllocator->_cudaMallocManaged((void**) &this->table, size * sizeof(struct TableEntry));
	CUDA_CHECK_ERROR();

	// Initialize the hash table with `KEY_INVALID`
	cudaMemset(this->table, KEY_INVALID, size * sizeof(struct TableEntry));
	CUDA_CHECK_ERROR();

	// Set the capacity and number of entries
	this->capacity = size;
	this->used_buckets = 0;
}

////////////////////////////// DESTRUCTOR //////////////////////////////
/**
 * Function desctructor GpuHashTable
 * Performs `free`
 */
GpuHashTable::~GpuHashTable() {
	// Free the memory allocated for the hash table
	glbGpuAllocator->_cudaFree(this->table);
	CUDA_CHECK_ERROR();
}

////////////////////////////// RESHAPE //////////////////////////////
/**
 * Kernel function for reshaping the hash table
 * @param old_table: the old hash table
 * @param old_capacity: the old capacity of the hash table
 * @param new_table: the new hash table
 * @param new_capacity: the new capacity of the hash table
 * @ref: Linear probing algorithm (https://en.wikipedia.org/wiki/Linear_probing)
 * @ref: CUDA atomicCAS (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomiccas)
*/
static __global__ void kernelReshape(struct TableEntry *old_table, uint32 old_capacity,
							  		 struct TableEntry *new_table, uint32 new_capacity) {
	// Calculate the index of the current thread
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the index is out of bounds or the entry is empty
	if (idx >= old_capacity || old_table[idx].key == KEY_INVALID)
		return;

	// Calculate a new hash for the current entry
	uint32 generated_hash = hashFunc(old_table[idx].key) % new_capacity;

	// Find the first empty bucket in the new hash table
	while (1) {
		// Set the `old_key` key to `KEY_INVALID` if the bucket is empty
		// `atomicCAS` returns the old value of the bucket, in an atomic manner (thread-safe, locking the bucket)
		uint32 old_key = atomicCAS(&new_table[generated_hash].key, KEY_INVALID, old_table[idx].key);

		// Check if the bucket is empty
		if (old_key == KEY_INVALID) {
			// Copy the value
			new_table[generated_hash].value = old_table[idx].value;
			return;
		}

		// If the bucket is not empty, try the next bucket
		generated_hash = (generated_hash + 1) % new_capacity;
		
		// Check if the hash table is full (it should not happen)
		if (generated_hash == hashFunc(old_table[idx].key) % new_capacity)
			return;
	}
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 * @param numBucketsReshape: the new capacity of the hash table
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Get the current `table`
	struct TableEntry *old_table = this->table;
	uint32 old_capacity = this->capacity;

	// Allocate `Unified Memory` for the new hash table
	glbGpuAllocator->_cudaMallocManaged((void**) &this->table, numBucketsReshape * sizeof(struct TableEntry));
	CUDA_CHECK_ERROR();

	// Initialize the new hash table with `KEY_INVALID`
	cudaMemset(this->table, KEY_INVALID, numBucketsReshape * sizeof(struct TableEntry));
	CUDA_CHECK_ERROR();

	// Set the new capacity of the hash table
	this->capacity = numBucketsReshape;

	// Estimate the number of blocks and threads
	size_t nBlocks, nThreads;
	estimateBlockSize(nBlocks, nThreads, old_capacity);
	
	// Reshape the hash table
	kernelReshape<<<nBlocks, nThreads>>>(old_table, old_capacity, this->table, this->capacity);
	
	// Wait for reshape to finish
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Free the memory allocated for the old hash table
	glbGpuAllocator->_cudaFree(old_table);
	CUDA_CHECK_ERROR();

	return;
}

////////////////////////////// INSERT //////////////////////////////
/**
 * Kernel function for inserting keys in the hash table
 * @param table: the hash table
 * @param capacity: the capacity of the hash table
 * @param device_keys: the keys to be inserted
 * @param device_values: the values to be inserted
 * @param numKeys: the number of keys to be inserted
 * @param count_updated_keys: the number of keys that were updated
 * @ref: Linear probing algorithm (https://en.wikipedia.org/wiki/Linear_probing)
*/
static __global__ void kernelInsert(struct TableEntry *table, uint32 capacity,
									uint32 *device_keys, uint32 *device_values,
									uint32 numKeys, uint32 *count_updated_keys) {
	// Get the index of the current thread
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the index is out of bounds
	if (idx >= numKeys)
		return;

	// Calculate the hash for the current key
	uint32 generated_hash = hashFunc(device_keys[idx]) % capacity;

	// Find the first empty bucket in the hash table
	while (1) {
		// Set the `old_key` key to `KEY_INVALID` if the bucket is empty
		// `atomicCAS` returns the old value of the bucket, in an atomic manner (thread-safe, locking the bucket)
		uint32 old_key = atomicCAS(&table[generated_hash].key, KEY_INVALID, device_keys[idx]);

		// Check if the bucket is empty
		if (old_key == KEY_INVALID || old_key == device_keys[idx]) {
			// Increment the number of updated keys
			if (old_key == device_keys[idx])
				atomicAdd(count_updated_keys, 1);

			// Copy the value
			table[generated_hash].value = device_values[idx];
			return;
		}

		// If the bucket is not empty, try the next bucket
		generated_hash = (generated_hash + 1) % capacity;
		
		// Check if the hash table is full (should not happen)
		if (generated_hash == hashFunc(device_keys[idx]) % capacity)
			return;
	}
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 * @param keys: array of keys to be inserted
 * @param values: array of values to be inserted
 * @param numKeys: number of keys to be inserted
 * @return: true if all keys were inserted, false otherwise
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (numKeys == 0)
		return false;

	// Resize the table if needed to keep the load factor between `MIN_LOAD_FACTOR` and `MAX_LOAD_FACTOR`
	float lf = loadFactor(this->used_buckets + numKeys, this->capacity);
	if (lf > MAX_LOAD_FACTOR)
		this->reshape((this->used_buckets + numKeys) / MIN_LOAD_FACTOR);

	// Allocate space for the batch of keys and values
	uint32 *device_keys, *device_values;
	glbGpuAllocator->_cudaMalloc((void**) &device_keys, numKeys * sizeof(uint32));
	CUDA_CHECK_ERROR();
	glbGpuAllocator->_cudaMalloc((void**) &device_values, numKeys * sizeof(uint32));
	CUDA_CHECK_ERROR();

	// Copy the batch of keys and values to the GPU
	cudaMemcpy(device_keys, keys, numKeys * sizeof(uint32), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(device_values, values, numKeys * sizeof(uint32), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();

	// There will not be 2 identical keys in the batch, but
	// succesive `insert` operations could lead to identical keys
	// Thus, we need to keep track of the number of updated keys
	uint32 *count_updated_keys;
	glbGpuAllocator->_cudaMallocManaged((void**) &count_updated_keys, sizeof(*count_updated_keys));
	CUDA_CHECK_ERROR();
	*count_updated_keys = 0;

	// Estimate the number of blocks and threads
	size_t nBlocks, nThreads;
	estimateBlockSize(nBlocks, nThreads, numKeys);
	
	// Insert the batch of keys and values
	kernelInsert<<<nBlocks, nThreads>>>(this->table, this->capacity, device_keys, device_values, numKeys, count_updated_keys);

	// Wait for the kernel to finish
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Update the number of used buckets
	this->used_buckets += numKeys - *count_updated_keys;

	// Free the memory allocated for the batch of keys and values
	glbGpuAllocator->_cudaFree(device_keys);
	CUDA_CHECK_ERROR();

	glbGpuAllocator->_cudaFree(device_values);
	CUDA_CHECK_ERROR();

	glbGpuAllocator->_cudaFree(count_updated_keys);
	CUDA_CHECK_ERROR();

	return true;
}

////////////////////////////// GET-BATCH //////////////////////////////
/**
 * Kernel function for getting a batch of values
 * @param table: the hash table
 * @param capacity: the capacity of the hash table
 * @param device_keys: the batch of keys
 * @param device_values: the batch of values to be returned
 * @param numKeys: the number of keys in the batch (chunk size)
*/
static __global__ void kernelGet(struct TableEntry *table, uint32 capacity,
								uint32 *device_keys, uint32 *device_values, uint32 numKeys) {
	// Get the index of the current thread
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the index is out of bounds
	if (idx >= numKeys)
		return;

	// Calculate the hash for the current key
	uint32 generated_hash = hashFunc(device_keys[idx]) % capacity;

	// Find the bucket with the current key
	while (1) {
		if (device_keys[idx] == table[generated_hash].key) {
			// Copy the value
			device_values[idx] = table[generated_hash].value;
			return;
		}

		// Check the next bucket
		generated_hash = (generated_hash + 1) % capacity;
	}
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 * @param keys: the batch of keys
 * @param numKeys: the number of keys in the batch (chunk size)
 * @return: the batch of values corresponding to the given keys
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	if (numKeys == 0)
		return NULL;

	// Allocate space for the batch of `keys` and `values`
	uint32 *device_keys, *device_values;
	glbGpuAllocator->_cudaMalloc((void**) &device_keys, numKeys * sizeof(*device_keys));
	CUDA_CHECK_ERROR();
	glbGpuAllocator->_cudaMalloc((void**) &device_values, numKeys * sizeof(*device_values));
	CUDA_CHECK_ERROR();

	// Copy the batch of `keys` to the GPU
	cudaMemcpy(device_keys, keys, numKeys * sizeof(*device_keys), cudaMemcpyHostToDevice);

	// Estimate the number of blocks and threads
	size_t nBlocks, nThreads;
	estimateBlockSize(nBlocks, nThreads, numKeys);
	
	// Get the batch of `values`
	kernelGet<<<nBlocks, nThreads>>>(this->table, this->capacity, device_keys, device_values, numKeys);

	// Wait for the kernel to finish
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Copy the batch of `values` back to the CPU
	uint32 *host_values = (uint32*) malloc(numKeys * sizeof(*host_values));
	cudaMemcpy(host_values, device_values, numKeys * sizeof(*host_values), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();

	// Free the memory allocated for the batch of `keys` and `values`
	glbGpuAllocator->_cudaFree(device_keys);
	CUDA_CHECK_ERROR();
	glbGpuAllocator->_cudaFree(device_values);
	CUDA_CHECK_ERROR();

	return (int*) host_values;
}