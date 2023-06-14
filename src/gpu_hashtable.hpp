#ifndef _HASHCPU_
#define _HASHCPU_

// Log the error message (if any) and exit the program
#define CUDA_CHECK_ERROR() do {												\
	cudaError_t _m_cudaStat = cudaGetLastError();							\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);        													\
	} } while(0)

// Keep the `load_factor` between these two values
#define MIN_LOAD_FACTOR 0.5f
#define MAX_LOAD_FACTOR 0.8f

typedef unsigned int uint32;

// (key, value) pair
struct TableEntry {
	uint32 key;
	uint32 value;
};

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();

	private:
		struct TableEntry *table;
		uint32 used_buckets;
		uint32 capacity;
};

#endif
