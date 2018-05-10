#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 512 

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct IsChar {
	__host__ __device__
	int operator()(const char x)
	{
		return x != '\n' ? 1 : 0;
	}
};
void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::device_ptr<const char> data_ptr(text);
	thrust::device_ptr<int> result_ptr(pos);
	thrust::transform(thrust::device, data_ptr, data_ptr + text_size,
		result_ptr, IsChar());
	thrust::inclusive_scan_by_key(thrust::device, result_ptr,
		result_ptr + text_size, result_ptr, result_ptr);
}

__global__ void scan(int *pos, int n)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int index_t = threadIdx.x;
	__shared__ int s[BLOCK_SIZE];
	if (index >= n)
		return;
	s[index_t] = pos[index];
	__syncthreads();
	for (int i = 0; i < 9; i++) {
		if (s[index_t] && index_t >= s[index_t]
			&& s[index_t - s[index_t]])
			s[index_t] += s[index_t - s[index_t]];
		else
			break;
		__syncthreads();
	}
	pos[index] = s[index_t];
}

__global__ void final_scan(int *pos, int n)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	if (pos[index] && threadIdx.x < pos[index] && index >= pos[index]
		&& pos[index - pos[index]])
		pos[index] += pos[index - pos[index]];
}

__global__ void init(const char *text, int *pos, int n)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n)
		return;
	if (text[index] != '\n')
		pos[index] = 1;
}
__global__ void slow(int *pos, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	int j;
	for (j = 0; i - j >= 0 &&  pos[i - j] != 0; j++);
	pos[i] = j;
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int grid_size = CeilDiv(text_size, BLOCK_SIZE);
	init<<<grid_size, BLOCK_SIZE>>>(text, pos, text_size);
	//slow<<<grid_size, BLOCK_SIZE>>>(pos, text_size);
	scan<<<grid_size, BLOCK_SIZE>>>(pos, text_size);
	final_scan<<<grid_size, BLOCK_SIZE>>>(pos, text_size);
}
