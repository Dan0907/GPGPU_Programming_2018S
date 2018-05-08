#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256
#define SWAP(a, b) do {\
	int *tmp_ptr = a;\
	a = b;\
	b = tmp_ptr;\
} while (0)

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

__global__ void scan(int *pos, int *tmp_arr, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	if (i >= tmp_arr[i]) {
		pos[i] = tmp_arr[i] + tmp_arr[i - tmp_arr[i]];
	} else {
		pos[i] = tmp_arr[i];
	}
}

__global__ void init(const char *text, int *pos, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	if (text[i] != '\n')
		pos[i] = 1;
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int i;
	int grid_size = CeilDiv(text_size, BLOCK_SIZE);
	int *ori_ptr = pos;
	int *tmp_arr;
	cudaMalloc(&tmp_arr, sizeof *tmp_arr * text_size);
	init<<<grid_size, BLOCK_SIZE>>>(text, pos, text_size);
	for (i = 0; i < 9; i++) {
		SWAP(pos, tmp_arr);
		scan<<<grid_size, BLOCK_SIZE>>>(pos, tmp_arr, text_size);
	}
	if (ori_ptr != pos) {
		cudaMemcpy(ori_ptr, pos, sizeof *ori_ptr * text_size,
			cudaMemcpyDeviceToDevice);
		cudaFree(pos);
	} else {
		cudaFree(tmp_arr);
	}
}
