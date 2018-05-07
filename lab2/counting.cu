#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::device_ptr<const char> data_ptr(text);
	thrust::device_ptr<int> result_ptr(pos);
	struct IsChar : public thrust::unary_function<const char, int>
	{
		__host__ __device__
		int operator()(const char x)
		{
			if (x != '\n')
				return 1;
			else
				return 0;
		}
	} isChar;
	thrust::transform(thrust::device, data_ptr, data_ptr + text_size, result_ptr, isChar);
	thrust::inclusive_scan_by_key(thrust::device, result_ptr, result_ptr + text_size, result_ptr, result_ptr);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
}
