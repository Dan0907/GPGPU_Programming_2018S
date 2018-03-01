#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 12;

__global__ void Draw(char *frame) {
	// TODO: draw more complex things here
	// Do not just submit the original file provided by the TA!
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H and x < W) {
		char c;
		if (x == W-1) {
			c = y == H-1 ? '\0' : '\n';
		} else if (y == 0 or y == H-1 or x == 0 or x == W-2) {
			c = ':';
		} else if ((x == 10 or x == 28) and y >= 4 and y <= 8) {
			c = 'o';
		} else if ((y == 4 or y == 8) and x > 10 and x < 28) {
			c = 'o';
		} else if (x == 19 and y >= 2 and y <=11) {
			c = 'o';
		} else {
			c = ' ';
		}
		frame[y*W+x] = c;
	}
}

int main(int argc, char **argv)
{
	MemoryBuffer<char> frame(W*H);
	auto frame_smem = frame.CreateSync(W*H);
	CHECK;

	Draw<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}
