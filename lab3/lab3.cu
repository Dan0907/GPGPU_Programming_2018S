#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt) {
		if (mask[curt] > 127.0f && yt < ht - 1 && yt > 0
			&& xt > 0 && xt < wt - 1) {
			fixed[curt*3+0] = target[curt*3+0];
			fixed[curt*3+1] = target[curt*3+1];
			fixed[curt*3+2] = target[curt*3+2];
		} else {
			const int yb = oy+yt, xb = ox+xt;
			const int curb = wb*yb+xb;
			if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
				fixed[curt*3+0] = background[curb*3+0];
				fixed[curt*3+1] = background[curb*3+1];
				fixed[curt*3+2] = background[curb*3+2];
			}
		}
	}
}

__global__ void PoissonImageCloningIteration(
	float *fixed,
	const float *mask,
	float *buf1,
	float *buf2,
	const int wt, const int ht
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	int Nt = wt*(yt-1)+xt;
	int St = wt*(yt+1)+xt;
	int Wt = wt*yt+(xt-1);
	int Et = wt*yt+(xt+1);
	float *a, *b, *c, *d, *e, *f, *g, *h;
	if (yt < ht and xt < wt) {
		if (mask[curt] > 127.0f && yt < ht - 1 && yt > 0
			&& xt > 0 && xt < wt - 1) {
			if (yt == 1 || mask[Nt] <= 127.0f) {
				if (yt == 1) {
					int t1 = wt*1+xt;
					int t2 = xt;
					a = fixed+3*t1;
					e = fixed+3*t2;
				} else {
					a = buf1+3*Nt;
					e = fixed+3*Nt;	
				}
			} else {
				a = fixed+3*Nt;
				e = buf1+3*Nt;
			}
			if (xt == 1 || mask[Wt] <= 127.0f) {
				if (xt == 1) {
					int t1 = wt*yt+1;
					int t2 = wt*yt;
					b = fixed+3*t1;
					f = fixed+3*t2;
				} else {
					b = buf1+3*Wt;
					f = fixed+3*Wt;	
				}
			} else {
				b = fixed+3*Wt;
				f = buf1+3*Wt;
			}
			if (yt == ht - 2 || mask[St] <= 127.0f) {
				if (yt == ht - 2) {
					int t1 = wt*(ht-2)+xt;
					int t2 = wt*(ht-1)+xt;
					c = fixed+3*t1;
					g = fixed+3*t2;
				} else {
					c = buf1+3*St;
					g = fixed+3*St;	
				}
			} else {
				c = fixed+3*St;
				g = buf1+3*St;
			}
			if (xt == wt - 2 || mask[Et] <= 127.0f) {
				if (xt == wt - 2) {
					int t1 = wt*yt+wt-2;
					int t2 = wt*yt+wt-1;
					d = fixed+3*t1;
					h = fixed+3*t2;
				} else {
					d = buf1+3*Et;
					h = fixed+3*Et;	
				}
			} else {
				d = fixed+3*Et;
				h = buf1+3*Et;
			}
			for (int i = 0; i < 3; i++) {
				buf2[3*curt+i] = 1.0f/4.0f
					*(4*fixed[3*curt+i]-a[i]-b[i]
						-c[i]-d[i]+e[i]
						+f[i]+g[i]+h[i]);
			}
		} else if (mask[curt] > 127.0f) {
			for (int i = 0; i < 3; i++)
				buf2[3*curt+i] = fixed[3*curt+i];
		} else {
			for (int i = 0; i < 3; i++)
				buf2[3*curt+i] = buf1[3*curt+i];
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht,
		cudaMemcpyDeviceToDevice);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed, wb, hb, wt, ht, oy, ox
	);

	for (int i = 0; i < 10000; i++) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
