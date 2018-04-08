#include "lab1.h"
#include "math.h"
#define PI 3.14159265358979323846
static const int W = 640;
static const int H = 480;
static const unsigned NFRAME = 720;

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__device__ void rgb2yuv(uint8_t r, uint8_t g, uint8_t b, uint8_t *y, uint8_t *u, uint8_t *v) {
	*y=(30*r+59*g+11*b)/100;
	*u=(-17*r-33*g+50*b)/100+128;
	*v=(50*r-42*g-8*b)/100+128;
}

__device__ void DrawRGB(uint8_t *frame, const int x, const int y,
	uint8_t r, uint8_t g, uint8_t b) {
	uint8_t Y, U, V;
	rgb2yuv(r,g,b,&Y,&U,&V);
	const int index_Y = y*W+x;
	frame[index_Y] = Y;
	if (x % 2 || y % 2)
		return;
	const int index_U = W*H+y*W/4+x/2;
	const int index_V = W*H+W*H/4+y*W/4+x/2;
	frame[index_U] = U;
	frame[index_V] = V;
}

__global__ void Draw(uint8_t *frame, int t) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	double speed = (double)t/20;
	if (x >= W || y >= H) {
		return;
	}
	double j, k;

	if (W>=H) {
		j = ((double)x + 0.5 * (H-W))/H;
		k = (double)y/H;
	} else {
		j = (double)x/H;
		k = ((double)y + 0.5 * (W-H))/H;
	}
	
	j = 2*j-1;
	k = -(2*k-1);

	double dist = sqrt(pow(j,2)+pow(k,2));
	
	double d_dist = pow(dist*5000, 0.3);
	
	if (k==0)
		k=0.001;
	if (j==0)
		j=0.001;
	double angle = fmod(fmod(atan(k/j) + d_dist - speed, PI) + PI, PI);
	
	uint8_t r, g, b;
	double ratio = angle/PI;

	r = (0.5+0.5*cos(speed/2 + PI*4/3))*255*ratio;
	g = (0.5+0.5*cos(speed/2 + PI*2/3))*255*ratio;
	b = (0.5+0.5*cos(speed/2))*255*ratio;

	DrawRGB(frame, x, y, r, g, b);	
	
}

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	Draw<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(yuv,impl->t);
	(impl->t)++;
}
	
