
.PHONY: all

all:
	nvcc -std=c++11 -I ../utils/ main.cu lab1.cu
	./a.out
	ffmpeg -y -i result.y4m a.mkv

