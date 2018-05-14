
.PHONY: all run run2 check check2 clean

all: a.out
a.out: lab3.cu
	nvcc -std=c++11 -O2 -arch sm_50 -I../utils/  main.cu lab3.cu pgm.cpp

run: a.out
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 600\
 output.ppm 

run2: a.out
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 900\
 output.ppm 

check: a.out
	cuda-memcheck a.out img_background.ppm img_target.ppm img_mask.pgm\
 130 600 output.ppm 

check2: a.out
	cuda-memcheck a.out img_background.ppm img_target.ppm img_mask.pgm\
 130 900 output.ppm 
	
clean:
	-rm -f a.out
