
.PHONY: all

all: a.out
a.out: counting.cu
	nvcc -std=c++11 -O2 -arch sm_30 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -I../utils/  main.cu counting.cu

run: a.out
	./a.out 
	
clean:
	-rm -f a.out
