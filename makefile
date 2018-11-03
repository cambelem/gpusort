NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30
CC_FLAGS = -c
#NVCCFLAGS = -m64 -O3 -Xptxas -v
#uncomment NVCCFLAGS below and comment out above, if you want to use cuda-gdb
NVCCFLAGS = -g -G -m64 --compiler-options -Wall -O3
OBJS = hybrid_sort.o wrappers.o d_hybrid_sort.o
.SUFFIXES: .cu .o .h
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -lcrypto $< -o $@

all: hybrid_sort

hybrid_sort: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -lcrypto -o hybrid_sort

hybrid_sort.o: hybrid_sort.cu wrappers.h d_hybrid_sort.h

d_hybrid_sort.o: d_hybrid_sort.cu d_hybrid_sort.h CHECK.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm hybrid_sort *.o
