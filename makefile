NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30
CC_FLAGS = -c
#NVCCFLAGS = -m64 -O3 -Xptxas -v
#uncomment NVCCFLAGS below and comment out above, if you want to use cuda-gdb
NVCCFLAGS = -g -G -m64 --compiler-options -Wall -O3
OBJS = cuda_radix.o wrappers.o d_radix.o
.SUFFIXES: .cu .o .h
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -lcrypto $< -o $@

all: cuda_radix

cuda_radix: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -lcrypto -o cuda_radix

cuda_radix.o: cuda_radix.cu wrappers.h d_radix.h

d_radix.o: d_radix.cu d_radix.h CHECK.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm cuda_radix *.o
