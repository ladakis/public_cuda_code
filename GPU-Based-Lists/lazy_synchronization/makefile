NVCC =  /usr/local/cuda-5.0/bin/nvcc
FLAGS = -arch=sm_20
GDB = /usr/local/cuda-5.0/bin/cuda-gdb

all:
	$(NVCC) testList.cu $(FLAGS) -o testList
debug:
	$(GDB) testList

clean:
	rm testList

