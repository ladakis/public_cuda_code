#include <stdio.h>
#include "fgsl.h"

#define NBLOCKS_TRUE 512
#define NTHREADS_TRUE 512 * 2

__global__ void kernel1(List list) {
	if(threadIdx.x == 0) {
		list.insert_node(blockIdx.x,NULL);
	
		if(blockIdx.x % 2 == 0)
			list.delete_node(blockIdx.x);
	}
}

__global__ void printList(List list) {
	//list.printlist();
}

int main() {

  int nblocks_host = 0;

  List list;
  float elapsedTime;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );

  ///kernel
  kernel1<<<NBLOCKS_TRUE,NTHREADS_TRUE>>>(list);

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &elapsedTime, start, stop );

  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  printf("blockCounter1 <<< %d, %d >>> () counted %d blocks in %f ms.\n",
        NBLOCKS_TRUE,
        NTHREADS_TRUE,
        nblocks_host,
        elapsedTime);

 printList<<<1,1>>>(list);
 if(cudaDeviceSynchronize()!=cudaSuccess)
 	printf("Error at GPU kernel \n");
}
