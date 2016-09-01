#ifndef __ONE_BIT_H__
#define __ONE_BIT_H__

#define TRUE 1
#define FALSE 0


struct Lock {
 unsigned int *arraybit;
 int nthreads;
 
 Lock(int n) {
	nthreads = n;
	cudaMalloc((void **)&arraybit, sizeof(unsigned int) * nthreads);
	cudaMemset(arraybit,0,sizeof(unsigned int) * nthreads);
	
 }
 
 ~Lock() {
	cudaFree(arraybit);
 }

 
 __device__ void lock() {
	int index = blockIdx.x;
	int j;
	do {	
		j = 1;
		atomicExch (&arraybit[index], TRUE);
		while (arraybit[index] == TRUE && j<1) 
		{
			if(atomicAdd( &arraybit[j],0) == TRUE) {
				atomicExch(&arraybit[index], FALSE);

				int loop = 1;
				do {
					unsigned int bj = atomicAdd(&arraybit[j],0);
					if(bj == FALSE)
						loop = 0;
				}while(loop);
			}
			j++;
		}

	}while(atomicAdd(&arraybit[index],0) == FALSE);
	
 	
	for(j = index+1; j<nthreads; j++)
	{
		int loop = 1;
		do{
			unsigned int bj = atomicAdd(&arraybit[j],0);
			if(bj == FALSE)
				loop = 0;
		}while(loop);
	}
 }

 __device__ void unlock () {
	int index = blockIdx.x;
	atomicExch(&arraybit[index],FALSE);
 }

};


#endif
