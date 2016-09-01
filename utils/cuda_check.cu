#ifndef __CUDA_CHECK_CU__
#define __CUDA_CHECK_CU__


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * Macro for printing error message
 */

# define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


/*
 * Synchronizing function for checking the cuda kernel 
 */

inline void device_synchronize(cudaStream_t stream)
{
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
                CUT_CHECK_ERROR("something gone wrong\n");
        }

}

/* 
 * Macro for cuda API checking. 
 * Example: checkCuda(cudaMallocHost()); 
 */

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %sn",
		cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


#endif

