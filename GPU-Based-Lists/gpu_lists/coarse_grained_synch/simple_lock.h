#ifndef __SIMPLE_LOCK_H__
#define __SIMPLE_LOCK_H__

// BEWARE DIVERGENCE CONDITIONS WHEN USING
// THIS LOCK IMPLEMENTATION. BEWARE OF WARPS.
// USE TO PREVENT RACE CONDITIONS AMONG BLOCKS
// BUT NOT AMONG THREADS.


struct Lock {
  int *mutex;
  Lock( void ) {
    int state = 0;
    cudaMalloc( (void**)& mutex,
                              sizeof(int) );
    cudaMemcpy( mutex, &state, sizeof(int),
                              cudaMemcpyHostToDevice );
  }
  
  ~Lock( void ) {
    //cudaFree( mutex );
  }

  __device__ void lock( void ) {
    while( atomicCAS( mutex, 0, 1 ) != 0 );
  }

  __device__ void unlock( void ) {
    atomicExch( mutex, 0 );
  }
};

#endif
