#ifndef __SIMPLE_LOCK_H__
#define __SIMPLE_LOCK_H__

// BEWARE DIVERGENCE CONDITIONS WHEN USING
// THIS LOCK IMPLEMENTATION. BEWARE OF WARPS.
// USE TO PREVENT RACE CONDITIONS AMONG BLOCKS
// BUT NOT AMONG THREADS.

typedef int* Lock;
  
__device__ void lock( Lock mutex ) {
    while( atomicCAS( (int *) mutex, 0, 1 ) != 0 );
}

__device__ void unlock( Lock mutex ) {
    atomicExch( (int *) mutex, 0 );
}


#endif
