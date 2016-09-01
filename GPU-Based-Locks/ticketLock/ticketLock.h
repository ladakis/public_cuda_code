#ifndef __TICKET__LOCK_H__
#define __TICKET__LOCK_H__

#define EBUSY 1

typedef union ticketlock ticketlock;

union ticketlock
{
	unsigned u;
	struct
	{
		unsigned int ticket;
		unsigned int users;
	} s;
};

struct Lock {

 ticketlock* t;
 Lock() {
	cudaMalloc((void**)&t, sizeof(ticketlock));
	cudaMemset((void *) t, 0, sizeof(ticketlock));
 }


 ~Lock() {
 	cudaFree(t);
 }


 __device__ void lock()
 {
	unsigned int me = atomicAdd(&t->s.users, 1);

	int loop = 1;
	do{
		unsigned int tick = atomicAdd(&t->s.ticket,0);
		if(tick == me)
			loop = 0;	
		
	}while(loop);
 }

 __device__ int trylock()
 {
	unsigned int me = t->s.users;
        unsigned int menew = me + 1;
        unsigned cmp = ((unsigned) me << 16) + me;
        unsigned cmpnew = ((unsigned) menew << 16) + me;

        if (atomicCAS(&t->u, cmp, cmpnew) == cmp) return 0;
        
        return EBUSY;
 }

 __device__ void unlock()
 {
        __syncthreads();
        t->s.ticket++;
 }

};

#endif
