#ifndef __FGSL_H__
#define __FGSL_H__

#define MININT -1
#define MAXINT 1025

#include "simple_lock.h"

typedef void* T;

typedef struct node {
        T value;
        int key;
	Lock lock;
        struct node *next;
}Node;


__global__ void initKey(Node *ptr,int key,Node *next) {
        ptr->value = NULL;
        ptr->key = key;
        ptr->next = next;
	ptr->lock = (int *) malloc(sizeof(int));
	*(ptr->lock) = 0;
}

struct List {
        Node *head,*tail;


        List() {
                cudaMalloc((void **)&head,sizeof(Node));
                cudaMalloc((void **)&tail,sizeof(Node));
                cudaMemset((void *)head,0,sizeof(Node));
                cudaMemset((void *)tail,0,sizeof(Node));

                initKey<<<1,1>>>(tail,MAXINT,NULL);
                initKey<<<1,1>>>(head,MININT,tail);
        }

        ~List() {
		//cudaFree(head);
                //cudaFree(tail);
        }

	__device__ void lockNode(Node *ptr) {
                lock(ptr->lock);
                atomicAdd((unsigned int *)ptr, 0); //force writeback
        }

        __device__ void unlockNode(Node *ptr) {
                unlock(ptr->lock);
                atomicAdd((unsigned int *)ptr, 0); //force writeback
        }

        __device__ bool search_node (int key) {
                Node *curr, *pred;
                bool result;

		lockNode(head);
		pred = head;
		curr = pred->next;
		lockNode(curr);
		while(curr->key < key) {
			unlockNode(pred);
			pred = curr;
			curr = curr->next;
			lockNode(curr);
		}
		
		if (key == curr->key) result = true;
		else result = false;
		unlockNode(pred); 
		unlockNode(curr);
		return result;
	}


	__device__ bool insert_node(int key, T value) {
		
		Node *pred, *curr; 
		bool result;
		
		lockNode(head);
		pred = head;
		curr = pred->next;		
		lockNode(curr);

		while (curr->key < key) {
			unlockNode(pred);

			pred = curr;
			curr = curr->next;

			lockNode(curr);
		}
		if (key == curr->key) result = false;
		else {
			Node *node = (Node *) malloc(sizeof(Node));
			node->lock = (int * ) malloc(sizeof(int));
                        *(node->lock) = 0;

			atomicExch((unsigned int *)&(node->next),(unsigned int)curr); //node->next = curr;

			node->value = value;
			node->key = key;

			atomicExch((unsigned int *)&pred->next,(unsigned int) node); //pred->next = node;
			//atomicAdd((unsigned int *)node,0); //force writeback
			result = true;
		}
		unlockNode(pred);
		unlockNode(curr);
		return result;
		
	}

	__device__ bool delete_node(int key) {
		// code for process p
		Node *pred, *curr;
		bool result;
		lockNode(head);
		pred = head;
		curr = pred->next;

		lockNode(curr);
		while (curr->key < key) {
			unlockNode(pred);
			pred = curr;
			curr = curr->next;

			lockNode(curr);
		}
		if (key == curr->key) {
			atomicExch((unsigned int *)&pred->next,(unsigned int) curr->next); //pred->next = curr->next;
			result = true;
		}
		else result = false;

		unlockNode(pred);
		unlockNode(curr);
		return result;
	}

	__device__ void  printlist() {
		Node * curr;
		curr = head;
		while(curr) {
			printf("curr->key = %d \n",curr->key);
			curr = curr->next;
		}
	
	}

};

#endif
