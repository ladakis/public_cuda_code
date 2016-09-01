#ifndef __LSL_H__
#define __LSL_H__

#define MININT -1
#define MAXINT 1025

#include "simple_lock.h"

typedef void* T;

typedef struct node {
        T value;
        int key;
	bool marked;
        Lock lock;
        struct node *next;
}Node;



__global__ void initKey(Node *ptr,int key,Node *next) {
        ptr->value = NULL;
        ptr->key = key;
	ptr->marked = false;
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

	__device__ bool validate(Node* pred, Node* curr) {
		atomicAdd((unsigned int *)pred, 0);
		atomicAdd((unsigned int *)curr, 0);
		if (pred->marked == false &&
			curr->marked == false &&
			pred->next == curr) return true;
		else	return false ;
		/*because of GPU's weak Memory model validate can't run succesfully. I don't know how to fix this.*/
	}

	__device__ bool search_node (int key) {
		Node *curr;
		while (true) {
			atomicAdd((unsigned int *)head, 0); //fetch to cache
			curr = head;
			while (curr->key < key) {
				atomicAdd((unsigned int *)curr, 0); //fetch to cache
				curr = curr->next;
			}
			if (curr->marked !=true && key == curr->key)
				return true;
			else return false;
		}

	}
	
	__device__ bool insert_node(int key, T value) {

		Node *pred, *curr;
                bool result;
                int return_flag = 0;

		while (true) {
                        atomicAdd((unsigned int *)head, 0);
			atomicAdd((unsigned int *)head->next, 0);
                        pred = head; curr = pred->next;
                        while (curr->key < key) {
				atomicAdd((unsigned int *)curr, 0);
                                pred = curr;
                                curr = curr->next;
				atomicAdd((unsigned int *)curr, 0);
                        }

			lockNode(pred); lockNode(curr);
			if (validate(pred, curr) == true) {
                                if (key == curr->key) {
                                        result = false; return_flag = 1;
                                }
				else {

					Node *node = (Node *) malloc(sizeof(Node));
                                        node->lock = (int * ) malloc(sizeof(int));
                                        *(node->lock) = 0;
                                        atomicExch((unsigned int *)&(node->next),(unsigned int)curr); //node->next = curr;
                                        node->value = value;
                                        node->key = key;
					node->marked = false;
                                        atomicExch((unsigned int *)&pred->next,(unsigned int) node); //pred->next = node;
                                        atomicAdd((unsigned int *)node,0); //force writeback
					result = true; return_flag = 1;

				}
			}
			unlockNode(pred); unlockNode(curr);
			if (return_flag) return result;

		}

	}

	__device__ bool delete_node(int key) {
		// code for process p
                Node *pred, *curr;
                bool result; int return_flag = 0;
                while (true) {
                        atomicAdd((unsigned int *)head, 0);
                        pred = head; curr = pred->next;
                        while (curr->key < key) {
                                atomicAdd((unsigned int *)curr, 0);
                                pred = curr;
                                curr = curr->next;
                        }
                        lockNode(pred); lockNode(curr);
			if(validate(pred,curr)) {
                        	if (key == curr->key) {
					atomicExch((unsigned int *)&curr->marked,(unsigned int)true);
                                	atomicExch((unsigned int *)&pred->next,(unsigned int) curr->next); //pred->next = curr->next;
                                	result = true;
                                	return_flag = 1;
                        	}
                        	else {
                                	result = false;
                                	return_flag = 1;
                        	}
			}
                        unlockNode(pred); unlockNode(curr);
                        if (return_flag == 1) return result;

                }
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
