#ifndef __CGSL_H__
#define __CGSL_H__

#define MININT -1
#define MAXINT 1025

#include "simple_lock.h"

typedef void* T;

typedef struct node {
	T value;
	int key;
	struct node *next;	
}Node;

__global__ void initKey(Node *ptr,int key,Node *next) {
	ptr->value = NULL;
	ptr->key = key;
	ptr->next = next;
}

struct List { 
	Lock lc;
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

	__device__ bool search_node (int key) {
		Node *curr;
		bool result;
		
		lc.lock();

		curr = head;
		while(curr->key < key) {
			//printf("curr->key %d\n",curr->key);
			curr = curr->next;
		}
		if(key == curr->key) result = true;
		else result = false;
		lc.unlock();
		
		return result;
	}

	__device__ bool insert_node (int key, T x) {
		//code for process p
		Node *pred, *curr;
		bool result;
		
		lc.lock();
		pred = head;
		
		curr = pred->next;
		while(curr->key < key) {
			pred = curr;
			curr = curr->next;
		}
		
		if(key == curr->key ) result = false;
		else {
			Node* node =(Node *) malloc(sizeof(Node));
			node->next = curr;
			node->value = x;
			node->key = key;
			pred->next = node;
			result = true;
			
		}

		lc.unlock();
		return result;
	}

	__device__ bool delete_node (int key) {
		//code for process p
		Node *pred,*curr;
		bool result;
		
		lc.lock();
		pred = head;
		curr = pred->next;

		while(curr->key < key) {
			pred = curr; 
			curr = curr->next;
		}
		
		if (key == curr->key) {
			pred->next = curr->next;
			free(curr);
			result = true;
		}
		else result = false;
		lc.unlock();
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
