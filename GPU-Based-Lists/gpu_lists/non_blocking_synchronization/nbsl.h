#ifndef __NBSL_H__
#define __NBSL_H__

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

__device__ Node* get_unmarked_reference(Node *ptr) {
	if( (unsigned int)ptr % 2 == 1) {
		ptr = (Node *) ((unsigned int) ptr - 1);
		return ptr;
	}
	return ptr;
}

__device__ Node* get_marked_reference (Node* ptr) {
        if( (unsigned int)ptr % 2 == 0) {
		ptr = (Node *) ((unsigned int) ptr + 1);
		return ptr;
	}
        return ptr;
}

__device__ bool is_marked_reference (Node* ptr) {
	if( (unsigned int)ptr % 2 == 0) return false;
	return true;
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

	__device__ Node* search (int key, Node **left_node) {
		Node *left_node_next, *right_node;
		search_again:
			do {
				Node *t = head;
				Node *t_next = head->next;
				/* 1: Find left_node and right_node */
				do {
					if (!is_marked_reference(t_next)) {
						(*left_node) = t;
						left_node_next = t_next;
					}
					t = get_unmarked_reference(t_next);
					if (t == tail) break;
					t_next = t->next;
				} while (is_marked_reference(t_next) || (t->key<key)); /*B1*/
				right_node = t;
				

				/* 2: Check nodes are adjacent */
				if (left_node_next == right_node)
					if ((right_node != tail) && is_marked_reference(right_node->next))
						goto search_again; /*G1*/
					else
						return right_node; /*R1*/
	
				/* 3: Remove one or more marked nodes */
				Node* old = (Node *) atomicCAS ((unsigned int *) &((*left_node)->next),
                                        (unsigned int)left_node_next,(unsigned int) right_node);

				if (old == left_node_next ) /*C1*/

					if ((right_node != tail) && is_marked_reference(right_node->next))
						goto search_again; /*G2*/
					else
						return right_node; /*R2*/
			} while(true);
	}

	__device__ bool insert_node(int key, T value) {
		Node *new_node = (Node *) malloc(sizeof (struct node));
		new_node->value = value;
		new_node->key = key;
		Node *right_node, *left_node;
		do {
			right_node = search (key, &left_node);
			if ((right_node != tail) && (right_node->key == key)) /*T1*/
				return false;
			new_node->next = right_node;

			Node* old = (Node *) atomicCAS ((unsigned int *)&(left_node->next),
                                (unsigned int)right_node,
                                (unsigned int)new_node);

			if (old == right_node) /*C2*/
				return true;
		} while (true); /*B3*/
	}

	__device__ bool delete_node(int key) {

		Node *right_node, *right_node_next, *left_node;
		do {
			right_node = search (key, &left_node);
			if ((right_node == tail) || (right_node->key != key)) /*T1*/
				return false;
		
			right_node_next = right_node->next;
			if (!is_marked_reference(right_node_next)) {
				Node* old = (Node *) atomicCAS ((unsigned int*) &(right_node->next), /*C3*/
                                        (unsigned int) right_node_next,
                                        (unsigned int) get_marked_reference (right_node_next));
				if (old == right_node_next)
					break;
			}
		} while (true); /*B4*/

		Node* old = (Node *) atomicCAS ((unsigned int *) &(left_node->next), (unsigned int)right_node,
                                        (unsigned int) right_node_next);
		if (old != right_node) /*C4*/
			right_node = search (right_node->key, &left_node);
		return true;
	
	}

	__device__ bool find_node (int key) {
		Node *right_node, *left_node;
		right_node = search (key, &left_node);
		if ((right_node == tail) ||
			(right_node->key != key))
			return false;
		else
		return true;
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
