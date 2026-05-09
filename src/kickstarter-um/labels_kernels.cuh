#ifndef LABELS_KERNELS_CUH
#define LABELS_KERNELS_CUH

#include "commons/globals.hpp"
#include "commons/dependencydata.cuh"

// mark label of activeNodes[from ... size] to be false
__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from);

//mark all label to be false
__global__ void clearLabel(bool *label, unsigned int size);

// mark label2 of activeNodes[from ... size] to be false and label1 <- (label11 or label2)
__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

// mark label2 of activeNodes[from ... size] to be false and label1 <- label2
__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

// mark label2 to be false and label1 <- label2
__global__ void moveUpLabels(bool *label1, bool *label2, unsigned int size);

__global__ void moveUpLabels(uint *queue1, uint *queue2, unsigned int* size);

__global__ void save_old(unsigned int numNodes,
							uint *value,
							uint *value_old,
							uint *level,
							uint *level_old,
							int *parent,
							int *parent_old);

__global__ void save_old(unsigned int numNodes,
							DependencyData * depends,
							DependencyData * depends_old);

__global__ void check_old(unsigned int numNodes,
							DependencyData * depends,
							DependencyData * depends_old);

__global__ void copyQSize(uint *qs1, uint *qs2);

// __global__ void bitmapToQueue(bool * label, uint * queue, uint numNodes, uint* size);

#endif	//	LABELS_KERNELS_HPP