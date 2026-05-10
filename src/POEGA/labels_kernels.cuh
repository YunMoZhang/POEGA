#ifndef LABELS_KERNELS_CUH
#define LABELS_KERNELS_CUH

#include "commons/globals.hpp"
#include "commons/dependencydata.cuh"

// mark label of activeNodes[from ... size] to be false
__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from);

//mark all label to be false
__global__ void clearLabel(bool *label, unsigned int size);

__global__ void flipLabel(bool *label1, bool *label2, unsigned int size);

// mark label2 of activeNodes[from ... size] to be false and label1 <- (label11 or label2)
__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

// mark label2 of activeNodes[from ... size] to be false and label1 <- label2
__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

// mark label2 to be false and label1 <- label2
__global__ void moveUpLabels(bool *label1, bool *label2, unsigned int size);

__global__ void moveUpLabels(uint *queue1, uint *queue2, unsigned int *size);

__global__ void diffLabels(bool *label1, bool *label2, unsigned int size);

__global__ void moveUpLabelsNoClear(bool *label1, bool *label2, unsigned int size);

__global__ void mergeLabels(bool *label1, bool *label2, unsigned int size);

__global__ void mergeLabelsNoClear(bool *label1, bool *label2, unsigned int size);

__global__ void mergeLabelsTotal(bool *label2, bool *label2s, uint numSnap, unsigned int size);

__global__ void resetLabels(bool *label2, unsigned int size);

__global__ void copyValues(uint *val1, uint *val2, unsigned int size);

__global__ void copyLabels(bool *val1, bool *val2, unsigned int size);

__global__ void setNFLabels(bool *label2, bool *far_label2, uint value, uint nf_threshold, unsigned int size);

// __global__ void copyQSize(uint *qs1, uint *qs2);

// __global__ void save_old(unsigned int numNodes,
// 							uint *value,
// 							uint *value_old,
// 							uint *level,
// 							uint *level_old,
// 							int *parent,
// 							int *parent_old);

__global__ void save_old(unsigned int numNodes,
							DependencyData * depends,
							DependencyData * depends_old);

__global__ void transfromValue(unsigned int numNodes,
							DependencyData * depends,
							uint *value);

__global__ void transfromLevel(unsigned int numNodes,
							   DependencyData *depends,
							   uint *level);

__global__ void transfromDependtoValue(unsigned int numNodes,
								DependencyData *depends,
								uint *value,
								 uint numSnap,
								 uint sid);

__global__ void transfromDependtoDepend(unsigned int numNodes,
								DependencyData *depends,
								DependencyData *depends2,
								uint numSnap,
								 uint sid);

// __global__ void bitmapToQueue(bool * label, uint * queue, uint numNodes, uint* size);

__global__ void sssp_setActive(uint *value, bool *label2, unsigned int size);

__global__ void copyValue(uint *value1, uint *value2, unsigned int size);

__global__ void set_same(bool *same, uint *value, uint num_nodes, uint num_snap);

__global__ void set_same(bool *same, uint *value, bool *label, uint num_nodes, uint num_snap);

__global__ void set_max_min(uint *max, uint *min, uint *value, uint num_nodes, uint num_snap);

__global__ void set_max_min(uint *max, uint *min, uint *value, uint * buffer, uint * buffer_total, uint num_nodes, uint num_snap);

__global__ void set_max_min_waitfree(uint *max, uint *min, uint *value, uint * buffer, uint * buffer_total, uint num_nodes, uint num_snap);

__global__ void update_max_min(uint *max, uint *min, uint *value, bool *label, uint num_nodes, uint num_snap);

__global__ void find_same(uint *max, uint *min, uint *off, uint num_nodes, uint num_snap);

__global__ void expand_same(uint *value, bool *label, uint num_nodes, uint num_snap);

__global__ void set_same_degree(bool *same, uint *deg, uint *nodePointer, uint num_nodes);

__global__ void update_value_old(uint *value, uint *old_value, uint *off, uint *deg, bool *label, uint num_nodes);

__global__ void makeQueueFromBitmap(unsigned int *activeNodes, bool *activeNodesLabeling,
							unsigned int *prefixLabeling, unsigned int numNodes);
__global__ void makeQueueFromBitmapPowerlaw(
    unsigned int *activeNodes,       
    bool *activeNodesLabeling,   
    unsigned int *prefixLabeling,    
	uint * chunkOffset,
    unsigned int numNodes);
#endif	//	LABELS_KERNELS_HPP