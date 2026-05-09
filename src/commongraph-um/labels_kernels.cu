#include "labels_kernels.cuh"

__global__ void clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[id] = false;
	}
}

__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		int nID = activeNodes[id+from];
		label1[nID] = label1[nID] || label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int nID;
	if(id < size){
		nID = activeNodes[id+from];
		label1[nID] = label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		label1[id] = label2[id];
		label2[id] = false;	
	}
}

__global__ void moveUpLabelsNoClear(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		label1[id] = label2[id];
		// label2[id] = false;	
	}
}

__global__ void moveUpLabels(uint *queue1, uint *queue2, unsigned int* size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < *size){
		queue1[id] = queue2[id];	
	}
}

__global__ void copyValue(uint *value1, uint *value2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		value1[id] = value2[id];
	}
}

__global__ void mergeLabelsNoClear(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		if (label2[id])
			label1[id] = label2[id];
	}
}

