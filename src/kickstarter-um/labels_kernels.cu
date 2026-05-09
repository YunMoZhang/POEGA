#include "labels_kernels.cuh"
// #include "commons/dependencydata.cuh"


__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[activeNodes[id+from]] = false;
	}
}

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

__global__ void moveUpLabels(uint *queue1, uint *queue2, unsigned int* size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < *size){
		queue1[id] = queue2[id];	
	}
}

__global__ void save_old(unsigned int numNodes,
							uint *value,
							uint *value_old,
							uint *level,
							uint *level_old,
							int *parent,
							int *parent_old)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes)
	{	
		value_old[tId] = value[tId];
		level_old[tId] = level[tId];
		parent_old[tId] = parent[tId];
	}	
}

__global__ void save_old(unsigned int numNodes,
							DependencyData * depends,
							DependencyData * depends_old)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes)
	{	
		depends_old[tId] = depends[tId];
	}	
}

// __global__ void check_old(unsigned int numNodes,
// 							DependencyData * depends,
// 							DependencyData * depends_old)
// {
// 	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(tId < numNodes)
// 	{	
// 		if(depends[tId].value < depends_old[tId].value)
// 			printf("[check error] %u: %u, %u\n", tId, depends[tId].value, depends_old[tId].value);
// 	}	
// }

__global__ void copyQSize(uint *qs1, uint *qs2)
{
	*qs1 = *qs2;
	*qs2 = 0;
}

// __global__ void bitmapToQueue(bool * label, uint * queue, uint numNodes, uint* size)
// {
// 	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(tId < numNodes)
// 	{	
// 		if(label[tId]){
// 			uint index = atomicAdd(size, 1);
// 			queue[index] = tId;
// 		}
// 	}	
// }

