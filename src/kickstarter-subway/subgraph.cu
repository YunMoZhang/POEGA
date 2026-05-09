#include "subgraph.cuh"
#include "commons/gpu_error_check.hpp"
// #include "graph.cuh"
#include <cuda_profiler_api.h>


template <class E>
Subgraph<E>::Subgraph(uint num_nodes, uint num_edges)
{	
	cudaProfilerStart();
	cudaError_t error;
	cudaDeviceProp dev;
	int deviceID;
	cudaGetDevice(&deviceID);
	error = cudaGetDeviceProperties(&dev, deviceID);
	if(error != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaProfilerStop();
	
	cout << "dev.total global memory: " << dev.totalGlobalMem << endl;
	// this->max_partition_size = 0.7 * (dev.totalGlobalMem - 32 * 4 * num_nodes) / sizeof(E);
	this->max_partition_size = 0.75 * (dev.totalGlobalMem - 28 * 4 * (size_t)num_nodes) / sizeof(E);
	//max_partition_size = 1000000000;
	
	// if(max_partition_size > DIST_INFINITY)
	// 	max_partition_size = DIST_INFINITY;
	if(max_partition_size > num_edges)
		max_partition_size = num_edges;
	cout << "Max Partition Size: " << this->max_partition_size << endl;

	this->num_nodes = num_nodes;
	this->num_edges = num_edges;
	
	gpuErrorcheck(cudaMallocHost(&activeNodes, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeNodesPointer, (num_nodes+1) * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeEdgeList, num_edges * sizeof(E)));
	
	gpuErrorcheck(cudaMalloc(&d_activeNodes, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeNodesPointer, (num_nodes+1) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeEdgeList, (max_partition_size) * sizeof(E)));
}

template class Subgraph<OutEdge_Evolving>;
template class Subgraph<OutEdgeWeighted_Evolving>;