#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
// #include "commons/dependencydata.cuh"

// #include "graph.cuh"
#include "graph_kernels.cuh"


//    ==================================================================
//                             SSSP kernels
//    ==================================================================						


__global__ void sssp_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = tId + fromNode;

		if (label1[id] == false)
			return;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			if(edgeList[i].bitmap & ((ull)1 << sid)){
				uint dst = edgeList[i].end;
				uint new_value = value[id] + edgeList[i].w8;

				if (new_value < value[dst])
				{
					atomicMin(&value[dst], new_value);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void ssnp_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId + fromNode;

		if (label1[id] == false)
			return;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				uint new_value = max(value[id], edgeList[i].w8);

				if (new_value < value[dst])
				{
					atomicMin(&value[dst], new_value);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sswp_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId + fromNode;

		if (label1[id] == false)
			return;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				uint new_value = min(value[id], edgeList[i].w8);

				if (new_value > value[dst])
				{
					atomicMax(&value[dst], new_value);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void viterbi_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId + fromNode;

		if (label1[id] == false)
			return;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				uint new_value = value[id] / edgeList[i].w8;

				if (new_value > value[dst])
				{
					atomicMax(&value[dst], new_value);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void bfs_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId + fromNode;

		if (label1[id] == false)
			return;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				uint new_value = value[id] + 1;

				if (new_value < value[dst])
				{
					atomicMin(&value[dst], new_value);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void cc_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId + fromNode;

		if (label1[id] == false)
			return;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				uint new_value = value[id];

				if (new_value < value[dst])
				{
					atomicMin(&value[dst], new_value);
					label2[dst] = true;
				}
			}
		}
	}
}

