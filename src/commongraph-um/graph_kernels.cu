#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
// #include "commons/dependencydata.cuh"

// #include "graph.cuh"
#include "graph_kernels.cuh"


//    ==================================================================
//                             SSSP kernels
//    ==================================================================						

__global__ void sssp_kernel_cg(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int num_snap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if(label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint finalVal;

		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
			{
				finalVal = value[id] + edgeList[i].w8;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				
				while (finalVal < cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}


// __global__ void sssp_kernel_cg(unsigned int numNodes,
// 							   unsigned int from,
// 							   unsigned int numPartitionedEdges,
// 							   unsigned int *activeNodes,
// 							   unsigned int *activeNodesPointer,
// 							   OutEdgeWeighted_Evolving *edgeList,
// 							   unsigned int *outDegree,
// 							   bool *label1,
// 							   bool *label2,
// 							   uint *value,
// 							   int num_snap)
// {
// 	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

// 	if(tId < numNodes)
// 	{
// 		unsigned int id = activeNodes[from + tId];

// 		if(label1[id] == false)
// 			return;

// 		label1[id] = false;

// 		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
// 		unsigned int degree = outDegree[id];
// 		unsigned int thisTo = thisFrom + degree;

// 		uint finalVal;

// 		for(unsigned int i=thisFrom; i<thisTo; i++)
// 		{
// 			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
// 			{
// 				finalVal = value[id] + edgeList[i].w8;
// 				uint dst = edgeList[i].end;
// 				uint cur_val = value[dst];
				
// 				while (finalVal < cur_val)
// 				{
// 					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
// 					{
// 						label2[dst] = true;
// 						break;
// 					}
// 					cur_val = value[dst];
// 				}
// 			}
// 		}
// 	}
// }


__global__ void sssp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint src_value = dist[id];
		
		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			if ((e.bitmap & mask) == mask)
			{
				finalDist = src_value + e.w8;
				uint dst = e.end;

				if (finalDist < dist[dst])
				{
					atomicMin(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sssp_kernel_concurrent_cg(unsigned int numNodes,
									   unsigned int from,
									   unsigned int numPartitionedEdges,
									   unsigned int *activeNodes,
									   unsigned int *activeNodesPointer,
									   OutEdgeWeighted_Evolving *edgeList,
									   unsigned int *outDegree,
									   bool *label1,
									   bool *label2,
									   uint **value,
									   uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					new_val = value[sid][id] + weight;
					if (new_val < value[sid][dst])
					{
						atomicMin(&value[sid][dst], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sssp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				finalDist = value[id] + edgeList[i].w8;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				while (finalDist < cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalDist) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}


__global__ void sssp_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint finalDist;

		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & ((ull)1 << sid)) && ((edgeList[i].bitmap & mask) != mask))
			{
				finalDist = value[id] + edgeList[i].w8;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				while (finalDist < cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalDist) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}

//    ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void sswp_kernel_cg(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int num_snap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint finalVal;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
			{
				finalVal = min(value[id], edgeList[i].w8);
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];

				while (finalVal > cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}

__global__ void sswp_kernel_concurrent_cg(unsigned int numNodes,
										  unsigned int from,
										  unsigned int numPartitionedEdges,
										  unsigned int *activeNodes,
										  unsigned int *activeNodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  unsigned int *outDegree,
										  bool *label1,
										  bool *label2,
										  uint **value,
										  uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					new_val = min(value[sid][id], weight);
					if (new_val > value[sid][dst])
					{
						atomicMax(&value[sid][dst], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}



__global__ void sswp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint src_value = dist[id];
		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			if ((e.bitmap & mask) == mask)
			{
				finalDist = min(src_value, e.w8);
				uint dst = e.end;

				if (finalDist > dist[dst])
				{
					atomicMax(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void sswp_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint finalDist;

		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & ((ull)1 << sid)) && ((edgeList[i].bitmap & mask) != mask))
			{
				finalDist = min(value[id], edgeList[i].w8);
				uint dst = edgeList[i].end;

				if (finalDist > value[dst])
				{
					atomicMax(&value[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void sswp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				finalDist = min(value[id], edgeList[i].w8);
				uint dst = edgeList[i].end;

				if (finalDist > value[dst])
				{
					atomicMax(&value[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}

//    ==================================================================
//                             SSNP kernels
//    ==================================================================


__global__ void ssnp_kernel_cg(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int num_snap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint finalVal;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
			{
				finalVal = max(value[id], edgeList[i].w8);
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];

				while (finalVal < cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}

__global__ void ssnp_kernel_concurrent_cg(unsigned int numNodes,
										  unsigned int from,
										  unsigned int numPartitionedEdges,
										  unsigned int *activeNodes,
										  unsigned int *activeNodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  unsigned int *outDegree,
										  bool *label1,
										  bool *label2,
										  uint **value,
										  uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					new_val = max(value[sid][id], weight);
					if (new_val < value[sid][dst])
					{
						atomicMin(&value[sid][dst], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}


__global__ void ssnp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint src_value = dist[id];
		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			if ((e.bitmap & mask) == mask)
			{
				finalDist = max(src_value, e.w8);
				uint dst = e.end;

				if (finalDist < dist[dst])
				{
					atomicMin(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}



__global__ void ssnp_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint finalDist;

		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & ((ull)1 << sid)) && ((edgeList[i].bitmap & mask) != mask))
			{
				finalDist = max(value[id], edgeList[i].w8);
				uint dst = edgeList[i].end;

				if (finalDist < value[dst])
				{
					atomicMin(&value[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void ssnp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				finalDist = max(value[id], edgeList[i].w8);
				uint dst = edgeList[i].end;

				if (finalDist < value[dst])
				{
					atomicMin(&value[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


//    ==================================================================
//                             Viterbi kernels
//    ==================================================================

__global__ void viterbi_kernel_cg(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int num_snap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint finalVal;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
			{
				finalVal = value[id] / edgeList[i].w8;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];

				while (finalVal > cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}

__global__ void viterbi_kernel_concurrent_cg(unsigned int numNodes,
										  unsigned int from,
										  unsigned int numPartitionedEdges,
										  unsigned int *activeNodes,
										  unsigned int *activeNodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  unsigned int *outDegree,
										  bool *label1,
										  bool *label2,
										  uint **value,
										  uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					new_val = value[sid][id]/ weight;
					if (new_val > value[sid][dst])
					{
						atomicMin(&value[sid][dst], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void viterbi_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint src_value = dist[id];
		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			if ((e.bitmap & mask) == mask)
			{
				finalDist = src_value / e.w8;
				uint dst = e.end;

				if (finalDist > dist[dst])
				{
					atomicMax(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void viterbi_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint finalDist;

		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & ((ull)1 << sid)) && ((edgeList[i].bitmap & mask) != mask))
			{
				finalDist = value[id] / edgeList[i].w8;
				uint dst = edgeList[i].end;

				if (finalDist > value[dst])
				{
					atomicMax(&value[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void viterbi_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				finalDist = value[id] / edgeList[i].w8;
				uint dst = edgeList[i].end;

				if (finalDist > value[dst])
				{
					atomicMax(&value[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}
//    ==================================================================
//                             BFS kernels
//    ==================================================================

__global__ void bfs_kernel_cg(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int num_snap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint finalVal;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
			{
				finalVal = value[id] + 1;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];

				while (finalVal < cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}

__global__ void bfs_kernel_concurrent_cg(unsigned int numNodes,
										  unsigned int from,
										  unsigned int numPartitionedEdges,
										  unsigned int *activeNodes,
										  unsigned int *activeNodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  unsigned int *outDegree,
										  bool *label1,
										  bool *label2,
										  uint **value,
										  uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		// uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			// weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					new_val = value[sid][id] + 1;
					if (new_val < value[sid][dst])
					{
						atomicMin(&value[sid][dst], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}



__global__ void bfs_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint finalDist;

		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & ((ull)1 << sid)) && ((edgeList[i].bitmap & mask) != mask))
			{
				finalDist = value[id] + 1;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				if(finalDist < cur_val)
				{
					atomicMin(&value[dst], finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void bfs_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				finalDist = value[id] + 1;
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				if(finalDist < cur_val)
				{
					atomicMin(&value[dst], finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void bfs_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint src_value = dist[id];
		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			if ((e.bitmap & mask) == mask)
			{
				finalDist = src_value + 1;
				uint dst = e.end;

				if (finalDist < dist[dst])
				{
					atomicMin(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}

//    ==================================================================
//                             CC kernels
//    ==================================================================

__global__ void cc_kernel_cg(unsigned int numNodes,
							  unsigned int from,
							  unsigned int numPartitionedEdges,
							  unsigned int *activeNodes,
							  unsigned int *activeNodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  uint *value,
							  int num_snap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint finalVal;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << num_snap) - 1)) == (((ull)1 << num_snap) - 1))
			{
				finalVal = value[id];
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];

				while (finalVal < cur_val)
				{
					if (atomicCAS(&value[dst], cur_val, finalVal) == cur_val)
					{
						label2[dst] = true;
						break;
					}
					cur_val = value[dst];
				}
			}
		}
	}
}

__global__ void cc_kernel_concurrent_cg(unsigned int numNodes,
										 unsigned int from,
										 unsigned int numPartitionedEdges,
										 unsigned int *activeNodes,
										 unsigned int *activeNodesPointer,
										 OutEdgeWeighted_Evolving *edgeList,
										 unsigned int *outDegree,
										 bool *label1,
										 bool *label2,
										 uint **value,
										 uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		// uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			// weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					new_val = value[sid][id];
					if (new_val < value[sid][dst])
					{
						atomicMin(&value[sid][dst], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}



__global__ void cc_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint src_value = dist[id];
		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			if ((e.bitmap & mask) == mask)
			{
				finalDist = src_value;
				uint dst = e.end;

				if (finalDist < dist[dst])
				{
					atomicMin(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}



__global__ void cc_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint finalDist;

		ull mask = ((ull)1 << num_snap) - 1;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & ((ull)1 << sid)) && ((edgeList[i].bitmap & mask) != mask))
			{
				finalDist = value[id];
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				if(finalDist < cur_val)
				{
					atomicMin(&value[dst], finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void cc_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		uint finalDist;
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				finalDist = value[id];
				uint dst = edgeList[i].end;
				uint cur_val = value[dst];
				if(finalDist < cur_val)
				{
					atomicMin(&value[dst], finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}

