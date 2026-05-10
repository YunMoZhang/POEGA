#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"

#include "graph_kernels.cuh"	

#define MSB_TAG_MASK 0x80000000 
#define DATA_MASK 0x7FFFFFFF
#define LOCKED_TOKEN 0xFFFFFFFF


#define WARPS_PER_BLOCK 16 // assume blockDim.x = 512
// const uint64_t kBitsPerWord = 64;

// Mask: highest 2 bits
#define MASK_STATE      0xC0000000 
// Mask: lowest 30 bits (Payload)
#define MASK_PAYLOAD    0x3FFFFFFF 

// // three states
#define STATE_COMPACT   0x00000000 // 00...
#define STATE_LOCKED    0x80000000 // 10... (locked)
#define STATE_BUFFER    0xC0000000 // 11... (point to Buffer)


__global__ void sssp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							uint *dist,
							bool *label1,
							bool *label2,
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
				finalDist = dist[id] + edgeList[i].w8;
				uint dst = edgeList[i].end;
				if (finalDist < dist[dst])
				{
					atomicMin(&dist[dst],finalDist);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void sssp_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE; 

    if (warpId >= numActiveNodes) return;
    
    int u = activeSet[warpId]; 

    uint start = nodesPointer[u];
    uint end   = nodesPointer[u+1];

    uint src_dist = dist[u];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = src_dist + e.w8;
             uint v = e.end;

             if (finalDist < dist[v]) {
                 atomicMin(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}

__global__ void sssp_kernel_common_warp(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE; 

    if (warpId >= numNodes) return;
    if (label1[warpId] == false) return;

    uint start = nodesPointer[warpId];
    uint end   = nodesPointer[warpId+1];

    uint src_dist = dist[warpId];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = src_dist + e.w8;
             uint v = e.end;

             if (finalDist < dist[v]) {
                 atomicMin(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}

__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData *depends,
							int sid)
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

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				
				DependencyData new_dep;
				new_dep.value = depends[id].value + edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void sssp_kernel(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
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
		unsigned int id = activeNodes[from + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint dst;
		uint new_val;
		uint sourceweight = value[id];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				dst = edgeList[i].end;
				new_val = sourceweight + edgeList[i].w8;

				if (new_val < value[dst])
				{
					atomicMin(&value[dst], new_val);
					// value[dst]= new_val;
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sssp_kernel_concurrent(unsigned int numNodes,
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


__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int fromNode,
							unsigned int fromEdge,
							unsigned int *nodePointer,
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

		label1[id] = false;

		unsigned int thisFrom = nodePointer[id] - fromEdge;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;

		uint dst;
		uint new_val;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				dst = edgeList[i].end;
				new_val = value[id] + edgeList[i].w8;
				if (new_val < value[dst])
				{
					atomicMin(&value[dst], new_val);
					// value[dst]= new_val;
					label2[dst] = true;
				}

			}
		}
	}
}

__global__ void sssp_incremental_cg_concurrent(uint numNodes,
									unsigned int *nodesPointer,
									OutEdgeWeighted_Evolving *edgeList,
									DependencyData *depends,
									bool *label2,
									int numSnap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1)){
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				for (uint sid = 0; sid < numSnap; sid++)
				{
					if (bmp & ((ull)1 << sid))
					{
						DependencyData cur_dep = depends[dst * numSnap + sid];
						DependencyData new_dep;
						new_dep.value = depends[id * numSnap + sid].value + weight;
						new_dep.parent = id;
						new_dep.level = depends[id * numSnap + sid].level + 1;

						while (new_dep.value < cur_dep.value)
						{
							if (atomicCAS((unsigned long long int *)&depends[dst * numSnap + sid], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
							{
								label2[dst] = true;
								break;
							}
							cur_dep = depends[dst * numSnap + sid];
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_incremental_cg_concurrent(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   int numSnap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				for (uint sid = 0; sid < numSnap; sid++)
				{
					if (bmp & ((ull)1 << sid))
					{
						uint src_idx = id * numSnap + sid;
						uint dst_idx = dst * numSnap + sid;

						new_val = value[src_idx] + weight;
						if (new_val < value[dst_idx])
						{
							atomicMin(&value[dst_idx], new_val);
							label2[dst] = true;
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_incremental_cg_concurrent_bound(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *max_min,
											   uint *value,
											   bool *label2,
											   int numSnap)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;
		uint src_bound = max_min[id];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				if(src_bound + weight >= max_min[dst])
					continue;

				for (uint sid = 0; sid < numSnap; sid++)
				{
					if (bmp & ((ull)1 << sid))
					{
						uint src_idx = id * numSnap + sid;
						uint dst_idx = dst * numSnap + sid;

						new_val = value[src_idx] + weight;
						if (new_val < value[dst_idx])
						{
							atomicMin(&value[dst_idx], new_val);
							label2[dst] = true;
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_incremental_cg_concurrent2(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				if (bmp & ((ull)1 << sid))
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;

					new_val = value[src_idx] + weight;
					if (new_val < value[dst_idx])
					{
						atomicMin(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sssp_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint src_value = value[id];
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}


		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint dst_value = value[dst];
					if((dst_value & MSB_TAG_MASK) != 0) // updated
					{
						if(dst_value == LOCKED_TOKEN)
						{
							while (true) 
							{
								dst_value = *(volatile uint*)&value[id];
								if (dst_value != LOCKED_TOKEN) {
									break;
								}
							}
						}
						if((dst_value & MSB_TAG_MASK) != 0)
							dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
					}


					uint new_val = src_value + weight;
					if (new_val < dst_value)
					{
						while (true) 
						{
							uint cur_dst_head = *(volatile uint*)&value[dst];

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_dst_head == LOCKED_TOKEN) {
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if ((cur_dst_head & MSB_TAG_MASK) != 0) {
								uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
								atomicMin(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// ---------------------------------------------------
							// C: (Need Allocation)
							// ---------------------------------------------------
							if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
							{
								uint new_slot = atomicAdd(buffer_counter, 1);
								if (new_slot < *buffer_total) {
									for(uint k = 0; k < numSnap; k++){
										buffer[new_slot * numSnap + k] = cur_dst_head; 
									}
									__threadfence();
									uint new_ptr_val = new_slot | MSB_TAG_MASK;
									atomicExch(&value[dst], new_ptr_val);
								} else {
									printf("==== Buffer Full! cur size: %u, total size: %u ====\n", *buffer_counter, *buffer_total);
									atomicExch(&value[dst], cur_dst_head); 
									break; 
								}
								continue;
							}
						}
						
					}
				}
			}
		}
	}
}

__global__ void sssp_incremental_cg_concurrent2_newstorage_waitfree(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint raw_src, src_value;

		while (true) 
		{
			raw_src = *(volatile uint*)&value[id];
			if ((raw_src & MASK_STATE) != STATE_LOCKED) {
				break; 
			}
		}

        if ((raw_src & MASK_STATE) == STATE_BUFFER) {
            uint buff_idx = raw_src & MASK_PAYLOAD;
            src_value = buffer[buff_idx * numSnap + sid];
        } else {
            src_value = raw_src & MASK_PAYLOAD;
        }


		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint raw_dst = *(volatile uint*)&value[dst];
					uint dst_value;

					if ((raw_dst & MASK_STATE) == STATE_BUFFER) {
						uint buff_idx = raw_dst & MASK_PAYLOAD;
						dst_value = buffer[buff_idx * numSnap + sid];
					} else {
						dst_value = raw_dst & MASK_PAYLOAD;
					}


					uint new_val = src_value + weight;
					if (new_val < dst_value)
					{
						while (true) 
						{
							uint cur_raw = *(volatile uint*)&value[dst];
                        	uint cur_state = cur_raw & MASK_STATE;
                        	uint cur_payload = cur_raw & MASK_PAYLOAD;

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_state == STATE_LOCKED) {
								if (new_val >= cur_payload) {
									break; 
								}
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if (cur_state == STATE_BUFFER) {
								uint dst_idx = cur_payload * numSnap + sid;
								atomicMin(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// ---------------------------------------------------
							// C: (Need Allocation)
							// ---------------------------------------------------
							uint locked_val = cur_payload | STATE_LOCKED;
							if (atomicCAS(&value[dst], cur_raw, locked_val) == cur_raw) 
							{
								uint new_slot = atomicAdd(buffer_counter, 1);
								
								if (new_slot < *buffer_total) {
									for(uint k = 0; k < numSnap; k++){
										buffer[new_slot * numSnap + k] = cur_payload; 
									}
									
									__threadfence();
									uint new_ptr_val = new_slot | STATE_BUFFER;
									atomicExch(&value[dst], new_ptr_val);
								} else {
									printf("Buffer Full!\n");
									atomicExch(&value[dst], cur_payload); 
								}
								continue;
							}
						}
						
					}
				}
			}
		}
	}
}

__global__ void extend_varray_to_edgelist(uint numNodes,
										  uint fromNode,
										  uint fromEdge,
										  uint *nodesPointer,
										  uint *nodesPointer_el,
										  unsigned int *outDegree)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes){
		uint id = tId + fromNode;
		unsigned int thisFrom = nodesPointer[id] - fromEdge;
		unsigned int thisTo = thisFrom + outDegree[id];

		for(uint i = thisFrom; i < thisTo; i++){
			nodesPointer_el[i] = id;
		}
	}
}


__global__ void sssp_incre_concurrent(unsigned int numEdges,
										  unsigned int *nodePointer_el,
										  OutEdgeWeighted_Evolving *edgeList,
										  uint **value,
										  bool *label2,
										  uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				finalVal = value[sid][src] + weight;
				if (finalVal < value[sid][dst])
				{
					atomicMin(&value[sid][dst], finalVal);
					label2[dst] = true;
					// changed = true;
				}
			}
		}
	}
}



__global__ void sssp_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid] + weight;
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void sssp_incre_concurrent_locality_bound(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint *max,
											   uint *min,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (min[src] + weight >= max[dst])
			return;

		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid] + weight;
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void sssp_incre_concurrent_locality_bound_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *max,
											   uint *min,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (min[src] + weight >= max[dst])
			return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0) 
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}

				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}

				new_val = src_value + weight;
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}


__global__ void sssp_incre_concurrent_locality_bound_newstorage_waitfree(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *max,
											   uint *min,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (min[src] + weight >= max[dst])
			return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint raw_src, src_value;

				while (true) 
				{
					raw_src = *(volatile uint*)&value[src];
					if ((raw_src & MASK_STATE) != STATE_LOCKED) {
						break; 
					}
				}

				if ((raw_src & MASK_STATE) == STATE_BUFFER) {
					uint buff_idx = raw_src & MASK_PAYLOAD;
					src_value = buffer[buff_idx * numSnap + sid];
				} else {
					src_value = raw_src & MASK_PAYLOAD;
				}

				uint raw_dst = *(volatile uint*)&value[dst];
				uint dst_value;

				if ((raw_dst & MASK_STATE) == STATE_BUFFER) {
					uint buff_idx = raw_dst & MASK_PAYLOAD;
					dst_value = buffer[buff_idx * numSnap + sid];
				} else {
					dst_value = raw_dst & MASK_PAYLOAD;
				}

				new_val = src_value + weight;
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_raw = *(volatile uint*)&value[dst];
						uint cur_state = cur_raw & MASK_STATE;
						uint cur_payload = cur_raw & MASK_PAYLOAD;

						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_state == STATE_LOCKED) {
							if (new_val >= cur_payload) {
								break; 
							}
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if (cur_state == STATE_BUFFER) {
							uint dst_idx = cur_payload * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						uint locked_val = cur_payload | STATE_LOCKED;
						if (atomicCAS(&value[dst], cur_raw, locked_val) == cur_raw) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_payload; 
								}
								
								__threadfence();
								uint new_ptr_val = new_slot | STATE_BUFFER;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("Buffer Full!\n");
								atomicExch(&value[dst], cur_payload); 
								return;
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}

__global__ void sssp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0)
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}
				
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0) 
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
				
				uint new_val = src_value + weight;
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							if(!label2[dst])
								label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: % u==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_incre_concurrent2_locality_newstorage_waitfree(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint raw_src, src_value;

				while (true) 
				{
					raw_src = *(volatile uint*)&value[src];
					if ((raw_src & MASK_STATE) != STATE_LOCKED) {
						break; 
					}
				}
				
				if ((raw_src & MASK_STATE) == STATE_BUFFER) {
					uint buff_idx = raw_src & MASK_PAYLOAD;
					src_value = buffer[buff_idx * numSnap + sid];
				} else {
					src_value = raw_src & MASK_PAYLOAD;
				}
				
				uint raw_dst = *(volatile uint*)&value[dst];
				uint dst_value;

				if ((raw_dst & MASK_STATE) == STATE_BUFFER) {
					uint buff_idx = raw_dst & MASK_PAYLOAD;
					dst_value = buffer[buff_idx * numSnap + sid];
				} else {
					dst_value = raw_dst & MASK_PAYLOAD;
				}
				
				uint new_val = src_value + weight;
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_raw = *(volatile uint*)&value[dst];
						uint cur_state = cur_raw & MASK_STATE;
						uint cur_payload = cur_raw & MASK_PAYLOAD;

						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_state == STATE_LOCKED) {

							if (new_val >= cur_payload) {
								break; 
							}
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if (cur_state == STATE_BUFFER) {
							uint dst_idx = cur_payload * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						uint locked_val = cur_payload | STATE_LOCKED;
						if (atomicCAS(&value[dst], cur_raw, locked_val) == cur_raw) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_payload; 
								}
								
								__threadfence();
								uint new_ptr_val = new_slot | STATE_BUFFER;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("Buffer Full!\n");
								atomicExch(&value[dst], cur_payload); 
							}
							continue;
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_incre_concurrent_locality(unsigned int numEdges,
													 unsigned int *nodePointer_el,
													 OutEdgeWeighted_Evolving *edgeList,
													 uint *value,
													 uint *old_value,
													 uint *val_off,
													 uint *val_deg,
													 bool *label2,
													 uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint src_val_off = val_off[src];
		uint dst_val_off = val_off[dst];

		if (val_deg[src] == 1 && val_deg[dst] == 1){
			if(value[src_val_off] + weight >= old_value[dst])
				return;
		}

		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				if (val_deg[src] == 1)
					finalVal = old_value[src] + weight;
				else
				{
					finalVal = value[src_val_off + sid] + weight;
				}

				if (val_deg[dst] != 1)
				{
					uint dst_idx = dst_val_off + sid;
					if (finalVal < value[dst_idx])
					{
						atomicMin(&value[dst_idx], finalVal);
						label2[dst] = true;
					}
				}
				else
				{
					if (finalVal < old_value[dst])
					{
						if (value[dst_val_off] == old_value[dst])
						{
							if (atomicCAS(&value[dst_val_off], old_value[dst], finalVal) == old_value[dst])
							{
								label2[dst] = true;
								continue;
							}
						}
						if (finalVal != value[dst_val_off])
						{ // need to be expanded
							// label2[src] = true;
							atomicMin(&value[dst_val_off], finalVal);
							label2[dst] = true;
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_incre_concurrent_locality_common(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap,
											   bool *same)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;

		if(same[src] && same[dst] && ((bmp & (((ull)1 << numSnap) - 1)) == (((ull)1 << numSnap) - 1)))
		{
			finalVal = value[src * numSnap] + weight;
			if (finalVal < value[dst * numSnap])
			{
				atomicMin(&value[dst * numSnap], finalVal);
				label2[dst] = true;
			
			}
		}
	}
}

__global__ void sssp_incre_concurrent_locality_opt(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;

		if ((bmp & (((ull)1 << numSnap) - 1)) == (((ull)1 << numSnap) - 1))
		{
			for (uint sid = 0; sid < numSnap; sid++)
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid] + weight;
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
		else{
			for (uint sid = 0; sid < numSnap; sid++)
			{
				if (bmp & ((ull)1 << sid))
				{
					uint dst_idx = dst * numSnap + sid;
					finalVal = value[src * numSnap + sid] + weight;
					if (finalVal < value[dst_idx])
					{
						atomicMin(&value[dst_idx], finalVal);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sssp_incre_concurrent(unsigned int numEdges,
									  EdgeWeighted *edgeList,
									  uint **value,
									  bool *label2,
									  uint snapid,
									  uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = edgeList[id].source;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;

		uint finalVal;
		for (uint sid = snapid; sid < numSnap; sid++)
		{
			finalVal = value[sid][src] + weight;
			if (finalVal < value[sid][dst])
			{
				atomicMin(&value[sid][dst], finalVal);
				label2[dst] = true;
			}
		}
	}
}

__global__ void sssp_incre_concurrent2(unsigned int numEdges,
									  unsigned int *nodePointer_el,
									  OutEdgeWeighted_Evolving *edgeList,
									  uint **value,
									//   DependencyData *depends,
									//   bool *label1,
									  bool *label2,
									  uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src]) 
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = value[sid][src] + weight;
				if (finalVal < value[sid][dst])
				{
					atomicMin(&value[sid][dst], finalVal);
					if (!label2[dst])
						label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sssp_incre_concurrent2_locality(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = value[src * numSnap + sid] + weight;
				if (finalVal < value[dst * numSnap + sid])
				{
					atomicMin(&value[dst * numSnap + sid], finalVal);
					if (!label2[dst]){
						label2[dst] = true;
					}
				}
			}
		}
	}
}




__global__ void sssp_incre_concurrent(unsigned int numEdges,
									  unsigned int *nodePointer_el,
									  OutEdgeWeighted_Evolving *edgeList,
									  uint *value,
									  bool *label2,
									  uint numSnap,
									  uint num_nodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		bool changed = false;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = value[sid * num_nodes + src] + weight;
				if (finalVal < value[sid * num_nodes + dst])
				{
					atomicMin(&value[sid * num_nodes + dst], finalVal);
					changed = true;
				}
			}
		}
		if (changed)
			label2[dst] = true;
	}
}

__global__ void sssp_incre_concurrent(unsigned int numEdges,
									  unsigned int *nodePointer_el,
									  OutEdgeWeighted_Evolving *edgeList,
									  uint *value,
									  bool *label2,
									  uint sid)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if(value[src] == DIST_INFINITY)
			return;

		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = edgeList[id].bitmap >> 1;
		if (bmp & ((ull)1 << sid))
		{
			uint finalVal = value[src] + weight;
			if (finalVal < value[dst])
			{
				atomicMin(&value[dst], finalVal);
				label2[dst] = true;
			}
		}
	}
}

__global__ void sssp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value + edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}


__global__ void sssp_kernel_ks_concurrent(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int numSnap)
{	
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			bmp = edgeList[i].bitmap >> 1; // skip the first bit
			weight = edgeList[i].w8;

			if((bmp & (((ull)1 << numSnap) - 1)) == (((ull)1 << numSnap) - 1)){
				for (uint sid = 0; sid < numSnap; sid++)
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;
					DependencyData new_dep;
					new_dep.value = depends[src_idx].value + weight;
					new_dep.parent = id;
					new_dep.level = depends[src_idx].level + 1;
					DependencyData cur_dep = depends[dst_idx];

					while (new_dep.value < cur_dep.value)
					{
						if (atomicCAS((unsigned long long int *)&depends[dst_idx], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
						{
							label2[dst] = true;
							break;
						}
						cur_dep = depends[dst_idx];
					}
				}
			}
			else{
				for (uint sid = 0; sid < numSnap; sid++)
				{
					if (bmp & ((ull)1 << sid))
					{
						uint src_idx = id * numSnap + sid;
						uint dst_idx = dst * numSnap + sid;

						DependencyData new_dep;
						new_dep.value = depends[src_idx].value + weight;
						new_dep.parent = id;
						new_dep.level = depends[src_idx].level + 1;
						DependencyData cur_dep = depends[dst_idx];

						while (new_dep.value < cur_dep.value)
						{
							if (atomicCAS((unsigned long long int *)&depends[dst_idx], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
							{
								label2[dst] = true;
								break;
							}
							cur_dep = depends[dst_idx];
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_kernel_ks_concurrent2(unsigned int numNodes,
										  unsigned int *nodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  bool *label1,
										  bool *label2,
										  DependencyData *depends,
										  int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			bmp = edgeList[i].bitmap >> 1; // skip the first bit
			weight = edgeList[i].w8;

			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				DependencyData new_dep;
				new_dep.value = depends[src_idx].value + weight;
				new_dep.parent = id;
				new_dep.level = depends[src_idx].level + 1;
				DependencyData cur_dep = depends[dst_idx];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst_idx], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst_idx];
				}
			}
		}
	}
}


__global__ void sssp_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;

				uint new_val = value[src_idx] + weight;
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}



__global__ void sssp_kernel_ks_concurrent2(unsigned int numActiveNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   uint *activeSet,
										   bool *label2,
										   uint *value,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numActiveNodes * numSnap)
	{
		unsigned int id = tId / numSnap;
		uint u = activeSet[id];

		unsigned int thisFrom = nodesPointer[u];
		unsigned int thisTo = nodesPointer[u + 1];

		uint sid = tId % numSnap;
		uint src_idx = u * numSnap + sid;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;

				uint new_val = value[src_idx] + weight;
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sssp_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_value = value[id];
		
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = src_value + weight;
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[id];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
		
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];
						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void sssp_kernel_concurrent_newstorage_waitfree(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint raw_src, src_value;

		while (true) 
		{
			raw_src = *(volatile uint*)&value[id];
			if ((raw_src & MASK_STATE) != STATE_LOCKED) {
				break; 
			}
		}
				
        if ((raw_src & MASK_STATE) == STATE_BUFFER) {
            // Buffer
            uint buff_idx = raw_src & MASK_PAYLOAD;
            src_value = buffer[buff_idx * numSnap + sid];
        } else {
            // Compact or Locked
            src_value = raw_src & MASK_PAYLOAD;
        }

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = src_value + weight;
				uint raw_dst = *(volatile uint*)&value[dst];
				uint dst_value;

				if ((raw_dst & MASK_STATE) == STATE_BUFFER) {
					uint buff_idx = raw_dst & MASK_PAYLOAD;
					dst_value = buffer[buff_idx * numSnap + sid];
				} else {
					dst_value = raw_dst & MASK_PAYLOAD;
				}
		
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_raw = *(volatile uint*)&value[dst];
						uint cur_state = cur_raw & MASK_STATE;
						uint cur_payload = cur_raw & MASK_PAYLOAD;

						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_state == STATE_LOCKED) {
							if (new_val >= cur_payload) {
								break; 
							}
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if (cur_state == STATE_BUFFER) {
							uint dst_idx = cur_payload * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						uint locked_val = cur_payload | STATE_LOCKED;
						if (atomicCAS(&value[dst], cur_raw, locked_val) == cur_raw) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_payload; 
								}
								
								__threadfence();

								uint new_ptr_val = new_slot | STATE_BUFFER;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								// printf("Buffer Full!\n");
								atomicExch(&value[dst], cur_payload); 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void sssp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;      
    int laneId = threadIdx.x % WARP_SIZE;      
    int globalWarpId = tId / WARP_SIZE;        

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {
        uint edge_idx = i + laneId;
        uint valid_edge = (edge_idx < end);

        if (valid_edge)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1; 
        }
        
        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = value[src_idx] + w;
                    
                    if (new_val < value[dst_idx])
                    {
                        atomicMin(&value[dst_idx], new_val);
                        if(!label2[dst]) 
							label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void sssp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
												uint *max,
												uint *min,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;     
    int laneId = threadIdx.x % WARP_SIZE;   
    int globalWarpId = tId / WARP_SIZE;      

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {  
        uint edge_idx = i + laneId;

        if (edge_idx < end)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1;
        }

        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

		uint min_u = min[u];

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

			if (min_u + w >= max[dst]) {
				continue;
			}

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
				
                    uint new_val = value[src_idx] + w;
                    
                    if (new_val < value[dst_idx])
                    {
                        atomicMin(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }

    }
}

__global__ void sssp_kernel_warp(unsigned int numActiveNodes,
                                        unsigned int *nodesPointer,
                                        OutEdgeWeighted_Evolving *edgeList,
                                        uint *activeSet,
                                        bool *label2,
                                        uint *value,
                                        int numSnap)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int laneId = threadIdx.x % WARP_SIZE;
    int globalWarpId = tId / WARP_SIZE;

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {
        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;
        for (uint k = 0; k < tile_size; k++)
        {
            uint cur_edge_idx = i + k;
            OutEdgeWeighted_Evolving e = edgeList[cur_edge_idx]; 
            
            ull bmp = e.bitmap >> 1;
            if (bmp == 0) continue;

            uint dst = e.end;
            uint w = e.w8;
            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = value[src_idx] + w; 
                    if (new_val < value[dst_idx])
                    {
                        atomicMin(&value[dst_idx], new_val);
                        if(!label2[dst]) 
							label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void sssp_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *old_value,
										   uint *val_off,
										   uint *val_deg,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;
		uint src_val_off = val_off[id];

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			bmp = edgeList[i].bitmap >> 1; // skip the first bit
			weight = edgeList[i].w8;

			if (bmp & ((ull)1 << sid))
			{
				if ((val_deg[id] == 1) && (val_deg[dst] == 1))
				{													   // optimization by bound check
					if (value[src_val_off] + weight >= old_value[dst]) // min[src] + weight >= max[src]
						continue;
				}

				if (val_deg[id] == 1)
					new_val = old_value[id] + weight;
				else
					new_val = value[src_val_off + sid] + weight;

				if (val_deg[dst] != 1)
				{
					uint dst_idx = val_off[dst] + sid;
					if (new_val < value[dst_idx])
					{
						atomicMin(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
				else
				{
					if (new_val < old_value[dst]) // otherwise, no need to chek
					{
						if (value[val_off[dst]] == old_value[dst]) // value not yet changed
						{
							if (atomicCAS(&value[val_off[dst]], old_value[dst], new_val) == old_value[dst])
							{
								label2[dst] = true;
								continue;
							}
						}
						if (new_val != value[val_off[dst]]) // already a new value, need to expand
						{
							label2[id] = true;
						}
					}
				}
			}
		}
	}
}

__global__ void sssp_kernel_ks_concurrent2_bound(unsigned int numNodes,
										   unsigned int * __restrict__ nodesPointer,
										   OutEdgeWeighted_Evolving * __restrict__ edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint * __restrict__ max,
										   uint * __restrict__ min,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;
		uint my_min = min[id];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			uint dst = e.end;
			ull bmp = e.bitmap >> 1; // skip the first bit
			uint weight = e.w8;

			if (my_min + weight >= max[dst])
				continue;

			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;

				uint new_val = value[src_idx] + weight;
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sssp_kernel_ks_concurrent2_bound(unsigned int numActiveNodes,
										   unsigned int * __restrict__ nodesPointer,
										   OutEdgeWeighted_Evolving * __restrict__ edgeList,
										   uint * activeSet,
										   bool *label2,
										   uint *value,
										   uint * __restrict__ max,
										   uint * __restrict__ min,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numActiveNodes * numSnap)
	{
		unsigned int vid = tId / numSnap;
		uint id = activeSet[vid];

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;
		uint my_min = min[id];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			uint dst = e.end;
			ull bmp = e.bitmap >> 1; // skip the first bit
			uint weight = e.w8;

			if (my_min + weight >= max[dst])
				continue;

			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;

				uint new_val = value[src_idx] + weight;
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}


__global__ void sssp_kernel_ks_concurrent_bound(unsigned int numNodes,
										   unsigned int * __restrict__ nodesPointer,
										   OutEdgeWeighted_Evolving * __restrict__ edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint * __restrict__ max,
										   uint * __restrict__ min,
										   int numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numNodes)
	{
		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint my_min = min[id];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			uint dst = e.end;
			ull bmp = e.bitmap >> 1; // skip the first bit
			uint weight = e.w8;

			if (my_min + weight >= max[dst])
				continue;

			for (uint sid = 0; sid < numSnap; sid++){
				if (bmp & ((ull)1 << sid))
				{
					uint dst_idx = dst * numSnap + sid;
					uint new_val = value[id * numSnap + sid] + weight;

					if (new_val < value[dst_idx])
					{
						atomicMin(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}


__global__ void sssp_kernel_ks(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							bool *label1,
							bool *label2,
							DependencyData * depends,
							bool * all_affected_vertices,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = tId;
		
		if(label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{

				uint dst = edgeList[i].end;
				
				DependencyData new_dep;
				new_dep.value = depends[id].value + edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while(new_dep.value < cur_dep.value)
				{				
					if(atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep))){
						label2[dst] = true;
						all_affected_vertices[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

// ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void sswp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *dist,
							bool *label1,
							bool *label2,
							int sid)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		uint id = tId;

		if(label1[id] == false)
			return;
		
		unsigned int finalDist;
		unsigned int thisFrom, thisTo;

		thisFrom = nodesPointer[id];
		thisTo = nodesPointer[id + 1];
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			if(edgeList[i].bitmap & ((ull)1 << sid)){ 
				finalDist = min(dist[id], edgeList[i].w8);
				uint dst = edgeList[i].end;
				uint cur_val = dist[dst];
				while(finalDist > cur_val)
				{
					if(atomicCAS(&dist[dst], cur_val, finalDist) == cur_val){
						label2[dst] = true;
						// parent[dst] = id;
						break;
					}
					cur_val = dist[dst];
				}
			}
		}
	}
}

//    ==================================================================
//                             SSNP kernels
//    ==================================================================

__global__ void ssnp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData *depends,
							int sid)
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

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = max(depends[id].value, edgeList[i].w8);
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void ssnp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{

				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = max(depends[id].value, edgeList[i].w8);
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void ssnp_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i]; 
			uint dst = e.end;
			uint weight = e.w8;
			ull bmp = e.bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{	
				uint dst_idx = dst * numSnap + sid;

				uint new_val = max(value[src_idx] , weight);
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void ssnp_kernel_concurrent2_locality(unsigned int numNodes,
												 unsigned int from,
												 unsigned int numPartitionedEdges,
												 unsigned int *activeNodes,
												 unsigned int *activeNodesPointer,
												 OutEdgeWeighted_Evolving *edgeList,
												 unsigned int *outDegree,
												 bool *label1,
												 bool *label2,
												 uint *value,
												 uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = activeNodes[from + tId / numSnap];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId / numSnap] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId / numSnap + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				new_val = max(value[src_idx], weight);
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void ssnp_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				if (bmp & ((ull)1 << sid))
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;

					new_val = max(value[src_idx], weight);
					if (new_val < value[dst_idx])
					{
						atomicMin(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void ssnp_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				finalVal = max(value[src * numSnap + sid], weight);
				if (finalVal < value[dst * numSnap + sid])
				{
					atomicMin(&value[dst * numSnap + sid], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void ssnp_incre_concurrent_locality_bound(unsigned int numEdges,
													 unsigned int *nodePointer_el,
													 OutEdgeWeighted_Evolving *edgeList,
													 uint *value,
													 bool *label2,
													 uint *value_max,
													 uint *value_min,
													 uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges) 
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		if (max(value_min[src], weight) >= value_max[dst])
			return;

		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = max(value[src * numSnap + sid], weight);
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void ssnp_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = max(value[src * numSnap + sid], weight);
				if (finalVal < value[dst * numSnap + sid])
				{
					atomicMin(&value[dst * numSnap + sid], finalVal);
					if (!label2[dst])
					{
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void ssnp_kernel_concurrent_locality(unsigned int numNodes,
												unsigned int from,
												unsigned int numPartitionedEdges,
												unsigned int *activeNodes,
												unsigned int *activeNodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												unsigned int *outDegree,
												bool *label1,
												bool *label2,
												uint *value,
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
					new_val = max(value[id * numSnap + sid], weight);
					if (new_val < value[dst * numSnap + sid])
					{
						atomicMin(&value[dst * numSnap + sid], new_val);
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


__global__ void ssnp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;    
    int laneId = threadIdx.x % WARP_SIZE;    
    int globalWarpId = tId / WARP_SIZE;


    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {
        uint edge_idx = i + laneId;
        uint valid_edge = (edge_idx < end); 

        if (valid_edge)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1; 
        }
        
        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = max(value[src_idx], w);
                    
                    if (new_val < value[dst_idx])
                    {
                        atomicMin(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void ssnp_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint src_value = value[id];
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}


		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint dst_value = value[dst];
					if((dst_value & MSB_TAG_MASK) != 0) // updated
					{
						if(dst_value == LOCKED_TOKEN)
						{
							while (true) 
							{
								dst_value = *(volatile uint*)&value[id];
								if (dst_value != LOCKED_TOKEN) {
									break;
								}
							}
						}
						if((dst_value & MSB_TAG_MASK) != 0)
							dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
					}


					uint new_val = max(src_value, weight);
					if (new_val < dst_value)
					{
						while (true) 
						{
							uint cur_dst_head = *(volatile uint*)&value[dst];

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_dst_head == LOCKED_TOKEN) {
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if ((cur_dst_head & MSB_TAG_MASK) != 0) {
								uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
								atomicMin(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// ---------------------------------------------------
							// C: (Need Allocation)
							// ---------------------------------------------------
							if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
							{
								uint new_slot = atomicAdd(buffer_counter, 1);
								if (new_slot < *buffer_total) {
									for(uint k = 0; k < numSnap; k++){
										buffer[new_slot * numSnap + k] = cur_dst_head; 
									}
									__threadfence();
									uint new_ptr_val = new_slot | MSB_TAG_MASK;
									atomicExch(&value[dst], new_ptr_val);
								} else {
									printf("==== Buffer Full! cur size: %u, total size: %u ====\n", *buffer_counter, *buffer_total);
									atomicExch(&value[dst], cur_dst_head); 
									break; 
								}
								continue;
							}
						}
						
					}
				}
			}
		}
	}
}


__global__ void ssnp_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_value = value[id];
		
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = max(src_value, weight);
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[id];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
		
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];
						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void ssnp_incre_concurrent_locality_bound_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *upper,
											   uint *lower,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (max(lower[src], weight)>= upper[dst])
			return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0) 
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}

				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}

				new_val = max(src_value, weight);
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}


__global__ void ssnp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0)
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}
				
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0) 
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
				
				uint new_val = max(src_value, weight);
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							if(!label2[dst])
								label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: % u==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void ssnp_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE;

    if (warpId >= numActiveNodes) return;
    
    int u = activeSet[warpId];

    uint start = nodesPointer[u];
    uint end   = nodesPointer[u+1];

    uint src_dist = dist[u];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = max(src_dist, e.w8);
             uint v = e.end;

             if (finalDist < dist[v]) {
                 atomicMin(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}



//    ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void sswp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData *depends,
							int sid)
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

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = min(depends[id].value, edgeList[i].w8);
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value > cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void sswp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{

				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = min(depends[id].value, edgeList[i].w8);
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value > cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}


__global__ void sswp_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = min(value[src * numSnap + sid], weight);
				if (finalVal > value[dst_idx])
				{
					atomicMax(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sswp_incre_concurrent_locality_bound(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *up,
											   uint *low,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;

		if (min(up[src], weight) <= low[dst]) {
			return;
		}

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = min(value[src * numSnap + sid], weight);
				if (finalVal > value[dst_idx])
				{
					atomicMax(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sswp_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = min(value[src * numSnap + sid], weight);
				uint dst_idx = dst * numSnap + sid;
				if (finalVal > value[dst_idx])
				{
					atomicMax(&value[dst_idx], finalVal);
					if (!label2[dst])
					{
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sswp_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				if (bmp & ((ull)1 << sid))
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;

					new_val = min(value[src_idx], weight);
					if (new_val > value[dst_idx])
					{
						atomicMax(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sswp_kernel_concurrent_locality(unsigned int numNodes,
												unsigned int from,
												unsigned int numPartitionedEdges,
												unsigned int *activeNodes,
												unsigned int *activeNodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												unsigned int *outDegree,
												bool *label1,
												bool *label2,
												uint *value,
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
					new_val = min(value[id * numSnap + sid], weight);
					if (new_val > value[dst * numSnap + sid])
					{
						atomicMax(&value[dst * numSnap + sid], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sswp_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i];
			uint dst = e.end;
			ull bmp = e.bitmap >> 1; // skip the first bit
			uint weight = e.w8;

			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;

				uint new_val = min(value[src_idx], weight);
				if (new_val > value[dst_idx])
				{
					atomicMax(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void sswp_kernel_concurrent2_locality(unsigned int numNodes,
												 unsigned int from,
												 unsigned int numPartitionedEdges,
												 unsigned int *activeNodes,
												 unsigned int *activeNodesPointer,
												 OutEdgeWeighted_Evolving *edgeList,
												 unsigned int *outDegree,
												 bool *label1,
												 bool *label2,
												 uint *value,
												 uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = activeNodes[from + tId / numSnap];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId / numSnap] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId / numSnap + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				new_val = min(value[src_idx], weight);
				if (new_val > value[dst_idx])
				{
					atomicMax(&value[dst_idx], new_val);
					label2[dst] = true;
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


__global__ void sswp_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE; 

    if (warpId >= numActiveNodes) return;
    
    int u = activeSet[warpId]; 

    uint start = nodesPointer[u];
    uint end   = nodesPointer[u+1];

    uint src_dist = dist[u];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = min(src_dist, e.w8);
             uint v = e.end;

             if (finalDist > dist[v]) {
                 atomicMax(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}


__global__ void sswp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;    
    int laneId = threadIdx.x % WARP_SIZE;
    int globalWarpId = tId / WARP_SIZE;       

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {  
        uint edge_idx = i + laneId;
        uint valid_edge = (edge_idx < end);

        if (valid_edge)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1;
        }
        
        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = min(value[src_idx], w);

                    if (new_val > value[dst_idx])
                    {
                        atomicMax(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void sswp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
												uint *up,
												uint *low,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;      
    int laneId = threadIdx.x % WARP_SIZE;     
    int globalWarpId = tId / WARP_SIZE;      

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {  
        uint edge_idx = i + laneId;

        if (edge_idx < end)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1;
        }

        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

		uint max_u = up[u];

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

			if (min(max_u, w) <= low[dst]) {
				continue; 
			}

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = min(value[src_idx], w);
                    
                    if (new_val > value[dst_idx])
                    {
                        atomicMax(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }

    }
}


__global__ void sswp_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint src_value = value[id];

		// int timeout = 0;
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}


		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint dst_value = value[dst];
					if((dst_value & MSB_TAG_MASK) != 0) // updated
					{
						if(dst_value == LOCKED_TOKEN)
						{
							while (true) 
							{
								dst_value = *(volatile uint*)&value[id];
								if (dst_value != LOCKED_TOKEN) {
									break;
								}
							}
						}
						if((dst_value & MSB_TAG_MASK) != 0)
							dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
					}


					uint new_val = min(src_value, weight);
					if (new_val > dst_value)
					{
						while (true) 
						{
							uint cur_dst_head = *(volatile uint*)&value[dst];

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_dst_head == LOCKED_TOKEN) {
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if ((cur_dst_head & MSB_TAG_MASK) != 0) {
								uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
								atomicMax(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// C: Need Allocation
                            if (new_val <= cur_dst_head) { 
                                break; 
                            }

                            if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
                            {
                                uint new_slot = atomicAdd(buffer_counter, 1);
                                if (new_slot < *buffer_total) {
                                    for(uint k = 0; k < numSnap; k++){
                                        buffer[new_slot * numSnap + k] = cur_dst_head; 
                                    }

                                    __threadfence();
                                    
                                    uint new_ptr_val = new_slot | MSB_TAG_MASK;
                                    atomicExch(&value[dst], new_ptr_val);
                                } else {
                                    printf("Buffer Full!\n");
                                    atomicExch(&value[dst], cur_dst_head); 
                                    break; 
                                }
								continue;
                            }
						}
						
					}
				}
			}
		}
	}
}

__global__ void sswp_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_value = value[id];
		
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = min(src_value, weight);
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[id];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
		
				if (new_val > dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];
						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMax(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								// buffer[new_slot * numSnap + sid] = new_val;
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void sswp_incre_concurrent_locality_bound_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *upper,
											   uint *lower,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges) 
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (min(upper[src], weight) <= lower[dst])
			return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0) 
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}

				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}

				new_val = min(src_value, weight);
				if (new_val > dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMax(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								// buffer[new_slot * numSnap + sid] = new_val;
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}


__global__ void sswp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0)
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}
				
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0) 
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
				
				uint new_val = min(src_value, weight);
				if (new_val > dst_value)
				{
					while (true) 
					{
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMax(&buffer[dst_idx], new_val);
							if(!label2[dst])
								label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								// buffer[new_slot * numSnap + sid] = new_val;
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: % u==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}

//    ==================================================================
//                             Viterbi kernels
//    ==================================================================

__global__ void viterbi_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData *depends,
							int sid)
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

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value / edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value > cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void viterbi_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{

				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value / edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value > cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void viterbi_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			OutEdgeWeighted_Evolving e = edgeList[i]; 
			uint dst = e.end;
			uint weight = e.w8;
			ull bmp = e.bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;

				uint new_val = value[src_idx] / weight;
				if (new_val > value[dst_idx])
				{
					atomicMax(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void viterbi_kernel_concurrent2_locality(unsigned int numNodes,
												 unsigned int from,
												 unsigned int numPartitionedEdges,
												 unsigned int *activeNodes,
												 unsigned int *activeNodesPointer,
												 OutEdgeWeighted_Evolving *edgeList,
												 unsigned int *outDegree,
												 bool *label1,
												 bool *label2,
												 uint *value,
												 uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = activeNodes[from + tId / numSnap];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId / numSnap] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId / numSnap + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		uint weight;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				new_val = value[src_idx] / weight;
				if (new_val > value[dst_idx])
				{
					atomicMax(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void viterbi_identifyDel(uint num_edges_del,
								 EdgeWeighted *del_edgeList,
								 bool *label2,
								 DependencyData *depends,
								 bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < num_edges_del)
	{
		uint src = del_edgeList[tId].source;
		uint dst = del_edgeList[tId].end;
		if (depends[dst].parent == src)
		{
			depends[dst].reset();
			depends[dst].value = 0;
			label2[dst] = true;
			all_affected_vertices[dst] = true;
		}
	}
}

__global__ void viterbi_incre(uint num_edges_add,
						   EdgeWeighted *add_edgeList,
						   DependencyData *depends,
						   bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < num_edges_add)
	{
		uint src = add_edgeList[tId].source;
		uint dst = add_edgeList[tId].end;
		if (depends[src].value == 0)
			return;

		DependencyData new_dep;
		new_dep.value = depends[src].value / add_edgeList[tId].w8;
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;
		DependencyData cur_dep = depends[dst];

		while (new_dep.value > cur_dep.value)
		{
			if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
			{
				all_affected_vertices[dst] = true;
				break;
			}
			cur_dep = depends[dst];
		}
	}
}

__global__ void viterbi_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint weight;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				weight = edgeList[i].w8;

				if (bmp & ((ull)1 << sid))
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;

					new_val = value[src_idx] / weight;
					if (new_val > value[dst_idx])
					{
						atomicMax(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void viterbi_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				finalVal = value[src * numSnap + sid] / weight;
				if (finalVal > value[dst * numSnap + sid])
				{
					atomicMax(&value[dst * numSnap + sid], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void viterbi_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = value[src * numSnap + sid] / weight;
				if (finalVal > value[dst * numSnap + sid])
				{
					atomicMax(&value[dst * numSnap + sid], finalVal);
					if (!label2[dst])
					{
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void viterbi_kernel_concurrent_locality(unsigned int numNodes,
												unsigned int from,
												unsigned int numPartitionedEdges,
												unsigned int *activeNodes,
												unsigned int *activeNodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												unsigned int *outDegree,
												bool *label1,
												bool *label2,
												uint *value,
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
					new_val = value[id * numSnap + sid] / weight;
					if (new_val > value[dst * numSnap + sid])
					{
						atomicMax(&value[dst * numSnap + sid], new_val);
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


__global__ void viterbi_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap)
{

    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;    
    int laneId = threadIdx.x % WARP_SIZE;    
    int globalWarpId = tId / WARP_SIZE;    

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {
        uint edge_idx = i + laneId;
        uint valid_edge = (edge_idx < end);

        if (valid_edge)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1; 
        }
        __syncwarp(); 
        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = value[src_idx] / w;
                    if (new_val > value[dst_idx])
                    {
                        atomicMax(&value[dst_idx], new_val);
                        if(!label2[dst]) 
							label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void viterbi_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
												uint *up,
												uint *low,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;   
    int laneId = threadIdx.x % WARP_SIZE;    
    int globalWarpId = tId / WARP_SIZE;       

    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {  
        uint edge_idx = i + laneId;

        if (edge_idx < end)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1;
        }

        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

		uint max_u = up[u];

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

			if (max_u / w <= low[dst]) {
				continue;
			}

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = value[src_idx] / w;
                    
                    if (new_val > value[dst_idx])
                    {
                        atomicMax(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }

    }
}


__global__ void viterbi_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint src_value = value[id];

		// int timeout = 0;
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}


		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint dst_value = value[dst];
					if((dst_value & MSB_TAG_MASK) != 0) // updated
					{
						if(dst_value == LOCKED_TOKEN)
						{
							while (true) 
							{
								dst_value = *(volatile uint*)&value[id];
								if (dst_value != LOCKED_TOKEN) {
									break;
								}
							}
						}
						if((dst_value & MSB_TAG_MASK) != 0)
							dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
					}


					uint new_val = src_value / weight;
					if (new_val > dst_value)
					{
						while (true) 
						{
							uint cur_dst_head = *(volatile uint*)&value[dst];

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_dst_head == LOCKED_TOKEN) {
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if ((cur_dst_head & MSB_TAG_MASK) != 0) {
								uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
								atomicMax(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// C: Need Allocation
                            if (new_val <= cur_dst_head) { 
                                break; 
                            }

                            if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
                            {
                                uint new_slot = atomicAdd(buffer_counter, 1);
                                if (new_slot < *buffer_total) {
                                    for(uint k = 0; k < numSnap; k++){
                                        buffer[new_slot * numSnap + k] = cur_dst_head; 
                                    }

                                    __threadfence();
                                    
                                    uint new_ptr_val = new_slot | MSB_TAG_MASK;
                                    atomicExch(&value[dst], new_ptr_val);
                                } else {
                                    printf("Buffer Full!\n");
                                    atomicExch(&value[dst], cur_dst_head); 
                                    break; 
                                }
								continue;
                            }
						}
						
					}
				}
			}
		}
	}
}


__global__ void viterbi_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_value = value[id];
		
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = src_value / weight;
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[id];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
		
				if (new_val > dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];
						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMax(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								// buffer[new_slot * numSnap + sid] = new_val;
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}



__global__ void viterbi_incre_concurrent_locality_bound_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *upper,
											   uint *lower,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges) 
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		// if (upper[src]/ weight <= lower[dst])
		// 	return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0) 
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}

				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}

				new_val = src_value / weight;
				if (new_val > dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMax(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								// buffer[new_slot * numSnap + sid] = new_val;
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}


__global__ void viterbi_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0)
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}
				
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0) 
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
				
				uint new_val = src_value / weight;
				if (new_val > dst_value)
				{
					while (true) 
					{
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMax(&buffer[dst_idx], new_val);
							if(!label2[dst])
								label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								// buffer[new_slot * numSnap + sid] = new_val;
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: % u==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void viterbi_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE; 

    if (warpId >= numActiveNodes) return;
    
    int u = activeSet[warpId]; 

    uint start = nodesPointer[u];
    uint end   = nodesPointer[u+1];

    uint src_dist = dist[u];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = src_dist / e.w8;
             uint v = e.end;

             if (finalDist > dist[v]) {
                 atomicMax(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}
//    ==================================================================
//                             BFS kernels
//    ==================================================================


__global__ void bfs_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData *depends,
							int sid)
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

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value + 1;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void bfs_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value + 1;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void bfs_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		// uint weight;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;
				// weight = edgeList[i].w8;

				if (bmp & ((ull)1 << sid))
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;

					new_val = value[src_idx] + 1;
					if (new_val < value[dst_idx])
					{
						atomicMin(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void bfs_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_idx = id * numSnap + sid;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				
				uint dst_idx = dst * numSnap + sid;

				uint new_val = value[src_idx] + 1;
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void bfs_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid] + 1;
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void bfs_incre_concurrent_locality_bound(unsigned int numEdges,
													 unsigned int *nodePointer_el,
													 OutEdgeWeighted_Evolving *edgeList,
													 uint *value,
													 bool *label2,
													 uint *max,
													 uint *min,
													 uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges) 
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		if (min[src] + 1 >= max[dst])
			return;

		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid] + 1;
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void bfs_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = value[src * numSnap + sid] + 1;
				if (finalVal < value[dst * numSnap + sid])
				{
					atomicMin(&value[dst * numSnap + sid], finalVal);
					if (!label2[dst])
					{
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void bfs_kernel_concurrent2_locality(unsigned int numNodes,
												 unsigned int from,
												 unsigned int numPartitionedEdges,
												 unsigned int *activeNodes,
												 unsigned int *activeNodesPointer,
												 OutEdgeWeighted_Evolving *edgeList,
												 unsigned int *outDegree,
												 bool *label1,
												 bool *label2,
												 uint *value,
												 uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = activeNodes[from + tId / numSnap];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId / numSnap] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId / numSnap + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		// uint weight;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			// weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				new_val = value[src_idx] + 1;
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
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


__global__ void bfs_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE; 

    if (warpId >= numActiveNodes) return;
    
    int u = activeSet[warpId]; 

    uint start = nodesPointer[u];
    uint end   = nodesPointer[u+1];

    uint src_dist = dist[u];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = src_dist + 1;
             uint v = e.end;

             if (finalDist < dist[v]) {
                 atomicMin(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}

__global__ void bfs_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;     
    int laneId = threadIdx.x % WARP_SIZE;      
    int globalWarpId = tId / WARP_SIZE;   


    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {
        uint edge_idx = i + laneId;
        uint valid_edge = (edge_idx < end); 

        if (valid_edge)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_w[warpId][laneId] = e.w8;
            sm_bmp[warpId][laneId] = e.bitmap >> 1; 
        }
        
        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            uint w = sm_w[warpId][k];
            ull bmp = sm_bmp[warpId][k];

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = value[src_idx] + 1;
                    
                    if (new_val < value[dst_idx])
                    {
                        atomicMin(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void bfs_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint src_value = value[id];
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				// uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint dst_value = value[dst];
					if((dst_value & MSB_TAG_MASK) != 0) // updated
					{
						if(dst_value == LOCKED_TOKEN)
						{
							while (true) 
							{
								dst_value = *(volatile uint*)&value[id];
								if (dst_value != LOCKED_TOKEN) {
									break;
								}
							}
						}
						if((dst_value & MSB_TAG_MASK) != 0)
							dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
					}

					uint new_val = src_value + 1;
					if (new_val < dst_value)
					{
						while (true) 
						{
							uint cur_dst_head = *(volatile uint*)&value[dst];

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_dst_head == LOCKED_TOKEN) {
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if ((cur_dst_head & MSB_TAG_MASK) != 0) {
								uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
								atomicMin(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// ---------------------------------------------------
							// C: (Need Allocation)
							// ---------------------------------------------------
							if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
							{
								uint new_slot = atomicAdd(buffer_counter, 1);
								if (new_slot < *buffer_total) {
									for(uint k = 0; k < numSnap; k++){
										buffer[new_slot * numSnap + k] = cur_dst_head; 
									}
									__threadfence();
									uint new_ptr_val = new_slot | MSB_TAG_MASK;
									atomicExch(&value[dst], new_ptr_val);
								} else {
									printf("==== Buffer Full! cur size: %u, total size: %u ====\n", *buffer_counter, *buffer_total);
									atomicExch(&value[dst], cur_dst_head); 
									break; 
								}
								continue;
							}
						}
					}
				}
			}
		}
	}
}


__global__ void bfs_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_value = value[id];
		
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			// uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = src_value + 1;
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[id];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
		
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];
						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void bfs_incre_concurrent_locality_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *max,
											   uint *min,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges) 
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (min[src] + 1 >= max[dst])
			return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0) 
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}

				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}

				new_val = src_value + 1;
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}


__global__ void bfs_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0)
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}
				
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0) 
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
				
				uint new_val = src_value + 1;
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							if(!label2[dst])
								label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: % u==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}

//    ==================================================================
//                             CC kernels
//    ==================================================================

__global__ void cc_kernel(unsigned int numNodes,
						   unsigned int from,
						   unsigned int numPartitionedEdges,
						   unsigned int *activeNodes,
						   unsigned int *activeNodesPointer,
						   OutEdgeWeighted_Evolving *edgeList,
						   unsigned int *outDegree,
						   bool *label1,
						   bool *label2,
						   DependencyData *depends,
						   int sid)
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

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void cc_kernel_ks(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = tId;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;

				DependencyData new_dep;
				new_dep.value = depends[id].value;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;
				DependencyData cur_dep = depends[dst];

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						label2[dst] = true;
						break;
					}
					cur_dep = depends[dst];
				}
			}
		}
	}
}

__global__ void cc_incremental_cg_concurrent2(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				dst = edgeList[i].end;
				bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint src_idx = id * numSnap + sid;
					uint dst_idx = dst * numSnap + sid;

					new_val = value[src_idx];
					if (new_val < value[dst_idx])
					{
						atomicMin(&value[dst_idx], new_val);
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void cc_kernel_ks_concurrent2(unsigned int numNodes,
										  unsigned int *nodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  bool *label1,
										  bool *label2,
										  uint *value,
										  int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint dst;
		ull bmp;
		// uint weight;
		uint new_val;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			bmp = edgeList[i].bitmap >> 1; // skip the first bit
			// weight = edgeList[i].w8;

			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				new_val = value[src_idx];
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void cc_kernel_common_warp(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	int warpId = tId / WARP_SIZE; 
    int laneId = tId % WARP_SIZE; 

    if (warpId >= numNodes) return;
    if (label1[warpId] == false) return;

    uint start = nodesPointer[warpId];
    uint end   = nodesPointer[warpId+1];

    uint src_dist = dist[warpId];
	ull mask = ((ull)1 << num_snap) - 1;

    for (uint i = start + laneId; i < end; i += WARP_SIZE) {
        OutEdgeWeighted_Evolving e = edgeList[i]; 

        if ((e.bitmap & mask) == mask) {
             uint finalDist = src_dist;
             uint v = e.end;

             if (finalDist < dist[v]) {
                 atomicMin(&dist[v], finalDist);
                 label2[v] = true;
             }
        }
    }
}

__global__ void cc_incre_concurrent_locality(unsigned int numEdges,
											  unsigned int *nodePointer_el,
											  OutEdgeWeighted_Evolving *edgeList,
											  uint *value,
											  bool *label2,
											  uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid];
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void cc_incre_concurrent_locality_bound(unsigned int numEdges,
													unsigned int *nodePointer_el,
													OutEdgeWeighted_Evolving *edgeList,
													uint *value,
													bool *label2,
													uint *max,
													uint *min,
													uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		ull bmp = (edgeList[id].bitmap >> 1);

		if (min[src] >= max[dst])
			return;

		uint finalVal;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint dst_idx = dst * numSnap + sid;
				finalVal = value[src * numSnap + sid];
				if (finalVal < value[dst_idx])
				{
					atomicMin(&value[dst_idx], finalVal);
					label2[dst] = true;
				}
			}
		}
	}
}

__global__ void cc_incre_concurrent2_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint finalVal = value[src * numSnap + sid];
				if (finalVal < value[dst * numSnap + sid])
				{
					atomicMin(&value[dst * numSnap + sid], finalVal);
					if (!label2[dst])
					{
						label2[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void cc_kernel_concurrent2_locality(unsigned int numNodes,
												unsigned int from,
												unsigned int numPartitionedEdges,
												unsigned int *activeNodes,
												unsigned int *activeNodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												unsigned int *outDegree,
												bool *label1,
												bool *label2,
												uint *value,
												uint numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = activeNodes[from + tId / numSnap];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int thisFrom = activeNodesPointer[from + tId / numSnap] - numPartitionedEdges;
		unsigned int thisTo = activeNodesPointer[from + tId / numSnap + 1] - numPartitionedEdges;

		uint new_val;
		uint dst;
		ull bmp;
		// uint weight;

		uint sid = tId % numSnap;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			dst = edgeList[i].end;
			// weight = edgeList[i].w8;
			bmp = edgeList[i].bitmap >> 1;

			// OutEdgeWeighted_Evolving dst_e = edgeList[i];
			if (bmp & ((ull)1 << sid))
			{
				uint src_idx = id * numSnap + sid;
				uint dst_idx = dst * numSnap + sid;

				new_val = value[src_idx];
				if (new_val < value[dst_idx])
				{
					atomicMin(&value[dst_idx], new_val);
					label2[dst] = true;
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


__global__ void cc_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap)
{
    __shared__ uint sm_dst[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint sm_w[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ unsigned long long sm_bmp[WARPS_PER_BLOCK][WARP_SIZE];

    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;    
    int laneId = threadIdx.x % WARP_SIZE;     
    int globalWarpId = tId / WARP_SIZE;     


    if (globalWarpId >= numActiveNodes) return;

    uint u = activeSet[globalWarpId]; 
    uint start = nodesPointer[u];
    uint end = nodesPointer[u + 1];

    for (uint i = start; i < end; i += WARP_SIZE)
    {
        uint edge_idx = i + laneId;
        uint valid_edge = (edge_idx < end); 

        if (valid_edge)
        {
            OutEdgeWeighted_Evolving e = edgeList[edge_idx];
            sm_dst[warpId][laneId] = e.end;
            sm_bmp[warpId][laneId] = e.bitmap >> 1; 
        }
        
        __syncwarp(); 

        uint tile_size = (end - i) < WARP_SIZE ? (end - i) : WARP_SIZE;

        for (uint k = 0; k < tile_size; k++)
        {
            uint dst = sm_dst[warpId][k];
            ull bmp = sm_bmp[warpId][k];

            for (int sid = laneId; sid < numSnap; sid += WARP_SIZE)
            {
                if (bmp & ((ull)1 << sid))
                {
                    uint src_idx = u * numSnap + sid;
                    uint dst_idx = dst * numSnap + sid;
                    uint new_val = value[src_idx];
                    
                    if (new_val < value[dst_idx])
                    {
                        atomicMin(&value[dst_idx], new_val);
                        if(!label2[dst]) label2[dst] = true; 
                    }
                }
            }
        }
    }
}


__global__ void cc_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap)
{
	uint tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;

		uint src_value = value[id];
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if ((edgeList[i].bitmap & (((ull)1 << (numSnap + 1)) - 1)) != (((ull)1 << (numSnap + 1)) - 1))
			{
				uint dst = edgeList[i].end;
				// uint weight = edgeList[i].w8;
				ull bmp = edgeList[i].bitmap >> 1;

				if (bmp & ((ull)1 << sid))
				{
					uint dst_value = value[dst];
					if((dst_value & MSB_TAG_MASK) != 0) // updated
					{
						if(dst_value == LOCKED_TOKEN)
						{
							while (true) 
							{
								dst_value = *(volatile uint*)&value[id];
								if (dst_value != LOCKED_TOKEN) {
									break;
								}
							}
						}
						if((dst_value & MSB_TAG_MASK) != 0)
							dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
					}

					uint new_val = src_value;
					if (new_val < dst_value)
					{
						while (true) 
						{
							uint cur_dst_head = *(volatile uint*)&value[dst];

							// ---------------------------------------------------
							// A: (Locked)
							// ---------------------------------------------------
							if (cur_dst_head == LOCKED_TOKEN) {
								continue; 
							}

							// ---------------------------------------------------
							// B: (Buffer Mode)
							// ---------------------------------------------------
							if ((cur_dst_head & MSB_TAG_MASK) != 0) {
								uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
								atomicMin(&buffer[dst_idx], new_val);
								label2[dst] = true;
								break; 
							}

							// ---------------------------------------------------
							// C: (Need Allocation)
							// ---------------------------------------------------
							if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
							{
								uint new_slot = atomicAdd(buffer_counter, 1);
								if (new_slot < *buffer_total) {
									for(uint k = 0; k < numSnap; k++){
										buffer[new_slot * numSnap + k] = cur_dst_head; 
									}
									__threadfence();
									uint new_ptr_val = new_slot | MSB_TAG_MASK;
									atomicExch(&value[dst], new_ptr_val);
								} else {
									printf("==== Buffer Full! cur size: %u, total size: %u ====\n", *buffer_counter, *buffer_total);
									atomicExch(&value[dst], cur_dst_head); 
									break; 
								}
								continue;
							}
						}
					}
				}
			}
		}
	}
}


__global__ void cc_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes * numSnap)
	{
		unsigned int id = tId / numSnap;

		if (label1[id] == false)
			return;

		unsigned int thisFrom = nodesPointer[id];
		unsigned int thisTo = nodesPointer[id + 1];

		uint sid = tId % numSnap;
		uint src_value = value[id];
		
		if((src_value & MSB_TAG_MASK) != 0) // updated
        {
			if(src_value == LOCKED_TOKEN)
            {
				while (true) 
				{
					src_value = *(volatile uint*)&value[id];
					if (src_value != LOCKED_TOKEN) {
						break;
					}
				}
			}
			if((src_value & MSB_TAG_MASK) != 0)
				src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
		}

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			uint dst = edgeList[i].end;
			// uint weight = edgeList[i].w8;
			ull bmp = edgeList[i].bitmap >> 1; // skip the first bit

			if (bmp & ((ull)1 << sid))
			{
				uint new_val = src_value;
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[id];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
		
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];
						// ---------------------------------------------------
						// A: (Locked)
						// ---------------------------------------------------
						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						// ---------------------------------------------------
						// B: (Buffer Mode)
						// ---------------------------------------------------
						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						// ---------------------------------------------------
						// C: (Need Allocation)
						// ---------------------------------------------------
						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}


__global__ void cc_incre_concurrent_locality_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *max,
											   uint *min,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		uint src = nodePointer_el[id];
		uint dst = edgeList[id].end;
		ull bmp = (edgeList[id].bitmap >> 1);
		
		if (min[src] >= max[dst])
			return;

		uint new_val;
		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0) 
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}

				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0)
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}

				new_val = src_value;
				if (new_val < dst_value)
				{
					while (true) 
					{
						
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: %u ==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
							
					}
				}
			}
		}
	}
}


__global__ void cc_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < numEdges)
	{
		// unsigned int id = tId + fromEdge;
		uint src = nodePointer_el[id];
		if (!label2[src])
			return;
		uint dst = edgeList[id].end;
		// uint weight = edgeList[id].w8;
		ull bmp = (edgeList[id].bitmap >> 1);

		for (uint sid = 0; sid < numSnap; sid++)
		{
			if (bmp & ((ull)1 << sid))
			{
				uint src_value = value[src];
				if((src_value & MSB_TAG_MASK) != 0) // updated
				{
					if(src_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							src_value = *(volatile uint*)&value[src];
							if (src_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((src_value & MSB_TAG_MASK) != 0)
						src_value = buffer[(src_value & DATA_MASK) * numSnap + sid];
				}
				
				uint dst_value = value[dst];
				if((dst_value & MSB_TAG_MASK) != 0) // updated
				{
					if(dst_value == LOCKED_TOKEN)
					{
						while (true) 
						{
							dst_value = *(volatile uint*)&value[dst];
							if (dst_value != LOCKED_TOKEN) {
								break;
							}
						}
					}
					if((dst_value & MSB_TAG_MASK) != 0) 
						dst_value = buffer[(dst_value & DATA_MASK) * numSnap + sid];
				}
				
				uint new_val = src_value;
				if (new_val < dst_value)
				{
					while (true) 
					{
						uint cur_dst_head = *(volatile uint*)&value[dst];

						if (cur_dst_head == LOCKED_TOKEN) {
							continue; 
						}

						if ((cur_dst_head & MSB_TAG_MASK) != 0) {
							uint dst_idx = (cur_dst_head & DATA_MASK) * numSnap + sid;
							atomicMin(&buffer[dst_idx], new_val);
							if(!label2[dst])
								label2[dst] = true;
							break; 
						}

						if (atomicCAS(&value[dst], cur_dst_head, LOCKED_TOKEN) == cur_dst_head) 
						{
							uint new_slot = atomicAdd(buffer_counter, 1);
							if (new_slot < *buffer_total) {
								for(uint k = 0; k < numSnap; k++){
									buffer[new_slot * numSnap + k] = cur_dst_head; 
								}
								__threadfence();
								uint new_ptr_val = new_slot | MSB_TAG_MASK;
								atomicExch(&value[dst], new_ptr_val);
							} else {
								printf("==== Buffer Full! cur size: %u, total size: % u==== \n", *buffer_counter, *buffer_total);
								atomicExch(&value[dst], cur_dst_head); 
								break; 
							}
							continue;
						}
					}
				}
			}
		}
	}
}