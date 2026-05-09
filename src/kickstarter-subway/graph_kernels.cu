#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
// #include "commons/dependencydata.cuh"

// #include "graph.cuh"
#include "graph_kernels.cuh"


//    ==================================================================
//                             SSSP kernels
//    ==================================================================						


__global__ void sssp_identifyDel(uint num_edges_del,
							EdgeWeighted *del_edgeList, 
							bool *label2,
							DependencyData * depends,
							bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < num_edges_del)
	{
		uint src = del_edgeList[tId].source;
		uint dst = del_edgeList[tId].end;
		if(depends[dst].parent == src){
			depends[dst].reset();
			depends[dst].value = DIST_INFINITY;
			label2[dst] = true;
			all_affected_vertices[dst] = true;
		}
	}
}



__global__ void sssp_jumpstart(unsigned int numNodes,
							uint fromNode,
							uint fromEdge, 
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData * depends,
							DependencyData * depends_old,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{	
		unsigned int id = activeNodes[fromNode + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if(edgeList[i].bitmap & ((ull)1 << sid)){
				uint dst = edgeList[i].end;
				if(depends_old[dst].level < depends_old[id].level){
					// newDist = depends[dst].value + edgeList[i].w8;
					DependencyData cur_dep = depends[id];
					DependencyData new_dep(cur_dep);
					new_dep.value = depends[dst].value + edgeList[i].w8;
					new_dep.parent = dst;
					new_dep.level = depends[dst].level + 1;

					while(new_dep.value < cur_dep.value)
					{				
						if(atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep))){
							break;
						}
						cur_dep = depends[id];
					}
				}
			}
		}
		if(depends[id].value > depends_old[id].value)
		{
			label2[id] = true;
		}
	}
}



__global__ void sssp_propogate(unsigned int numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData * depends,
							bool *all_affected_vertices,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{	
		unsigned int id = activeNodes[fromNode + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if(edgeList[i].bitmap & ((ull)1 << sid)){
				uint dst = edgeList[i].end;
				if(depends[dst].parent == id){
					unsigned int old_val = depends[dst].value;
					depends[dst].reset();
					depends[dst].value = DIST_INFINITY;

					DependencyData cur_dep = depends[dst];
					DependencyData new_dep(depends[dst]);
					new_dep.value = depends[id].value + edgeList[i].w8;
					new_dep.parent = id;
					new_dep.level = depends[id].level + 1;

					if(new_dep.value < cur_dep.value){				
						atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep));
					}

					if(old_val < depends[dst].value){
						label2[dst] = true;
						all_affected_vertices[dst] = true;
					}
				}
			}
		}
	}
}


__global__ void sssp_pull_once(unsigned int numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							DependencyData * depends,
							bool *label1,
							int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{	
		unsigned int id = activeNodes[fromNode + tId];

		if(label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;


		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if(edgeList[i].bitmap & ((ull)1 << sid)){
				uint dst = edgeList[i].end;
				// newDist = dist[dst] + edgeList[i].w8;
				DependencyData cur_dep = depends[id];
				DependencyData new_dep(depends[id]);
				new_dep.value = depends[dst].value + edgeList[i].w8;
				new_dep.parent = dst;
				new_dep.level = depends[dst].level + 1;

				while(new_dep.value < cur_dep.value)
				{				
					if(atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep))){
						break;
					}
					cur_dep = depends[id];
				}
			}
		}
	}
}


__global__ void sssp_incre(uint num_edges_add,
							EdgeWeighted *add_edgeList, 
							DependencyData * depends,
							bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < num_edges_add)
	{	
		uint src = add_edgeList[tId].source;
		uint dst = add_edgeList[tId].end;
		if(depends[src].value == DIST_INFINITY)
			return;

		DependencyData cur_dep = depends[dst];
		DependencyData new_dep(cur_dep);
		new_dep.value = depends[src].value + add_edgeList[tId].w8;
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;
		
		while(new_dep.value < cur_dep.value)
		{				
			if(atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep))){
				all_affected_vertices[dst] = true;
				break;
			}
			cur_dep = depends[dst];
		}
	}
}


__global__ void sssp_kernel_ks(unsigned int numNodes,
									unsigned int from,
									unsigned int numPartitionedEdges,
									unsigned int *activeNodes,
									unsigned int *activeNodesPointer,
									OutEdgeWeighted_Evolving *edgeList,
									unsigned int *outDegree,
									bool *label1,
									bool *label2,
									DependencyData * depends,
									int sid)
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


		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			if(edgeList[i].bitmap & ((ull)1 << sid)){
				uint dst = edgeList[i].end;
				DependencyData cur_dep = depends[dst];
				DependencyData new_dep(depends[dst]);
				new_dep.value = depends[id].value + edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;

				while(new_dep.value < cur_dep.value)
				{				
					if(atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep))){
						label2[dst] = true;
						break;
					}
				cur_dep = depends[dst];
				}
			}
		}
	}
}



//    ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void sswp_kernel_ks(unsigned int numNodes,
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
				DependencyData cur_dep = depends[dst];
				DependencyData new_dep(depends[dst]);
				new_dep.value = min(depends[id].value, edgeList[i].w8);
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;

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

__global__ void sswp_identifyDel(uint num_edges_del,
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

__global__ void sswp_jumpstart(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   DependencyData *depends_old,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends_old[dst].level < depends_old[id].level)
				{
					DependencyData cur_dep = depends[id];
					DependencyData new_dep(cur_dep);
					new_dep.value = min(depends[dst].value, edgeList[i].w8);
					new_dep.parent = dst;
					new_dep.level = depends[dst].level + 1;

					while (new_dep.value > cur_dep.value)
					{
						if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
						{
							break;
						}
						cur_dep = depends[id];
					}
				}
			}
		}
		if (depends[id].value < depends_old[id].value)
		{
			label2[id] = true;
		}
	}
}

__global__ void sswp_propogate(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends[dst].parent == id)
				{
					unsigned int old_val = depends[dst].value;
					depends[dst].reset();
					depends[dst].value = 0;

					DependencyData cur_dep = depends[dst];
					DependencyData new_dep(depends[dst]);
					new_dep.value = min(depends[id].value, edgeList[i].w8);
					new_dep.parent = id;
					new_dep.level = depends[id].level + 1;

					if (new_dep.value > cur_dep.value)
					{
						atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep));
					}

					if (old_val > depends[dst].value)
					{
						label2[dst] = true;
						all_affected_vertices[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void sswp_pull_once(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   DependencyData *depends,
							   bool *label1,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		// unsigned int newDist;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				DependencyData cur_dep = depends[id];
				DependencyData new_dep(depends[id]);
				new_dep.value = min(depends[dst].value, edgeList[i].w8);
				new_dep.parent = dst;
				new_dep.level = depends[dst].level + 1;

				while (new_dep.value > cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						break;
					}
					cur_dep = depends[id];
				}
			}
		}
	}
}

__global__ void sswp_incre(uint num_edges_add,
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

		DependencyData cur_dep = depends[dst];
		DependencyData new_dep(cur_dep);
		new_dep.value = min(depends[src].value, add_edgeList[tId].w8);
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;

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

//    ==================================================================
//                             SSNP kernels
//    ==================================================================

__global__ void ssnp_kernel_ks(unsigned int numNodes,
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
				DependencyData cur_dep = depends[dst];
				DependencyData new_dep(depends[dst]);
				new_dep.value = max(depends[id].value, edgeList[i].w8);
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;

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

__global__ void ssnp_identifyDel(uint num_edges_del,
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
			depends[dst].value = DIST_INFINITY;
			label2[dst] = true;
			all_affected_vertices[dst] = true;
		}
	}
}

__global__ void ssnp_jumpstart(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   DependencyData *depends_old,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends_old[dst].level < depends_old[id].level)
				{
					DependencyData cur_dep = depends[id];
					DependencyData new_dep(cur_dep);
					new_dep.value = max(depends[dst].value, edgeList[i].w8);
					new_dep.parent = dst;
					new_dep.level = depends[dst].level + 1;

					while (new_dep.value < cur_dep.value)
					{
						if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
						{
							break;
						}
						cur_dep = depends[id];
					}
				}
			}
		}
		if (depends[id].value > depends_old[id].value)
		{
			label2[id] = true;
		}
	}
}

__global__ void ssnp_propogate(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends[dst].parent == id)
				{
					unsigned int old_val = depends[dst].value;
					depends[dst].reset();
					depends[dst].value = DIST_INFINITY;

					DependencyData cur_dep = depends[dst];
					DependencyData new_dep(depends[dst]);
					new_dep.value = max(depends[id].value, edgeList[i].w8);
					new_dep.parent = id;
					new_dep.level = depends[id].level + 1;

					if (new_dep.value < cur_dep.value)
					{
						atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep));
					}

					if (old_val < depends[dst].value)
					{
						label2[dst] = true;
						all_affected_vertices[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void ssnp_pull_once(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   DependencyData *depends,
							   bool *label1,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		// unsigned int newDist;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				// newDist = dist[dst] + edgeList[i].w8;
				DependencyData cur_dep = depends[id];
				DependencyData new_dep(depends[id]);
				new_dep.value = max(depends[dst].value, edgeList[i].w8);
				new_dep.parent = dst;
				new_dep.level = depends[dst].level + 1;

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						break;
					}
					cur_dep = depends[id];
				}
			}
		}
	}
}

__global__ void ssnp_incre(uint num_edges_add,
						   EdgeWeighted *add_edgeList,
						   DependencyData *depends,
						   bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < num_edges_add)
	{
		uint src = add_edgeList[tId].source;
		uint dst = add_edgeList[tId].end;
		if (depends[src].value == DIST_INFINITY)
			return;

		DependencyData cur_dep = depends[dst];
		DependencyData new_dep(cur_dep);
		new_dep.value = max(depends[src].value, add_edgeList[tId].w8);
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;

		while (new_dep.value < cur_dep.value)
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

//    ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void viterbi_kernel_ks(unsigned int numNodes,
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
				DependencyData cur_dep = depends[dst];
				DependencyData new_dep(depends[dst]);
				new_dep.value = depends[id].value / edgeList[i].w8;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;

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

__global__ void viterbi_jumpstart(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   DependencyData *depends_old,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends_old[dst].level < depends_old[id].level)
				{
					DependencyData cur_dep = depends[id];
					DependencyData new_dep(cur_dep);
					new_dep.value = depends[dst].value / edgeList[i].w8;
					new_dep.parent = dst;
					new_dep.level = depends[dst].level + 1;

					while (new_dep.value > cur_dep.value)
					{
						if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
						{
							break;
						}
						cur_dep = depends[id];
					}
				}
			}
		}
		if (depends[id].value < depends_old[id].value)
		{
			label2[id] = true;
		}
	}
}

__global__ void viterbi_propogate(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends[dst].parent == id)
				{
					unsigned int old_val = depends[dst].value;
					depends[dst].reset();
					depends[dst].value = 0;

					DependencyData cur_dep = depends[dst];
					DependencyData new_dep(depends[dst]);
					new_dep.value = depends[id].value / edgeList[i].w8;
					new_dep.parent = id;
					new_dep.level = depends[id].level + 1;

					if (new_dep.value > cur_dep.value)
					{
						atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep));
					}

					if (old_val > depends[dst].value)
					{
						label2[dst] = true;
						all_affected_vertices[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void viterbi_pull_once(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   DependencyData *depends,
							   bool *label1,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		// unsigned int newDist;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				DependencyData cur_dep = depends[id];
				DependencyData new_dep(depends[id]);
				new_dep.value = depends[dst].value / edgeList[i].w8;
				new_dep.parent = dst;
				new_dep.level = depends[dst].level + 1;

				while (new_dep.value > cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						break;
					}
					cur_dep = depends[id];
				}
			}
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

		DependencyData cur_dep = depends[dst];
		DependencyData new_dep(cur_dep);
		new_dep.value = depends[src].value / add_edgeList[tId].w8;
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;

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

//    ==================================================================
//                             BFS kernels
//    ==================================================================

__global__ void bfs_kernel_ks(unsigned int numNodes,
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
				DependencyData cur_dep = depends[dst];
				DependencyData new_dep;
				new_dep.value = depends[id].value + 1;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;

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

__global__ void bfs_identifyDel(uint num_edges_del,
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
			depends[dst].value = DIST_INFINITY;
			label2[dst] = true;
			all_affected_vertices[dst] = true;
		}
	}
}

__global__ void bfs_jumpstart(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   DependencyData *depends_old,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends_old[dst].level < depends_old[id].level)
				{
					DependencyData cur_dep = depends[id];
					DependencyData new_dep;
					new_dep.value = depends[dst].value + 1;
					new_dep.parent = dst;
					new_dep.level = depends[dst].level + 1;

					while (new_dep.value < cur_dep.value)
					{
						if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
						{
							break;
						}
						cur_dep = depends[id];
					}
				}
			}
		}
		if (depends[id].value > depends_old[id].value)
		{
			label2[id] = true;
		}
	}
}

__global__ void bfs_propogate(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends[dst].parent == id)
				{
					unsigned int old_val = depends[dst].value;
					depends[dst].reset();
					depends[dst].value = DIST_INFINITY;

					DependencyData cur_dep = depends[dst];
					DependencyData new_dep;
					new_dep.value = depends[id].value + 1;
					new_dep.parent = id;
					new_dep.level = depends[id].level + 1;

					if (new_dep.value < cur_dep.value)
					{
						atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep));
					}

					if (old_val < depends[dst].value)
					{
						label2[dst] = true;
						all_affected_vertices[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void bfs_pull_once(unsigned int numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   DependencyData *depends,
							   bool *label1,
							   int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		// unsigned int newDist;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				DependencyData cur_dep = depends[id];
				DependencyData new_dep;
				new_dep.value = depends[dst].value + 1;
				new_dep.parent = dst;
				new_dep.level = depends[dst].level + 1;

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						break;
					}
					cur_dep = depends[id];
				}
			}
		}
	}
}

__global__ void bfs_incre(uint num_edges_add,
						   EdgeWeighted *add_edgeList,
						   DependencyData *depends,
						   bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < num_edges_add)
	{
		uint src = add_edgeList[tId].source;
		uint dst = add_edgeList[tId].end;
		if (depends[src].value == DIST_INFINITY)
			return;

		DependencyData cur_dep = depends[dst];
		DependencyData new_dep;
		new_dep.value = depends[src].value + 1;
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;

		while (new_dep.value < cur_dep.value)
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

//    ==================================================================
//                             CC kernels
//    ==================================================================

__global__ void cc_kernel_ks(unsigned int numNodes,
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
				DependencyData cur_dep = depends[dst];
				DependencyData new_dep;
				new_dep.value = depends[id].value;
				new_dep.parent = id;
				new_dep.level = depends[id].level + 1;

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

__global__ void cc_identifyDel(uint num_edges_del,
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
			depends[dst].value = dst;
			label2[dst] = true;
			all_affected_vertices[dst] = true;
		}
	}
}

__global__ void cc_jumpstart(unsigned int numNodes,
							  uint fromNode,
							  uint fromEdge,
							  unsigned int *activeNodes,
							  unsigned int *activeNodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  DependencyData *depends_old,
							  int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends_old[dst].level < depends_old[id].level)
				{
					DependencyData cur_dep = depends[id];
					DependencyData new_dep;
					new_dep.value = depends[dst].value;
					new_dep.parent = dst;
					new_dep.level = depends[dst].level + 1;

					while (new_dep.value < cur_dep.value)
					{
						if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
						{
							break;
						}
						cur_dep = depends[id];
					}
				}
			}
		}
		if (depends[id].value > depends_old[id].value)
		{
			label2[id] = true;
		}
	}
}

__global__ void cc_propogate(unsigned int numNodes,
							  uint fromNode,
							  uint fromEdge,
							  unsigned int *activeNodes,
							  unsigned int *activeNodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  bool *all_affected_vertices,
							  int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				if (depends[dst].parent == id)
				{
					unsigned int old_val = depends[dst].value;
					depends[dst].reset();
					depends[dst].value = dst;

					DependencyData cur_dep = depends[dst];
					DependencyData new_dep;
					new_dep.value = depends[id].value;
					new_dep.parent = id;
					new_dep.level = depends[id].level + 1;

					if (new_dep.value < cur_dep.value)
					{
						atomicCAS((unsigned long long int *)&depends[dst], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep));
					}

					if (old_val < depends[dst].value)
					{
						label2[dst] = true;
						all_affected_vertices[dst] = true;
					}
				}
			}
		}
	}
}

__global__ void cc_pull_once(unsigned int numNodes,
							  uint fromNode,
							  uint fromEdge,
							  unsigned int *activeNodes,
							  unsigned int *activeNodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  DependencyData *depends,
							  bool *label1,
							  int sid)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[fromNode + tId];

		if (label1[id] == false)
			return;

		label1[id] = false;

		unsigned int degree = outDegree[id];
		unsigned int thisFrom = activeNodesPointer[fromNode + tId] - fromEdge;
		unsigned int thisTo = thisFrom + degree;

		// unsigned int newDist;

		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (edgeList[i].bitmap & ((ull)1 << sid))
			{
				uint dst = edgeList[i].end;
				DependencyData cur_dep = depends[id];
				DependencyData new_dep;
				new_dep.value = depends[dst].value;
				new_dep.parent = dst;
				new_dep.level = depends[dst].level + 1;

				while (new_dep.value < cur_dep.value)
				{
					if (atomicCAS((unsigned long long int *)&depends[id], *((unsigned long long int *)(&cur_dep)), *((unsigned long long int *)(&new_dep))) == *((unsigned long long int *)(&cur_dep)))
					{
						break;
					}
					cur_dep = depends[id];
				}
			}
		}
	}
}

__global__ void cc_incre(uint num_edges_add,
						  EdgeWeighted *add_edgeList,
						  DependencyData *depends,
						  bool *all_affected_vertices)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < num_edges_add)
	{
		uint src = add_edgeList[tId].source;
		uint dst = add_edgeList[tId].end;
		// if (depends[src].value == DIST_INFINITY)
		// 	return;

		DependencyData cur_dep = depends[dst];
		DependencyData new_dep;
		new_dep.value = depends[src].value;
		new_dep.parent = src;
		new_dep.level = depends[src].level + 1;

		while (new_dep.value < cur_dep.value)
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
