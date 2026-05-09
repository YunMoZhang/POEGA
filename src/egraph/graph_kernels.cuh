#ifndef	GRAPH_KERNELS_CUH
#define	GRAPH_KERNELS_CUH

#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/dependencydata.cuh"

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
							int sid);

//    ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void sswp_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);

//    ==================================================================
//                             SSNP kernels
//    ==================================================================

__global__ void ssnp_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);

//    ==================================================================
//                             Viterbi kernels
//    ==================================================================

__global__ void viterbi_kernel(uint numNodes,
							   uint fromNode,
							   uint fromEdge,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int sid);

__global__ void bfs_kernel(uint numNodes,
						   uint fromNode,
						   uint fromEdge,
						   unsigned int *nodesPointer,
						   OutEdgeWeighted_Evolving *edgeList,
						   unsigned int *outDegree,
						   bool *label1,
						   bool *label2,
						   uint *value,
						   int sid);

__global__ void cc_kernel(uint numNodes,
							uint fromNode,
							uint fromEdge,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);
#endif // 