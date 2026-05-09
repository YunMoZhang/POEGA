#ifndef	GRAPH_KERNELS_CUH
#define	GRAPH_KERNELS_CUH

#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/dependencydata.cuh"

//    ==================================================================
//                             SSSP kernels
//    ==================================================================


__global__ void sssp_incre(uint num_edges_add,
							EdgeWeighted *add_edgeList, 
							DependencyData * depends,
							bool *all_affected_vertices);


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
							   int num_snap);

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
									   uint numSnap);

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
							   int sid);

__global__ void sssp_incremental_kernel(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int sid,
							uint num_snap);
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
							   int num_snap);

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
										  uint numSnap);

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
							   int num_snap);

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
										  uint numSnap);

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
							   int num_snap);
							   
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
											 uint numSnap);

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
							  int num_snap);

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
										 uint numSnap);

__global__ void bfs_incremental_kernel(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int sid,
							uint num_snap);

__global__ void bfs_kernel(unsigned int numNodes,
							   unsigned int from,
							   unsigned int numPartitionedEdges,
							   unsigned int *activeNodes,
							   unsigned int *activeNodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   uint *value,
							   int sid);
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
							 int num_snap);

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
										uint numSnap);
#endif //