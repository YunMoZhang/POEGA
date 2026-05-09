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

__global__ void sssp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void sssp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);	
							
__global__ void sssp_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
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

__global__ void sswp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void sswp_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap);

__global__ void sswp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);
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

__global__ void ssnp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void ssnp_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap);

__global__ void ssnp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);
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
											 
__global__ void viterbi_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void viterbi_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap);

__global__ void viterbi_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);
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

__global__ void bfs_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap);

__global__ void bfs_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);

__global__ void bfs_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);
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
__global__ void cc_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void cc_incremental_cg(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label2,
							uint *value,
							uint sid,
							uint num_snap);

__global__ void cc_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);
#endif //