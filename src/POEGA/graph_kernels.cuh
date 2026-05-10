#ifndef	GRAPH_KERNELS_CUH
#define	GRAPH_KERNELS_CUH

#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/dependencydata.cuh"
#include <curand_kernel.h>

#define WARP_SIZE 32

__global__ void bfs_kernel(uint numNodes,
						   unsigned int *nodesPointer,
						   OutEdgeWeighted_Evolving *edgeList,
						   uint *level,
						   bool *label1,
						   bool *label2,
						   bool *visited,
						   uint cur_level);

__global__ void bfs_kernel(uint numNodes,
						   unsigned int *children,
						   uint *level,
						   bool *label1,
						   bool *label2,
						   bool *visited,
						   uint cur_level);

__global__ void sssp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							uint *dist,
							bool *label1,
							bool *label2,
							// int *parent,
							int sid,
							bool *changed);

__global__ void sssp_kernel(uint numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							uint *dist,
							bool *label1,
							bool *label2,
							int sid);

__global__ void sssp_kernel_common(uint numNodes,
								unsigned int *nodesPointer,
								OutEdgeWeighted_Evolving *edgeList,
								uint *dist,
								bool *label1,
								bool *label2,
								uint num_snap);

__global__ void sssp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap);

__global__ void sssp_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap);


__global__ void sssp_kernel_warp(unsigned int numActiveNodes,
                                        unsigned int *nodesPointer,
                                        OutEdgeWeighted_Evolving *edgeList,
                                        uint *activeSet,
                                        bool *label2,
                                        uint *value,
                                        int numSnap);

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
							int sid);

__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							//    DependencyData *depends,
							uint *value,
							int sid);

// __global__ void sssp_kernel(unsigned int numNodes,
// 							unsigned int from,
// 							unsigned int numPartitionedEdges,
// 							unsigned int *activeNodes,
// 							unsigned int *activeNodesPointer,
// 							OutEdgeWeighted_Evolving *edgeList,
// 							unsigned int *outDegree,
// 							bool *label1,
// 							bool *label2,
// 							DependencyData *depends,
// 							uint *value,
// 							int sid);

__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int fromNode,
							unsigned int fromEdge,
							unsigned int *NodesPointer,
							OutEdgeWeighted_Evolving *edgeList,
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							uint *value,
							int sid);

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
									   uint numSnap);


__global__ void sssp_incremental_cg_concurrent(uint numNodes,
										unsigned int *nodesPointer,
										OutEdgeWeighted_Evolving *edgeList,
										DependencyData *depends,
										bool *label2,
										int num_snap);

__global__ void sssp_incremental_cg_concurrent(uint numNodes,
										unsigned int *nodesPointer,
										OutEdgeWeighted_Evolving *edgeList,
										uint *value,
										bool *label2,
										int numSnap);

__global__ void sssp_incremental_cg_concurrent_bound(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *max_min,
											   uint *value,
											   bool *label2,
											   int numSnap);

__global__ void sssp_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap);

__global__ void sssp_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												uint *old_value,
												uint *val_off,
												uint * val_deg,
												bool *label2,
												int numSnap);

__global__ void sssp_incre_concurrent_locality_common(unsigned int numEdges,
													  unsigned int *nodePointer_el,
													  OutEdgeWeighted_Evolving *edgeList,
													  uint *value,
													  bool *label2,
													  uint numSnap,
													  bool *same);


__global__ void extend_varray_to_edgelist(uint numNodes,
										  uint fromNode,
										  uint fromEdge,
										  uint *nodesPointer,
										  uint *nodesPointer_el,
										  unsigned int *outDegree);

__global__ void sssp_incre_concurrent(unsigned int numEdges,
									  unsigned int *nodePointer_el,
									  OutEdgeWeighted_Evolving *edgeList,
									  uint **value,
									  //   DependencyData *depends,
									  bool *label2,
									  uint numSnap);

__global__ void sssp_incre_concurrent2(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint **value,
									//    DependencyData *depends,
									//    bool *label1,
									   bool *label2,
									   uint numSnap);


__global__ void sssp_incre_concurrent2_locality(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   //   DependencyData *depends,
									   bool *label2,
									   uint numSnap);

// __global__ void sssp_incre_concurrent(unsigned int numEdges,
// 										unsigned int *nodePointer_el,
// 										OutEdgeWeighted_Evolving *edgeList,
// 										uint *value,
// 										bool *label2,
// 										uint sid);

__global__ void sssp_incre_concurrent(unsigned int numEdges,
									  unsigned int *nodePointer_el,
									  OutEdgeWeighted_Evolving *edgeList,
									  uint *value,
									  bool *label2,
									  uint numSnap,
									  uint num_nodes);

__global__ void sssp_incre_concurrent_locality(unsigned int numEdges,
									  unsigned int *nodePointer_el,
									  OutEdgeWeighted_Evolving *edgeList,
									  uint *value,
									  //   DependencyData *depends,
									  bool *label2,
									  uint numSnap);


__global__ void sssp_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *old_value,
											   uint *val_off,
											   uint *val_deg,
											   bool *label2,
											   uint numSnap);

__global__ void sssp_incre_concurrent_locality_bound(unsigned int numEdges,
														 unsigned int *nodePointer_el,
														 OutEdgeWeighted_Evolving *edgeList,
														 uint *value,
														 bool *label2,
														 uint *max,
														 uint *min,
														 uint numSnap);
														 

__global__ void sssp_incre_concurrent_locality_opt(unsigned int numEdges,
													   unsigned int *nodePointer_el,
													   OutEdgeWeighted_Evolving *edgeList,
													   uint *value,
													   bool *label2,
													   uint numSnap);


// __global__ void sssp_kernel_ks(unsigned int numNodes,
// 								unsigned int *nodesPointer,
// 								OutEdgeWeighted_Evolving *edgeList,
// 								unsigned int *dist,
// 								bool *label1,
// 								bool *label2,
// 								uint *level,
// 								int *parent,
// 								int sid);

__global__ void sssp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid);

__global__ void sssp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid);

__global__ void sssp_kernel_ks_concurrent(unsigned int numNodes,
								unsigned int *nodesPointer,
								OutEdgeWeighted_Evolving *edgeList,
								bool *label1,
								bool *label2,
								DependencyData *depends,
								int numSnap);

__global__ void sssp_kernel_ks_concurrent2(unsigned int numNodes,
								unsigned int *nodesPointer,
								OutEdgeWeighted_Evolving *edgeList,
								bool *label1,
								bool *label2,
								DependencyData *depends,
								int numSnap);

__global__ void sssp_kernel_ks_concurrent2(unsigned int numNodes,
								unsigned int *nodesPointer,
								OutEdgeWeighted_Evolving *edgeList,
								bool *label1,
								bool *label2,
								uint *value,
								int numSnap);

__global__ void sssp_kernel_ks_concurrent2(unsigned int numActiveNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   uint *activeSet,
										   bool *label2,
										   uint *value,
										   int numSnap);


__global__ void sssp_kernel_ks_concurrent2_bound(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *max,
										   uint *min,
										   int numSnap);

__global__ void sssp_kernel_ks_concurrent2_bound(unsigned int numActiveNodes,
										   unsigned int * __restrict__ nodesPointer,
										   OutEdgeWeighted_Evolving * __restrict__ edgeList,
										   uint *activeSet,
										   bool *label2,
										   uint *value,
										   uint * __restrict__ max,
										   uint * __restrict__ min,
										   int numSnap);
__global__ void sssp_kernel_ks_concurrent_bound(unsigned int numNodes,
										   unsigned int * __restrict__ nodesPointer,
										   OutEdgeWeighted_Evolving * __restrict__ edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint * __restrict__ max,
										   uint * __restrict__ min,
										   int numSnap);

__global__ void sssp_kernel_ks_concurrent2(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *old_values,
										   uint *value_off,
										   uint *value_deg,
										   int numSnap);

__global__ void sssp_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);

												
__global__ void sssp_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);
											   
__global__ void sssp_incremental_cg_concurrent2_newstorage_waitfree(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);

__global__ void sssp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap);

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
							int sid);

__global__ void ssnp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid);

__global__ void ssnp_kernel_ks_concurrent2(unsigned int numNodes,
								unsigned int *nodesPointer,
								OutEdgeWeighted_Evolving *edgeList,
								bool *label1,
								bool *label2,
								uint *value,
								int numSnap);

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
												 uint numSnap);

// __global__ void ssnp_identifyDel(uint num_edges_del,
// 									 EdgeWeighted *del_edgeList,
// 									 bool *label2,
// 									 DependencyData *depends,
// 									 bool *all_affected_vertices);

// __global__ void ssnp_incre(uint num_edges_add,
// 							   EdgeWeighted *add_edgeList,
// 							   DependencyData *depends,
// 							   bool *all_affected_vertices);


__global__ void ssnp_incre_concurrent_locality(unsigned int numEdges,
												   unsigned int *nodePointer_el,
												   OutEdgeWeighted_Evolving *edgeList,
												   uint *value,
												   bool *label2,
												   uint numSnap);



__global__ void ssnp_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap);
												
__global__ void sssp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);
__global__ void sssp_kernel_concurrent_newstorage_waitfree(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);
__global__ void ssnp_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap);

__global__ void ssnp_incre_concurrent_locality_bound(unsigned int numEdges,
													 unsigned int *nodePointer_el,
													 OutEdgeWeighted_Evolving *edgeList,
													 uint *value,
													 bool *label2,
													 uint *max,
													 uint *min,
													 uint numSnap);

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
											   uint numSnap);

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
											   uint numSnap);

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
													uint numSnap);

__global__ void ssnp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void ssnp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap);

__global__ void ssnp_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);

__global__ void ssnp_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);

__global__ void ssnp_incre_concurrent_locality_bound_newstorage(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   bool *label2,
											   uint *max,
											   uint *min,
											   uint *buffer_counter,
											   uint *buffer_total,
											   uint numSnap);

__global__ void ssnp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void sssp_incre_concurrent2_locality_newstorage_waitfree(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void ssnp_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap);


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
							int sid);

__global__ void sswp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid);


__global__ void sswp_incre_concurrent_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap);

__global__ void sswp_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap);

__global__ void sswp_incremental_cg_concurrent2(uint numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												int numSnap);

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
													uint numSnap);

__global__ void sswp_kernel_ks_concurrent2(unsigned int numNodes,
												unsigned int *nodesPointer,
												OutEdgeWeighted_Evolving *edgeList,
												bool *label1,
												bool *label2,
												uint *value,
										   		int numSnap);

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
												 uint numSnap);


__global__ void sswp_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void sswp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap);

__global__ void sswp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
												uint *up,
												uint *low,
                                                int numSnap);

__global__ void sswp_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);

__global__ void sswp_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);

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
											   uint numSnap);

__global__ void sswp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void sswp_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void sswp_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap);

__global__ void sswp_incre_concurrent_locality_bound(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *up,
											   uint *low,
											   bool *label2,
											   uint numSnap);
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
								int sid);

__global__ void viterbi_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid);

__global__ void viterbi_incre(uint num_edges_add,
								  EdgeWeighted *add_edgeList,
								  DependencyData *depends,
								  bool *all_affected_vertices);


__global__ void viterbi_kernel_ks_concurrent2(unsigned int numNodes,
											  unsigned int *nodesPointer,
											  OutEdgeWeighted_Evolving *edgeList,
											  bool *label1,
											  bool *label2,
											  uint *value,
											  int numSnap);

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
													uint numSnap);

__global__ void viterbi_incre_concurrent_locality(unsigned int numEdges,
													  unsigned int *nodePointer_el,
													  OutEdgeWeighted_Evolving *edgeList,
													  uint *value,
													  bool *label2,
													  uint numSnap);

__global__ void viterbi_incre_concurrent2_locality(unsigned int numEdges,
												unsigned int *nodePointer_el,
												OutEdgeWeighted_Evolving *edgeList,
												uint *value,
												bool *label2,
												uint numSnap);

__global__ void viterbi_incremental_cg_concurrent2(uint numNodes,
												   unsigned int *nodesPointer,
												   OutEdgeWeighted_Evolving *edgeList,
												   uint *value,
												   bool *label2,
												   int numSnap);

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
													uint numSnap);

__global__ void viterbi_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void viterbi_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap);

__global__ void viterbi_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
												uint *up,
												uint *low,
                                                int numSnap);

__global__ void viterbi_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);

__global__ void viterbi_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);

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
											   uint numSnap);

__global__ void viterbi_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void viterbi_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap);
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
						   int sid);

__global__ void bfs_kernel_ks(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  int sid);

__global__ void bfs_incremental_cg_concurrent2(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   int numSnap);

__global__ void bfs_kernel_ks_concurrent2(unsigned int numNodes,
										  unsigned int *nodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  bool *label1,
										  bool *label2,
										  uint *value,
										  int numSnap);

__global__ void bfs_incre_concurrent_locality(unsigned int numEdges,
											  unsigned int *nodePointer_el,
											  OutEdgeWeighted_Evolving *edgeList,
											  uint *value,
											  bool *label2,
											  uint numSnap);

__global__ void bfs_incre_concurrent_locality_bound(unsigned int numEdges,
													unsigned int *nodePointer_el,
													OutEdgeWeighted_Evolving *edgeList,
													uint *value,
													bool *label2,
													uint *max,
													uint *min,
													uint numSnap);

__global__ void bfs_incre_concurrent2_locality(unsigned int numEdges,
												   unsigned int *nodePointer_el,
												   OutEdgeWeighted_Evolving *edgeList,
												   uint *value,
												   bool *label2,
												   uint numSnap);

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
												uint numSnap);

__global__ void bfs_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void bfs_kernel_common_warp(uint numActiveNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   uint *activeSet,
								   bool *label2,
								   uint num_snap);

__global__ void bfs_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap);

__global__ void bfs_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);

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
											   uint numSnap);

__global__ void bfs_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void bfs_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);
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
						   int sid);

__global__ void cc_kernel_ks(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  int sid);

__global__ void cc_incremental_cg_concurrent2(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   int numSnap);

__global__ void cc_kernel_ks_concurrent2(unsigned int numNodes,
										  unsigned int *nodesPointer,
										  OutEdgeWeighted_Evolving *edgeList,
										  bool *label1,
										  bool *label2,
										  uint *value,
										  int numSnap);

__global__ void cc_incre_concurrent_locality(unsigned int numEdges,
											  unsigned int *nodePointer_el,
											  OutEdgeWeighted_Evolving *edgeList,
											  uint *value,
											  bool *label2,
											  uint numSnap);

__global__ void cc_incre_concurrent_locality_bound(unsigned int numEdges,
													unsigned int *nodePointer_el,
													OutEdgeWeighted_Evolving *edgeList,
													uint *value,
													bool *label2,
													uint *max,
													uint *min,
													uint numSnap);

__global__ void cc_incre_concurrent2_locality(unsigned int numEdges,
											   unsigned int *nodePointer_el,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   bool *label2,
											   uint numSnap);

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
												uint numSnap);

__global__ void cc_kernel_common(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);	
								   
__global__ void cc_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
                                                int numSnap);
__global__ void cc_incremental_cg_concurrent2_newstorage(uint numNodes,
											   unsigned int *nodesPointer,
											   OutEdgeWeighted_Evolving *edgeList,
											   uint *value,
											   uint *buffer,
											   uint *buffer_counter,
											   uint *buffer_total,
											   bool *label2,
											   int numSnap);

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
											   uint numSnap);

__global__ void cc_incre_concurrent2_locality_newstorage(unsigned int numEdges,
									   unsigned int *nodePointer_el,
									   OutEdgeWeighted_Evolving *edgeList,
									   uint *value,
									   uint *buffer,
									   uint * buffer_counter,
									   uint * buffer_total,
									   bool *label2,
									   uint numSnap);

__global__ void cc_kernel_concurrent_newstorage(unsigned int numNodes,
										   unsigned int *nodesPointer,
										   OutEdgeWeighted_Evolving *edgeList,
										   bool *label1,
										   bool *label2,
										   uint *value,
										   uint *buffer,
										   uint *buffer_counter,
										   uint *buffer_total,
										   int numSnap);

__global__ void cc_kernel_common_warp(uint numNodes,
								   unsigned int *nodesPointer,
								   OutEdgeWeighted_Evolving *edgeList,
								   uint *dist,
								   bool *label1,
								   bool *label2,
								   uint num_snap);

__global__ void sssp_kernel_concurrent_optimized_transpose(unsigned int numActiveNodes,
                                                unsigned int *nodesPointer,
                                                OutEdgeWeighted_Evolving *edgeList,
                                                uint *activeSet,
                                                bool *label2,
                                                uint *value,
												uint *max,
												uint *min,
                                                int numSnap);


#endif //