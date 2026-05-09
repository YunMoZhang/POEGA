#ifndef	GRAPH_KERNELS_CUH
#define	GRAPH_KERNELS_CUH

#include "commons/globals.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/dependencydata.cuh"


//    ==================================================================
//                             SSSP kernels
//    ==================================================================


__global__ void sssp_kernel_ks(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							unsigned int *dist,
							bool *label1,
							bool *label2,
							uint *level,
							int *parent,
							int sid);

__global__ void sssp_identifyDel(uint num_edges_del,
							EdgeWeighted *del_edgeList, 
							unsigned int *dist,
							bool *label2,
							uint *level,
							int *parent,
							bool *d_all_affected_vertices);

__global__ void sssp_identifyDel(uint num_edges_del,
							EdgeWeighted *del_edgeList, 
							bool *label2,
							DependencyData * depends,
							bool *all_affected_vertices);

__global__ void sssp_jumpstart(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *dist_old,
							bool *label1,
							bool *label2,
							uint *level,
							uint *level_old,
							int *parent,
							int sid, bool debug);

__global__ void sssp_jumpstart(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							bool *label1,
							bool *label2,
							DependencyData * depends,
							DependencyData * depends_old,
							int sid);

__global__ void sssp_jumpstart(unsigned int *q_size1,
							unsigned int *q_size2,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *dist,
							unsigned int *dist_old,
							uint *queue1,
							uint *queue2,
							uint *level,
							uint *level_old,
							int *parent,
							int sid);

__global__ void sssp_propogate(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *dist_old,
							bool *label1,
							bool *label2,
							uint *level,
							int *parent,
							bool *all_affected_vertices,
							int sid, bool debug);

__global__ void sssp_propogate(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							// unsigned int *dist,
							// unsigned int *dist_old,
							bool *label1,
							bool *label2,
							// uint *level,
							// int *parent,
							DependencyData * depends,
							bool *all_affected_vertices,
							int sid);

__global__ void sssp_propogate(unsigned int *q_size1,
							unsigned int *q_size2,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *dist,
							unsigned int *dist_old,
							uint *queue1,
							uint *queue2,
							uint *level,
							int *parent,
							bool *all_affected_vertices,
							int sid);

__global__ void sssp_pull_once(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							unsigned int *dist,
							uint *level,
							int *parent,
							bool *all_affected_vertices,
							int sid);

__global__ void sssp_pull_once(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							DependencyData * depends,
							bool *all_affected_vertices,
							int sid);

__global__ void sssp_incre(uint num_edges_add,
							EdgeWeighted *add_edgeList, 
							unsigned int *dist,
							uint *level,
							int *parent,
							bool *all_affected_vertices);

__global__ void sssp_incre(uint num_edges_add,
							EdgeWeighted *add_edgeList, 
							// unsigned int *dist,
							// uint *level,
							// int *parent,
							DependencyData * depends,
							bool *all_affected_vertices);

__global__ void sssp_kernel_ks(unsigned int *q_size1,
							unsigned int *q_size2,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							unsigned int *dist,
							uint *queue1,
							uint *queue2,
							uint *level,
							int *parent,
							int sid);

__global__ void sssp_kernel_ks(unsigned int numNodes,
							unsigned int *nodesPointer,
							OutEdgeWeighted_Evolving *edgeList, 
							unsigned int *outDegree,
							// unsigned int *dist,
							bool *label1,
							bool *label2,
							// uint *level,
							// int *parent,
							DependencyData * depends,
							int sid);

//    ==================================================================
//                             SSWP kernels
//    ==================================================================

__global__ void sswp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid);

__global__ void sswp_identifyDel(uint num_edges_del,
								 EdgeWeighted *del_edgeList,
								 bool *label2,
								 DependencyData *depends,
								 bool *all_affected_vertices);

__global__ void sswp_jumpstart(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   DependencyData *depends_old,
							   int sid);

__global__ void sswp_propogate(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid);

__global__ void sswp_pull_once(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid);

__global__ void sswp_incre(uint num_edges_add,
						   EdgeWeighted *add_edgeList,
						   DependencyData *depends,
						   bool *all_affected_vertices);

//    ==================================================================
//                             SSNP kernels
//    ==================================================================

__global__ void ssnp_kernel_ks(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   int sid);

__global__ void ssnp_identifyDel(uint num_edges_del,
								 EdgeWeighted *del_edgeList,
								 bool *label2,
								 DependencyData *depends,
								 bool *all_affected_vertices);

__global__ void ssnp_jumpstart(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   DependencyData *depends_old,
							   int sid);

__global__ void ssnp_propogate(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   bool *label1,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid);

__global__ void ssnp_pull_once(unsigned int numNodes,
							   unsigned int *nodesPointer,
							   OutEdgeWeighted_Evolving *edgeList,
							   unsigned int *outDegree,
							   DependencyData *depends,
							   bool *all_affected_vertices,
							   int sid);

__global__ void ssnp_incre(uint num_edges_add,
						   EdgeWeighted *add_edgeList,
						   DependencyData *depends,
						   bool *all_affected_vertices);

//    ==================================================================
//                             Viterbi kernels
//    ==================================================================

__global__ void viterbi_kernel_ks(unsigned int numNodes,
								  unsigned int *nodesPointer,
								  OutEdgeWeighted_Evolving *edgeList,
								  unsigned int *outDegree,
								  bool *label1,
								  bool *label2,
								  DependencyData *depends,
								  int sid);

__global__ void viterbi_identifyDel(uint num_edges_del,
									EdgeWeighted *del_edgeList,
									bool *label2,
									DependencyData *depends,
									bool *all_affected_vertices);

__global__ void viterbi_jumpstart(unsigned int numNodes,
								  unsigned int *nodesPointer,
								  OutEdgeWeighted_Evolving *edgeList,
								  unsigned int *outDegree,
								  bool *label1,
								  bool *label2,
								  DependencyData *depends,
								  DependencyData *depends_old,
								  int sid);

__global__ void viterbi_propogate(unsigned int numNodes,
								  unsigned int *nodesPointer,
								  OutEdgeWeighted_Evolving *edgeList,
								  unsigned int *outDegree,
								  bool *label1,
								  bool *label2,
								  DependencyData *depends,
								  bool *all_affected_vertices,
								  int sid);

__global__ void viterbi_pull_once(unsigned int numNodes,
								  unsigned int *nodesPointer,
								  OutEdgeWeighted_Evolving *edgeList,
								  unsigned int *outDegree,
								  DependencyData *depends,
								  bool *all_affected_vertices,
								  int sid);

__global__ void viterbi_incre(uint num_edges_add,
							  EdgeWeighted *add_edgeList,
							  DependencyData *depends,
							  bool *all_affected_vertices);

//    ==================================================================
//                             BFS kernels
//    ==================================================================

__global__ void bfs_kernel_ks(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  // unsigned int *dist,
							  bool *label1,
							  bool *label2,
							  // uint *level,
							  // int *parent,
							  DependencyData *depends,
							  int sid);

__global__ void bfs_identifyDel(uint num_edges_del,
								EdgeWeighted *del_edgeList,
								bool *label2,
								DependencyData *depends,
								bool *all_affected_vertices);

__global__ void bfs_jumpstart(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  DependencyData *depends_old,
							  int sid);

__global__ void bfs_propogate(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  bool *all_affected_vertices,
							  int sid);

__global__ void bfs_pull_once(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  DependencyData *depends,
							  bool *all_affected_vertices,
							  int sid);

__global__ void bfs_incre(uint num_edges_add,
						  EdgeWeighted *add_edgeList,
						  DependencyData *depends,
						  bool *all_affected_vertices);

//    ==================================================================
//                             CC kernels
//    ==================================================================

__global__ void cc_kernel_ks(unsigned int numNodes,
							 unsigned int *nodesPointer,
							 OutEdgeWeighted_Evolving *edgeList,
							 unsigned int *outDegree,
							 bool *label1,
							 bool *label2,
							 DependencyData *depends,
							 int sid);

__global__ void cc_identifyDel(uint num_edges_del,
							   EdgeWeighted *del_edgeList,
							   bool *label2,
							   DependencyData *depends,
							   bool *all_affected_vertices);

__global__ void cc_jumpstart(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  DependencyData *depends_old,
							  int sid);	
							  
__global__ void cc_propogate(unsigned int numNodes,
							  unsigned int *nodesPointer,
							  OutEdgeWeighted_Evolving *edgeList,
							  unsigned int *outDegree,
							  bool *label1,
							  bool *label2,
							  DependencyData *depends,
							  bool *all_affected_vertices,
							  int sid);

__global__ void cc_pull_once(unsigned int numNodes,
							 unsigned int *nodesPointer,
							 OutEdgeWeighted_Evolving *edgeList,
							 unsigned int *outDegree,
							 DependencyData *depends,
							 bool *all_affected_vertices,
							 int sid);

__global__ void cc_incre(uint num_edges_add,
						  EdgeWeighted *add_edgeList,
						  DependencyData *depends,
						  bool *all_affected_vertices);							 
#endif //