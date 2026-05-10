#ifndef Evolving_GRAPH_HPP
#define Evolving_GRAPH_HPP

#include "commons/globals.hpp"
#include "commons/pvector.hpp"
#include "commons/bitmap.hpp"
#include "commons/dependencydata.cuh"
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thread>

template <class E, class EL, class DL>
class Evolving_Graph
{
public:
    Evolving_Graph(string graphFilePath, bool isWeighted, 
		int num_snap, double init_p, double delta_r_a, double delta_r_d);
    string GetFileExtension(string fileName);
    void AssignW8(E &elem, uint w8);
	void AssignW8(DL &elem, uint w8);
	void AssignW8(DL &elem1, DL &elem2);

	void Set_Bit_Snap(uint pos, int sid);
	void Reset_Bit_Snap(uint pos, int sid);
	void Set_Bitmap(E el[], uint pos, ull bmp);
	void Set_Bit_Snap(E el[], uint pos, int sid);
	void Reset_Bit_Snap(E el[], uint pos, int sid);
	bool Get_Bit_Snap(uint pos, int sid);
	uint64_t word_offset(uint pos);
	uint64_t bit_offset(size_t pos);
	uint num_words(uint size);
	void Set_Bit(uint64_t *bmp, uint pos);
	void Reset_Bit(uint64_t *bmp, uint pos);

	void CreateCSR(uint num_edges, pvector<EL> &inputEdgeList, uint *nodePointer, E * &edgeList, uint degrees[], uint8_t sid_a[], uint8_t sid_d[]);
	void ReadGraph(bool fileContainsWeight, int seed);
	void calCurNumActiveNodes();
	void calCurNumActiveNodes(bool * d_labels);
	void calCurNumActiveNodes(cudaStream_t stream);
	void calCurNumActiveNodes(bool * d_label, cudaStream_t stream);

	void SetAsWeight(bool isWeighted);
	void TransferValuesToDevice();

	void LoadCoreUnionCSR();

	void CreateProxyGraphUnionCSR(int degree_limit = 800, uint bridge_thresh = 800);

	void CreateLowWeightCoreUnionCSR(int seed); // for testing wonderland
	void CreateRandomCoreUnionCSR(int seed);
	
public:
	string graphFilePath;
	string graphFormat;
	bool isWeighted;

	uint num_nodes;
	
	uint num_edges_base;
	ull num_edges_input;
	uint num_edges_total;  // the total number of edges, including base and dynamic
	uint num_edges_add;
	uint num_edges_del;
	uint num_edges_core;
	uint num_edges_max_core;
	uint * num_edges_add_core;
	uint num_nodes_power_law_csg; // currently useless
	uint numActiveNodes;
	int MaxDegree;

	pvector<EL> edgesInput; // the dynamic graph edge list

	uint *nodePointer; // using NEW with size (num_nodes+1)
	E *edgeList;		// using cudaMallocHost with size num_edges

	DL **add_edgeList;
	DL **del_edgeList;
	DL **add_edgeList_core;

	// used for proxy graphs 
	uint **nodePointer_csg;  // separate csg;
	E **edgeList_csg;
	uint *nodePointer_csg_u; // union of csgs
	E *edgeList_csg_u;

	// using NEW for initialization, with size num_nodes
	bool *label1; 
	bool *label2;
	uint *outDegree;
	uint *core_outDegree;
	bool *core_active_vertices;
	bool *direct_affected_vertices;

	uint *value;
	uint **value_core;
	uint *value_data; // 1-D array represents 2-D values for all snapshots
	uint *buffer;

	DependencyData * depends;
	DependencyData * depends_core;
	DependencyData ** depends_proxy;

	uint *d_nodePointer; 	// using cudaMalloc with size (num_nodes+1)
	E *d_edgeList;		 	// using cudaMalloc with size num_edges
	uint *d_nodePointer_core = nullptr;// core graph csr
	E *d_edgeList_core = nullptr;
	DL *d_add_edgeList;
	DL *d_del_edgeList;
	uint * d_outDegree;
	uint * d_core_outDegree;

	bool *d_label1;
	bool *d_label2;   	// frontiers of nodes in grid

	uint *d_value;    	// values of nodes in grid
	uint *d_activeNodes;

	DependencyData * d_depends;
	DependencyData * d_depends_old;

	int top_k_degree;  // for pre-evaluation
	uint *top_k_degree_nodes;

	// parameters of setting evolving graphs
	int numSnapshots;
	double delta_rate_add;
	double delta_rate_del;
	double init_percentage;
};

#endif	//	Evolving_GRAPH_HPP