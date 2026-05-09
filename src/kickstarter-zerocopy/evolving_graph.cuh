#ifndef Evolving_GRAPH_HPP
#define Evolving_GRAPH_HPP

#include "commons/globals.hpp"
#include "commons/pvector.hpp"
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
    void AssignW8(E & elem, uint w8);
	void AssignW8(DL & elem, uint w8);
	void CopyW8(DL &elem1, E& elem2);
	void Set_Bit_Snap(uint pos, int sid);
	void Reset_Bit_Snap(uint pos, int sid);
	// void GenerateAdd(int sid);
	// void ApplyBatch();
	void CreateCSR(uint num_edges, uint *nodePointer, E *edgeList, int sid_a[], int sid_d[]);
	void ReadGraph(bool fileContainsWeight, int seed);
	void calCurNumActiveNodes();
	void SetAsWeight(bool isWeighted);
	void copy_to_device_values();
	void copy_to_device_delta(uint i);
	void countTotalDegree(bool *label);

public : 
	string graphFilePath;
	string graphFormat;
	bool isWeighted;
	bool isLarge;

	uint num_nodes;
	uint num_edges_base;   // the number of edges in curret snapshot
	uint num_edges_input;
	uint num_edges_total;  // the total number of edges, including base and dynamic
	uint num_edges_add;
	uint num_edges_del;
	uint numActiveNodes;

	pvector<EL> edgesInput; // the dynamic graph edge list

	uint *nodePointer; // using NEW with size (num_nodes+1)
	E *edgeList;		// using cudaMallocHost with size num_edges
	DL **add_edgeList;
	DL **del_edgeList;
	uint* deleted_idx;

	// using NEW for initialization, with size num_nodes
	uint *outDegree; 
	bool *label1; 
	bool *label2; 
	
	DependencyData * depends;

	uint *d_nodePointer; // using cudaMalloc with size (num_nodes+1)
	E *d_edgeList;		// using cudaMalloc with size num_edges
	// DL **d_add_edgeList;
	// DL **d_del_edgeList;
	DL *d_add_edgeList;
	DL *d_del_edgeList;

	// using cudaMalloc for initialization, with size num_nodes
	uint *d_outDegree;
	bool *d_label1;
	bool *d_label2;
	bool *d_all_affected_vertices;

	DependencyData * d_depends;
	DependencyData * d_depends_old;

	// parameters of setting evolving graphs
	// int cur_snap;
	int numSnapshots;
	double delta_rate_add;
	double delta_rate_del;
	double init_percentage;
private:

};

#endif	//	Evolving_GRAPH_HPP