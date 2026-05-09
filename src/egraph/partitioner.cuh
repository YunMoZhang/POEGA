#ifndef PARTITIONER_CUH
#define PARTITIONER_CUH


#include "commons/globals.hpp"
// #include "subgraph.cuh"
#include "evolving_graph.cuh"

template <class E, class EL, class DL>
class Partitioner
{
private:

public:
	uint numPartitions;
	vector<uint> fromNode;
	vector<uint> fromEdge;
	vector<uint> partitionNodeSize;
	vector<uint> partitionEdgeSize;
	Partitioner();
    void partition(Evolving_Graph<E, EL, DL> &egraph);
    void reset();
};

#endif	//	PARTITIONER_CUH