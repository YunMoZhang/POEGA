#ifndef PARTITIONER_CUH
#define PARTITIONER_CUH


#include "commons/globals.hpp"
#include "subgraph.cuh"
#include "evolving_graph.cuh"

template <class E>
class Partitioner
{
private:

public:
	uint numPartitions;
	vector<uint> fromNode;
	vector<uint> fromEdge;
	vector<uint> partitionNodeSize;
	vector<uint> partitionEdgeSize;
	uint num_nodes;
	uint num_edges;
	ull max_partition_size;

	Partitioner(uint num_nodes_, uint num_edges_);
	Partitioner(uint num_nodes_, uint num_edges_, double ratio);
	void partition(Subgraph<E> &subgraph, uint numActiveNodes);

	void partition(Evolving_Graph<E, EdgeWeighted_Union, EdgeWeighted> &graph);
	void reset();
};

#endif	//	PARTITIONER_CUH