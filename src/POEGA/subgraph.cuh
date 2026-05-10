#ifndef SUBGRAPH_HPP
#define SUBGRAPH_HPP


#include "commons/globals.hpp"


template <class E>
class Subgraph
{
private:

public:
	uint num_nodes;
	uint num_edges;
	uint numActiveNodes;
	
	uint *activeNodes;
	uint *activeNodesPointer;
	E *activeEdgeList;
	
	uint *d_activeNodes;
	uint *d_activeNodesPointer;
	E *d_activeEdgeList;
	
	ull max_partition_size;
	
	Subgraph(uint num_nodes, uint num_edges, ull max_paritition_size);

	~Subgraph();
};

#endif	//	SUBGRAPH_HPP