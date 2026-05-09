#include "partitioner.cuh"
#include "commons/gpu_error_check.hpp"

template <class E>
Partitioner<E>::Partitioner()
{
	reset();
}

template <class E>
void Partitioner<E>::partition(Subgraph<E> &subgraph, uint numActiveNodes)
{
	reset();
	// cout << "partitioning..." << endl;
	unsigned int from, to;
	unsigned int left, right, mid;
	unsigned int partitionSize;
	unsigned int numNodesInPartition;
	unsigned int numPartitionedEdges;
	bool foundTo;
	unsigned int accurCount;
	// uint local_max_partition_size = 0;
	
	
	from = 0;
	to = numActiveNodes; // last in pointers
	numPartitionedEdges = 0;
	
	do
	{
		left = from;
		right = numActiveNodes;

		// cout << "#active nodes: " << numActiveNodes << endl;
		// cout << "left: " << left << "    right: " << right << endl;
		// cout << "pointer to left: " << subgraph.activeNodesPointer[left] << "    pointer to right: " << subgraph.activeNodesPointer[right] << endl;

		partitionSize = subgraph.activeNodesPointer[right] - subgraph.activeNodesPointer[left];
		if(partitionSize <= subgraph.max_partition_size)
		{
			to = right;
		}
		else
		{
			foundTo = false;
			accurCount = 10;
			while(foundTo == false || accurCount>0)
			{
				mid = (left + right)/2;
				partitionSize = subgraph.activeNodesPointer[mid] - subgraph.activeNodesPointer[from];
				if(foundTo == true)
					accurCount--;
				if(partitionSize <= subgraph.max_partition_size)
				{
					left = mid;
					to = mid;
					foundTo = true;
				}
				else
				{
					right = mid;  
				}
			}
			if(!foundTo && from != numActiveNodes){
                to = from + 1;   
            }
			

			if(to == numActiveNodes)
			{
				cout << "Error in Partitioning...\n";
				exit(-1);
			}

		}

		partitionSize = subgraph.activeNodesPointer[to] - subgraph.activeNodesPointer[from];
		numNodesInPartition = to - from;
		assert(to >= from);

		// if(subgraph.activeNodesPointer[to] - subgraph.activeNodesPointer[from] > local_max_partition_size)
        //     local_max_partition_size = partitionSize;

		// cout << "from: " << from << "   to: " << to << endl;
		// cout << "#nodes in P: " << numNodesInPartition << "    #edges in P: " << partitionSize << endl;
		
		fromNode.push_back(from);
		fromEdge.push_back(numPartitionedEdges);
		partitionNodeSize.push_back(numNodesInPartition);
		partitionEdgeSize.push_back(partitionSize);
		
		from = to;
		numPartitionedEdges += partitionSize;
	
	} while (to != numActiveNodes);

	
	// if(local_max_partition_size != subgraph.max_partition_size){
    //     subgraph.max_partition_size = local_max_partition_size;
    //     cout << "max partition size is adjusted to: " << local_max_partition_size << endl;
    // }
    // gpuErrorcheck(cudaMalloc(&egraph.d_partitionEdgeList, (local_max_partition_size + 1) * sizeof(E)));
	numPartitions = fromNode.size();
    // cout << "#Partition: " << numPartitions << endl;
}


template <class E>
void Partitioner<E>::reset()
{
	fromNode.clear();
	fromEdge.clear();
	partitionNodeSize.clear();
	partitionEdgeSize.clear();
	numPartitions = 0;
}


template class Partitioner<OutEdge_Evolving>;
template class Partitioner<OutEdgeWeighted_Evolving>;