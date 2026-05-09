#include "partitioner.cuh"
#include "commons/gpu_error_check.hpp"

template <class E, class EL, class DL>
Partitioner<E, EL, DL>::Partitioner()
{
	reset();
}

template <class E, class EL, class DL>
void Partitioner<E, EL, DL>::partition(Evolving_Graph<E, EL, DL> &egraph)
{
	reset();
	cout << "Partition..."<< endl;
	unsigned int from, to;
	unsigned int left, right, mid;
	unsigned int partitionSize;
	unsigned int numNodesInPartition;
	unsigned int numPartitionedEdges;
	bool foundTo;
	unsigned int accurCount;
    uint local_max_partition_size = 0;
	
	
	from = 0;
	to = egraph.num_nodes; // last in pointers
	numPartitionedEdges = 0;
	
	do
	{
		left = from;
		right = egraph.num_nodes;

		//cout << "#active nodes: " << numActiveNodes << endl;
		// cout << "left: " << left << "    right: " << right << endl;
		//cout << "pointer to left: " << subgraph.activeNodesPointer[left] << "    pointer to right: " << subgraph.activeNodesPointer[right] << endl;

		partitionSize = egraph.nodePointer[right] - egraph.nodePointer[left];
		if(partitionSize <= egraph.max_partition_size)
		{
			to = right;
		}
		else
		{
			foundTo = false;
			accurCount = 10;
			while((right > left + 2) &&(foundTo == false || accurCount > 0))
			{
				mid = (left + right)/2;   
                // if(mid == from){
                //     cout << "Error in partition... left" << left << " mid: " << mid << " right: " << right << endl;
                //     exit(-1);
                // }
				partitionSize = egraph.nodePointer[mid] - egraph.nodePointer[from];
                // cout << "right : " << right << "\t left: " << left << "\t mid: " << mid<< " from: " << from << " to: " << to << " partition size: " << partitionSize << endl;
				if(foundTo == true)
					accurCount--;
				if(partitionSize <= egraph.max_partition_size)
				{
					left = mid;
					to = mid;
					foundTo = true;
                    // cout << "found to: " << to << endl;
				}
				else
				{
					right = mid;  
				}
			}
			if(!foundTo && from != egraph.num_nodes){
                to = from + 1;
            }

			if(to == egraph.num_nodes)
			{
				cout << "Error in Partitioning...\n";
				exit(-1);
			}

		}

		partitionSize = egraph.nodePointer[to] - egraph.nodePointer[from];
		numNodesInPartition = to - from;

		if(egraph.nodePointer[to] - egraph.nodePointer[from] > local_max_partition_size)
            local_max_partition_size = partitionSize;

		// cout << "from: " << from << "   to: " << to << endl;
		// cout << "#nodes in P: " << numNodesInPartition << "    #edges in P: " << partitionSize << endl;
        if(numNodesInPartition == 0){
            cout << "Error in partition... left" << left << " mid: " << mid << " right: " << right << endl;
            exit(-1);
        }
		fromNode.push_back(from);
		fromEdge.push_back(numPartitionedEdges);
		partitionNodeSize.push_back(numNodesInPartition);
		partitionEdgeSize.push_back(partitionSize);
		
		from = to;
		numPartitionedEdges += partitionSize;
	
	} while (to != egraph.num_nodes);
	
    if(local_max_partition_size != egraph.max_partition_size){
        egraph.max_partition_size = local_max_partition_size;
        cout << "max partition size is adjusted to: " << local_max_partition_size << endl;
    }
    gpuErrorcheck(cudaMalloc(&egraph.d_partitionEdgeList, (local_max_partition_size + 1) * sizeof(E)));
	numPartitions = fromNode.size();
    cout << "#Partition: " << numPartitions << endl;
}

template <class E, class EL, class DL>
void Partitioner<E, EL, DL>::reset()
{
	fromNode.clear();
	fromEdge.clear();
	partitionNodeSize.clear();
	partitionEdgeSize.clear();
	numPartitions = 0;
}


template class Partitioner<OutEdge_Evolving, Edge_Union, Edge>;
template class Partitioner<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>;