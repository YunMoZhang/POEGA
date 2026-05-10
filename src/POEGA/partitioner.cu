#include "partitioner.cuh"
#include "commons/gpu_error_check.hpp"
#include <cuda_profiler_api.h>

template <class E>
Partitioner<E>::Partitioner(uint num_nodes_, uint num_edges_)
{
	reset();
	this->num_nodes = num_nodes_;
	this->num_edges = num_edges_;
	cudaProfilerStart();
	cudaError_t error;
	cudaDeviceProp dev;
	int deviceID;
	cudaGetDevice(&deviceID);
	error = cudaGetDeviceProperties(&dev, deviceID);
	if (error != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaProfilerStop();

	size_t free, total;
	cudaMemGetInfo(&free, &total);

	cout << "dev.total global memory: " << dev.totalGlobalMem << endl;
	// cout << "dev.totalGlobalMem - 20 * 4 * (size_t) num_nodes: " << dev.totalGlobalMem - 20 * 4 * (size_t) num_nodes << endl;
	// this->max_partition_size = 0.3 * (dev.totalGlobalMem - 64 * 4 * (size_t)num_nodes) / sizeof(E); // for 64
	this->max_partition_size = 0.4 * (free -  6 * 4 * (size_t) num_nodes) / sizeof(E);

	if (max_partition_size > num_edges_){
		max_partition_size = num_edges_;
		cout << "Reset max martition size to num_edges: " << num_edges_ << endl;
	}
	cout << "Max Partition Size: " << this->max_partition_size << endl;
}

template <class E>
Partitioner<E>::Partitioner(uint num_nodes_, uint num_edges_, double ratio)
{
	reset();
	this->num_nodes = num_nodes_;
	this->num_edges = num_edges_;
	cudaProfilerStart();
	cudaError_t error;
	cudaDeviceProp dev;
	int deviceID;
	cudaGetDevice(&deviceID);
	error = cudaGetDeviceProperties(&dev, deviceID);
	if (error != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaProfilerStop();

	cout << "dev.total global memory: " << dev.totalGlobalMem << endl;
	// cout << "dev.totalGlobalMem - 20 * 4 * (size_t) num_nodes: " << dev.totalGlobalMem - 20 * 4 * (size_t) num_nodes << endl;
	// this->max_partition_size = ratio * (dev.totalGlobalMem - 28 * 4 * (size_t)num_nodes) / sizeof(E); // normal case
	this->max_partition_size = ratio * (dev.totalGlobalMem - 35 * 4 * (size_t)num_nodes) / sizeof(E); // normal case
	// this->max_partition_size = ratio * (dev.totalGlobalMem - 64 * 4 * (size_t)num_nodes) / sizeof(E); // for 64

	if (max_partition_size > num_edges_)
	{
		max_partition_size = num_edges_;
		cout << "Reset max martition size to num_edges: " << num_edges_ << endl;
	}
	cout << "Max Partition Size: " << this->max_partition_size << endl;
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
void Partitioner<E>::partition(Evolving_Graph<E, EdgeWeighted_Union, EdgeWeighted> &egraph)
{
	reset();
	// cout << "Partition..." << endl;
	unsigned int from, to;
	unsigned int left, right, mid;
	unsigned int partitionSize;
	unsigned int numNodesInPartition;
	unsigned int numPartitionedEdges;
	bool foundTo;
	unsigned int accurCount;
	// uint local_max_partition_size = 0;

	from = 0;
	to = egraph.num_nodes; // last in pointers
	numPartitionedEdges = 0;

	do
	{
		left = from;
		right = egraph.num_nodes;

		// cout << "#active nodes: " << numActiveNodes << endl;
		//  cout << "left: " << left << "    right: " << right << endl;
		// cout << "pointer to left: " << subgraph.activeNodesPointer[left] << "    pointer to right: " << subgraph.activeNodesPointer[right] << endl;

		partitionSize = egraph.nodePointer[right] - egraph.nodePointer[left];
		if (partitionSize <= max_partition_size)
		{
			to = right;
		}
		else
		{
			foundTo = false;
			accurCount = 10;
			while (foundTo == false && accurCount > 0)
			{
				mid = (left + right) / 2;
				// if(mid == from){
				//     cout << "Error in partition... left" << left << " mid: " << mid << " right: " << right << endl;
				//     exit(-1);
				// }
				partitionSize = egraph.nodePointer[mid] - egraph.nodePointer[from];
				// cout << "right : " << right << "\t left: " << left << "\t mid: " << mid<< " from: " << from << " to: " << to << " partition size: " << partitionSize << endl;
				if (foundTo == true)
					accurCount--;
				if (partitionSize <= max_partition_size)
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
			if (!foundTo && from != egraph.num_nodes)
			{
				to = from + 1;
			}

			if (to == egraph.num_nodes)
			{
				cout << "Error in Partitioning...\n";
				exit(-1);
			}
		}

		partitionSize = egraph.nodePointer[to] - egraph.nodePointer[from];
		numNodesInPartition = to - from;


		// cout << "from: " << from << "   to: " << to << endl;
		// cout << "#nodes in P: " << numNodesInPartition << "    #edges in P: " << partitionSize << endl;

		fromNode.push_back(from);
		fromEdge.push_back(numPartitionedEdges);
		partitionNodeSize.push_back(numNodesInPartition);
		partitionEdgeSize.push_back(partitionSize);

		from = to;
		numPartitionedEdges += partitionSize;

	} while (to != egraph.num_nodes);

	// if (local_max_partition_size != egraph.max_partition_size)
	// {
	// 	egraph.max_partition_size = local_max_partition_size;
	// 	cout << "max partition size is adjusted to: " << local_max_partition_size << endl;
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