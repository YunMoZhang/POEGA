#include "subgraph_generator.cuh"
// #include "graph.cuh"
#include "evolving_graph.cuh"
#include "subgraph.cuh"
#include "commons/gpu_error_check.hpp"
#include "commons/timer.hpp"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

const unsigned int NUM_THREADS = 32;

const unsigned int THRESHOLD_THREAD = 50000;
// const unsigned int THRESHOLD_THREAD = 1000;

__global__ void prePrefix(unsigned int *activeNodesLabeling, unsigned int *activeNodesDegree, 
	unsigned int *outDegree, bool *label1, bool *label2, unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes){
		activeNodesLabeling[id] = label1[id] || label2[id]; // label1 is always zero in sync
		//activeNodesLabeling[id] = label[id];
		//activeNodesLabeling[id] = 1;
		activeNodesDegree[id] = 0;
		if(activeNodesLabeling[id] == 1)
			activeNodesDegree[id] = outDegree[id];	
	}	
}

__global__ void prePrefix(unsigned int *activeNodesLabeling, unsigned int *activeNodesDegree, 
							unsigned int *outDegree, float *delta, unsigned int numNodes, float acc)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes){
		if(delta[id] > acc)
		{
			activeNodesLabeling[id] = 1;
		}
		else
		{
			activeNodesLabeling[id] = 0;
		}
		activeNodesDegree[id] = 0;
		if(activeNodesLabeling[id] == 1)
			activeNodesDegree[id] = outDegree[id];	
	}
}

__global__ void makeQueue(unsigned int *activeNodes, unsigned int *activeNodesLabeling,
							unsigned int *prefixLabeling, unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes && activeNodesLabeling[id] == 1){
		activeNodes[prefixLabeling[id]] = id;
	}
}

__global__ void makeActiveNodesPointer(unsigned int *activeNodesPointer, unsigned int *activeNodesLabeling, 
											unsigned int *prefixLabeling, unsigned int *prefixSumDegrees, 
											unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes && activeNodesLabeling[id] == 1){
		activeNodesPointer[prefixLabeling[id]] = prefixSumDegrees[id];
	}
}

// pthread
template <class E, class EL, class DL>
void dynamic(unsigned int tId,
				unsigned int numThreads,	
				unsigned int numActiveNodes,
				unsigned int *activeNodes,
				unsigned int *outDegree, 
				unsigned int *activeNodesPointer,
				unsigned int *nodePointer, 
				E *activeEdgeList,
				E *edgeList)
{

	// unsigned int chunkSize = ceil(numActiveNodes / numThreads);
	unsigned int chunkSize = (unsigned int)ceil((double)numActiveNodes / numThreads);
	unsigned int left, right;
	left = tId * chunkSize;
	right = min(left+chunkSize, numActiveNodes);	
	
	unsigned int thisNode;
	unsigned int thisDegree;
	unsigned int fromHere;
	unsigned int fromThere;

	for(unsigned int i=left; i<right; i++)
	{
		thisNode = activeNodes[i];
		thisDegree = outDegree[thisNode];
		fromHere = activeNodesPointer[i];
		fromThere = nodePointer[thisNode];
		for(unsigned int j=0; j<thisDegree; j++)
		{
			activeEdgeList[fromHere+j] = edgeList[fromThere+j];
		}
	}
	
}

template <class E, class EL, class DL>
SubgraphGenerator<E, EL, DL>::SubgraphGenerator(Evolving_Graph<E, EL, DL> &graph)
{
	gpuErrorcheck(cudaMallocHost(&activeNodesLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocHost(&activeNodesDegree, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocHost(&prefixLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocHost(&prefixSumDegrees, (graph.num_nodes+1) * sizeof(unsigned int)));
	
	gpuErrorcheck(cudaMalloc(&d_activeNodesLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeNodesDegree, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_prefixLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_prefixSumDegrees , (graph.num_nodes+1) * sizeof(unsigned int)));
}

template <class E, class EL, class DL>
void SubgraphGenerator<E, EL, DL>::generate(Evolving_Graph<E, EL, DL> &graph, Subgraph<E> &subgraph)
{
	// std::chrono::time_point<std::chrono::system_clock> startDynG, finishDynG;
	// startDynG = std::chrono::system_clock::now();
	// Timer t;
	// float time;
	// t.Start();
	prePrefix<<<graph.num_nodes / 512 + 1, 512>>>(d_activeNodesLabeling, d_activeNodesDegree, graph.d_outDegree, graph.d_label1, graph.d_label2, graph.num_nodes);
	// time = t.Finish();
	// cout << "preprocess: " << time << "\t";
	thrust::device_ptr<unsigned int> ptr_labeling(d_activeNodesLabeling);
	thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(d_prefixLabeling);

	subgraph.numActiveNodes = thrust::reduce(ptr_labeling, ptr_labeling + graph.num_nodes);
	// cout << "Number of Active Nodes = " << subgraph.numActiveNodes << endl;
	// t.Start();
	thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph.num_nodes, ptr_labeling_prefixsum);
	// time = t.Finish();
	// cout << "prefixsum: " << time << "\t";

	// t.Start();
	makeQueue<<<graph.num_nodes / 512 + 1, 512>>>(subgraph.d_activeNodes, d_activeNodesLabeling, d_prefixLabeling, graph.num_nodes);
	// time = t.Finish();
	// cout << "makequeue: " << time << "\t";
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodes, subgraph.d_activeNodes, subgraph.numActiveNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// time = t.Finish();
	// cout << "Active nodes vector: " << time << "\t";

	thrust::device_ptr<unsigned int> ptr_degrees(d_activeNodesDegree);
	thrust::device_ptr<unsigned int> ptr_degrees_prefixsum(d_prefixSumDegrees);

	thrust::exclusive_scan(ptr_degrees, ptr_degrees + graph.num_nodes, ptr_degrees_prefixsum);

	makeActiveNodesPointer<<<graph.num_nodes / 512 + 1, 512>>>(subgraph.d_activeNodesPointer, d_activeNodesLabeling, d_prefixLabeling, d_prefixSumDegrees, graph.num_nodes);
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, subgraph.numActiveNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	unsigned int numActiveEdges = 0;
	if (subgraph.numActiveNodes > 0)
		numActiveEdges = subgraph.activeNodesPointer[subgraph.numActiveNodes - 1] + graph.outDegree[subgraph.activeNodes[subgraph.numActiveNodes - 1]];

	unsigned int last = numActiveEdges;
	gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer + subgraph.numActiveNodes, &last, sizeof(unsigned int), cudaMemcpyHostToDevice));

	gpuErrorcheck(cudaMemcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, (subgraph.numActiveNodes + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// cout << "Number of Active Edges = " << subgraph.activeNodesPointer[subgraph.numActiveNodes] << endl;
	// finishDynG = std::chrono::system_clock::now();
	// std::chrono::duration<double> elapsed_seconds_dyng = finishDynG-startDynG;
	// std::time_t finish_time_dyng = std::chrono::system_clock::to_time_t(finishDynG);
	// std::cout << "Dynamic GPU Time = " << elapsed_seconds_dyng.count() << std::endl;

	// td::chrono::time_point<std::chrono::system_clock> startDynC, finishDynC;
	// startDynC = std::chrono::system_clock::now();

	unsigned int numThreads = NUM_THREADS;

	if (subgraph.numActiveNodes < THRESHOLD_THREAD)
		numThreads = 1;

	thread runThreads[numThreads];


	for (unsigned int t = 0; t < numThreads; t++)
	{
		runThreads[t] = thread(dynamic<E, EL, DL>,
							   t,
							   numThreads,
							   subgraph.numActiveNodes,
							   subgraph.activeNodes,
							   graph.outDegree,
							   subgraph.activeNodesPointer,
							   graph.nodePointer,
							   subgraph.activeEdgeList,
							   graph.edgeList);
	}

	for (unsigned int t = 0; t < numThreads; t++)
		runThreads[t].join();
	// cout << "CPU time: " << cpu_time << "\t";

	// finishDynC = std::chrono::system_clock::now();
	// std::chrono::duration<double> elapsed_seconds_dync = finishDynC-startDynC;
	// std::time_t finish_time_dync = std::chrono::system_clock::to_time_t(finishDynC);
	// std::cout << "Dynamic CPU Time = " << elapsed_seconds_dync.count() << std::endl;
}

template <class E, class EL, class DL>
void SubgraphGenerator<E, EL, DL>::generate(Evolving_Graph<E, EL, DL> &graph, Subgraph<E> &subgraph, float & cpu_time)
{
	//std::chrono::time_point<std::chrono::system_clock> startDynG, finishDynG;
	//startDynG = std::chrono::system_clock::now();
	Timer t;
	// float time;
	// t.Start();
	prePrefix<<<graph.num_nodes/512+1, 512>>>(d_activeNodesLabeling, d_activeNodesDegree, graph.d_outDegree, graph.d_label1, graph.d_label2, graph.num_nodes);
	// time = t.Finish();
	// cout << "preprocess: " << time << "\t";
	thrust::device_ptr<unsigned int> ptr_labeling(d_activeNodesLabeling);
	thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(d_prefixLabeling);
	
	subgraph.numActiveNodes = thrust::reduce(ptr_labeling, ptr_labeling + graph.num_nodes);
	// cout << "Number of Active Nodes = " << subgraph.numActiveNodes << endl;
	// t.Start();
	thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph.num_nodes, ptr_labeling_prefixsum);
	// time = t.Finish();
	// cout << "prefixsum: " << time << "\t";

	// t.Start();
	makeQueue<<<graph.num_nodes/512+1, 512>>>(subgraph.d_activeNodes, d_activeNodesLabeling, d_prefixLabeling, graph.num_nodes);
	// time = t.Finish();
	// cout << "makequeue: " << time << "\t";
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodes, subgraph.d_activeNodes, subgraph.numActiveNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// time = t.Finish();
	// cout << "Active nodes vector: " << time << "\t";

	thrust::device_ptr<unsigned int> ptr_degrees(d_activeNodesDegree);
	thrust::device_ptr<unsigned int> ptr_degrees_prefixsum(d_prefixSumDegrees);

	thrust::exclusive_scan(ptr_degrees, ptr_degrees + graph.num_nodes, ptr_degrees_prefixsum);
	
	makeActiveNodesPointer<<<graph.num_nodes/512+1, 512>>>(subgraph.d_activeNodesPointer, d_activeNodesLabeling, d_prefixLabeling, d_prefixSumDegrees, graph.num_nodes);
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, subgraph.numActiveNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	unsigned int numActiveEdges = 0;
	if(subgraph.numActiveNodes > 0)
		numActiveEdges = subgraph.activeNodesPointer[subgraph.numActiveNodes-1] + graph.outDegree[subgraph.activeNodes[subgraph.numActiveNodes-1]];	
	
	unsigned int last = numActiveEdges;
	gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer+subgraph.numActiveNodes, &last, sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, (subgraph.numActiveNodes+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	// cout << "Number of Active Edges = " << subgraph.activeNodesPointer[subgraph.numActiveNodes] << endl;
	//finishDynG = std::chrono::system_clock::now();
	//std::chrono::duration<double> elapsed_seconds_dyng = finishDynG-startDynG;
	//std::time_t finish_time_dyng = std::chrono::system_clock::to_time_t(finishDynG);
	//std::cout << "Dynamic GPU Time = " << elapsed_seconds_dyng.count() << std::endl;
	
	//td::chrono::time_point<std::chrono::system_clock> startDynC, finishDynC;
	//startDynC = std::chrono::system_clock::now();
	
	unsigned int numThreads = NUM_THREADS;

	if(subgraph.numActiveNodes < THRESHOLD_THREAD)
		numThreads = 1;

	thread runThreads[numThreads];

	t.Start();
	
	for(unsigned int t=0; t<numThreads; t++)
	{
		runThreads[t] = thread(dynamic<E, EL, DL>,
								t,
								numThreads,
								subgraph.numActiveNodes,
								subgraph.activeNodes,
								graph.outDegree, 
								subgraph.activeNodesPointer,
								graph.nodePointer, 
								subgraph.activeEdgeList,
								graph.edgeList);

	}
		
	for(unsigned int t=0; t<numThreads; t++)
		runThreads[t].join();

	cpu_time += t.Finish();
	// cout << "CPU time: " << cpu_time << "\t";
	
	//finishDynC = std::chrono::system_clock::now();
	//std::chrono::duration<double> elapsed_seconds_dync = finishDynC-startDynC;
	//std::time_t finish_time_dync = std::chrono::system_clock::to_time_t(finishDynC);
	//std::cout << "Dynamic CPU Time = " << elapsed_seconds_dync.count() << std::endl;
	
}



template class SubgraphGenerator<OutEdge_Evolving, Edge_Union, Edge>;
template class SubgraphGenerator<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>;