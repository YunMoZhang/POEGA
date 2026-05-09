#include "commons/globals.hpp"
#include "commons/timer.hpp"
#include "commons/argument_parser.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/output_utilities.hpp"

#include "evolving_graph.cuh"
#include "labels_kernels.cuh"
#include "graph_kernels.cuh"
#include "partitioner.cuh"

int main(int argc, char** argv)
{
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer, timer_sub, timer_sub2, timer_sub3;
	float sub_time;
	timer.Start();
	
	Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> graph(arguments.input, true, arguments.numSnapshots, arguments.init_percentage, arguments.delta_rate_add, arguments.delta_rate_del);
	int seed = 999;
	graph.ReadGraph(false, seed);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

	for(uint i = 0; i <= graph.numSnapshots; i++){
		graph.value[i] = new uint[graph.num_nodes];
		graph.label1[i] = new bool[graph.num_nodes];
		graph.label2[i] = new bool[graph.num_nodes];
	}

	/// Init values
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.label1[0][i] = false;
		graph.label2[0][i] = false;
		graph.value[0][i] = 0;
	}
	graph.numActiveNodes = 1;
	graph.label2[0][arguments.sourceNode] = true;
	graph.value[0][arguments.sourceNode] = DIST_INFINITY;
	for(uint i = 1; i <= graph.numSnapshots; i++){
		memcpy(graph.value[i], graph.value[0], graph.num_nodes * sizeof(uint));
		memcpy(graph.label1[i], graph.label1[0], graph.num_nodes * sizeof(bool));
		memcpy(graph.label2[i], graph.label2[0], graph.num_nodes * sizeof(bool));
	}

	// graph.copy_to_device_values();

	Partitioner<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> partitioner;
	partitioner.partition(graph);
	
	/// Init computation
	timer.Start();
	uint itr = 0;
	bool finished = false;
	while (!finished)
	{
		itr++;
		finished = true;
		cout << "(Init) Running itr " << itr << endl;
		for(uint sid = 0; sid <= graph.numSnapshots; sid++){
			memcpy(graph.label1[sid], graph.label2[sid], graph.num_nodes * sizeof(bool));
			memset(graph.label2[sid], false, graph.num_nodes * sizeof(bool));
		}
		float comm_time = 0, comp_time = 0;
		timer_sub.Start();	
		for(int pi = 0; pi < partitioner.numPartitions; pi++)
		{	
			// cout << "=== Partition "<< pi << endl;
			timer_sub3.Start();
			timer_sub2.Start();
			gpuErrorcheck(cudaMemcpy(graph.d_partitionEdgeList, graph.edgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
			comm_time += timer_sub2.Finish();
			// clearLabel<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label2, graph.num_nodes);

			for(uint sid = 0; sid <= graph.numSnapshots; sid++){
				timer_sub2.Start();
				gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1[sid], graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
				gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2[sid], graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
				gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value[sid], graph.num_nodes * sizeof(uint), cudaMemcpyHostToDevice));
				
				comm_time += timer_sub2.Finish();

				timer_sub2.Start();
				sswp_kernel<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[pi],
															partitioner.fromNode[pi],
															partitioner.fromEdge[pi],
															graph.d_nodePointer,
															graph.d_partitionEdgeList,
															graph.d_outDegree,
															graph.d_label1,
															graph.d_label2,
															graph.d_value,
															sid);
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaPeekAtLastError());	
				comp_time += timer_sub2.Finish();
				sub_time = timer_sub2.Finish();

				gpuErrorcheck(cudaMemcpy(graph.value[sid], graph.d_value, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));
				gpuErrorcheck(cudaMemcpy(graph.label2[sid], graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));

				graph.calCurNumActiveNodes();
				cout << "snapshot " << sid << " #active vertices: " << graph.numActiveNodes << endl;
				if(graph.numActiveNodes > 0) 
					finished = false;
			}
			float partition_time = timer_sub3.Finish();
			// cout << partition_time << endl;
		}
		
		sub_time = timer_sub.Finish();
		cout << sub_time << " comm time: " << comm_time << "  comp time: " << comp_time <<endl;
	}	
	
	float runtime = timer.Finish();
	cout << "(SSWP) Processing finished in " << runtime << " (ms).\n";
	cout << "Number of iterations = " << itr << endl;
}