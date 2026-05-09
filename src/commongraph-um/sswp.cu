#include "commons/globals.hpp"
#include "commons/timer.hpp"
#include "commons/argument_parser.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/output_utilities.hpp"

#include "evolving_graph.cuh"
#include "labels_kernels.cuh"
#include "graph_kernels.cuh"

int main(int argc, char** argv)
{
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer, timer_sub;
	float sub_time;
	timer.Start();
	
	Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> graph(arguments.input, true, arguments.numSnapshots, arguments.init_percentage, arguments.delta_rate_add, arguments.delta_rate_del);
	int seed = 999;
	graph.ReadGraph(false, seed);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

	/// Init values
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.label1[i] = false;
		graph.label2[i] = false;
		graph.value[i] = 0;
	}
	graph.numActiveNodes = 1;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;
	graph.value [arguments.sourceNode] = DIST_INFINITY;

	graph.copy_to_device_values();

	timer.Start();
	uint itr = 0;

	while (graph.numActiveNodes>0)
	{
		itr++;
		cout << "(Init) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);
		cudaDeviceSynchronize();
		timer_sub.Start();	
		sswp_kernel_common<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes,
																  graph.d_nodePointer,
																  graph.d_edgeList,
																  graph.d_value,
																  graph.d_label1,
																  graph.d_label2,
		  														  graph.numSnapshots + 1);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());
		graph.calCurNumActiveNodes();
		sub_time = timer_sub.Finish();
		cout << sub_time << endl;
	}	

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";
	cout << "Number of iterations = " << itr << endl;
	cudaStream_t *streams = new cudaStream_t[graph.numSnapshots];
	for (uint i = 0; i < graph.numSnapshots; i++)
		cudaStreamCreate(&streams[i]);

	uint **d_values = new uint *[graph.numSnapshots];
	bool **d_label1s = new bool *[graph.numSnapshots];
	bool **d_label2s = new bool *[graph.numSnapshots];

	for (uint i = 0; i < graph.numSnapshots; i++)
	{
		gpuErrorcheck(cudaMallocManaged(&d_values[i], graph.num_nodes * sizeof(uint)));
		copyValue<<<graph.num_nodes / 512 + 1, 512>>>(d_values[i], graph.d_value, graph.num_nodes);

		gpuErrorcheck(cudaMallocManaged(&d_label1s[i], graph.num_nodes * sizeof(bool)));
		gpuErrorcheck(cudaMallocManaged(&d_label2s[i], graph.num_nodes * sizeof(bool)));
	}

	timer.Start();

	for (uint i = 0; i < graph.numSnapshots; i++){
		sswp_incremental_cg<<<graph.num_nodes / 512 + 1, 512, 0, streams[i]>>>(graph.num_nodes,
																	graph.d_nodePointer,
																	graph.d_edgeList,
																	d_label2s[i],
																	d_values[i],
																	i+1,
															    graph.numSnapshots + 1);
	}
	for (uint sid = 0; sid < graph.numSnapshots; sid++)
	{
		cudaStreamSynchronize(streams[sid]);
	}
	for(uint i = 0; i < graph.numSnapshots; i++){
		mergeLabelsNoClear<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label2, d_label2s[i], graph.num_nodes);
	}

	
	graph.calCurNumActiveNodes();
	itr = 0;
	while (graph.numActiveNodes > 0)
	{
		clearLabel<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label2, graph.num_nodes);
		itr++;

		timer_sub.Start();
		cout << "(Snapshot) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		// moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);
		for (uint i = 0; i < graph.numSnapshots; i++){
			moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(d_label1s[i], d_label2s[i], graph.num_nodes);
			
			sswp_kernel<<<graph.num_nodes / 512 + 1, 512, 0, streams[i]>>>(graph.num_nodes,
																graph.d_nodePointer,
																graph.d_edgeList,
																d_label1s[i],
																d_label2s[i],
																d_values[i],
																i+1);
			
		}
		for (uint sid = 0; sid < graph.numSnapshots; sid++)
		{
			cudaStreamSynchronize(streams[sid]);
		}
		gpuErrorcheck(cudaPeekAtLastError());
		for(uint i = 0; i < graph.numSnapshots; i++){
			mergeLabelsNoClear<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label2, d_label2s[i], graph.num_nodes);
		}
		graph.calCurNumActiveNodes();
		float sub_time = timer_sub.Finish();
		cout << sub_time << endl;
	}

	runtime = timer.Finish();
	cout << ">>Processing finished in " << runtime << " (ms).\n";
}