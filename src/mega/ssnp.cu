#include "commons/globals.hpp"
#include "commons/timer.hpp"
#include "commons/argument_parser.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/output_utilities.hpp"

#include "evolving_graph.cuh"
#include "labels_kernels.cuh"
#include "graph_kernels.cuh"
#include "partitioner.cuh"
#include "subgraph.cuh"
#include "subgraph_generator.cuh"

int main(int argc, char** argv)
{
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer, timer_sub, timer_sub2;
	float sub_time;
	timer.Start();
	
	Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> graph(arguments.input, true, arguments.numSnapshots, arguments.init_percentage, arguments.delta_rate_add, arguments.delta_rate_del);
	int seed = 999;
	graph.ReadGraph(false, seed);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

	/// Init values
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		graph.label1[i] = false;
		graph.label2[i] = false;
		graph.value[i] = DIST_INFINITY;
	}
	graph.numActiveNodes = 1;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;
	graph.value[arguments.sourceNode] = 0;

	graph.copy_to_device_values();

	Subgraph<OutEdgeWeighted_Evolving> subgraph(graph.num_nodes, graph.num_edges_total);
	SubgraphGenerator<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> subgen(graph);
	subgen.generate(graph, subgraph);

	Partitioner<OutEdgeWeighted_Evolving> partitioner;

	/// Init computation
	float total_comm = 0, total_comp, total_gen = 0;
	timer.Start();
	uint itr = 0;

	while (subgraph.numActiveNodes > 0)
	{
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		itr++;
		cout << "(Init) Running itr " << itr << " with #nodes " << subgraph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		float comm_time = 0, comp_time = 0;
		timer_sub.Start();
		for (int pi = 0; pi < partitioner.numPartitions; pi++)
		{
			// cout << "=== Partition "<< pi << "\t";
			timer_sub2.Start();
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			comm_time += timer_sub2.Finish();

			moveUpLabels<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);

			timer_sub2.Start();
			ssnp_kernel_cg<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512>>>(partitioner.partitionNodeSize[pi],
																				 partitioner.fromNode[pi],
																				 partitioner.fromEdge[pi],
																				 subgraph.d_activeNodes,
																				 subgraph.d_activeNodesPointer,
																				 subgraph.d_activeEdgeList,
																				 graph.d_outDegree,
																				 graph.d_label1,
																				 graph.d_label2,
																				 graph.d_value,
																				 graph.numSnapshots + 1);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			comp_time += timer_sub2.Finish();
		}
		timer_sub2.Start();
		subgen.generate(graph, subgraph);
		float gen_time = timer_sub2.Finish();

		sub_time = timer_sub.Finish();
		cout << sub_time << "  (comm: " << comm_time << "; comp: " << comp_time << "; subCSR: " << gen_time << ")" << endl;
		total_comm += comm_time;
		total_comp += comp_time;
		total_gen += gen_time;
	}

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";
	cout << "Number of iterations = " << itr << endl;
	cout << ">> (comm: " << total_comm << "; comp time: " << total_comp << "; subCSR: " << total_gen << ")" << endl;

	uint **d_values;
	uint **d_values_data = new uint *[graph.numSnapshots];
	cudaMalloc(&d_values, graph.numSnapshots * sizeof(uint *));
	for (uint i = 0; i < graph.numSnapshots; i++)
	{
		cudaMalloc(&d_values_data[i], graph.num_nodes * sizeof(uint));
		copyValue<<<graph.num_nodes / 512 + 1, 512>>>(d_values_data[i], graph.d_value, graph.num_nodes);
	}
	cudaMemcpy(d_values, d_values_data, graph.numSnapshots * sizeof(uint *), cudaMemcpyHostToDevice);

	timer.Start();
	sssp_setActive<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_value, graph.d_label2, graph.num_nodes);

	subgen.generate(graph, subgraph);
	while (subgraph.numActiveNodes > 0)
	{
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		itr++;

		timer_sub.Start();
		cout << "(Snapshot) Running itr " << itr << " with #nodes " << subgraph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);

		float comm_time = 0, comp_time = 0;
		for (int pi = 0; pi < partitioner.numPartitions; pi++)
		{
			timer_sub2.Start();
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			comm_time += timer_sub2.Finish();

			timer_sub2.Start();
			ssnp_kernel_concurrent_cg<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512>>>(partitioner.partitionNodeSize[pi],
																							partitioner.fromNode[pi],
																							partitioner.fromEdge[pi],
																							subgraph.d_activeNodes,
																							subgraph.d_activeNodesPointer,
																							subgraph.d_activeEdgeList,
																							graph.d_outDegree,
																							graph.d_label1,
																							graph.d_label2,
																							d_values,
																							graph.numSnapshots);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			comp_time += timer_sub2.Finish();
		}
		timer_sub2.Start();
		subgen.generate(graph, subgraph);
		// subgen.generate(graph, subgraph);
		float gen_time = timer_sub2.Finish();

		float sub_time = timer_sub.Finish();
		cout << sub_time << "  (comm: " << comm_time << "; comp: " << comp_time << "; subCSR: " << gen_time << ")" << endl;
	}

	runtime = timer.Finish();
	cout << ">>Processing finished in " << runtime << " (ms).\n";
}