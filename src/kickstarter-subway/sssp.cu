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

#include <cuda_profiler_api.h>

uint neighbor_cnt(Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> &graph, bool *label2)
{
	uint cnt = 0;
	for (uint i = 0; i < graph.num_nodes; i++)
	{
		if (label2)
		{
			cnt += graph.nodePointer[i + 1] - graph.nodePointer[i];
		}
	}
	return cnt;
}

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

	/// Init values
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.label1[i] = false;
		graph.label2[i] = false;
		graph.depends[i].reset();
		graph.depends[i].value = DIST_INFINITY;
	}
	graph.numActiveNodes = 1;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;

	graph.depends[arguments.sourceNode].value = 0;
	graph.depends[arguments.sourceNode].level = 0;
	graph.depends[arguments.sourceNode].parent = arguments.sourceNode;

	graph.copy_to_device_values();

	Subgraph<OutEdgeWeighted_Evolving> subgraph(graph.num_nodes, graph.num_edges_total);
	SubgraphGenerator<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> subgen(graph);
	subgen.generate(graph, subgraph);

	Partitioner<OutEdgeWeighted_Evolving> partitioner;

	/// Init computation
	float total_comm = 0, total_comp, total_gen = 0, total_cpu = 0;
	timer.Start();
	uint itr = 0;
	
	cudaProfilerStart();
	while (subgraph.numActiveNodes>0)
	{
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		itr++;
		cout << "(Init) Running itr " << itr << " with #nodes " << subgraph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		// moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);	
		// gpuErrorcheck(cudaPeekAtLastError());	
		float comm_time = 0, comp_time = 0;
		timer_sub.Start();	
		for(int pi = 0; pi < partitioner.numPartitions; pi++)
		{	
			// cout << "=== Partition "<< pi << "\t";
			timer_sub2.Start();
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			comm_time += timer_sub2.Finish();

			moveUpLabels<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);
			
			timer_sub2.Start();
			sssp_kernel_ks<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[pi],
														partitioner.fromNode[pi],
														partitioner.fromEdge[pi],
														subgraph.d_activeNodes,
														subgraph.d_activeNodesPointer,
														subgraph.d_activeEdgeList,
														graph.d_outDegree,
														graph.d_label1,
														graph.d_label2,
														graph.d_depends,
														0);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());	
			comp_time += timer_sub2.Finish();
			// sub_time = timer_sub2.Finish();
			// if(sub_time > 100)
				// cout << sub_time << "  " << partitioner.partitionEdgeSize[pi] << " " << partitioner.partitionNodeSize[pi] << "\t\t";
		}
		// graph.calCurNumActiveNodes();
		
		timer_sub2.Start();
		subgen.generate(graph, subgraph, total_cpu);
		float gen_time = timer_sub2.Finish();

		sub_time = timer_sub.Finish();
		cout << sub_time << "  (comm: " << comm_time << "; comp: " << comp_time  << "; subCSR: " << gen_time << ")" << endl;
		total_comm += comm_time;
		total_comp += comp_time;
		total_gen += gen_time;
	}	
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";
	cout << "Number of iterations = " << itr << endl;
	cout << ">> (comm: " << total_comm << "  comp time: " << total_comp << "  subCSR: " << total_gen << "  cpu: "<< total_cpu << " )" << endl;

	gpuErrorcheck(cudaMemcpy(graph.depends, graph.d_depends, graph.num_nodes * sizeof(DependencyData), cudaMemcpyDeviceToHost));
	utilities::PrintResults(graph.depends, min(30, graph.num_nodes));
	// utilities::PrintParents(graph.depends, min(30, graph.num_nodes));
	
	total_comm = 0;
	total_comp = 0;
	total_gen = 0;
	total_cpu = 0;

	double transfer_total = 0;

	/// Incremental Analysis
	for(uint i = 1; i <= graph.numSnapshots; i++)
	{
		graph.copy_to_device_delta(i - 1);
		transfer_total += (graph.num_edges_add + graph.num_edges_del) * sizeof(EdgeWeighted);
		cout << "\n===========snapshot " << i << "==============" << endl;
		timer.Start();
		clearLabel<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.num_nodes);	
		clearLabel<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_all_affected_vertices, graph.num_nodes);	
		save_old<<< graph.num_nodes/512 + 1, 512 >>>(graph.num_nodes, graph.d_depends, graph.d_depends_old);
		gpuErrorcheck(cudaPeekAtLastError());		
		//==== Phase 1: identify the direct affect of deletions ====
		timer_sub.Start();
		sssp_identifyDel<<< graph.num_nodes/512 + 1, 512 >>>(graph.num_edges_del, 
															graph.d_del_edgeList, 
															graph.d_label2, 
															graph.d_depends,
															graph.d_all_affected_vertices);	
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());		
		graph.calCurNumActiveNodes();
		sub_time = timer_sub.Finish();
		cout << ">Phase 1: identify the direct affect of deletions: " << sub_time << " (ms).\n";

		//==== Phase 2: Trimming ====
		itr = 0;
		timer_sub.Start();

		// timer_sub2.Start();
		subgen.generate(graph, subgraph);
		// gen_time = timer_sub2.Finish();
		// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
		// transfer_total+= neighbor_cnt(graph, graph.label2);

		while (subgraph.numActiveNodes > 0)
		{
			partitioner.partition(subgraph, subgraph.numActiveNodes);
			itr++;
			cout << "(Trimming) Running itr " << itr << " with #nodes " << subgraph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			float comm_time = 0, comp_time = 0, gen_time = 0;

			timer_sub3.Start();
			
			// moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);	
			// gpuErrorcheck(cudaPeekAtLastError());	
			for(int pi = 0; pi < partitioner.numPartitions; pi++)
			{
				// cout << "=== Partition "<< pi << "\t";
				timer_sub2.Start();
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
				cudaDeviceSynchronize();
				comm_time += timer_sub2.Finish();
				transfer_total += partitioner.partitionEdgeSize[pi] * sizeof(OutEdgeWeighted_Evolving);

				moveUpLabels<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);
				
				timer_sub2.Start();
				sssp_jumpstart<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[pi],
																partitioner.fromNode[pi],
																partitioner.fromEdge[pi],
																subgraph.d_activeNodes,
																subgraph.d_activeNodesPointer,
																subgraph.d_activeEdgeList,
																graph.d_outDegree, 
																graph.d_label1, 
																graph.d_label2, 
																graph.d_depends,
																graph.d_depends_old,
																i);
				gpuErrorcheck(cudaPeekAtLastError());	
				comp_time += timer_sub2.Finish();
			}
			timer_sub2.Start();
			subgen.generate(graph, subgraph, total_cpu);
			gen_time = timer_sub2.Finish();
			partitioner.partition(subgraph, subgraph.numActiveNodes);
			// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
			// transfer_total += neighbor_cnt(graph, graph.label2);

			// moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			// gpuErrorcheck(cudaPeekAtLastError());

			for(int pi = 0; pi < partitioner.numPartitions; pi++)
			{
				timer_sub2.Start();
				gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
				comm_time += timer_sub2.Finish();
				transfer_total += partitioner.partitionEdgeSize[pi] * sizeof(OutEdgeWeighted_Evolving);

				moveUpLabels<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);

				timer_sub2.Start();
				sssp_propogate<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[pi],
																partitioner.fromNode[pi],
																partitioner.fromEdge[pi],
																subgraph.d_activeNodes,
																subgraph.d_activeNodesPointer,
																subgraph.d_activeEdgeList,
																graph.d_outDegree, 
																graph.d_label1, 
																graph.d_label2, 
																graph.d_depends,
																graph.d_all_affected_vertices,
																i);
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaPeekAtLastError());	
				// sub_time = timer_sub2.Finish();
				comp_time += timer_sub2.Finish();
			}

			timer_sub2.Start();
			subgen.generate(graph, subgraph, total_cpu);
			gen_time += timer_sub2.Finish();
			// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
			// transfer_total += neighbor_cnt(graph, graph.label2);

			sub_time = timer_sub3.Finish();
			cout << sub_time << " (comm: " << comm_time << "; comp time: " << comp_time << "; subCSR: " << gen_time << ")" << endl;
			total_comm += comm_time;
			total_comp += comp_time;
			total_gen += gen_time;
		}

		timer_sub3.Start();
		moveUpLabelsNoClear<<< graph.num_nodes/512 + 1 , 512 >>>(graph.d_label2, graph.d_all_affected_vertices, graph.num_nodes);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());
		timer_sub2.Start();	
		subgen.generate(graph, subgraph, total_cpu);

		// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
		// transfer_total += neighbor_cnt(graph, graph.label2);

		float gen_time = timer_sub2.Finish();
		partitioner.partition(subgraph, subgraph.numActiveNodes);

		float comm_time = 0, comp_time = 0;
		for(int pi = 0; pi < partitioner.numPartitions; pi++)
		{
			timer_sub2.Start();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
			comm_time += timer_sub2.Finish();
			transfer_total+= partitioner.partitionEdgeSize[pi] * sizeof(OutEdgeWeighted_Evolving);

			moveUpLabels<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);

			timer_sub2.Start();
			sssp_pull_once<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[pi],
															partitioner.fromNode[pi],
															partitioner.fromEdge[pi],
															subgraph.d_activeNodes,
															subgraph.d_activeNodesPointer,
															subgraph.d_activeEdgeList,
															graph.d_outDegree, 
															graph.d_depends,
															graph.d_label1,
															i);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());	
			comp_time += timer_sub2.Finish();
		}
		sub_time = timer_sub3.Finish();
		cout << " pull once: " << sub_time <<  " (comm: " << comm_time << "; comp time: " << comp_time << "; subCSR: " << gen_time << ")" << endl;
		total_comm += comm_time;
		total_comp += comp_time;
		total_gen += gen_time;

		sub_time = timer_sub.Finish();
		cout << ">Phase 2: trimming: " << sub_time << " (ms).\n";

		//==== Phase 3: Handling Additions ====
		timer_sub.Start();
		sssp_incre<<< graph.num_nodes/512 + 1, 512 >>>(graph.num_edges_add, 
													graph.d_add_edgeList, 
													graph.d_depends,
													graph.d_all_affected_vertices);	
		gpuErrorcheck(cudaPeekAtLastError());	

		moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label2, graph.d_all_affected_vertices, graph.num_nodes);	
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());	
		graph.calCurNumActiveNodes();
		sub_time = timer_sub.Finish();
		cout << ">Phase 3: processing additions: " << sub_time << " (ms).\n";

		//==== Phase 4: Re-converge ====
		itr = 0;
		timer_sub.Start();
		timer_sub2.Start();
		subgen.generate(graph, subgraph, total_cpu);
		total_gen += timer_sub2.Finish();

		// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
		// transfer_total += neighbor_cnt(graph, graph.label2);

		while (subgraph.numActiveNodes > 0)
		{
			partitioner.partition(subgraph, subgraph.numActiveNodes);
			itr++;
			timer_sub2.Start();
			cout << "(Re-converge) Running itr " << itr << " with #nodes " << subgraph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			comm_time = 0, comp_time = 0;
			
			for(int pi=0; pi<partitioner.numPartitions; pi++)
			{
				timer_sub3.Start();
				gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
				comm_time += timer_sub3.Finish();
				transfer_total += partitioner.partitionEdgeSize[pi] * sizeof(OutEdgeWeighted_Evolving);

				moveUpLabels<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);

				timer_sub3.Start();
				sssp_kernel_ks<<< partitioner.partitionNodeSize[pi]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[pi],
															partitioner.fromNode[pi],
															partitioner.fromEdge[pi],
															subgraph.d_activeNodes,
															subgraph.d_activeNodesPointer,
															subgraph.d_activeEdgeList,
															graph.d_outDegree,
															graph.d_label1,
															graph.d_label2,
															graph.d_depends,
															i);
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaPeekAtLastError());	
				comp_time += timer_sub3.Finish();
			}
			
			timer_sub3.Start();
			subgen.generate(graph, subgraph, total_cpu);
			gen_time = timer_sub3.Finish();

			// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
			// transfer_total += neighbor_cnt(graph, graph.label2);

			sub_time = timer_sub2.Finish();
			cout << sub_time <<  " (comm: " << comm_time << "; comp time: " << comp_time  << "; subCSR: " << gen_time << ")" << endl;
			// graph.calCurNumActiveNodes();
			total_comm += comm_time;
			total_comp += comp_time;
			total_gen += gen_time;
		}
		sub_time = timer_sub.Finish();
		cout << ">Phase 4: re-converge: " << sub_time << " (ms).\n";

		runtime = timer.Finish();
		cout << ">>Processing finished in " << runtime << " (ms).\n";
		cout << ">> (comm: " << total_comm << "; comp time: " << total_comp << "; subCSR: " << total_gen << "  total cpu: " << total_cpu << " )" << endl;
		gpuErrorcheck(cudaMemcpy(graph.depends, graph.d_depends, graph.num_nodes * sizeof(DependencyData), cudaMemcpyDeviceToHost));
		utilities::PrintResults(graph.depends, min(30, graph.num_nodes));
		cout << "Total transfer: " << transfer_total << " bytes" << endl; 
	}
	cudaProfilerStop();
}