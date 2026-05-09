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
	
	Timer timer, timer_sub, timer_sub2;
	float sub_time;
	timer.Start();
	
	Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> graph(arguments.input, true, arguments.numSnapshots, arguments.init_percentage, arguments.delta_rate_add, arguments.delta_rate_del);
	int seed = 101;
	graph.ReadGraph(false, seed);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

	/// Init values
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.label1[i] = false;
		graph.label2[i] = false;
		graph.depends[i].reset();
		graph.depends[i].value = 0;
	}
	graph.numActiveNodes = 1;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;
	
	graph.depends[arguments.sourceNode].value = DIST_INFINITY;
	graph.depends[arguments.sourceNode].parent = arguments.sourceNode;
	graph.depends[arguments.sourceNode].level = 0;

	graph.copy_to_device_values();
	
	/// Init computation
	timer.Start();
	uint itr = 0;
		
	while (graph.numActiveNodes>0)
	{
		itr++;
		cout << "(Init) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);	
		gpuErrorcheck(cudaPeekAtLastError());	
		timer_sub.Start();
		sswp_kernel_ks<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes,
														   graph.d_nodePointer,
														   graph.d_edgeList,
														   graph.d_outDegree,
														   graph.d_label1,
														   graph.d_label2,
														   graph.d_depends,
														   0);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());	
		graph.calCurNumActiveNodes();
		sub_time = timer_sub.Finish();
		cout << sub_time << endl;
	}	
	
	float runtime = timer.Finish();
	cout << "(SSWP) Processing finished in " << runtime << " (ms).\n";
	cout << "Number of iterations = " << itr << endl;

	gpuErrorcheck(cudaMemcpy(graph.depends, graph.d_depends, graph.num_nodes * sizeof(DependencyData), cudaMemcpyDeviceToHost));
	utilities::PrintResults(graph.depends, min(30, graph.num_nodes));
	// utilities::PrintParents(graph.depends, min(30, graph.num_nodes));

	/// Incremental Analysis
	for (uint i = 1; i <= graph.numSnapshots; i++)
	{
		cout << "\n===========snapshot " << i << "==============" << endl;
		graph.copy_to_device_delta(i - 1);
		timer.Start();
		clearLabel<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.num_nodes);
		clearLabel<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_all_affected_vertices, graph.num_nodes);
		save_old<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes, graph.d_depends, graph.d_depends_old);
		gpuErrorcheck(cudaPeekAtLastError());
		//==== Phase 1: identify the direct affect of deletions ====
		timer_sub.Start();
		sswp_identifyDel<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_edges_del,
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
		while (graph.numActiveNodes > 0)
		{
			itr++;
			cout << "(Trimming) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			// gpuErrorcheck(cudaMemcpy(graph.label2, graph.d_label2, graph.num_nodes * sizeof(bool), cudaMemcpyDeviceToHost));
			// graph.countTotalDegree(graph.label2);

			timer_sub2.Start();
			moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			gpuErrorcheck(cudaPeekAtLastError());
			// timer_sub3.Start();
			sswp_jumpstart<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes,
															   graph.d_nodePointer,
															   graph.d_edgeList,
															   graph.d_outDegree,
															   graph.d_label1,
															   graph.d_label2,
															   graph.d_depends,
															   graph.d_depends_old,
															   i);
			gpuErrorcheck(cudaPeekAtLastError());
			moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);

			gpuErrorcheck(cudaPeekAtLastError());
			// timer_sub3.Start();
			sswp_propogate<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes,
															   graph.d_nodePointer,
															   graph.d_edgeList,
															   graph.d_outDegree,
															   graph.d_label1,
															   graph.d_label2,
															   graph.d_depends,
															   graph.d_all_affected_vertices,
															   i);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			graph.calCurNumActiveNodes();
			sub_time = timer_sub2.Finish();
			cout << sub_time << endl;
			if (graph.numActiveNodes == 1)
				break;
		}

		timer_sub2.Start();
		sswp_pull_once<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes,
														   graph.d_nodePointer,
														   graph.d_edgeList,
														   graph.d_outDegree,
														   graph.d_depends,
														   graph.d_all_affected_vertices,
														   i);
		cudaDeviceSynchronize();
		sub_time = timer_sub2.Finish();
		cout << " pull once: " << sub_time << endl;
		gpuErrorcheck(cudaPeekAtLastError());
		sub_time = timer_sub.Finish();
		cout << ">Phase 2: trimming: " << sub_time << " (ms).\n";

		//==== Phase 3: Handling Additions ====
		timer_sub.Start();
		moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label2, graph.d_all_affected_vertices, graph.num_nodes);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());

		graph.calCurNumActiveNodes();
		cout << "#all affected nodes: " << graph.numActiveNodes << endl;

		sswp_incre<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_edges_add,
													   graph.d_add_edgeList,
													   graph.d_depends,
													   graph.d_label2);
		gpuErrorcheck(cudaPeekAtLastError());
		graph.calCurNumActiveNodes();
		sub_time = timer_sub.Finish();
		cout << ">Phase 3: processing additions: " << sub_time << " (ms).\n";

		//==== Phase 4: Re-converge ====
		itr = 0;
		timer_sub.Start();
		while (graph.numActiveNodes > 0)
		{
			itr++;
			timer_sub2.Start();
			cout << "(Re-converge) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			sswp_kernel_ks<<<graph.num_nodes / 512 + 1, 512>>>(graph.num_nodes,
															   graph.d_nodePointer,
															   graph.d_edgeList,
															   graph.d_outDegree,
															   graph.d_label1,
															   graph.d_label2,
															   graph.d_depends,
															   i);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			graph.calCurNumActiveNodes();
			sub_time = timer_sub2.Finish();
			cout << sub_time << endl;
		}
		sub_time = timer_sub.Finish();
		cout << ">Phase 4: re-converge: " << sub_time << " (ms).\n";

		runtime = timer.Finish();
		cout << ">>Processing finished in " << runtime << " (ms).\n";
		gpuErrorcheck(cudaMemcpy(graph.depends, graph.d_depends, graph.num_nodes * sizeof(DependencyData), cudaMemcpyDeviceToHost));
		utilities::PrintResults(graph.depends, min(30, graph.num_nodes));
	}
}