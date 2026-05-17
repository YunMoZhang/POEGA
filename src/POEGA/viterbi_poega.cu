// #include <stack>
#include "commons/globals.hpp"
#include "commons/timer.hpp"
#include "commons/argument_parser.hpp"
#include "commons/gpu_error_check.hpp"
#include "commons/output_utilities.hpp"
#include <string>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "evolving_graph.cuh"
#include "labels_kernels.cuh"
#include "graph_kernels.cuh"
#include "partitioner.cuh"
#include "subgraph.cuh"
#include "subgraph_generator.cuh"


void two_stage_analysis(Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> &graph, ArgumentParser& arguments, int seed){
	Timer timer, sub_timer, timer_sub2;
	float sub_runtime, analysis_time;
	graph.value = new uint[graph.num_nodes];
	cudaFree(graph.d_depends);
	cudaFree(graph.d_depends_old);

	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	if (!prop.canMapHostMemory)
    	exit(0);

	graph.CreateProxyGraphUnionCSR(arguments.degree_limit);
	// graph.CreateRandomCoreUnionCSR(seed);

	cout << "\n============Beginning two-stage analysis============" << endl;
	cout << "Analyzing the csg" << endl;
	for(uint i = 0; i < graph.num_nodes; i++){
		graph.label1[i] = false;
		graph.label2[i] = false;
		graph.value[i] = 0;
	}
	graph.numActiveNodes = 1;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;
	graph.value[arguments.sourceNode] = DIST_INFINITY;
	
	graph.LoadCoreUnionCSR();
	graph.TransferValuesToDevice();
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(uint), cudaMemcpyHostToDevice));


	uint itr = 0;	
	sub_timer.Start();
	uint *d_prefixLabeling;
	cudaMalloc(&graph.d_activeNodes, graph.num_nodes * sizeof(uint));
	cudaMalloc(&d_prefixLabeling, (graph.num_nodes + 1) * sizeof(uint));
	cudaMemset(d_prefixLabeling, 0, sizeof(unsigned int)); // set 1st as 0
	thrust::device_ptr<bool> ptr_label(graph.d_label2);
	thrust::device_ptr<unsigned int> ptr_degree(graph.d_core_outDegree);
	thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(d_prefixLabeling);
	thrust::device_ptr<uint> ptr_degrees(graph.d_core_outDegree);
	// uint * activeNodes = new uint[graph.num_nodes];

	thrust::exclusive_scan(ptr_label, ptr_label + graph.num_nodes, ptr_labeling_prefixsum, (unsigned int)0);
	makeQueueFromBitmap<<<graph.num_nodes/512+1, 512>>>(graph.d_activeNodes, graph.d_label2, d_prefixLabeling, graph.num_nodes);

	graph.calCurNumActiveNodes();
	while (graph.numActiveNodes > 0)
	{
		itr++;
		timer_sub2.Start();
		cout << "(P1-1) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
		moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);
		gpuErrorcheck(cudaPeekAtLastError());
		viterbi_kernel_common_warp<<< (graph.numActiveNodes * WARP_SIZE) / 512 + 1, 512>>>(graph.numActiveNodes,
																  graph.d_nodePointer_core,
																  graph.d_edgeList_core,
																  graph.d_value,
																  graph.d_activeNodes,
																  graph.d_label2,
		  														  graph.numSnapshots + 1);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());
		graph.calCurNumActiveNodes();

		if(graph.numActiveNodes > 0){
			thrust::exclusive_scan(ptr_label, ptr_label + graph.num_nodes, ptr_labeling_prefixsum, (unsigned int)0);
			makeQueueFromBitmap<<<graph.num_nodes/512+1, 512>>>(graph.d_activeNodes, graph.d_label2, d_prefixLabeling, graph.num_nodes);
		}
		sub_runtime = timer_sub2.Finish();
		cout << sub_runtime << endl;
	}
	float sub_time = sub_timer.Finish();
	cout << ">>Processing Common graph finished in " << sub_time << " (ms).\n";

	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));

	cout << "\n==Concurrent Analysis from CG to Proxy graphs==" << endl;

	size_t free_mem = 0, total_mem = 0;
	gpuErrorcheck(cudaMemGetInfo(&free_mem, &total_mem));
	size_t value_bytes = (size_t)graph.numSnapshots * graph.num_nodes * sizeof(uint);
	if (value_bytes < free_mem * 0.8) 
	{
		uint *d_values;
		uint *value_data = new uint[graph.numSnapshots * graph.num_nodes];
		// gpuErrorcheck(cudaHostAlloc(&value_data, (graph.numSnapshots * graph.num_nodes) * sizeof(uint), cudaHostAllocMapped));
		
		
	#pragma omp parallel for
		for (uint i = 0; i < graph.num_nodes; i++)
		{
			for (uint j = 0; j < graph.numSnapshots; j++)
			{
				value_data[i * graph.numSnapshots + j] = graph.value[i];
			}
		}

		cudaMalloc(&d_values, graph.num_nodes * graph.numSnapshots * sizeof(uint));
		gpuErrorcheck(cudaMemcpy(d_values, value_data, graph.num_nodes * graph.numSnapshots * sizeof(uint), cudaMemcpyHostToDevice));
		// cudaHostGetDevicePointer(&d_values, value_data, 0);

		timer.Start();
		clearLabel<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.num_nodes);
		gpuErrorcheck(cudaPeekAtLastError());
		sub_timer.Start();
		viterbi_incremental_cg_concurrent2<<<(graph.num_nodes * graph.numSnapshots) / 512 + 1, 512>>>(graph.num_nodes,
																								graph.d_nodePointer_core,
																								graph.d_edgeList_core,
																								d_values,
																								graph.d_label2,
																								graph.numSnapshots);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());
		graph.calCurNumActiveNodes();
		float tracing_time = sub_timer.Finish();
		cout << ">P1-2: Tracing: " << tracing_time << " (ms).\n";


		thrust::exclusive_scan(ptr_label, ptr_label + graph.num_nodes, ptr_labeling_prefixsum, (unsigned int)0);
		makeQueueFromBitmap<<<graph.num_nodes/512+1, 512>>>(graph.d_activeNodes, graph.d_label2, d_prefixLabeling, graph.num_nodes);

		itr = 0;
		sub_timer.Start();
		// graph.calCurNumActiveNodes();
		while (graph.numActiveNodes > 0)
		{
			itr++;
			cout << "(P1-2: Re-converge) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			timer_sub2.Start();
			moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			viterbi_kernel_concurrent_optimized_transpose<<<(graph.numActiveNodes * 32) / 512 + 1, 512>>>(graph.numActiveNodes,
																									graph.d_nodePointer_core,
																									graph.d_edgeList_core,
																									graph.d_activeNodes,
																									graph.d_label2,
																									d_values,
																									graph.numSnapshots);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			graph.calCurNumActiveNodes();
			if(graph.numActiveNodes > 0){
				thrust::exclusive_scan(ptr_label, ptr_label + graph.num_nodes, ptr_labeling_prefixsum, (unsigned int)0);
				makeQueueFromBitmap<<<graph.num_nodes/512+1, 512>>>(graph.d_activeNodes, graph.d_label2, d_prefixLabeling, graph.num_nodes);
			}
			sub_time = timer_sub2.Finish();
			cout << sub_time << endl;	
		}
		sub_time = sub_timer.Finish();
		cout << ">P1-2: Re-converge: " << sub_time << " (ms).\n";

		float p1_runtime = timer.Finish() + tracing_time;
		cout << ">>> P1 total: " << p1_runtime << " (ms).\n";

		gpuErrorcheck(cudaMemcpy(value_data, d_values, graph.num_nodes * graph.numSnapshots * sizeof(uint), cudaMemcpyDeviceToHost));

		cudaFree(graph.d_nodePointer_core);
		cudaFree(graph.d_edgeList_core);
		cudaFree(graph.d_value);
		
		//================================================================
		cout << "\n==Concurrent P2=="<< endl;
		//================================================================

		float total_time = 0;

		Partitioner<OutEdgeWeighted_Evolving> partitioner(graph.num_nodes, graph.num_edges_total);
		partitioner.partition(graph);

		size_t free, total;
		cudaMemGetInfo(&free, &total);
		cout << "free:  " << free << "\ntotal: " << total << endl;

		uint *d_max, *d_min;
		gpuErrorcheck(cudaMalloc(&d_max, graph.num_nodes * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_min, graph.num_nodes * sizeof(uint)));
		set_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, d_values, graph.num_nodes, graph.numSnapshots);

		uint *d_nodePointer_el0, *d_nodePointer_el1;
		OutEdgeWeighted_Evolving *d_edgeList0, *d_edgeList1;
		// uint *d_nodePointer_el;
		// OutEdgeWeighted_Evolving *d_edgeList;
		cout << "allocating nodePointer edgelist: " << (partitioner.max_partition_size) * sizeof(uint) << "  edge array: " <<  (partitioner.max_partition_size) * sizeof(OutEdgeWeighted_Evolving) << endl;
		gpuErrorcheck(cudaMalloc(&d_nodePointer_el0, (partitioner.max_partition_size) * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_edgeList0, (partitioner.max_partition_size) * sizeof(OutEdgeWeighted_Evolving)));
		gpuErrorcheck(cudaMalloc(&d_nodePointer_el1, (partitioner.max_partition_size ) * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_edgeList1, (partitioner.max_partition_size) * sizeof(OutEdgeWeighted_Evolving)));

		// update_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, d_values, graph.d_label2, graph.num_nodes, graph.numSnapshots);

		timer.Start();
		cout << "#partition: " << partitioner.numPartitions << endl;
		cudaStream_t stream0, stream1;
		cudaStreamCreate(&stream0);
		cudaStreamCreate(&stream1);
		for (int pi = 0; pi < partitioner.numPartitions; pi++)
		{
			uint *d_nodePointer_el = (pi % 2 == 0) ? d_nodePointer_el0 : d_nodePointer_el1;
			// uint *d_nodePointer_el =  d_nodePointer_el0;
			OutEdgeWeighted_Evolving *d_edgeList = (pi % 2 == 0) ? d_edgeList0 : d_edgeList1;
			// OutEdgeWeighted_Evolving *d_edgeList = d_edgeList0;
			cudaStream_t stream = (pi % 2 == 0) ? stream0 : stream1;

			// timer_sub2.Start();
			gpuErrorcheck(cudaMemcpyAsync(d_edgeList, graph.edgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice, stream));
			// transfer_total += partitioner.partitionEdgeSize[pi] * sizeof(OutEdgeWeighted_Evolving);
			extend_varray_to_edgelist<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512, 0, stream>>>(partitioner.partitionNodeSize[pi],
																									partitioner.fromNode[pi],
																									partitioner.fromEdge[pi],
																									graph.d_nodePointer,
																									d_nodePointer_el,
																									graph.d_outDegree);
			viterbi_incre_concurrent_locality<<<partitioner.partitionEdgeSize[pi] / 512 + 1, 512, 0, stream>>>(partitioner.partitionEdgeSize[pi],
																											d_nodePointer_el,
																											d_edgeList,
																											d_values,
																											graph.d_label2,
																											graph.numSnapshots);
			viterbi_incre_concurrent2_locality<<<partitioner.partitionEdgeSize[pi] / 512 + 1, 512, 0, stream>>>(partitioner.partitionEdgeSize[pi],
																											d_nodePointer_el,
																											d_edgeList,
																											d_values,
																											graph.d_label2,
																											graph.numSnapshots);
			if (pi > 0)
			{
				cudaStream_t prev_stream = (pi % 2 == 0) ? stream1 : stream0;
				cudaStreamSynchronize(prev_stream);
			}
			// cudaDeviceSynchronize();
		}
		cudaStreamSynchronize(stream0);
		cudaStreamSynchronize(stream1);
		graph.calCurNumActiveNodes();
		total_time = timer.Finish();
		cout << "P2 Tracing: " << total_time << endl;
		cudaFree(d_nodePointer_el0);
		cudaFree(d_edgeList0);
		cudaFree(d_nodePointer_el1);
		cudaFree(d_edgeList1);
		cudaStreamDestroy(stream0);
		cudaStreamDestroy(stream1);

		update_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, d_values, graph.d_label2, graph.num_nodes, graph.numSnapshots);

		thrust::exclusive_scan(ptr_label, ptr_label + graph.num_nodes, ptr_labeling_prefixsum, (unsigned int)0);
		makeQueueFromBitmap<<<graph.num_nodes/512+1, 512>>>(graph.d_activeNodes, graph.d_label2, d_prefixLabeling, graph.num_nodes);

		//========== Re-converge: rest iterations==========
		itr = 0;
		timer.Start();
		while (graph.numActiveNodes > 0){
			itr++;
			sub_timer.Start();
			cout << "(P2) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			viterbi_kernel_concurrent_optimized_transpose<<<(graph.numActiveNodes * 32) / 512 + 1, 512>>>(graph.numActiveNodes,
																								graph.d_nodePointer,
																								graph.d_edgeList,
																								graph.d_activeNodes,
																								graph.d_label2,
																								d_values,
																								d_max,
																								d_min,
																								graph.numSnapshots);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			graph.calCurNumActiveNodes();

			if(graph.numActiveNodes > 0){
				thrust::exclusive_scan(ptr_label, ptr_label + graph.num_nodes, ptr_labeling_prefixsum, (unsigned int)0);
				makeQueueFromBitmap<<<graph.num_nodes/512+1, 512>>>(graph.d_activeNodes, graph.d_label2, d_prefixLabeling, graph.num_nodes);
			}

			update_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, d_values, graph.d_label2, graph.num_nodes, graph.numSnapshots);
			
			float sub_time = sub_timer.Finish();
			cout << sub_time << endl;
		}
		analysis_time = timer.Finish();
		cout << ">>P2 re-converge time: " << analysis_time << " (ms)" << endl;
		cout << ">>> P2 total: " << total_time + analysis_time << endl;
		cout << "============ Total Analysis Time: " << p1_runtime + total_time + analysis_time << " (ms) ============" << endl;
		cudaFree(d_max);
		cudaFree(d_min);
		cudaFree(d_values);
	}
	else
	{
		cudaFree(d_prefixLabeling);
		cudaFree(graph.d_activeNodes);
		gpuErrorcheck(cudaPeekAtLastError());

		uint *d_buffer;
		uint h_total = (graph.num_nodes * 0.5);
		cudaMalloc(&d_buffer, (size_t)h_total * graph.numSnapshots  * sizeof(uint));
		uint *d_counter, *d_total, h_counter;
		cudaMalloc(&d_counter, sizeof(uint));
		cudaMemset(d_counter, 0, sizeof(uint));
		cudaMalloc(&d_total, sizeof(uint));
		cudaMemset(d_total, h_total, sizeof(uint));
		cout << "-- buffer init size: "<< h_total  << endl;

		timer.Start();
		clearLabel<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.num_nodes);
		gpuErrorcheck(cudaPeekAtLastError());
		sub_timer.Start();
		viterbi_incremental_cg_concurrent2_newstorage<<<(graph.num_nodes * graph.numSnapshots) / 512 + 1, 512>>>(graph.num_nodes,
																								graph.d_nodePointer_core,
																								graph.d_edgeList_core,
																								graph.d_value,
																								d_buffer,
																								d_counter,
																								d_total,
																								graph.d_label2,
																								graph.numSnapshots);
		cudaDeviceSynchronize();
		gpuErrorcheck(cudaPeekAtLastError());
		graph.calCurNumActiveNodes();
		float tracing_time = sub_timer.Finish();
		cout << ">P2: Tracing: " << tracing_time << " (ms).\n";

		gpuErrorcheck(cudaMemcpy(&h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "--buffer counter: " << h_counter << endl;
		
		itr = 0;
		sub_timer.Start();
		// graph.calCurNumActiveNodes();
		while (graph.numActiveNodes > 0)
		{
			itr++;
			cout << "(P2-Re-converge) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			timer_sub2.Start();
			moveUpLabels<<<graph.num_nodes / 512 + 1, 512>>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			viterbi_kernel_concurrent_newstorage<<<(graph.num_nodes * graph.numSnapshots) / 512 + 1, 512>>>(graph.num_nodes,
																								graph.d_nodePointer_core,
																								graph.d_edgeList_core,
																								graph.d_label1,
																								graph.d_label2,
																								graph.d_value,
																								d_buffer,
																								d_counter,
																								d_total,
																								graph.numSnapshots);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			graph.calCurNumActiveNodes();
			sub_time = timer_sub2.Finish();
			cout << sub_time << endl;	
		}
		sub_time = sub_timer.Finish();
		cout << ">Phase 2: re-converge: " << sub_time << " (ms).\n";

		float p1_runtime = timer.Finish() + tracing_time;
		cout << ">>> P1 total: " << p1_runtime << " (ms).\n";

		gpuErrorcheck(cudaMemcpy(&h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "--buffer counter: " << h_counter << endl;

		// gpuErrorcheck(cudaMemcpy(value_data, d_values, graph.num_nodes * graph.numSnapshots * sizeof(uint), cudaMemcpyDeviceToHost));

		cudaFree(graph.d_nodePointer_core);
		cudaFree(graph.d_edgeList_core);

		//================================================================
		cout << "\n==Concurrent P2=="<< endl;
		//================================================================
		
		float total_time = 0;

		Partitioner<OutEdgeWeighted_Evolving> partitioner(graph.num_nodes, graph.num_edges_total);
		partitioner.partition(graph);

		size_t free, total;
		cudaMemGetInfo(&free, &total);
		cout << "free:  " << free << "\ntotal: " << total << endl;

		uint *d_nodePointer_el0, *d_nodePointer_el1;
		OutEdgeWeighted_Evolving *d_edgeList0, *d_edgeList1;
		cout << "allocating nodePointer edgelist: " << (partitioner.max_partition_size) * sizeof(uint) << "  edge array: " <<  (partitioner.max_partition_size) * sizeof(OutEdgeWeighted_Evolving) << endl;
		gpuErrorcheck(cudaMalloc(&d_nodePointer_el0, (partitioner.max_partition_size) * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_edgeList0, (partitioner.max_partition_size) * sizeof(OutEdgeWeighted_Evolving)));
		gpuErrorcheck(cudaMalloc(&d_nodePointer_el1, (partitioner.max_partition_size ) * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_edgeList1, (partitioner.max_partition_size) * sizeof(OutEdgeWeighted_Evolving)));

		uint *d_max, *d_min;
		gpuErrorcheck(cudaMalloc(&d_max, graph.num_nodes * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_min, graph.num_nodes * sizeof(uint)));
		set_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, graph.d_value, d_buffer, d_total, graph.num_nodes, graph.numSnapshots);
		gpuErrorcheck(cudaPeekAtLastError());

		timer.Start();
		cout << "#partition: " << partitioner.numPartitions << endl;
		cudaStream_t stream0, stream1;
		cudaStreamCreate(&stream0);
		cudaStreamCreate(&stream1);
		for (int pi = 0; pi < partitioner.numPartitions; pi++)
		{
			uint *d_nodePointer_el = (pi % 2 == 0) ? d_nodePointer_el0 : d_nodePointer_el1;
			OutEdgeWeighted_Evolving *d_edgeList = (pi % 2 == 0) ? d_edgeList0 : d_edgeList1;
			cudaStream_t stream = (pi % 2 == 0) ? stream0 : stream1;

			gpuErrorcheck(cudaMemcpyAsync(d_edgeList, graph.edgeList + partitioner.fromEdge[pi], (size_t)(partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice, stream));
			extend_varray_to_edgelist<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512, 0, stream>>>(partitioner.partitionNodeSize[pi],
																									partitioner.fromNode[pi],
																									partitioner.fromEdge[pi],
																									graph.d_nodePointer,
																									d_nodePointer_el,
																									graph.d_outDegree);
			viterbi_incre_concurrent_locality_bound_newstorage<<<partitioner.partitionEdgeSize[pi] / 512 + 1, 512, 0, stream>>>(partitioner.partitionEdgeSize[pi],
																											d_nodePointer_el,
																											d_edgeList,
																											graph.d_value,
																											d_buffer,
																											graph.d_label2,
																											d_max,
																											d_min,
																											d_counter,
																											d_total,
																											graph.numSnapshots);
			viterbi_incre_concurrent2_locality_newstorage<<<partitioner.partitionEdgeSize[pi] / 512 + 1, 512, 0, stream>>>(partitioner.partitionEdgeSize[pi],
																											d_nodePointer_el,
																											d_edgeList,
																											graph.d_value,
																											d_buffer,
																											d_counter,
																											d_total,
																											graph.d_label2,
																											graph.numSnapshots);
			if (pi > 0)
			{
				cudaStream_t prev_stream = (pi % 2 == 0) ? stream1 : stream0;
				cudaStreamSynchronize(prev_stream);
			}
			// cudaDeviceSynchronize();
		}
		cudaStreamSynchronize(stream0);
		cudaStreamSynchronize(stream1);
		graph.calCurNumActiveNodes();
		total_time = timer.Finish();
		cout << "P2 Tracing: " << total_time << endl;
		cudaFree(d_nodePointer_el0);
		cudaFree(d_edgeList0);
		cudaFree(d_nodePointer_el1);
		cudaFree(d_edgeList1);
		cudaStreamDestroy(stream0);
		cudaStreamDestroy(stream1);

		// update_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, d_values, graph.d_label2, graph.num_nodes, graph.numSnapshots);

		gpuErrorcheck(cudaMemcpy(&h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "--buffer counter: " << h_counter << endl;

		//========== Re-converge: rest iterations==========
		itr = 0;
		timer.Start();
		while (graph.numActiveNodes > 0){
			itr++;
			sub_timer.Start();
			cout << "(P2) Running itr " << itr << " with #nodes " << graph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			moveUpLabels<<< graph.num_nodes/512 + 1, 512 >>>(graph.d_label1, graph.d_label2, graph.num_nodes);
			viterbi_kernel_concurrent_newstorage<<<(graph.num_nodes * graph.numSnapshots) / 512 + 1, 512>>>(graph.num_nodes,
																								graph.d_nodePointer,
																								graph.d_edgeList,
																								graph.d_label1,
																								graph.d_label2,
																								graph.d_value,
																								d_buffer,
																								d_counter,
																								d_total,
																								graph.numSnapshots);
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());
			graph.calCurNumActiveNodes();
			// update_max_min<<<graph.num_nodes / 512 + 1, 512>>>(d_max, d_min, d_values, graph.d_label2, graph.num_nodes, graph.numSnapshots);
			
			float sub_time = sub_timer.Finish();
			cout << sub_time << endl;
		}
		analysis_time = timer.Finish();
		cout << ">>P2 re-converge time: " << analysis_time << " (ms)" << endl;
		cout << ">>> P2 total: " << total_time + analysis_time << endl;
		cout << "============ Total Analysis Time: " << p1_runtime + total_time + analysis_time << " (ms) ============" << endl;
		gpuErrorcheck(cudaMemcpy(&h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost));
		cout << "--buffer counter: " << h_counter << endl;
		cudaFree(d_max);
		cudaFree(d_min);
		cudaFree(d_buffer);
	}
}

void preAnalysisFull(Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> &graph){
	Timer timer, timer_sub, timer_sub2;
	float analysis_time;

	graph.depends_proxy = new DependencyData *[graph.top_k_degree];
	Partitioner<OutEdgeWeighted_Evolving> partitioner(graph.num_nodes, graph.num_edges_total);

	Subgraph<OutEdgeWeighted_Evolving> subgraph(graph.num_nodes, graph.num_edges_total, partitioner.max_partition_size);
	SubgraphGenerator<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> subgen(graph);
	// getting the dependency tree of a full snapshot
	for(uint top_i = 0; top_i < graph.top_k_degree; top_i++){
		cout << "Pre-analyzing " << top_i;
		for (uint i = 0; i < graph.num_nodes; i++)
		{
			graph.label1[i] = false;
			graph.label2[i] = false;
			graph.depends[i].reset();
			graph.depends[i].value = 0;
		}

		graph.numActiveNodes = 1;
		graph.label1[graph.top_k_degree_nodes[top_i]] = false;
		graph.label2[graph.top_k_degree_nodes[top_i]] = true;
		graph.depends[graph.top_k_degree_nodes[top_i]].level = 0;
		graph.depends[graph.top_k_degree_nodes[top_i]].parent = graph.top_k_degree_nodes[top_i];
		graph.depends[graph.top_k_degree_nodes[top_i]].value = DIST_INFINITY;

		graph.TransferValuesToDevice();
		gpuErrorcheck(cudaMemcpy(graph.d_depends, graph.depends, graph.num_nodes * sizeof(DependencyData), cudaMemcpyHostToDevice));

		subgen.generate(graph, subgraph);
		
		timer.Start();
		uint itr = 0;
		while (subgraph.numActiveNodes > 0)
		{
			partitioner.partition(subgraph, subgraph.numActiveNodes);
			itr++;
			// cout << "(G0) Running itr " << itr << " with #nodes " << subgraph.numActiveNodes << " out of " << graph.num_nodes << "\t";
			float comm_time = 0, comp_time = 0;
			timer_sub.Start();
			for (int pi = 0; pi < partitioner.numPartitions; pi++)
			{
				timer_sub2.Start();
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[pi], (partitioner.partitionEdgeSize[pi]) * sizeof(OutEdgeWeighted_Evolving), cudaMemcpyHostToDevice));
				cudaDeviceSynchronize();
				comm_time += timer_sub2.Finish();

				moveUpLabels<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[pi], partitioner.fromNode[pi]);

				timer_sub2.Start();
				viterbi_kernel<<<partitioner.partitionNodeSize[pi] / 512 + 1, 512>>>(partitioner.partitionNodeSize[pi],
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
			}
			timer_sub2.Start();
			subgen.generate(graph, subgraph);
			float gen_time = timer_sub2.Finish();
			float sub_time = timer_sub.Finish();
		}
		analysis_time = timer.Finish();
		cout << ", pre-analysis time: " << analysis_time << endl;
		graph.depends_proxy[top_i] = new DependencyData[graph.num_nodes];
		gpuErrorcheck(cudaMemcpy(graph.depends_proxy[top_i], graph.d_depends, graph.num_nodes * sizeof(DependencyData), cudaMemcpyDeviceToHost));
	}
}

int main(int argc, char** argv)
{
	cudaFree(0);
	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer;
	timer.Start();
	
	Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted> graph(arguments.input, true, arguments.numSnapshots, arguments.init_percentage, arguments.delta_rate_add, arguments.delta_rate_del);
	int seed = 999;
	graph.ReadGraph(false, seed);

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime / 1000 << " (s).\n";

	preAnalysisFull(graph);
	two_stage_analysis(graph, arguments, seed);	
}