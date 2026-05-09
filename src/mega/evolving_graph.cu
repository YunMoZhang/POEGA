#include <algorithm>
#include <random>
#include <math.h>

#include "evolving_graph.cuh"
#include "gpu_error_check.hpp"
#include "commons/pvector.hpp"
#include "commons/bitmap.hpp"
#include "commons/sampling.hpp"
#include "commons/timer.hpp"
#include <cuda_profiler_api.h>


#define WEIGHT(x, y) ((x + y) % 50 + 1)
// #define WEIGHT(x,y) (((int)log2(x+y)) % x + 1)
#define QueueLength 30000

template <class E, class EL, class DL>
Evolving_Graph<E, EL, DL>::Evolving_Graph(string graphFilePath, bool isWeighted, int num_snap, double init_p, double delta_r_a, double delta_r_d)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
	this->numSnapshots = num_snap;
	this->init_percentage = init_p;
	this->delta_rate_add = delta_r_a;
	this->delta_rate_del = delta_r_d;
	cout << "Eolving graph settings: init_precent=" << init_p << " add_delta_rate=" << delta_r_a << " del_delta_rate=" << delta_r_d << " #snapshot=" << num_snap <<endl;
}

template <class E, class EL, class DL>
string Evolving_Graph<E, EL, DL>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::calCurNumActiveNodes()
{
	thrust::device_ptr<bool> ptr_labeling(d_label2);	
	numActiveNodes = thrust::count(ptr_labeling, ptr_labeling + num_nodes, true);// thrust::reduce(ptr_labeling, ptr_labeling + graph.num_nodes);
	// cout << "Number of Active Nodes = " << numActiveNodes << endl;
}

template <>
void Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>::AssignW8(OutEdgeWeighted_Evolving &elem1, uint w8)
{
    elem1.w8 = w8;
}

template <>
void Evolving_Graph<OutEdge_Evolving, Edge_Union, Edge>::AssignW8(OutEdge_Evolving &elem, uint w8)
{
    ; //edgeList[index].end = edgeList[index].end; // do nothing
}

template <>
void Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>::AssignW8(EdgeWeighted &elem1, uint w8)
{
    elem1.w8 = w8;
}

template <>
void Evolving_Graph<OutEdge_Evolving, Edge_Union, Edge>::AssignW8(Edge &elem, uint w8)
{
    ; //edgeList[index].end = edgeList[index].end; // do nothing
}

template <>
void Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>::CopyW8(EdgeWeighted &elem1, OutEdgeWeighted_Evolving & elem2)
{
    elem1.w8 = elem2.w8;
}

template <>
void Evolving_Graph<OutEdge_Evolving, Edge_Evolving, Edge>::CopyW8(Edge &elem, OutEdge_Evolving & elem2)
{
    ; //edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::SetAsWeight(bool isWeighted_)
{
	this->isWeighted = isWeighted_;
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::Set_Bit_Snap(uint pos, int sid)
{
	edgeList[pos].bitmap = edgeList[pos].bitmap | ((ull)1 << sid);
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::Reset_Bit_Snap(uint pos, int sid)
{
	edgeList[pos].bitmap = edgeList[pos].bitmap & ~((ull)1 << sid);
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::CreateCSR(uint num_edges, uint *nodePointer, E *edgeList, int sid_a[], int sid_d[])
{
	outDegree = new uint[num_nodes]();
	for(uint i=0; i<num_nodes; i++)
		outDegree[i] = 0;
	for(uint i=0; i<num_edges; i++)
		outDegree[edgesInput[i].source]++;

	uint counter=0;
	for(uint i=0; i<num_nodes; i++)
	{
		nodePointer[i] = counter;
		counter = counter + outDegree[i];
	}
	nodePointer[num_nodes] = num_edges;
	uint *outDegreeCounter  = new uint[num_nodes]{0};
	uint location;  
	EL *edgeList_tmp = new EL[num_edges_total];
	for(uint i=0; i<num_edges; i++)
	{
		assert(edgesInput[i].source < num_nodes);
		location = nodePointer[edgesInput[i].source] + outDegreeCounter[edgesInput[i].source];
		assert(location < num_edges);
		edgeList_tmp[location].source = edgesInput[i].source;
		edgeList_tmp[location].end = edgesInput[i].end;
		edgeList_tmp[location].snap_a = edgesInput[i].snap_a;
		edgeList_tmp[location].snap_d = edgesInput[i].snap_d;
		outDegreeCounter[edgesInput[i].source]++;  
	}
	delete[] outDegreeCounter;

#pragma omp parallel for
	for(uint i=0; i < num_nodes; i++){
		if(nodePointer[i+1] - nodePointer[i] > 1)
			std::sort(edgeList_tmp + nodePointer[i], edgeList_tmp + nodePointer[i+1], 
			[] (const EL& lhs, const EL& rhs) {
				return lhs.end < rhs.end;
			});
	}
	// edgesInput.pop_front(num_edges);
	edgesInput.ReleaseResources();

#pragma omp parallel for
	for(uint i=0; i < num_edges_total; i++){
		edgeList[i].end = edgeList_tmp[i].end;
		AssignW8(edgeList[i], WEIGHT(edgeList_tmp[i].source, edgeList_tmp[i].end));
		sid_a[i] = edgeList_tmp[i].snap_a;
		sid_d[i] = edgeList_tmp[i].snap_d;
	}
	delete[] edgeList_tmp;
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::ReadGraph(bool fileContainsWeight, int seed)
{
	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	Timer timer;
	
	if(graphFormat == "bin")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		num_edges_input = 0;
		infile.read ((char*)&num_edges_input, sizeof(uint));
		num_edges_input *= 2;
		cout << "Begin file reading (" << num_nodes << " Nodes, " <<num_edges_input << " Edges) ..." << endl;

		num_edges_base = num_edges_input * init_percentage;
		if(num_edges_base % 2 != 0) num_edges_base -= 1; // make sure graph is symmetric
		cout << "Base Graph:\n";
		cout << "Number of nodes = " << num_nodes << endl;
		cout << "Number of edges = " << num_edges_base << endl;

		num_edges_add = num_edges_input * delta_rate_add;
		if (num_edges_add % 2 != 0)
			num_edges_add -= 1; // ensuring addition delta is symmetric
		num_edges_total = num_edges_base + numSnapshots * num_edges_add;
		num_edges_del = num_edges_input * delta_rate_del;
		if (num_edges_del % 2 != 0)
			num_edges_del -= 1; // ensuring deletion delta is symmetric
		cout << "addition delta size: " << num_edges_add << "\tdeletion delta size: " << num_edges_del << endl;

		edgesInput.alloc(num_edges_total + 1000);
		edgesInput.clear();

		timer.Start();
		EL newEdge, newEdge2;
		// for(uint i = 0;i < num_edges_input; i++){
		for (uint i = 0; i < num_edges_total + 10; i++){
			infile.read((char*)&(newEdge.source), sizeof(uint));
			infile.read((char*)&(newEdge.end), sizeof(uint));
			newEdge.snap_a = numeric_limits<int>::max();
			newEdge.snap_d = numeric_limits<int>::max();
			edgesInput.push_back(newEdge);
			newEdge2.source = newEdge.end;
			newEdge2.end = newEdge.source;
			newEdge2.snap_a = numeric_limits<int>::max();
			newEdge2.snap_d = numeric_limits<int>::max();
			edgesInput.push_back(newEdge2);
		}
		float read_time = timer.Finish();
		cout << "All edges are stored. Time: " << read_time / 1000 << " (s)" << endl;
		infile.close();
	}
	else if(graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;

		if(isWeighted && !fileContainsWeight)
		{
			/* Read edges from file and put them into vector<> edges*/
			// vector<EdgeWeighted> edges; // move to "edgesInput" in graph struct
			EL newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				// ss >> newEdge.w8;
				
				edgesInput.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges_total = edgeCounter;
			printf("Finished file reading and all edges are stored (%d Nodes, %d Edges) ...\n", num_nodes, num_edges_total);
			std::shuffle(std::begin(edgesInput), std::end(edgesInput), std::default_random_engine(seed));
		}
		else
		{
			;
			/*
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			*/					
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}

	for(uint i = 0; i < num_edges_base; i++){
		edgesInput[i].snap_a = 0;
	}

	for(int sid = 1; sid <= numSnapshots; sid++){
		cout << "generating deltas for snapshot " << sid << ": " << num_edges_base + sid * num_edges_add << endl;
		uint *deleted_idx = new uint[num_edges_del];
		uint num_edges_valid = num_edges_base + (sid-1) * num_edges_add - (sid-1) * num_edges_del;
		sampling_range(num_edges_valid/2, num_edges_del/2, deleted_idx, seed);
		std::sort(deleted_idx, deleted_idx + num_edges_del/2);
		uint valid_b = 0, p = 0;
		for(uint i = 0; i < num_edges_base + (sid-1) * num_edges_add; i+=2){
			if(edgesInput[i].snap_d > sid){
				if(p < num_edges_del/2 && valid_b == deleted_idx[p]){
					p++;
					edgesInput[i].snap_d = sid;
					edgesInput[i+1].snap_d = sid;
				}
				valid_b++;
			}
		}
		delete [] deleted_idx;
		assert(p == num_edges_del/2);
		for(uint i = num_edges_base + (sid - 1) * num_edges_add; i < num_edges_base + sid * num_edges_add; i++){
			edgesInput[i].snap_a = sid;
		}
	}


	gpuErrorcheck(cudaMallocHost(&nodePointer, (num_nodes + 1) * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges_total) * sizeof(E)));
	int *sid_a = new int[num_edges_total];
	int *sid_d = new int[num_edges_total];
	CreateCSR(num_edges_total, nodePointer, edgeList, sid_a, sid_d);
	edgesInput.ReleaseResources();
	cout << "finish creating CSR. Total_edges: " << num_edges_total << endl;


	/// Setting bitmaps for snapshots
	for(int sid = 0; sid <= numSnapshots; sid++){
#pragma omp parallel for
		for(uint i = 0; i < num_edges_total; i++){
			if(sid >= sid_a[i] && sid < sid_d[i]){
				Set_Bit_Snap(i, sid);
			}else{
				Reset_Bit_Snap(i, sid);
			}
		}
	}
	gpuErrorcheck(cudaMalloc(&d_nodePointer, (num_nodes+1) * sizeof(uint)));
	// gpuErrorcheck(cudaMalloc(&d_edgeList, num_edges_total * sizeof(E)));
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMemcpy(d_nodePointer, nodePointer, (num_nodes+1) * sizeof(uint), cudaMemcpyHostToDevice));
	// gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, num_edges_total * sizeof(E), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_outDegree, outDegree, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));

	/// deltas' edgeList
	add_edgeList = new DL*[numSnapshots];
	del_edgeList = new DL*[numSnapshots];
	// d_add_edgeList = new DL*[numSnapshots]; // each element is allocated on device, but pointer to pointer array is on host
	// d_del_edgeList = new DL*[numSnapshots];
	int *cnt_add_el = new int[numSnapshots]();
	int *cnt_del_el = new int[numSnapshots]();

	for(int i = 0; i < numSnapshots; i++){
		gpuErrorcheck(cudaMallocHost(&add_edgeList[i], num_edges_add * sizeof(DL)));
		gpuErrorcheck(cudaMallocHost(&del_edgeList[i], num_edges_del * sizeof(DL)));	
	}
	// gpuErrorcheck(cudaMalloc(&d_add_edgeList, num_edges_add * sizeof(DL)));
	// gpuErrorcheck(cudaMalloc(&d_del_edgeList, num_edges_del * sizeof(DL)));

	for(uint src = 0; src < num_nodes; src++){
		for(uint dst_p = nodePointer[src]; dst_p < nodePointer[src+1]; dst_p++){
			if(sid_a[dst_p] > 0){
				DL edge_;
				edge_.source = src;
				edge_.end = edgeList[dst_p].end;
				AssignW8(edge_, WEIGHT(edge_.source, edge_.end));
				add_edgeList[sid_a[dst_p] - 1][cnt_add_el[sid_a[dst_p] - 1]++] = edge_;
			}
			if(sid_d[dst_p] <= numSnapshots){
				DL edge_;
				edge_.source = src;
				edge_.end = edgeList[dst_p].end;
				AssignW8(edge_, WEIGHT(edge_.source, edge_.end));
				del_edgeList[sid_d[dst_p] - 1][cnt_del_el[sid_d[dst_p] - 1]++] = edge_;
			}
		}
	}

	delete []sid_a;
	delete []sid_d;
	delete[] cnt_add_el;
	delete[] cnt_del_el;

	// for(int i = 0; i < numSnapshots; i++){
	// 	gpuErrorcheck(cudaMemcpy(d_add_edgeList[i], add_edgeList[i], num_edges_add * sizeof(DL), cudaMemcpyHostToDevice));
	// 	gpuErrorcheck(cudaMemcpy(d_del_edgeList[i], del_edgeList[i], num_edges_del * sizeof(DL), cudaMemcpyHostToDevice));
	// }
	label1 = new bool[num_nodes]();
	label2 = new bool[num_nodes]();

	// depends = new DependencyData[num_nodes];
	value = new uint[num_nodes];

	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));

	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(uint)));
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::copy_to_device_values()
{
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_value, value, num_nodes * sizeof(uint), cudaMemcpyHostToDevice));
}

// template <class E, class EL, class DL>
// void Evolving_Graph<E, EL, DL>::countTotalDegree(bool * label){
// 	uint total_degree = 0;
// 	for(uint i = 0; i < num_nodes; i++){
// 		if(label[i])
// 			total_degree += (nodePointer[i+1] - nodePointer[i]);
// 	}
// 	cout << "\t total degree count: " << total_degree << endl;
// }


template class Evolving_Graph<OutEdge_Evolving, Edge_Union, Edge>;
template class Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>;
