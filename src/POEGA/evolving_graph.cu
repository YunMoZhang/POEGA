#include <algorithm>
#include <random>
#include <limits>
#include <vector>
#include <queue>
#include <set>
#include <map>

#include "omp.h"
#include "evolving_graph.cuh"
#include "gpu_error_check.hpp"
#include "commons/pvector.hpp"
#include "commons/sliding_queue.hpp"
#include "commons/sampling.hpp"
#include "commons/timer.hpp"
#include "commons/platform_atomics.hpp"
#include <thrust/system/cuda/execution_policy.h> // For CUDA-specific policies

#include <utility>

#define WEIGHT(x, y) ((x + y) % 50 + 1)
// #define WEIGHT(x,y) (((int)log2(x+y)) % + 1)
#define kBitsPerWord 64
#define QueueLength 30000
// #define HighDegreeThresh 200
#define PowerLawThresh 10000

#define MSB_TAG_MASK 0x80000000 
#define DATA_MASK 0x7FFFFFFF

template <class E, class EL, class DL>
Evolving_Graph<E, EL, DL>::Evolving_Graph(string graphFilePath, bool isWeighted, int num_snap, double init_p, double delta_r_a, double delta_r_d)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
	this->numSnapshots = num_snap;
	this->init_percentage = init_p;
	this->delta_rate_add = delta_r_a;
	this->delta_rate_del = delta_r_d;
	cout << "\n>>> Evolving graph settings: init_precent=" << init_p << " add_delta_rate=" << delta_r_a << " del_delta_rate=" << delta_r_d << " #snapshot=" << num_snap << endl;
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

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::calCurNumActiveNodes(bool * d_labels)
{
	thrust::device_ptr<bool> ptr_labeling(d_labels);
	numActiveNodes = thrust::count(ptr_labeling, ptr_labeling + num_nodes, true); 
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::calCurNumActiveNodes(cudaStream_t stream)
{
	thrust::device_ptr<bool> ptr_labeling(d_label2);
	numActiveNodes = thrust::count(thrust::cuda::par.on(stream), ptr_labeling, ptr_labeling + num_nodes, true); 
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::calCurNumActiveNodes(bool * d_label, cudaStream_t stream)
{
	thrust::device_ptr<bool> ptr_labeling(d_label);
	numActiveNodes = thrust::count(thrust::cuda::par.on(stream), ptr_labeling, ptr_labeling + num_nodes, true); 
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
void Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>::AssignW8(EdgeWeighted &elem1, EdgeWeighted &elem2)
{
	elem1.w8 = elem2.w8;
}

template <>
void Evolving_Graph<OutEdge_Evolving, Edge_Union, Edge>::AssignW8(Edge &elem1, Edge &elem2)
{
	; // edgeList[index].end = edgeList[index].end; // do nothing
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
void Evolving_Graph<E, EL, DL>::Set_Bit_Snap(E el[], uint pos, int sid)
{
	el[pos].bitmap = el[pos].bitmap | ((ull)1 << sid);
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::Reset_Bit_Snap(E el[], uint pos, int sid)
{
	el[pos].bitmap = el[pos].bitmap & ~((ull)1 << sid);
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::Set_Bitmap(E el[], uint pos, ull bmp)
{
	el[pos].bitmap = bmp;
}

template <class E, class EL, class DL>
bool Evolving_Graph<E, EL, DL>::Get_Bit_Snap(uint pos, int sid)
{
	return edgeList[pos].bitmap & ((ull)1 << sid);
}

template <class E, class EL, class DL>
uint64_t Evolving_Graph<E, EL, DL>::word_offset(uint pos) 
{ 
	return pos / kBitsPerWord; 
}

template <class E, class EL, class DL>
uint64_t Evolving_Graph<E, EL, DL>::bit_offset(size_t pos) 
{ 
	return pos & (kBitsPerWord - 1); 
}


template <class E, class EL, class DL>
uint Evolving_Graph<E, EL, DL>::num_words(uint size) 
{ 
	return (size + kBitsPerWord - 1) / kBitsPerWord;
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::Set_Bit(uint64_t *bmp, uint pos)
{
	bmp[word_offset(pos)] |= ((uint64_t) 1l << bit_offset(pos));
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::Reset_Bit(uint64_t *bmp, uint pos)
{
	bmp[word_offset(pos)] &= ~((uint64_t) 1l << bit_offset(pos));
}


template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::SetAsWeight(bool isWeighted_)
{
	this->isWeighted = isWeighted_;
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::CreateCSR(uint num_edges, pvector<EL> &inputEdgeList, uint *nodePointer, E * &edgeList, uint degrees [], uint8_t sid_a[], uint8_t sid_d[])
{
	// outDegree = new uint[num_nodes]();
	uint *pos_arr = new uint[num_edges]();

	memset(degrees, 0, num_nodes * sizeof(uint));
	for (uint i = 0; i < num_edges; i++)
	{
		pos_arr[i] = degrees[inputEdgeList[i].source]++;
	}

	uint counter=0;
	std::priority_queue<std::pair<uint, uint>, std::vector<std::pair<uint, uint>>, std::greater<std::pair<uint, uint>>> min_heap;
	top_k_degree = 10;
	
	MaxDegree = 10000;
	for(uint i = 0; i < num_nodes; i++)
	{
		if (min_heap.size() < top_k_degree) {
            min_heap.push({degrees[i], i});
        }
		else if (degrees[i] > min_heap.top().first) {
            min_heap.pop();
            min_heap.push({degrees[i], i});
        }
		nodePointer[i] = counter;
		counter = counter + degrees[i];
		if (degrees[i] > MaxDegree)	
			MaxDegree = degrees[i];
	}
	nodePointer[num_nodes] = num_edges;
	cout << "Graph Max degree: " << MaxDegree << endl;

	top_k_degree_nodes = new uint[top_k_degree];
    int heap_idx = top_k_degree - 1;
    while (!min_heap.empty()) {
        top_k_degree_nodes[heap_idx] = min_heap.top().second;
        min_heap.pop();
        heap_idx--;
    }

	// uint *outDegreeCounter  = new uint[num_nodes]{0};
	uint location;  
	EL *edgeList_tmp = new EL[num_edges_total];
#pragma omp parallel for
	for(uint i=0; i < num_edges; i++)
	{
		// location = nodePointer[edgesInput[i].source] + outDegreeCounter[edgesInput[i].source];
		location = nodePointer[inputEdgeList[i].source] + pos_arr[i];
		edgeList_tmp[location].source = inputEdgeList[i].source;
		edgeList_tmp[location].end = inputEdgeList[i].end;
		edgeList_tmp[location].snap_a = inputEdgeList[i].snap_a;
		edgeList_tmp[location].snap_d = inputEdgeList[i].snap_d;
		// outDegreeCounter[edgesInput[i].source]++;  
	}
	delete[] pos_arr;
	// inputEdgeList.pop_front(num_edges);
	inputEdgeList.ReleaseResources();
	cout << "finished edge array, sorting..." << endl;

	gpuErrorcheck(cudaHostAlloc(&edgeList, (num_edges_total) * sizeof(E), cudaHostAllocMapped));


#pragma omp parallel for
	for(uint i=0; i < num_nodes; i++){
		if(nodePointer[i+1] - nodePointer[i] > 1)
			std::sort(edgeList_tmp + nodePointer[i], edgeList_tmp + nodePointer[i+1], 
			[] (const EL& lhs, const EL& rhs) {
				return lhs.end < rhs.end;
			});
	}
	
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

	if (graphFormat == "bin")
	{
		ifstream infile(graphFilePath, ios::in | ios::binary);

		infile.read((char *)&num_nodes, sizeof(uint));
		num_edges_input = 0;
		infile.read((char *)&num_edges_input, sizeof(uint));
		num_edges_input *= 2;
		cout << "Begin file reading (" << num_nodes << " Nodes, " << num_edges_input << " Edges) ..." << endl;

		num_edges_base = num_edges_input * init_percentage;
		if (num_edges_base % 2 != 0)
			num_edges_base -= 1; // make sure graph is symmetric

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
		for (uint i = 0; i < num_edges_total + 10; i++)
		{
			infile.read((char *)&(newEdge.source), sizeof(uint));
			infile.read((char *)&(newEdge.end), sizeof(uint));
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
	else if (graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;

		if (isWeighted && !fileContainsWeight)
		{
			/* Read edges from file and put them into vector<> edges*/
			EL newEdge;
			timer.Start();
			while (getline(infile, line))
			{
				ss.str("");
				ss.clear();
				ss << line;

				ss >> newEdge.source;
				ss >> newEdge.end;
				// ss >> newEdge.w8;

				edgesInput.push_back(newEdge);
				edgeCounter++;

				if (max < newEdge.source)
					max = newEdge.source;
				if (max < newEdge.end)
					max = newEdge.end;
			}
			infile.close();
			num_nodes = max + 1;
			num_edges_input = edgeCounter;
			float runtime = timer.Finish();
			cout << "Time: " << runtime / 1000 << " seconds." << endl;

			timer.Start();
			std::shuffle(std::begin(edgesInput), std::end(edgesInput), std::default_random_engine(seed));
			runtime = timer.Finish();
			cout << "Shuffling time: " << runtime / 1000 << " seconds." << endl;
		}
		else
		{
			;
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}

	num_nodes_power_law_csg = 0;

	for (uint i = 0; i < num_edges_base; i++)
	{
		edgesInput[i].snap_a = 0;
	}

	for (int sid = 1; sid <= numSnapshots; sid++)
	{
		cout << "generating deltas for snapshot " << sid << ": " << num_edges_base + sid * num_edges_add << endl;
		uint *deleted_idx = new uint[num_edges_del / 2 + 1];
		uint num_edges_valid = num_edges_base + (sid - 1) * num_edges_add - (sid - 1) * num_edges_del;
		sampling_range(num_edges_valid / 2, num_edges_del / 2, deleted_idx, seed);
		std::sort(deleted_idx, deleted_idx + num_edges_del / 2);
		uint valid_b = 0, p = 0;
		for (uint i = 0; i < num_edges_base + (sid - 1) * num_edges_add; i += 2)
		{
			if (edgesInput[i].snap_d > sid)
			{ // not yet deleted, aka, exists in curret snapshot
				if (p < num_edges_del / 2 && valid_b == deleted_idx[p])
				{
					p++;
					edgesInput[i].snap_d = sid;
					edgesInput[i + 1].snap_d = sid;
				}
				valid_b++;
			}
		}
		delete[] deleted_idx;
		assert(p == num_edges_del / 2);
	#pragma omp parallel for
		for (uint i = num_edges_base + (sid - 1) * num_edges_add; i < num_edges_base + sid * num_edges_add; i++)
		{
			edgesInput[i].snap_a = sid;
		}
	}

	gpuErrorcheck(cudaMallocHost(&nodePointer, (num_nodes + 1) * sizeof(uint)));
	uint8_t *sid_a = new uint8_t[num_edges_total];
	uint8_t *sid_d = new uint8_t[num_edges_total];
	outDegree = new uint[num_nodes]();
	CreateCSR(num_edges_total, edgesInput, nodePointer, edgeList, outDegree, sid_a, sid_d);
	// edgesInput.ReleaseResources();
	cout << "finish creating CSR. Total_edges: " << num_edges_total << endl;

	/// Setting bitmaps for snapshots
	for (int sid = 0; sid <= numSnapshots; sid++)
	{
		// for deletions
#pragma omp parallel for
		for (uint i = 0; i < num_edges_total; i++)
		{
			if (sid >= sid_a[i] && sid < sid_d[i])
				Set_Bit_Snap(i, sid);
			else
				Reset_Bit_Snap(i, sid);
		}
	}

	gpuErrorcheck(cudaMalloc(&d_nodePointer, (num_nodes + 1) * sizeof(uint)));
	// gpuErrorcheck(cudaMalloc(&d_edgeList, num_edges_total * sizeof(E)));
	cudaHostGetDevicePointer(&d_edgeList, edgeList, 0);
	gpuErrorcheck(cudaMemcpy(d_nodePointer, nodePointer, (num_nodes + 1) * sizeof(uint), cudaMemcpyHostToDevice));
	// gpuErrorcheck(cudaMemcpy(d_edgeList, edgeList, num_edges_total * sizeof(E), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMemcpy(d_outDegree, outDegree, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));

	/// deltas' edgeList
	add_edgeList = new DL *[numSnapshots];
	del_edgeList = new DL *[numSnapshots];
	// d_add_edgeList = new DL*[numSnapshots]; // each element is allocated on device, but pointer to pointer array is on host
	// d_del_edgeList = new DL*[numSnapshots];
	int *cnt_add_el = new int[numSnapshots]();
	int *cnt_del_el = new int[numSnapshots]();
	direct_affected_vertices = new bool[num_nodes]();

	for (int i = 0; i < numSnapshots; i++)
	{
		gpuErrorcheck(cudaMallocHost(&add_edgeList[i], num_edges_add * sizeof(DL)));
		gpuErrorcheck(cudaMallocHost(&del_edgeList[i], num_edges_del * sizeof(DL)));
		// gpuErrorcheck(cudaMalloc(&d_add_edgeList[i], num_edges_add * sizeof(DL)));
	}
	// gpuErrorcheck(cudaMalloc(&d_add_edgeList, num_edges_add * sizeof(DL)));
	// gpuErrorcheck(cudaMalloc(&d_del_edgeList, num_edges_del * sizeof(DL)));

	for (uint src = 0; src < num_nodes; src++)
	{
		for (uint dst_p = nodePointer[src]; dst_p < nodePointer[src + 1]; dst_p++)
		{
			if (sid_a[dst_p] > 0)
			{
				DL edge_;
				edge_.source = src;
				edge_.end = edgeList[dst_p].end;
				AssignW8(edge_, WEIGHT(edge_.source, edge_.end));
				add_edgeList[sid_a[dst_p] - 1][cnt_add_el[sid_a[dst_p] - 1]++] = edge_;
			}
			if (sid_d[dst_p] <= numSnapshots)
			{
				DL edge_;
				edge_.source = src;
				edge_.end = edgeList[dst_p].end;
				AssignW8(edge_, WEIGHT(edge_.source, edge_.end));
				del_edgeList[sid_d[dst_p] - 1][cnt_del_el[sid_d[dst_p] - 1]++] = edge_;
				direct_affected_vertices[edge_.end] = true;
			}
		}
	}

	delete[] sid_a;
	delete[] sid_d;
	delete[] cnt_add_el;
	delete[] cnt_del_el;

	/// Allocating necessary runtime data

	label1 = new bool[num_nodes]();
	label2 = new bool[num_nodes]();

	value = new uint[num_nodes]();
	depends = new DependencyData[num_nodes];

	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(uint)));

	gpuErrorcheck(cudaMalloc(&d_depends, num_nodes * sizeof(DependencyData)));
	gpuErrorcheck(cudaMalloc(&d_depends_old, num_nodes * sizeof(DependencyData)));

	// depends_core = new DependencyData *[numSnapshots + 1];
	depends_core = new DependencyData[num_nodes];
	value_core = new uint *[numSnapshots + 1];
	for (uint i = 0; i <= numSnapshots; i++)
	{
		value_core[i] = new uint[num_nodes];
	}
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::TransferValuesToDevice()
{
	gpuErrorcheck(cudaMemcpy(d_label1, label1, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_label2, label2, num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaPeekAtLastError());
}

template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::CreateProxyGraphUnionCSR(int degree_limit, uint bridge_thresh)
{
	gpuErrorcheck(cudaPeekAtLastError());
	cout << "\nCreating proxy graph CSR..." << endl;
	Timer timer;
	timer.Start();
	pvector<EL> edgeL(num_edges_base);
	edgeL.clear();
	uint *degrees = new uint[num_nodes]();

	num_edges_core = 0;
	EL newEdge;
	newEdge.snap_a = 0;
	newEdge.snap_d = numeric_limits<int>::max();
	std::vector<uint> low_level_vertices;
	for (uint i = 0; i < num_nodes; i++)
	{
		uint low_level_times = 0;
		std::set<uint> sorted_parents; 
		for(uint top_i = 0; top_i < top_k_degree; top_i++)
		{
			if (depends_proxy[top_i][i].parent != PARENT_INFINITY && depends_proxy[top_i][i].parent != i)
			{
				sorted_parents.insert(depends_proxy[top_i][i].parent);
				if(outDegree[i] > 100 && depends_proxy[top_i][i].level <= 2)
					low_level_times++;
			}
			// if(num_edges_core >= num_edges_base * 0.07 && sorted_parents.size() > 1)
			// 	break;
		}
		for (const auto& elem : sorted_parents) {
			newEdge.source = elem;
			newEdge.end = i;
			edgeL.push_back(newEdge);
			degrees[elem]++;
		}
		num_edges_core += sorted_parents.size();
		if(low_level_times == 10)
			low_level_vertices.push_back(i);
	}
	cout << "\t proxy graph size: " << num_edges_core << "\t num_edges_base: " << num_edges_base << endl;

	// if(num_edges_core < num_edges_base * 0.05){
	// 	cout << "Proxy graph is too small, add more edges. " << endl;
	// 	for (uint i = 0; i < num_nodes; i++)
	// 	{
	// 		if (depends_proxy[0][i].parent != PARENT_INFINITY && depends_proxy[0][i].parent != i) // add edges of DT to CSG
	// 		{
	// 			uint prnt = depends_proxy[0][i].parent;
	// 			for (uint j = nodePointer[prnt]; j < nodePointer[prnt + 1]; j++)
	// 			{
	// 				if (edgeList[j].bitmap & ((ull)1))
	// 				{
	// 					uint dst = edgeList[j].end;
	// 					if(outDegree[prnt] > degree_limit){
	// 						if (dst != i && depends_proxy[0][dst].value == depends_proxy[0][i].value)  // depndency graph
	// 						{
	// 							newEdge.source = prnt;
	// 							newEdge.end = dst;
	// 							edgeL.push_back(newEdge);
	// 							num_edges_core++;
	// 						}
	// 					}
	// 					else if(depends_proxy[0][prnt].level < 5) // low level vertices
	// 					{
	// 						newEdge.source = prnt;
	// 						newEdge.end = dst;
	// 						edgeL.push_back(newEdge);
	// 						num_edges_core++;
	// 					}
	// 				}
	// 			}
	// 		}
	// 		if(num_edges_core >= num_edges_base * 0.15)
	// 			break;
	// 	}
	// }

	if(num_edges_core < num_edges_base * 0.05){
		int cnt = 0;
		int max_deg = 0;
		int max_deg_node = 0;
		for(uint v : low_level_vertices){
			if(outDegree[v] > max_deg)
			{
				max_deg = outDegree[v];
				max_deg_node = v;
			}
		}
		std::cout << "Proxy graph size is a little small, consider manually run more pre-analysis. Recommand node: " << max_deg_node << std::endl;
	}
	
	// make proxy graph symmetric
	for (uint i = 0; i < num_edges_core; i++)
	{
		newEdge.source = edgeL[i].end;
		newEdge.end = edgeL[i].source;
		edgeL.push_back(newEdge);
	}
	num_edges_core *= 2;
	cout << "proxy graph size after symm: " << num_edges_core << endl;

	///  ============ creating CSG for the rest snapshots  ===========
	// Include some edges from ADD delta to proxy graph
	add_edgeList_core = new DL*[numSnapshots];
	num_edges_add_core = new uint[numSnapshots]();

	DL newEdgeAdd;

	for (uint sid = 1; sid <= numSnapshots; sid++){
		add_edgeList_core[sid - 1] = new DL[num_edges_add];
		for (uint i = 0; i < num_edges_add; i++)
		{
			uint src = add_edgeList[sid - 1][i].source;
			uint dst = add_edgeList[sid - 1][i].end;
			bool cond1 = outDegree[src] > degree_limit && outDegree[dst] > degree_limit;
			bool cond2 = (depends[src].parent == PARENT_INFINITY) || (depends[dst].parent == PARENT_INFINITY);
			if (cond1 || cond2) // for IT-2004
			{
				newEdge.source = src;
				newEdge.end = dst;
				newEdge.snap_a = sid;
				edgeL.push_back(newEdge);
				num_edges_core++;

				newEdgeAdd.source = src;
				newEdgeAdd.end = dst;
				AssignW8(newEdgeAdd, add_edgeList[sid - 1][i]);
				add_edgeList_core[sid - 1][num_edges_add_core[sid - 1]++] = newEdgeAdd;
			}
		}
	}

	// for(uint i = 0; i < numSnapshots; i++){
	// 	cout << "proxy graph add delta " << i << ": "<< num_edges_add_core[i] << endl;
	// }

	cout << "proxy graph size after include addition delta: " << num_edges_core << endl;

	uint *pos_arr = new uint[num_edges_core]();
	
	memset(degrees, 0, num_nodes * sizeof(uint));
	for (uint i = 0; i < num_edges_core; i++)
	{
		pos_arr[i] = degrees[edgeL[i].source]++;
	}

	uint counter = 0;
	gpuErrorcheck(cudaMallocHost(&nodePointer_csg_u, (num_nodes + 1) * sizeof(uint)));
	
	for (uint i = 0; i < num_nodes; i++)
	{
		nodePointer_csg_u[i] = counter;
		counter = counter + degrees[i];
	}
	assert(counter == num_edges_core);
	nodePointer_csg_u[num_nodes] = counter;

	uint location;
	EL *edgeList_tmp = new EL[num_edges_core];
	
#pragma omp parallel for
	for (uint i = 0; i < num_edges_core; i++)
	{
		location = nodePointer_csg_u[edgeL[i].source] + pos_arr[i];
		edgeList_tmp[location].source = edgeL[i].source;
		edgeList_tmp[location].end = edgeL[i].end;
		edgeList_tmp[location].snap_a = edgeL[i].snap_a;
		edgeList_tmp[location].snap_d = edgeL[i].snap_d;
	}
#pragma omp parallel for
	for (uint i = 0; i < num_nodes; i++)
	{
		if (nodePointer_csg_u[i + 1] - nodePointer_csg_u[i] > 1)
			std::sort(edgeList_tmp + nodePointer_csg_u[i], edgeList_tmp + nodePointer_csg_u[i + 1],
					  [](const EL &lhs, const EL &rhs)
					  {
						  return lhs.end < rhs.end;
					  });
	}
	
	for (uint sid = 1; sid <= numSnapshots; sid++)  // find edges from deletion delta
	{
		// removing edges in deletion delta
#pragma omp parallel for
		for (uint i = 0; i < num_edges_del; i++)
		{
			uint src = del_edgeList[sid - 1][i].source, dst = del_edgeList[sid - 1][i].end;
			auto lower = std::lower_bound(edgeList_tmp + nodePointer_csg_u[src], edgeList_tmp + nodePointer_csg_u[src + 1], dst,
										  [](const EL &e1, const uint &val)
										  {
											  return e1.end < val;
										  });
			if (lower != edgeList_tmp + nodePointer_csg_u[src + 1] && (lower->end == dst))
			{
				edgeList_tmp[lower - edgeList_tmp].snap_d = sid;
				// num_edges_del_core[sid - 1]++;
			}
		}
	}

	gpuErrorcheck(cudaMallocHost(&edgeList_csg_u, num_edges_core * sizeof(E)));


#pragma omp parallel for
	for (uint i = 0; i < num_edges_core; i++)
	{
		edgeList_csg_u[i].end = edgeList_tmp[i].end;
		AssignW8(edgeList_csg_u[i], WEIGHT(edgeList_tmp[i].source, edgeList_tmp[i].end));
	}


	/// Setting bitmaps for snapshots
	for (int sid = 0; sid <= numSnapshots; sid++)
	{
		// for deletions
#pragma omp parallel for
		for (uint i = 0; i < num_edges_core; i++)
		{
			if (sid >= edgeList_tmp[i].snap_a && sid < edgeList_tmp[i].snap_d)
				Set_Bit_Snap(edgeList_csg_u, i, sid);
			else
				Reset_Bit_Snap(edgeList_csg_u, i, sid);
		}
	}

	uint * nodePointer_tmp = new uint[num_nodes + 1];
	memcpy(nodePointer_tmp, nodePointer_csg_u, (num_nodes+1) * sizeof(uint));

	core_outDegree = new uint[num_nodes]();
	memset(core_outDegree, 0, num_nodes);
	
	cout << "remove the replicate elements in edgelist " << endl;
	nodePointer_csg_u[0] = 0;
	uint idx = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		if (nodePointer_tmp[i + 1] > nodePointer_tmp[i])
		{
			edgeList_csg_u[idx++] = edgeList_csg_u[nodePointer_tmp[i]];
		}
		for (uint j = nodePointer_tmp[i] + 1; j < nodePointer_tmp[i + 1]; j++)
		{
			if (edgeList_csg_u[j].end != edgeList_csg_u[j - 1].end)
			{
				edgeList_csg_u[idx++] = edgeList_csg_u[j];
			}
		}
		nodePointer_csg_u[i + 1] = idx;
		core_outDegree[i] = nodePointer_csg_u[i + 1] - nodePointer_csg_u[i];
	}

	num_edges_core = nodePointer_csg_u[num_nodes];

	cout << "after dedup, union proxy graph size: " << nodePointer_csg_u[num_nodes] << " takes ratios: " << (double)num_edges_core / num_edges_total << endl;

	float gen_time = timer.Finish();
	cout << "Generating CSGs time: " << gen_time << endl;
	cout << endl;

	edgeL.ReleaseResources();
	delete[] degrees;
	delete[] pos_arr;
	gpuErrorcheck(cudaPeekAtLastError());
	for(int i = 0; i < top_k_degree; i++){
		delete[] depends_proxy[i];
	}
	delete[] depends_proxy;
}


template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::CreateLowWeightCoreUnionCSR(int seed)
{
	cout << "\nCreating Low-weight-based proxy graph CSR"  << endl;
	Timer timer;
	
	pvector<EL> edgeL(num_edges_base);
	edgeL.clear();
	uint *degrees = new uint[num_nodes]();

	num_edges_core = 0;
	EL newEdge;
	newEdge.snap_a = 0;
	newEdge.snap_d = numeric_limits<int>::max();

	uint *weights = new uint[nodePointer[num_nodes]];
	uint edge_cnt = 0;

	for(uint i = 0; i < num_nodes; i++){
		for (uint j = nodePointer[i]; j < nodePointer[i + 1]; j++)
		{
			uint dst = edgeList[j].end;
			weights[edge_cnt++] = WEIGHT(i, dst);
		}
	}
	timer.Start();
	std::sort(weights, weights + edge_cnt);
	uint weight_thresh = weights[(uint)(edge_cnt * 0.1)]; // take the top 5% of edges
	delete[] weights;

	num_nodes_power_law_csg = 0; // used for optimizing power-law CSG

	for (uint i = 0; i < num_nodes; i++)
	{
		for (uint j = nodePointer[i]; j < nodePointer[i + 1]; j++)
		{
			if (edgeList[j].bitmap & ((ull)1) && WEIGHT(i, edgeList[j].end) < weight_thresh)
			{
				newEdge.source = i;
				newEdge.end = edgeList[j].end;
				// AssignW8(newEdge, WEIGHT(i, edgeList[j].end));
				edgeL.push_back(newEdge);
				num_edges_core++;
			}
		}
		if (outDegree[i] > PowerLawThresh / 2)
			num_nodes_power_law_csg++;
	}
	cout << "\t dependency tree size: " << num_edges_core << "\t num_edges_base: " << num_edges_base << endl;

	// make random core graph symmetric
	for (uint i = 0; i < num_edges_core; i++)
	{
		newEdge.source = edgeL[i].end;
		newEdge.end = edgeL[i].source;
		edgeL.push_back(newEdge);
	}
	num_edges_core *= 2;
	cout << "dependency tree core graph size after symm: " << num_edges_core << endl;

	///  ============ creating CSG for the rest snapshots  ===========
	// Include some edges from ADD delta to Core graph
	add_edgeList_core = new DL *[numSnapshots];
	num_edges_add_core = new uint[numSnapshots]();

	DL newEdgeAdd;

	for (uint sid = 1; sid <= numSnapshots; sid++)
	{
		add_edgeList_core[sid - 1] = new DL[num_edges_add];
		for (uint i = 0; i < num_edges_add; i++)
		{
			uint src = add_edgeList[sid - 1][i].source;
			uint dst = add_edgeList[sid - 1][i].end;
			if (WEIGHT(src,dst) < weight_thresh) 
			{
				newEdge.source = src;
				newEdge.end = dst;
				newEdge.snap_a = sid;
				edgeL.push_back(newEdge);
				num_edges_core++;

				newEdgeAdd.source = src;
				newEdgeAdd.end = dst;
				AssignW8(newEdgeAdd, add_edgeList[sid - 1][i]);
				add_edgeList_core[sid - 1][num_edges_add_core[sid - 1]++] = newEdgeAdd;
			}
		}
	}

	uint *pos_arr = new uint[num_edges_core]();

	memset(degrees, 0, num_nodes * sizeof(uint));
	for (uint i = 0; i < num_edges_core; i++)
	{
		pos_arr[i] = degrees[edgeL[i].source]++;
	}

	cout << "num_nodes_power_law_csg : " << num_nodes_power_law_csg << endl;
	uint counter = 0;
	gpuErrorcheck(cudaMallocHost(&nodePointer_csg_u, (num_nodes + 1 + num_nodes_power_law_csg + 100) * sizeof(uint)));
	uint max_deg = 10000;
	// uint equal_cnt = 0;
	uint power_law_cnt = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		nodePointer_csg_u[i] = counter;
		counter = counter + degrees[i];
		if (degrees[i] > max_deg)
			max_deg = degrees[i];
		if (degrees[i] > PowerLawThresh)
		{
			nodePointer_csg_u[num_nodes + 1 + power_law_cnt] = i;
			power_law_cnt++;
		}
	}
	assert(counter == num_edges_core);
	cout << "max deg in CSG: " << max_deg << endl;

	cout << "power_law cnt: " << power_law_cnt << "\t num_nodes_power_law_csg: " << num_nodes_power_law_csg << endl;
	// assert(power_law_cnt == num_nodes_power_law_csg);
	num_nodes_power_law_csg = power_law_cnt;
	cout << "#power law nodes in CSG: " << num_nodes_power_law_csg << endl;
	nodePointer_csg_u[num_nodes] = counter;

	uint location;
	EL *edgeList_tmp = new EL[num_edges_core];

#pragma omp parallel for
	for (uint i = 0; i < num_edges_core; i++)
	{
		location = nodePointer_csg_u[edgeL[i].source] + pos_arr[i];
		edgeList_tmp[location].source = edgeL[i].source;
		edgeList_tmp[location].end = edgeL[i].end;
		edgeList_tmp[location].snap_a = edgeL[i].snap_a;
		edgeList_tmp[location].snap_d = edgeL[i].snap_d;
	}
#pragma omp parallel for
	for (uint i = 0; i < num_nodes; i++)
	{
		if (nodePointer_csg_u[i + 1] - nodePointer_csg_u[i] > 1)
			std::sort(edgeList_tmp + nodePointer_csg_u[i], edgeList_tmp + nodePointer_csg_u[i + 1],
					  [](const EL &lhs, const EL &rhs)
					  {
						  return lhs.end < rhs.end;
					  });
	}

	for (uint sid = 1; sid <= numSnapshots; sid++) // find edges from deletion delta
	{
		// removing edges in deletion delta
#pragma omp parallel for
		for (uint i = 0; i < num_edges_del; i++)
		{
			uint src = del_edgeList[sid - 1][i].source, dst = del_edgeList[sid - 1][i].end;
			auto lower = std::lower_bound(edgeList_tmp + nodePointer_csg_u[src], edgeList_tmp + nodePointer_csg_u[src + 1], dst,
										  [](const EL &e1, const uint &val)
										  {
											  return e1.end < val;
										  });
			if (lower != edgeList_tmp + nodePointer_csg_u[src + 1] && (lower->end == dst))
			{
				edgeList_tmp[lower - edgeList_tmp].snap_d = sid;
			}
		}
	}

	gpuErrorcheck(cudaMallocHost(&edgeList_csg_u, num_edges_core * sizeof(E)));

#pragma omp parallel for
	for (uint i = 0; i < num_edges_core; i++)
	{
		edgeList_csg_u[i].end = edgeList_tmp[i].end;
		AssignW8(edgeList_csg_u[i], WEIGHT(edgeList_tmp[i].source, edgeList_tmp[i].end));
	}

	/// Setting bitmaps for snapshots
	for (int sid = 0; sid <= numSnapshots; sid++)
	{
		// for deletions
#pragma omp parallel for
		for (uint i = 0; i < num_edges_core; i++)
		{
			if (sid >= edgeList_tmp[i].snap_a && sid < edgeList_tmp[i].snap_d)
				Set_Bit_Snap(edgeList_csg_u, i, sid);
			else
				Reset_Bit_Snap(edgeList_csg_u, i, sid);
		}
	}

	uint *nodePointer_tmp = new uint[num_nodes + 1];
	memcpy(nodePointer_tmp, nodePointer_csg_u, (num_nodes + 1) * sizeof(uint));

	// remove the replicate elements in edgelist
	nodePointer_csg_u[0] = 0;
	uint idx = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		if (nodePointer_tmp[i + 1] > nodePointer_tmp[i])
		{
			edgeList_csg_u[idx++] = edgeList_csg_u[nodePointer_tmp[i]];
		}
		for (uint j = nodePointer_tmp[i] + 1; j < nodePointer_tmp[i + 1]; j++)
		{
			if (edgeList_csg_u[j].end != edgeList_csg_u[j - 1].end)
			{
				edgeList_csg_u[idx++] = edgeList_csg_u[j];
			}
		}
		nodePointer_csg_u[i + 1] = idx;
	}

	num_edges_core = nodePointer_csg_u[num_nodes];

	cout << "after dedup, union csg size: " << nodePointer_csg_u[num_nodes] << " takes ratios: " << (double)num_edges_core / num_edges_total << endl;

	float gen_time = timer.Finish();
	cout << "Generating CSGs time: " << gen_time << endl;

	// delete[] edgeL;
	edgeL.ReleaseResources();
	// delete[] nodePointer_tmp;
	delete[] degrees;
	delete[] pos_arr;

	bool *deg_same_vertices = new bool[num_nodes]();
	uint num_same = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		if (outDegree[i] == nodePointer_csg_u[i + 1] - nodePointer_csg_u[i])
		{
			deg_same_vertices[i] = true;
			num_same++;
		}
	}
	cout << "num_same vertices: " << num_same << endl;
	delete[] deg_same_vertices;
}



template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::CreateRandomCoreUnionCSR(int seed)
{
	cout << "\nCreating Random-based proxy graph CSR" << endl;
	Timer timer;
	timer.Start();
	pvector<EL> edgeL(num_edges_base);
	edgeL.clear();
	uint *degrees = new uint[num_nodes]();

	num_edges_core = 0;
	EL newEdge;
	newEdge.snap_a = 0;
	newEdge.snap_d = numeric_limits<int>::max();

	uint *index = new uint[nodePointer[num_nodes]];
	uint edge_cnt = nodePointer[num_nodes];

#pragma omp parallel for
	for (uint i = 0; i < edge_cnt; i++)
	{
		index[i] = i;
	}
	shuffle(index, index+edge_cnt, default_random_engine(seed));
	edge_cnt = (uint)(edge_cnt * 0.1); // take the top 10% of edges
	std::sort(index, index + edge_cnt);


	num_nodes_power_law_csg = 0; // used for optimizing power-law CSG

	uint cnt = 0;
	uint idx = 0;

	for (uint i = 0; i < num_nodes; i++)
	{
		for (uint j = nodePointer[i]; j < nodePointer[i + 1]; j++)
		{
			if (edgeList[j].bitmap & ((ull)1) )
			{
				cnt++;
				if(cnt == index[idx]){
					newEdge.source = i;
					newEdge.end = edgeList[j].end;
					// AssignW8(newEdge, WEIGHT(i, edgeList[j].end));
					edgeL.push_back(newEdge);
					num_edges_core++;
					idx++;
					if(idx >= edge_cnt) break; // stop when we have enough edges
				}
			}
		}
		if (outDegree[i] > PowerLawThresh / 2)
			num_nodes_power_law_csg++;
	}
	cout << "\t dependency tree size: " << num_edges_core << "\t num_edges_base: " << num_edges_base << endl;

	// make random core graph symmetric
	for (uint i = 0; i < num_edges_core; i++)
	{
		newEdge.source = edgeL[i].end;
		newEdge.end = edgeL[i].source;
		edgeL.push_back(newEdge);
	}
	num_edges_core *= 2;
	cout << "dependency tree core graph size after symm: " << num_edges_core << endl;

	///  ============ creating CSG for the rest snapshots  ===========
	// Include some edges from ADD delta to Core graph
	add_edgeList_core = new DL *[numSnapshots];
	num_edges_add_core = new uint[numSnapshots]();

	DL newEdgeAdd;

	for (uint sid = 1; sid <= numSnapshots; sid++)
	{
		add_edgeList_core[sid - 1] = new DL[num_edges_add];
		for (uint i = 0; i < num_edges_add; i++)
		{
			uint src = add_edgeList[sid - 1][i].source;
			uint dst = add_edgeList[sid - 1][i].end;
			if (rand()%10 == 0)
			{
				newEdge.source = src;
				newEdge.end = dst;
				newEdge.snap_a = sid;
				edgeL.push_back(newEdge);
				num_edges_core++;

				newEdgeAdd.source = src;
				newEdgeAdd.end = dst;
				AssignW8(newEdgeAdd, add_edgeList[sid - 1][i]);
				add_edgeList_core[sid - 1][num_edges_add_core[sid - 1]++] = newEdgeAdd;
			}
		}
	}

	uint *pos_arr = new uint[num_edges_core]();

	memset(degrees, 0, num_nodes * sizeof(uint));
	for (uint i = 0; i < num_edges_core; i++)
	{
		pos_arr[i] = degrees[edgeL[i].source]++;
	}

	cout << "num_nodes_power_law_csg : " << num_nodes_power_law_csg << endl;
	uint counter = 0;
	gpuErrorcheck(cudaMallocHost(&nodePointer_csg_u, (num_nodes + 1 + num_nodes_power_law_csg + 100) * sizeof(uint)));
	uint max_deg = 10000;
	// uint equal_cnt = 0;
	uint power_law_cnt = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		nodePointer_csg_u[i] = counter;
		counter = counter + degrees[i];
		if (degrees[i] > max_deg)
			max_deg = degrees[i];
		if (degrees[i] > PowerLawThresh)
		{
			nodePointer_csg_u[num_nodes + 1 + power_law_cnt] = i;
			power_law_cnt++;
		}
	}
	assert(counter == num_edges_core);
	cout << "max deg in CSG: " << max_deg << endl;

	cout << "power_law cnt: " << power_law_cnt << "\t num_nodes_power_law_csg: " << num_nodes_power_law_csg << endl;
	// assert(power_law_cnt == num_nodes_power_law_csg);
	num_nodes_power_law_csg = power_law_cnt;
	cout << "#power law nodes in CSG: " << num_nodes_power_law_csg << endl;
	nodePointer_csg_u[num_nodes] = counter;

	uint location;
	EL *edgeList_tmp = new EL[num_edges_core];

#pragma omp parallel for
	for (uint i = 0; i < num_edges_core; i++)
	{
		location = nodePointer_csg_u[edgeL[i].source] + pos_arr[i];
		edgeList_tmp[location].source = edgeL[i].source;
		edgeList_tmp[location].end = edgeL[i].end;
		edgeList_tmp[location].snap_a = edgeL[i].snap_a;
		edgeList_tmp[location].snap_d = edgeL[i].snap_d;
	}
#pragma omp parallel for
	for (uint i = 0; i < num_nodes; i++)
	{
		if (nodePointer_csg_u[i + 1] - nodePointer_csg_u[i] > 1)
			std::sort(edgeList_tmp + nodePointer_csg_u[i], edgeList_tmp + nodePointer_csg_u[i + 1],
					  [](const EL &lhs, const EL &rhs)
					  {
						  return lhs.end < rhs.end;
					  });
	}

	for (uint sid = 1; sid <= numSnapshots; sid++) // find edges from deletion delta
	{
		// removing edges in deletion delta
#pragma omp parallel for
		for (uint i = 0; i < num_edges_del; i++)
		{
			uint src = del_edgeList[sid - 1][i].source, dst = del_edgeList[sid - 1][i].end;
			auto lower = std::lower_bound(edgeList_tmp + nodePointer_csg_u[src], edgeList_tmp + nodePointer_csg_u[src + 1], dst,
										  [](const EL &e1, const uint &val)
										  {
											  return e1.end < val;
										  });
			if (lower != edgeList_tmp + nodePointer_csg_u[src + 1] && (lower->end == dst))
			{
				edgeList_tmp[lower - edgeList_tmp].snap_d = sid;
			}
		}
	}

	gpuErrorcheck(cudaMallocHost(&edgeList_csg_u, num_edges_core * sizeof(E)));

#pragma omp parallel for
	for (uint i = 0; i < num_edges_core; i++)
	{
		edgeList_csg_u[i].end = edgeList_tmp[i].end;
		AssignW8(edgeList_csg_u[i], WEIGHT(edgeList_tmp[i].source, edgeList_tmp[i].end));
	}

	/// Setting bitmaps for snapshots
	for (int sid = 0; sid <= numSnapshots; sid++)
	{
		// for deletions
#pragma omp parallel for
		for (uint i = 0; i < num_edges_core; i++)
		{
			if (sid >= edgeList_tmp[i].snap_a && sid < edgeList_tmp[i].snap_d)
				Set_Bit_Snap(edgeList_csg_u, i, sid);
			else
				Reset_Bit_Snap(edgeList_csg_u, i, sid);
		}
	}

	uint *nodePointer_tmp = new uint[num_nodes + 1];
	memcpy(nodePointer_tmp, nodePointer_csg_u, (num_nodes + 1) * sizeof(uint));

	// remove the replicate elements in edgelist
	nodePointer_csg_u[0] = 0;
	idx = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		if (nodePointer_tmp[i + 1] > nodePointer_tmp[i])
		{
			edgeList_csg_u[idx++] = edgeList_csg_u[nodePointer_tmp[i]];
		}
		for (uint j = nodePointer_tmp[i] + 1; j < nodePointer_tmp[i + 1]; j++)
		{
			if (edgeList_csg_u[j].end != edgeList_csg_u[j - 1].end)
			{
				edgeList_csg_u[idx++] = edgeList_csg_u[j];
			}
		}
		nodePointer_csg_u[i + 1] = idx;
	}

	num_edges_core = nodePointer_csg_u[num_nodes];

	cout << "after dedup, union csg size: " << nodePointer_csg_u[num_nodes] << " takes ratios: " << (double)num_edges_core / num_edges_total << endl;

	float gen_time = timer.Finish();
	cout << "Generating CSGs time: " << gen_time << endl;

	// delete[] edgeL;
	edgeL.ReleaseResources();
	// delete[] nodePointer_tmp;
	delete[] degrees;
	delete[] pos_arr;

	bool *deg_same_vertices = new bool[num_nodes]();
	uint num_same = 0;
	for (uint i = 0; i < num_nodes; i++)
	{
		if (outDegree[i] == nodePointer_csg_u[i + 1] - nodePointer_csg_u[i])
		{
			deg_same_vertices[i] = true;
			num_same++;
		}
	}
	cout << "num_same vertices: " << num_same << endl;
	delete[] deg_same_vertices;
}


template <class E, class EL, class DL>
void Evolving_Graph<E, EL, DL>::LoadCoreUnionCSR()
{
	gpuErrorcheck(cudaMalloc(&d_nodePointer_core, (num_nodes + 1) * sizeof(uint)));
	gpuErrorcheck(cudaMalloc(&d_edgeList_core, num_edges_core * sizeof(E)));
	gpuErrorcheck(cudaMalloc(&d_core_outDegree, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMemcpy(d_nodePointer_core, nodePointer_csg_u, (num_nodes + 1) * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList_core, edgeList_csg_u, num_edges_core * sizeof(E), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_core_outDegree, core_outDegree, num_nodes * sizeof(uint), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaPeekAtLastError());
}


template class Evolving_Graph<OutEdge_Evolving, Edge_Union, Edge>;
template class Evolving_Graph<OutEdgeWeighted_Evolving, EdgeWeighted_Union, EdgeWeighted>;