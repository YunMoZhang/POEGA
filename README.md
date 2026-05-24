# POEGA

This repository includes the source code for POEGA, an efficient and scalable framework for out-of-GPU-memory evolving graph processing. It also provides implementations of other state-of-the-art comparative approaches. For more technical details, please refer to our paper: *Efficient GPU-centric Evolving Graph Processing at Scale* (Paper Link TBD).

## Citing POEGA
```
@inproceedings{
    TBD
}
```

## Prerequisites

Ensure your system meets the following requirements before building POEGA:

- CUDA Toolkit 12.4 or higher
- CMake >= 3.18
- C++ compiler with C++11 support (GCC/Clang) and OpenMP
- Build tool: `make` or `ninja`

**Hardware Environment**

This framework has been evaluated on an NVIDIA RTX A4000 GPU (16 GB memory) and an NVIDIA A6000 Ada GPU (48 GB memory). It might support other NVIDIA GPUs with comparable memory capacities.

## Directory Structure
The repository layout and brief descriptions:

```
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # Project overview and usage guidelines
├── run.sh                      # Script to run the benchmark suite and experiments.
├── datasets/                   # Graph dataset preparation guides and a sampled input format
├── include/                    # Public header files
├── src/                        # POEGA and seven compared methods source code
│   ├── commons/                # General shared source utilities
│   ├── egraph/                 
│   ├── kickstarter-um/        
│   ├── kickstarter-subway/     
│   ├── kickstarter-zerocopy/  
│   ├── commongraph-um/        
│   ├── commongraph-zerocopy/   
│   ├── mega/              
│   └── POEGA/                 
└── tools/                      # Utility programs and optional result processing scripts
```

## Quick Start

Clone the repository and navigate to the project root:
```
git clone https://github.com/YunMoZhang/POEGA.git
cd POEGA
```

## Datasets

The graph datasets used to evaluate POEGA are detailed in [Datasets](./datasets/Datasets.md). The downloaded datasets are typically provided in an edge list format.

### Input graph data format
POEGA requires the input graph format being **Edge List** and stored in a `.bin` file. 

Edge List file (```.el```) example:
```
[src1] [dst1]
[src2] [dst2]
...
```
### Format conversion
We provide a conversion tool to transform raw `.el` edge list files into the required `.bin` format.

To build and run the converter:
```
cd tools
make
./converter <input_.el_file> <output_file_name> <#edges_bypassed>
```
(The third parameter specifies the number of header lines to skip in the raw data file.)

**Example**

To convert the provided sample file in `../datasets/graph_data_sample.el`, run
```
./converter ../datasets/graph_data_sample.el ../datasets/output 1
```
This will generate ```output.bin```.

Note on Evolving Graphs: The steps above preprocess the full, static graph dataset. The evolving graph generation (which involves random delta sampling from the full graph) is handled internally by the runtime code, requiring no manual preprocessing.


## Building & Running

### Compilation

Build the project from the root directory using CMake:
```
mkdir build && cd build
cmake ..
make -j
```
### Running
Before executing the experiments, set the `DATA_PATH` environment variable to point to the directory containing your preprocessed `.bin` graph files:
```
export DATA_PATH=/absolute/path/to/graph_data 
```
Then, run the script to execute the entire benchmark suite:
```
bash ./run.sh
```
This script executes 6 graph algorithms across 5 different datasets for **POEGA** and 7 other comparative baselines.

**Supported Algorithms and Baselines**

Graph algorithms:
- SSSP (Single-Source Shortest Path)
- SSNP (Single-Source Nearest Path)
- SSWP (Single-Source Widest Path)
- BFS (Breadth-First Search)
- CC (Weakly Connected Components)
- Viterbi

Evaluated Frameworks:
- POEGA
- Re-evaluation based method: `egraph`
- Streaming-based incremental analysis methods: `kickstarter-um` (KS-UM), `kickstarter-subway` (KS-SW), `kickstarter-zerocopy` (Grapin)
- Batch-based incremental analysis methods: `commongraph-um` (CG-UM), `commongraph-zerocopy` (CG-ZC), `mega`

Note that each benchmark includes the evolving graph generation (offline) and evolving graph analytics (online).

**Parameter Explanation**

All frameworks accept a standard set of parameters to configure the benchmark:
```
./<framework>-<algo> --input <graph_data> --source <source_node> --init_percent <init_ratio> --delta_rate_add <add_rate> --delta_rate_del <del_rate> --snap <num_snapshots>
```
- `--init_percent`: The percentage of the graph used as the base graph (e.g., 50 for 50%).

- `--delta_rate_add / --delta_rate_del`: The ratio of edges added or deleted per snapshot relative to the full graph (e.g., 0.05 for 0.05%).

For POEGA-specific executions, the optional `--degree_limit` parameter can be specified to define the degree threshold for identifying high-degree nodes in proxy graph generation.



## References
1. Subway (Out-of-GPU-Memory Graph Processing with Minimal Data Transfer). [Github](https://github.com/AutomataLab/Subway)
2. Kickstarter/Graphbolt. [Github](https://github.com/pdclab/graphbolt)
2. Scott Beamer. GAP Benchmark Suite. [Github](https://github.com/sbeamer/gapbs)