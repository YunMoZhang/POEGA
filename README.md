# POEGA

## Prerequisites

Ensure your system meets the following requirements before building POEGA:

- CUDA Toolkit 12.4 or higher
- CMake >= 3.18
- C++ compiler with C++11 support (GCC/Clang) and OpenMP
- Build tool: `make` or `ninja`

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

**Compilation**

Build the project from the root directory
```
mkdir build
cd build
cmake ..
make -j
```
**Running**
(to be updated)
```
bash ./run.sh
```

## References
1. Subway (Out-of-GPU-Memory Graph Processing with Minimal Data Transfer). [Github](https://github.com/AutomataLab/Subway)
2. Kickstarter/Graphbolt. [Github](https://github.com/pdclab/graphbolt)
2. Scott Beamer. GAP Benchmark Suite. [Github](https://github.com/sbeamer/gapbs)