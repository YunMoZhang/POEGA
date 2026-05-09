# POEGA

## Prerequisites

- CUDA Toolkit 12.4
- CMake >= 3.18
- C++ compiler with C++11 support (GCC/Clang) and OpenMP
- Build tool: `make` or `ninja`

## Datasets
The graph datasets used in the evaluation of POEGA can be found in [Datasets](./datasets/Datasets.md). The downloaded graph datasets are typically in edge list format.

POEGA requires the input graph format being **Edge List** stored in a ```.bin``` file. 

Edge List ```(.el)``` example:
```
[src1] [dst1]
[src2] [dst2]
```

To generate ```.bin``` file from the raw ```.el``` Edge List graph data, we provide the conversion tool.
```
cd tools
make
./converter [input .el file] [output file name] [#edges bypassed]
```
For example, to process the sample file provided in ```../datasets/graph_data_sample.el```, run
```
./converter ../datasets/graph_data_sample.el ../datasets/output 1
```
It will generate ```output.bin```.

Note that the above steps preprocess the full graph dataset. The evolving graph generation steps that involve the random delta sampling from the full graph are written in the code introduced below, thus not requiring additional processing steps.


## Running the Experiments

**Compilation**
```
cd POEGA
mkdir build
cd build
cmake ..
```
**Running**
(to be updated)
```
bash ./run.sh
```

## References
1. Subway (Out-of-GPU-Memory Graph Processing with Minimal Data Transfer). https://github.com/AutomataLab/Subway.
2. Kickstarter/Graphbolt. https://github.com/pdclab/graphbolt
2. Scott Beamer. GAP Benchmark Suite. https://github.com/sbeamer/gapbs.