cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(GGraph)
enable_language(CXX CUDA)
# cmake_policy(SET CMP0126 NEW)

find_package(CUDA 12.4 REQUIRED)
find_package(OpenMP REQUIRED)

# set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Specify the C++ standard (C++11 is fully supported by GCC 5.4.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 -Wno-deprecated-gpu-targets -g -Xcompiler -fopenmp")

include_directories(.)
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

# a common CUDA library
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/commons/*")
cuda_add_library(commons ${SOURCES})
target_include_directories(commons PRIVATE include/commons)
set_property(TARGET commons PROPERTY CUDA_STANDARD 11)
set_property(TARGET commons PROPERTY CUDA_ARCHITECTURES "86")
target_link_libraries(commons OpenMP::OpenMP_CXX)

# generate executable of kickstarter for add-del incremental on evolving graphs
SET(KICKSTARTER "kickstarter-um")
# file(GLOB_RECURSE SOURCES_KICKSTARTER RELATIVE ${CMAKE_SOURCE_DIR} "src/${KICKSTARTER}/*")
SET(SOURCES_KICKSTARTER 
    src/${KICKSTARTER}/evolving_graph.cu
    src/${KICKSTARTER}/evolving_graph.cuh
    src/${KICKSTARTER}/graph_kernels.cu
    src/${KICKSTARTER}/graph_kernels.cuh
    src/${KICKSTARTER}/labels_kernels.cu
    src/${KICKSTARTER}/labels_kernels.cuh)
cuda_add_library(ks_graphs ${SOURCES_KICKSTARTER})
set_property(TARGET ks_graphs PROPERTY CUDA_STANDARD 11)
set_property(TARGET ks_graphs PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(ks_graphs PRIVATE include/commons)

cuda_add_executable(ks-sssp src/${KICKSTARTER}/sssp.cu)
set_property(TARGET ks-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-sssp commons)
target_link_libraries(ks-sssp ks_graphs)
target_link_libraries(ks-sssp OpenMP::OpenMP_CXX)

cuda_add_executable(ks-sswp src/${KICKSTARTER}/sswp.cu)
set_property(TARGET ks-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-sswp commons)
target_link_libraries(ks-sswp ks_graphs)
target_link_libraries(ks-sswp OpenMP::OpenMP_CXX)

cuda_add_executable(ks-ssnp src/${KICKSTARTER}/ssnp.cu)
set_property(TARGET ks-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-ssnp commons)
target_link_libraries(ks-ssnp ks_graphs)
target_link_libraries(ks-ssnp OpenMP::OpenMP_CXX)

cuda_add_executable(ks-viterbi src/${KICKSTARTER}/viterbi.cu)
set_property(TARGET ks-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-viterbi commons)
target_link_libraries(ks-viterbi ks_graphs)
target_link_libraries(ks-viterbi OpenMP::OpenMP_CXX)

cuda_add_executable(ks-bfs src/${KICKSTARTER}/bfs.cu)
set_property(TARGET ks-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-bfs commons)
target_link_libraries(ks-bfs ks_graphs)
target_link_libraries(ks-bfs OpenMP::OpenMP_CXX)

cuda_add_executable(ks-cc src/${KICKSTARTER}/cc.cu)
set_property(TARGET ks-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-cc commons)
target_link_libraries(ks-cc ks_graphs)
target_link_libraries(ks-cc OpenMP::OpenMP_CXX)

# target_include_directories(ks-sssp PRIVATE include/commons)

# generate executable of kickstarter for add-del incremental on evolving graphs
SET(KICKSTARTER_ZC "kickstarter-zerocopy")
# file(GLOB_RECURSE SOURCES_KICKSTARTER RELATIVE ${CMAKE_SOURCE_DIR} "src/${KICKSTARTER}/*")
SET(SOURCES_KICKSTARTER_ZC
    src/${KICKSTARTER_ZC}/evolving_graph.cu
    src/${KICKSTARTER_ZC}/evolving_graph.cuh
    src/${KICKSTARTER_ZC}/graph_kernels.cu
    src/${KICKSTARTER_ZC}/graph_kernels.cuh
    src/${KICKSTARTER_ZC}/labels_kernels.cu
    src/${KICKSTARTER_ZC}/labels_kernels.cuh)
cuda_add_library(ks_zc_graphs ${SOURCES_KICKSTARTER_ZC})
set_property(TARGET ks_zc_graphs PROPERTY CUDA_STANDARD 11)
set_property(TARGET ks_zc_graphs PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(ks_zc_graphs PRIVATE include/commons)

cuda_add_executable(ks-zc-sssp src/${KICKSTARTER_ZC}/sssp.cu)
set_property(TARGET ks-zc-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-zc-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-zc-sssp commons)
target_link_libraries(ks-zc-sssp ks_zc_graphs)
target_link_libraries(ks-zc-sssp OpenMP::OpenMP_CXX)

cuda_add_executable(ks-zc-sswp src/${KICKSTARTER_ZC}/sswp.cu)
set_property(TARGET ks-zc-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-zc-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-zc-sswp commons)
target_link_libraries(ks-zc-sswp ks_zc_graphs)
target_link_libraries(ks-zc-sswp OpenMP::OpenMP_CXX)

cuda_add_executable(ks-zc-ssnp src/${KICKSTARTER_ZC}/ssnp.cu)
set_property(TARGET ks-zc-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-zc-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-zc-ssnp commons)
target_link_libraries(ks-zc-ssnp ks_zc_graphs)
target_link_libraries(ks-zc-ssnp OpenMP::OpenMP_CXX)

cuda_add_executable(ks-zc-viterbi src/${KICKSTARTER_ZC}/viterbi.cu)
set_property(TARGET ks-zc-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-zc-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-zc-viterbi commons)
target_link_libraries(ks-zc-viterbi ks_zc_graphs)
target_link_libraries(ks-zc-viterbi OpenMP::OpenMP_CXX)

cuda_add_executable(ks-zc-bfs src/${KICKSTARTER_ZC}/bfs.cu)
set_property(TARGET ks-zc-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-zc-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-zc-bfs commons)
target_link_libraries(ks-zc-bfs ks_zc_graphs)
target_link_libraries(ks-zc-bfs OpenMP::OpenMP_CXX)

cuda_add_executable(ks-zc-cc src/${KICKSTARTER_ZC}/cc.cu)
set_property(TARGET ks-zc-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-zc-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-zc-cc commons)
target_link_libraries(ks-zc-cc ks_zc_graphs)
target_link_libraries(ks-zc-cc OpenMP::OpenMP_CXX)

# generate executable of kickstarter_partition for add-del incremental on evolving graphs
SET(KICKSTARTER_SUBWAY "kickstarter-subway")
SET(SOURCES_KICKSTARTER_SUBWAY 
    src/${KICKSTARTER_SUBWAY}/evolving_graph.cu
    src/${KICKSTARTER_SUBWAY}/evolving_graph.cuh
    src/${KICKSTARTER_SUBWAY}/graph_kernels.cu
    src/${KICKSTARTER_SUBWAY}/graph_kernels.cuh
    src/${KICKSTARTER_SUBWAY}/labels_kernels.cu
    src/${KICKSTARTER_SUBWAY}/labels_kernels.cuh
    src/${KICKSTARTER_SUBWAY}/partitioner.cu
    src/${KICKSTARTER_SUBWAY}/partitioner.cuh
    src/${KICKSTARTER_SUBWAY}/subgraph.cu
    src/${KICKSTARTER_SUBWAY}/subgraph.cuh
    src/${KICKSTARTER_SUBWAY}/subgraph_generator.cu
    src/${KICKSTARTER_SUBWAY}/subgraph_generator.cuh)
cuda_add_library(ks_subway_graphs ${SOURCES_KICKSTARTER_SUBWAY})
set_property(TARGET ks_subway_graphs PROPERTY CUDA_STANDARD 11)
set_property(TARGET ks_subway_graphs PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(ks_subway_graphs PRIVATE include/commons)

cuda_add_executable(ks-subway-sssp src/${KICKSTARTER_SUBWAY}/sssp.cu)
set_property(TARGET ks-subway-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-subway-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-subway-sssp commons)
target_link_libraries(ks-subway-sssp ks_subway_graphs)

cuda_add_executable(ks-subway-sswp src/${KICKSTARTER_SUBWAY}/sswp.cu)
set_property(TARGET ks-subway-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-subway-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-subway-sswp commons)
target_link_libraries(ks-subway-sswp ks_subway_graphs)

cuda_add_executable(ks-subway-ssnp src/${KICKSTARTER_SUBWAY}/ssnp.cu)
set_property(TARGET ks-subway-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-subway-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-subway-ssnp commons)
target_link_libraries(ks-subway-ssnp ks_subway_graphs)

cuda_add_executable(ks-subway-viterbi src/${KICKSTARTER_SUBWAY}/viterbi.cu)
set_property(TARGET ks-subway-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-subway-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-subway-viterbi commons)
target_link_libraries(ks-subway-viterbi ks_subway_graphs)

cuda_add_executable(ks-subway-bfs src/${KICKSTARTER_SUBWAY}/bfs.cu)
set_property(TARGET ks-subway-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-subway-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-subway-bfs commons)
target_link_libraries(ks-subway-bfs ks_subway_graphs)
# target_include_directories(ks-sssp PRIVATE include/commons)

cuda_add_executable(ks-subway-cc src/${KICKSTARTER_SUBWAY}/cc.cu)
set_property(TARGET ks-subway-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-subway-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-subway-cc commons)
target_link_libraries(ks-subway-cc ks_subway_graphs)


# generate executable of kickstarter_partition for add-del incremental on evolving graphs
SET(COMMONGRAPH_SUBWAY "mega")
SET(SOURCES_COMMONGRAPH_SUBWAY 
    src/${COMMONGRAPH_SUBWAY}/evolving_graph.cu
    src/${COMMONGRAPH_SUBWAY}/evolving_graph.cuh
    src/${COMMONGRAPH_SUBWAY}/graph_kernels.cu
    src/${COMMONGRAPH_SUBWAY}/graph_kernels.cuh
    src/${COMMONGRAPH_SUBWAY}/labels_kernels.cu
    src/${COMMONGRAPH_SUBWAY}/labels_kernels.cuh
    src/${COMMONGRAPH_SUBWAY}/partitioner.cu
    src/${COMMONGRAPH_SUBWAY}/partitioner.cuh
    src/${COMMONGRAPH_SUBWAY}/subgraph.cu
    src/${COMMONGRAPH_SUBWAY}/subgraph.cuh
    src/${COMMONGRAPH_SUBWAY}/subgraph_generator.cu
    src/${COMMONGRAPH_SUBWAY}/subgraph_generator.cuh)
cuda_add_library(cg_subway_graphs ${SOURCES_COMMONGRAPH_SUBWAY})
set_property(TARGET cg_subway_graphs PROPERTY CUDA_STANDARD 11)
set_property(TARGET cg_subway_graphs PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(cg_subway_graphs PRIVATE include/commons)

cuda_add_executable(cg-subway-sssp src/${COMMONGRAPH_SUBWAY}/sssp.cu)
set_property(TARGET cg-subway-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-subway-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-subway-sssp commons)
target_link_libraries(cg-subway-sssp cg_subway_graphs)

cuda_add_executable(cg-subway-ssnp src/${COMMONGRAPH_SUBWAY}/ssnp.cu)
set_property(TARGET cg-subway-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-subway-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-subway-ssnp commons)
target_link_libraries(cg-subway-ssnp cg_subway_graphs)

cuda_add_executable(cg-subway-sswp src/${COMMONGRAPH_SUBWAY}/sswp.cu)
set_property(TARGET cg-subway-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-subway-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-subway-sswp commons)
target_link_libraries(cg-subway-sswp cg_subway_graphs)

cuda_add_executable(cg-subway-viterbi src/${COMMONGRAPH_SUBWAY}/viterbi.cu)
set_property(TARGET cg-subway-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-subway-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-subway-viterbi commons)
target_link_libraries(cg-subway-viterbi cg_subway_graphs)

cuda_add_executable(cg-subway-bfs src/${COMMONGRAPH_SUBWAY}/bfs.cu)
set_property(TARGET cg-subway-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-subway-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-subway-bfs commons)
target_link_libraries(cg-subway-bfs cg_subway_graphs)

cuda_add_executable(cg-subway-cc src/${COMMONGRAPH_SUBWAY}/cc.cu)
set_property(TARGET cg-subway-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-subway-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-subway-cc commons)
target_link_libraries(cg-subway-cc cg_subway_graphs)


SET(COMMONGRAPH_ZEROCOPY "commongraph-zerocopy")
SET(SOURCES_COMMONGRAPH_ZEROCOPY 
    src/${COMMONGRAPH_ZEROCOPY}/evolving_graph.cu
    src/${COMMONGRAPH_ZEROCOPY}/evolving_graph.cuh
    src/${COMMONGRAPH_ZEROCOPY}/graph_kernels.cu
    src/${COMMONGRAPH_ZEROCOPY}/graph_kernels.cuh
    src/${COMMONGRAPH_ZEROCOPY}/labels_kernels.cu
    src/${COMMONGRAPH_ZEROCOPY}/labels_kernels.cuh)
cuda_add_library(cg_zero_graphs ${SOURCES_COMMONGRAPH_ZEROCOPY})
set_property(TARGET cg_zero_graphs PROPERTY CUDA_STANDARD 11)
set_property(TARGET cg_zero_graphs PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(cg_zero_graphs PRIVATE include/commons)

cuda_add_executable(cg-zero-sssp src/${COMMONGRAPH_ZEROCOPY}/sssp.cu)
set_property(TARGET cg-zero-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-zero-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-zero-sssp commons)
target_link_libraries(cg-zero-sssp cg_zero_graphs)

cuda_add_executable(cg-zero-sswp src/${COMMONGRAPH_ZEROCOPY}/sswp.cu)
set_property(TARGET cg-zero-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-zero-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-zero-sswp commons)
target_link_libraries(cg-zero-sswp cg_zero_graphs)

cuda_add_executable(cg-zero-bfs src/${COMMONGRAPH_ZEROCOPY}/bfs.cu)
set_property(TARGET cg-zero-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-zero-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-zero-bfs commons)
target_link_libraries(cg-zero-bfs cg_zero_graphs)

cuda_add_executable(cg-zero-ssnp src/${COMMONGRAPH_ZEROCOPY}/ssnp.cu)
set_property(TARGET cg-zero-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-zero-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-zero-ssnp commons)
target_link_libraries(cg-zero-ssnp cg_zero_graphs)

cuda_add_executable(cg-zero-viterbi src/${COMMONGRAPH_ZEROCOPY}/viterbi.cu)
set_property(TARGET cg-zero-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-zero-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-zero-viterbi commons)
target_link_libraries(cg-zero-viterbi cg_zero_graphs)

cuda_add_executable(cg-zero-cc src/${COMMONGRAPH_ZEROCOPY}/cc.cu)
set_property(TARGET cg-zero-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-zero-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-zero-cc commons)
target_link_libraries(cg-zero-cc cg_zero_graphs)


SET(COMMONGRAPH_UM "commongraph-um")
SET(SOURCES_COMMONGRAPH_UM 
    src/${COMMONGRAPH_UM}/evolving_graph.cu
    src/${COMMONGRAPH_UM}/evolving_graph.cuh
    src/${COMMONGRAPH_UM}/graph_kernels.cu
    src/${COMMONGRAPH_UM}/graph_kernels.cuh
    src/${COMMONGRAPH_UM}/labels_kernels.cu
    src/${COMMONGRAPH_UM}/labels_kernels.cuh)
cuda_add_library(cg_um_graphs ${SOURCES_COMMONGRAPH_UM})
set_property(TARGET cg_um_graphs PROPERTY CUDA_STANDARD 11)
set_property(TARGET cg_um_graphs PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(cg_um_graphs PRIVATE include/commons)

cuda_add_executable(cg-um-sssp src/${COMMONGRAPH_UM}/sssp.cu)
set_property(TARGET cg-um-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-um-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-um-sssp commons)
target_link_libraries(cg-um-sssp cg_um_graphs)

cuda_add_executable(cg-um-sswp src/${COMMONGRAPH_UM}/sswp.cu)
set_property(TARGET cg-um-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-um-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-um-sswp commons)
target_link_libraries(cg-um-sswp cg_um_graphs)

cuda_add_executable(cg-um-bfs src/${COMMONGRAPH_UM}/bfs.cu)
set_property(TARGET cg-um-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-um-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-um-bfs commons)
target_link_libraries(cg-um-bfs cg_um_graphs)

cuda_add_executable(cg-um-ssnp src/${COMMONGRAPH_UM}/ssnp.cu)
set_property(TARGET cg-um-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-um-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-um-ssnp commons)
target_link_libraries(cg-um-ssnp cg_um_graphs)

cuda_add_executable(cg-um-viterbi src/${COMMONGRAPH_UM}/viterbi.cu)
set_property(TARGET cg-um-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-um-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-um-viterbi commons)
target_link_libraries(cg-um-viterbi cg_um_graphs)

cuda_add_executable(cg-um-cc src/${COMMONGRAPH_UM}/cc.cu)
set_property(TARGET cg-um-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(cg-um-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(cg-um-cc commons)
target_link_libraries(cg-um-cc cg_um_graphs)

# generate executable of kickstarter_partition for add-del incremental on evolving graphs
SET(EGRAPH "egraph")
SET(SOURCES_EGRAPH 
    src/${EGRAPH}/evolving_graph.cu
    src/${EGRAPH}/evolving_graph.cuh
    src/${EGRAPH}/graph_kernels.cu
    src/${EGRAPH}/graph_kernels.cuh
    src/${EGRAPH}/labels_kernels.cu
    src/${EGRAPH}/labels_kernels.cuh
    src/${EGRAPH}/partitioner.cu
    src/${EGRAPH}/partitioner.cuh)
cuda_add_library(ks_egraph ${SOURCES_EGRAPH})
set_property(TARGET ks_egraph PROPERTY CUDA_STANDARD 11)
set_property(TARGET ks_egraph PROPERTY CUDA_ARCHITECTURES "86")
target_include_directories(ks_egraph PRIVATE include/commons)

cuda_add_executable(ks-egraph-sssp src/${EGRAPH}/sssp.cu)
set_property(TARGET ks-egraph-sssp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-egraph-sssp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-egraph-sssp commons)
target_link_libraries(ks-egraph-sssp ks_egraph)

cuda_add_executable(ks-egraph-ssnp src/${EGRAPH}/ssnp.cu)
set_property(TARGET ks-egraph-ssnp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-egraph-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-egraph-ssnp commons)
target_link_libraries(ks-egraph-ssnp ks_egraph)

cuda_add_executable(ks-egraph-sswp src/${EGRAPH}/sswp.cu)
set_property(TARGET ks-egraph-sswp PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-egraph-sswp PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-egraph-sswp commons)
target_link_libraries(ks-egraph-sswp ks_egraph)

cuda_add_executable(ks-egraph-viterbi src/${EGRAPH}/viterbi.cu)
set_property(TARGET ks-egraph-viterbi PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-egraph-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-egraph-viterbi commons)
target_link_libraries(ks-egraph-viterbi ks_egraph)

cuda_add_executable(ks-egraph-bfs src/${EGRAPH}/bfs.cu)
set_property(TARGET ks-egraph-bfs PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-egraph-bfs PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-egraph-bfs commons)
target_link_libraries(ks-egraph-bfs ks_egraph)

cuda_add_executable(ks-egraph-cc src/${EGRAPH}/cc.cu)
set_property(TARGET ks-egraph-cc PROPERTY CUDA_STANDARD 11)
set_target_properties(ks-egraph-cc PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(ks-egraph-cc commons)
target_link_libraries(ks-egraph-cc ks_egraph)



# generate executable of core graph for incremental analysis on evolving graphs
# SET(POEGA "POEGA")
# SET(SOURCES_COREZEROCOPY 
#     src/${POEGA}/evolving_graph.cu
#     src/${POEGA}/evolving_graph.cuh
#     src/${POEGA}/graph_kernels.cu
#     src/${POEGA}/graph_kernels.cuh
#     src/${POEGA}/labels_kernels.cu
#     src/${POEGA}/labels_kernels.cuh
#     src/${POEGA}/partitioner.cu
#     src/${POEGA}/partitioner.cuh
#     src/${POEGA}/subgraph.cu
#     src/${POEGA}/subgraph.cuh
#     src/${POEGA}/subgraph_generator.cu
#     src/${POEGA}/subgraph_generator.cuh)
# cuda_add_library(core_zerocopy ${SOURCES_COREZEROCOPY})
# set_property(TARGET core_zerocopy PROPERTY CUDA_STANDARD 11)
# set_property(TARGET core_zerocopy PROPERTY CUDA_ARCHITECTURES "86")
# target_include_directories(core_zerocopy PRIVATE include/commons)


# cuda_add_executable(poega-sssp src/${POEGA}/sssp_poega.cu)
# set_property(TARGET poega-sssp PROPERTY CUDA_STANDARD 11)
# set_target_properties(poega-sssp PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(poega-sssp commons)
# target_link_libraries(poega-sssp OpenMP::OpenMP_CXX)
# target_link_libraries(poega-sssp core_zerocopy)


# # cuda_add_executable(core-zero-sssp-multistream src/${POEGA}/sssp_cg_concur_multistream.cu)
# # set_property(TARGET core-zero-sssp-multistream PROPERTY CUDA_STANDARD 11)
# # set_target_properties(core-zero-sssp-multistream PROPERTIES CUDA_ARCHITECTURES "86")
# # target_link_libraries(core-zero-sssp-multistream commons)
# # target_link_libraries(core-zero-sssp-multistream OpenMP::OpenMP_CXX)
# # target_link_libraries(core-zero-sssp-multistream core_zerocopy)


# cuda_add_executable(poega-ssnp src/${POEGA}/ssnp_poega.cu)
# set_property(TARGET poega-ssnp PROPERTY CUDA_STANDARD 11)
# set_target_properties(poega-ssnp PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(poega-ssnp commons)
# target_link_libraries(poega-ssnp OpenMP::OpenMP_CXX)
# target_link_libraries(poega-ssnp core_zerocopy)


# cuda_add_executable(poega-sswp src/${POEGA}/sswp_poega.cu)
# set_property(TARGET poega-sswp PROPERTY CUDA_STANDARD 11)
# set_target_properties(poega-sswp PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(poega-sswp commons)
# target_link_libraries(poega-sswp OpenMP::OpenMP_CXX)
# target_link_libraries(poega-sswp core_zerocopy)


# cuda_add_executable(poega-viterbi src/${POEGA}/viterbi_poega.cu)
# set_property(TARGET poega-viterbi PROPERTY CUDA_STANDARD 11)
# set_target_properties(poega-viterbi PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(poega-viterbi commons)
# target_link_libraries(poega-viterbi OpenMP::OpenMP_CXX)
# target_link_libraries(poega-viterbi core_zerocopy)


# cuda_add_executable(poega-bfs src/${POEGA}/bfs_poega.cu)
# set_property(TARGET poega-bfs PROPERTY CUDA_STANDARD 11)
# set_target_properties(poega-bfs PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(poega-bfs commons)
# target_link_libraries(poega-bfs OpenMP::OpenMP_CXX)
# target_link_libraries(poega-bfs core_zerocopy)

# cuda_add_executable(poega-cc src/${POEGA}/cc_poega.cu)
# set_property(TARGET poega-cc PROPERTY CUDA_STANDARD 11)
# set_target_properties(poega-cc PROPERTIES CUDA_ARCHITECTURES "86")
# target_link_libraries(poega-cc commons)
# target_link_libraries(poega-cc OpenMP::OpenMP_CXX)
# target_link_libraries(poega-cc core_zerocopy)
