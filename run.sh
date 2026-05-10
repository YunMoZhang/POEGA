cd build
make clean
make -j
cd ..
# Set DATA_PATH before running to point at your graph data directory.
# export DATA_PATH=<your graph data directory>
# (Optional) specify the GPU device to run
# export CUDA_VISIBLE_DEVICES=<your device id>

#===========================================================================
# ================================Kickstarter-UM=======================
# echo "Kickstarter-UM"
# echo "SSSP"
./build/bin/ks-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add  0.05 --delta_rate_del  0.05 --snap 32
./build/bin/ks-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add  0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSNP"
./build/bin/ks-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

# echo "SSWP"
./build/bin/ks-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "viterbi"
./build/bin/ks-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

#  echo "BFS"
./build/bin/ks-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

## CC
#  echo "CC"
./build/bin/ks-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

#===========================================================================
# ================================Kickstarter-ZeroCopy (Grapin)=======================
# echo "Kickstarter-ZeroCopy"
# echo "SSSP"
./build/bin/ks-zc-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add  0.05 --delta_rate_del  0.05 --snap 32
./build/bin/ks-zc-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add  0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSNP"
./build/bin/ks-zc-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSWP"
./build/bin/ks-zc-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "viterbi"
./build/bin/ks-zc-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


#  echo "BFS"
./build/bin/ks-zc-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

## CC
#  echo "CC"
./build/bin/ks-zc-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-zc-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


#=========================================================================
# ================================Kickstarter-Subway=======================
# echo "Kickstarter-Subway"
# echo "SSSP"
./build/bin/ks-subway-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSNP"
./build/bin/ks-subway-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSWP"
./build/bin/ks-subway-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


## Viterbi
# echo "viterbi"
./build/bin/ks-subway-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

## BFS
# echo "BFS"
./build/bin/ks-subway-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


## CC
# echo "CC"
./build/bin/ks-subway-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-subway-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32



#=========================================================================
# ======================================MEGA==============================
# echo "MEGA"
# echo "SSSP"
./build/bin/cg-subway-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32



## SSNP
# echo "SSNP"
./build/bin/cg-subway-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

# echo "SSWP"
./build/bin/cg-subway-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "viterbi"
./build/bin/cg-subway-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "BFS"
./build/bin/cg-subway-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "CC"
./build/bin/cg-subway-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-subway-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

#===========================================================================
# ================================Commongraph-Zerocopy=======================
# echo "Commongraph-Zerocopy"
# echo "SSSP"
./build/bin/cg-zero-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSWP"
./build/bin/cg-zero-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "SSNP"
./build/bin/cg-zero-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "Viterbi"
./build/bin/cg-zero-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

## BFS
# echo "BFS"
./build/bin/cg-zero-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

## CC
# echo "CC"
./build/bin/cg-zero-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-zero-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


#=====================================================================
# ================================Commongraph-UM=======================
# echo "Commongraph-UM"
# echo "SSSP"
./build/bin/cg-um-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

# echo "SSNP"
./build/bin/cg-um-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

# echo "SSWP"
./build/bin/cg-um-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "Viterbi"
./build/bin/cg-um-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "BFS"
./build/bin/cg-um-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


# echo "CC"
./build/bin/cg-um-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/cg-um-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

#===================================================================
# ================================Egraph=======================
## SSSP
# echo "SSSP"
./build/bin/ks-egraph-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

##SSNP
# echo "SSNP"
./build/bin/ks-egraph-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


##SSWP
# echo "SSWP"
./build/bin/ks-egraph-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


##Viterbi
# echo "Viterbi"
./build/bin/ks-egraph-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32


##bfs
# echo "BFS"
./build/bin/ks-egraph-bfs --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32

##cc
# echo "CC"
./build/bin/ks-egraph-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32
./build/bin/ks-egraph-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32



#===================================================================
# ================================  POEGA  =========================
# SSSP

# echo "SSSP"
./build/bin/poega-sssp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800 
./build/bin/poega-sssp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800 
./build/bin/poega-sssp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 8000 
./build/bin/poega-sssp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --degree_limit 800 --snap 32 --bridge_thresh 800
./build/bin/poega-sssp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 2000 --bridge_thresh 10000


# SSNP
# echo "SSNP"
./build/bin/poega-ssnp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800
./build/bin/poega-ssnp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800
./build/bin/poega-ssnp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 8000
./build/bin/poega-ssnp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --degree_limit 800 --snap 32 --bridge_thresh 800
./build/bin/poega-ssnp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 2000 --bridge_thresh 10000 

# SSWP
./build/bin/poega-sswp --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800 
./build/bin/poega-sswp --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800 
./build/bin/poega-sswp --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 8000
./build/bin/poega-sswp --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --degree_limit 800 --snap 32 --bridge_thresh 800 
./build/bin/poega-sswp --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 2000 --bridge_thresh 10000 


# Viterbi
./build/bin/poega-viterbi --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800
./build/bin/poega-viterbi --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800 
./build/bin/poega-viterbi --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 8000
./build/bin/poega-viterbi --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --degree_limit 800 --snap 32 --bridge_thresh 800
./build/bin/poega-viterbi --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 2000 --bridge_thresh 10000

# BFS
./build/bin/poega-bfs --input "$DATA_PATH/uk-2005.bin" --source  34054405  --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800
./build/bin/poega-bfs --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800
./build/bin/poega-bfs --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 8000
./build/bin/poega-bfs --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --degree_limit 800 --snap 32 --bridge_thresh 800
./build/bin/poega-bfs --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 2000 --bridge_thresh 10000 
 

# CC
./build/bin/poega-cc --input "$DATA_PATH/uk-2005.bin" --source 34054405 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800
./build/bin/poega-cc --input "$DATA_PATH/it-2004.bin" --source 26424103 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 800 
./build/bin/poega-cc --input "$DATA_PATH/twitter.bin" --source 16190898 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 800 --bridge_thresh 8000 
./build/bin/poega-cc --input "$DATA_PATH/sk-2005.bin" --source 48932207 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --degree_limit 800 --snap 32 --bridge_thresh 800
./build/bin/poega-cc --input "$DATA_PATH/subdomain.bin" --source 100640172 --init_percent 50 --delta_rate_add 0.05 --delta_rate_del 0.05 --snap 32 --degree_limit 2000 --bridge_thresh 10000