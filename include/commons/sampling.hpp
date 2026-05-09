#include <cstdlib>
/// author: Yunmo

/*
    return k random numbers from [0, n)
    reservoir sampling
*/
void sampling_range(uint n, uint k, uint *out, int seed){ 
    // Initialize output with first k elements from s[0, n) 
    uint i;
    for (i = 0; i < k; i++) 
        out[i] = i; 
 
    srand(seed); 
 
    // Iterate from the (k+1)th element to nth element 
    for (; i < n; i++) { 
        // Pick a random index from 0 to i. 
        int j = rand() % (i + 1); 
 
        // If the randomly picked index is smaller than k, 
        // then replace the element present at the index 
        // with new element from stream 
        if (j < k) 
            out[j] = i; 
    }
} 
 