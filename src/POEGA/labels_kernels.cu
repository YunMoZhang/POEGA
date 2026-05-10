#include "labels_kernels.cuh"

#define MSB_TAG_MASK 0x80000000 
#define DATA_MASK 0x7FFFFFFF

#define MASK_STATE      0xC0000000 
// 掩码：提取低 30 位数据 (Payload)
#define MASK_PAYLOAD    0x3FFFFFFF 

// // 三种状态标志
#define STATE_COMPACT   0x00000000 // 00...
#define STATE_LOCKED    0x80000000 // 10... (锁住，但带数据)
#define STATE_BUFFER    0xC0000000 // 11... (指向 Buffer)

__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[activeNodes[id+from]] = false;
	}
}

__global__ void clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[id] = false;
	}
}

__global__ void flipLabel(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		if(label2[id])
			label1[id] = false;
		else
			label1[id] = true;
	}
}

__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		int nID = activeNodes[id+from];
		label1[nID] = label1[nID] || label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int nID;
	if(id < size){
		nID = activeNodes[id+from];
		label1[nID] = label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		label1[id] = label2[id];
		label2[id] = false;	
	}
}

__global__ void diffLabels(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		if(label1[id] && label2[id])
			label2[id] = false;	
	}
}


__global__ void moveUpLabelsNoClear(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		label1[id] = label2[id];
		// label2[id] = false;
	}
}

// __global__ void moveUpLabels(uint *queue1, uint *queue2, unsigned int* size)
// {
// 	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(id < *size){
// 		queue1[id] = queue2[id];	
// 	}
// }

__global__ void mergeLabels(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		if(label2[id])
			label1[id] = label2[id];
		label2[id] = false;
	}
}

__global__ void mergeLabelsNoClear(bool *label1, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		if (label2[id])
			label1[id] = label2[id];
	}
}


__global__ void mergeLabelsTotal(bool *label2, bool *label2s, uint numSnap, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		for(uint sid = 0; sid < numSnap; sid++)
			if(label2s[id * numSnap + sid]){
				label2[id] = true;
				label2s[id * numSnap + sid] = false;
				break;
			}

	}
}

__global__ void resetLabels(bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		label2[id] = false;	
	}
}

__global__ void copyValues(uint *val1, uint *val2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		val1[id] = val2[id];
	}
}

__global__ void copyLabels(bool *val1, bool *val2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		val1[id] = val2[id];
	}
}

__global__ void setNFLabels(bool *label2, bool *far_label2, uint value, uint nf_threshold, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		if(label2[id] && value > nf_threshold){
			far_label2[id] = true;
			label2[id] = false;
		}
	}
}

// __global__ void copyQSize(uint *qs1, uint *qs2)
// {
// 	*qs1 = *qs2;
// 	*qs2 = 0;
// }

// __global__ void save_old(unsigned int numNodes,
// 							uint *value,
// 							uint *value_old,
// 							uint *level,
// 							uint *level_old,
// 							int *parent,
// 							int *parent_old)
// {
// 	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(tId < numNodes)
// 	{	
// 		value_old[tId] = value[tId];
// 		level_old[tId] = level[tId];
// 		parent_old[tId] = parent[tId];
// 	}	
// }

__global__ void save_old(unsigned int numNodes,
							DependencyData * depends,
							DependencyData * depends_old)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes)
	{	
		depends_old[tId] = depends[tId];
	}	
}

__global__ void transfromValue(unsigned int numNodes,
							DependencyData * depends,
							uint *value)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes)
	{	
		value[id] = (uint)depends[id].value;
	}	
}

__global__ void transfromLevel(unsigned int numNodes,
							   DependencyData *depends,
							   uint *level)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < numNodes)
	{
		level[id] = depends[id].level;
	}
}

__global__ void transfromDependtoValue(unsigned int numNodes,
									   DependencyData *depends,
									   uint *value,
										uint numSnap,
										uint sid)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < numNodes)
	{
		value[id] = (uint)depends[id * numSnap +sid].value;
	}
}

__global__ void transfromDependtoDepend(unsigned int numNodes,
									   DependencyData *depends,
									   DependencyData *depends2,
									   uint numSnap,
									uint sid)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < numNodes)
	{
		depends2[id] = depends[id * numSnap + sid];
	}
}

// __global__ void bitmapToQueue(bool * label, uint * queue, uint numNodes, uint* size)
// {
// 	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
// 	if(tId < numNodes)
// 	{	
// 		if(label[tId]){
// 			uint index = atomicAdd(size, 1);
// 			queue[index] = tId;
// 			label[tId] = false;
// 		}
// 	}	
// }

__global__ void sssp_setActive(uint *value, bool *label2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		if (value[id] < DIST_INFINITY)
		{
			label2[id] = true;
		}
	}
}

__global__ void copyValue(uint *value1, uint *value2, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < size)
	{
		value1[id] = value2[id];
	}
}

__global__ void set_same(bool *same, uint * value, uint num_nodes, uint num_snap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes)
	{
		same[id] = true;
		for(uint i = 1; i < num_snap; i++){
			if(value[id * num_snap + i] != value[id * num_snap + i-1]){
				same[id] = false;
				break;
			}
		}
	}
}

__global__ void set_same(bool *same, uint *value, bool * label, uint num_nodes, uint num_snap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes && same[id] && label[id])
	{
        uint elem = id * num_snap;
        for (uint i = 1; i < num_snap; i++)
        {
			if (value[elem + i] != value[elem + i - 1])
			{
				same[id] = false;
				break;
			}
		}
	}
}

__global__ void set_max_min(uint *max, uint *min, uint *value, uint num_nodes, uint num_snap)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_nodes)
    {
        uint from = id * num_snap;
        uint to = (id+1) * num_snap;
        max[id] = value[from];
        min[id] = value[from];
        for (uint i = from + 1; i < to; i++)
        {
            if (value[i] > max[id])
            {
                max[id] = value[i];
            }
            if (value[i] < min[id])
            {
                min[id] = value[i];
            }
        }
    }
}

__global__ void set_max_min(uint *max, uint *min, uint *value, uint * buffer, uint * buffer_total, uint num_nodes, uint num_snap)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_nodes)
    {
		if((value[id] & MSB_TAG_MASK) != 0){
			uint dst_idx_bg = (value[id] & DATA_MASK) * num_snap;
			max[id] = buffer[dst_idx_bg];
			min[id] = buffer[dst_idx_bg];
			for (uint i = dst_idx_bg + 1; i < dst_idx_bg + num_snap; i++)
			{
				uint val = buffer[i];
				if (val > max[id])
				{
					max[id] = val;
				}
				if (val < min[id])
				{
					min[id] = val;
				}
			}
		}else{
			max[id] = value[id];
			min[id] = value[id];
		}
    }
}


__global__ void set_max_min_waitfree(uint *max, uint *min, uint *value, uint * buffer, uint * buffer_total, uint num_nodes, uint num_snap)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_nodes)
    {
		if((value[id] & MASK_STATE)  == STATE_BUFFER){
			uint dst_idx_bg = (value[id] & MASK_PAYLOAD) * num_snap;
			max[id] = buffer[dst_idx_bg];
			min[id] = buffer[dst_idx_bg];
			for (uint i = dst_idx_bg + 1; i < dst_idx_bg + num_snap; i++)
			{
				uint val = buffer[i];
				if (val > max[id])
				{
					max[id] = val;
				}
				if (val < min[id])
				{
					min[id] = val;
				}
			}
		}else{
			max[id] = value[id];
			min[id] = value[id];
		}
    }
}

__global__ void clear_waitfree_buffer(uint *value, uint * buffer, uint * counter, uint * buffer_total, uint num_nodes, uint num_snap)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_nodes)
    {
		if((value[id] & MASK_STATE)  == STATE_BUFFER){
			bool same = true;
			uint dst_idx_bg = (value[id] & MASK_PAYLOAD) * num_snap;
			uint val = buffer[dst_idx_bg];
			for (uint i = dst_idx_bg + 1; i < dst_idx_bg + num_snap; i++)
			{
				if(buffer[i] != val){
					same = false;
					break;
				}
			}
			if(same){
				value[id] = val;
				uint old_counter = atomicSub(counter, 1);
			}
		}
    }
}



__global__ void update_max_min(uint *max, uint *min, uint *value, bool * label, uint num_nodes, uint num_snap)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_nodes)
    {
        if(label[id] == false)
            return;
        uint from = id * num_snap;
        uint to = (id + 1) * num_snap;
        max[id] = value[from];
        min[id] = value[from];
        for (uint i = from; i < to; i++)
        {
            if (value[i] > max[id])
            {
                max[id] = value[i];
            }
            if (value[i] < min[id])
            {
                min[id] = value[i];
            }
        }
    }
}

__global__ void find_same(uint *max, uint *min, uint * off,  uint num_nodes, uint num_snap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes)
	{
		if(max[id] == min[id]){
			off[id] = 1;
		}
		else{
			off[id] = num_snap;
		}
	}
}

__global__ void expand_same(uint *value, bool * label, uint num_nodes, uint num_snap)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes)
	{
		if(label[id]){
			label[id] = false;
			for (uint i = 1; i < num_snap; i++)
			{
				value[id * num_snap + i] = value[id * num_snap];
			}
		}
	}
}

__global__ void set_same_degree(bool *same, uint *deg, uint *nodePointer, uint num_nodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes)
	{
		if (deg[id] == nodePointer[id + 1] - nodePointer[id])
		{
			same[id] = true;
		}
		else
		{
			same[id] = false;
		}
	}
}

__global__ void update_value_old(uint *value, uint *old_value, uint *off, uint * deg, bool* label, uint num_nodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < num_nodes)
	{
		if(label[id] && deg[id] == 1){
			old_value[id] = value[off[id]];
		}
	}
}

__global__ void makeQueueFromBitmap(unsigned int *activeNodes, bool *activeNodesLabeling,
							unsigned int *prefixLabeling, unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes && activeNodesLabeling[id]){
		activeNodes[prefixLabeling[id]] = id;
	}
}

__global__ void makeQueueFromBitmapPowerlaw(
    unsigned int *activeNodes,       
    bool *activeNodesLabeling,   
    unsigned int *prefixLabeling,    
	uint * chunkOffset,
    unsigned int numNodes)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < numNodes && activeNodesLabeling[id]) {
        unsigned int from = prefixLabeling[id];
		unsigned int to = prefixLabeling[id + 1];
		uint count = to - from;

        for (unsigned int i = 0; i < count; ++i) {
            activeNodes[from + i] = id;
			chunkOffset[from + i] = i;
        }
    }
}