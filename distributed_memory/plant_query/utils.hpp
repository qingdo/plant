// This file implements helper functions for 
// maintaining label tables and computing state of a
// node during pre-processing. 
//
// Functions include parallel prefix sum, label partitioning,
// moving labels between MPI buffers and label tables
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu

#include <vector>
#include <omp.h>
#include "graph.hpp"

namespace hl {

void prefixSum (std::vector<hl::lCounts> &Out, std::vector<int> &In, unsigned int start, unsigned int end)
{
	Out[start] = In[start];
	for (unsigned int i=start+1; i<end; i++)
	{
		Out[i] = Out[i-1] + In[i];
	}
}

//exclusive prefix sum
//integer data types for MPI compatibility
int prefixSumExc (std::vector<int> &Out, std::vector<int> &In, unsigned int start, unsigned int end)
{
	Out[start] = 0;
	for (unsigned int i=start+1; i<end; i++)
	{
		Out[i] = Out[i-1] + In[i-1];
	}
    return Out[end-1] + In[end-1];
}

void parallelPrefixSum (std::vector<hl::lCounts> &Out, std::vector<int> &In, unsigned int start, unsigned int end, unsigned int NUM_THREAD)
{
    unsigned batch = (end-start-1)/NUM_THREAD + 1;
	assert(batch > 0);
    //std::cout <<"start = "<<start<<" end= "<<end<<" batch size = " <<batch<<std::endl;

    std::vector<lCounts> tVal (NUM_THREAD+1, 0);
    #pragma omp parallel for num_threads(NUM_THREAD)
    for (unsigned tid=0; tid<NUM_THREAD; tid++)
    {
        unsigned first = start + tid*batch;
        unsigned last = std::min(start + (tid+1)*batch, end);
        for (unsigned i=first; i<last; i++)
        {
            if (i==first)
                Out[i] = In[i];
            else
                Out[i] = In[i] + Out[i-1];
        }
        tVal[tid+1] = Out[last-1];
    }
    //std::cout <<"tval " <<1<<" is "<< tVal[1]<<std::endl;
    for (unsigned i=1; i<NUM_THREAD; i++)
        tVal[i] += tVal[i-1];
    //std::cout <<"tval " <<0<<" is "<< tVal[0]<<std::endl;
    //std::cout <<"tval " <<1<<" is "<< tVal[1]<<std::endl;
    #pragma omp parallel for num_threads(NUM_THREAD)
    for (unsigned tid=0; tid<NUM_THREAD; tid++)
    {
        unsigned first = start + tid*batch;
        unsigned last = std::min(start + (tid+1)*batch, end);
        for (unsigned i=first; i<last; i++)
            Out[i] += tVal[tid];
    }

}

void partition (std::vector<Vertex> &parts, std::vector<hl::lCounts> &offset, unsigned int num_parts, std::vector<int> &In, unsigned int size)
{
	std::vector<hl::lCounts> Out (size);
	parallelPrefixSum(Out, In, 0, size, num_parts);
    //std::cout <<" num_parts " << num_parts<<std::endl;
    //std::cout <<" out[size-1] " << Out[size-1]<<std::endl;
	hl::lCounts partSize = (Out[size-1]-1)/num_parts + 1;
	assert(partSize > 0 && num_parts>0);
    //std::cout <<" lCounts " << partSize<<std::endl;
	for (unsigned i=0; i<num_parts; i++)
	{
		offset[i] = 0;
		parts[i] = 0;
	}
	#pragma omp parallel for num_threads(num_parts)
	for (unsigned i=0; i<size-1; i++)
	{
		unsigned q = Out[i]/partSize;
		unsigned qNxt = Out[i+1]/partSize;
		if (qNxt > q)
		{
			parts[q] = i+1;
			offset[qNxt] = Out[i];
		}
	}
	parts[num_parts-1] = size;
   // std::cout <<" out " <<0<<" is "<< parts[0]<<std::endl;
   // std::cout <<" out " <<1<<" is "<< parts[1]<<std::endl;
   // std::cout <<" out " <<2<<" is "<< parts[2]<<std::endl;
}


// load labels from local labeling into label list//
void parallelLoad (std::vector<int> &label_list, hl::Labeling &local_labeling, unsigned int NUM_THREAD)
{
    unsigned N = local_labeling.n;
	std::vector<int> label_count_per_vertex (N);
    #pragma omp parallel for //num_threads(NUM_THREAD)
    for (Vertex v=0; v<N; ++v)
    {
    	label_count_per_vertex[v] = local_labeling.label_v[v][0].size() + local_labeling.label_v[v][1].size();
    }

    std::vector<Vertex> vDiv (NUM_THREAD);
    std::vector<hl::lCounts> offset (NUM_THREAD);
    partition(vDiv, offset, NUM_THREAD, label_count_per_vertex, N);
    #pragma omp parallel for num_threads (NUM_THREAD)
    for (unsigned tid = 0; tid < NUM_THREAD; tid++)
    {
    	lCounts cnt = 0;
        lCounts wr_offset = 3*offset[tid];
    	Vertex start = (tid==0) ? 0 : vDiv[tid-1];
    	Vertex end = vDiv[tid];
    	for (Vertex v = start; v < end; ++v) {
    	    for (int side = 0; side < 2; ++side) {
    	        for (size_t ii = 0; ii < local_labeling.label_v[v][side].size(); ++ii) {
    				Vertex i = local_labeling.label_v[v][side][ii];
    				Distance d = local_labeling.label_d[v][side][ii];
                    label_list[wr_offset + cnt] = i*2 + side;
                    label_list[wr_offset + cnt + 1] = v;
                    label_list[wr_offset + cnt + 2] = d;
                    cnt += 3;
    	        }
    	    }
    	}
    }
}

lCounts computeLabelsPerVertex (std::vector<int> &label_count_per_vertex, const hl::Labeling &local_labeling, std::vector<int> &loadOffset, Vertex lastV)
{
    lCounts label_list_size = 0;
	Vertex N = local_labeling.n;
    #pragma omp parallel for reduction (+:label_list_size)
    for (Vertex v=0; v<N; v++)
    {
        label_count_per_vertex[v] = 0;
        for (int side=0; side<2; side++)
        {
            unsigned int labels_of_v = 0;
            for (size_t ii=loadOffset[2*v + side]; ii<local_labeling.label_v[v][side].size(); ii++)
            {
                if (local_labeling.label_v[v][side][ii] >= lastV)
                    break;
                labels_of_v++;
            }
            label_count_per_vertex[v] += labels_of_v;
        }
        label_list_size += label_count_per_vertex[v];
    }
    //std::cout << "compute labels per vertex" <<label_list_size<<std::endl;
    return label_list_size;
}

// load labels from local labeling into label list upto a given vertex//
void parallelLoadWOffset (std::vector<int> &label_list, const hl::Labeling &local_labeling, std::vector<int> &loadOffset, Vertex lastV, unsigned int NUM_THREAD)
{

	unsigned N = local_labeling.n;
	 //compute labels per spt
    std::vector<int> label_count_per_vertex(N);
    //total labels to be included in this broadcast
    //std::cout << "labels per vertex 0 " <<label_count_per_vertex[0]<<std::endl;
    lCounts label_list_size = 3*computeLabelsPerVertex(label_count_per_vertex, local_labeling, loadOffset, lastV);

    //std::cout << "labels per vertex 0 " <<label_count_per_vertex[0]<<std::endl;
    //std::cout << "label_list_size" <<label_list_size<<std::endl;
    //std::cout << "N: " <<N<<std::endl;
    //initialize the size of label_list//
    label_list.resize(label_list_size);

    //copy desired labels from local labeling to label_list
    std::vector<Vertex> vDiv (NUM_THREAD);
    std::vector<lCounts> offset (NUM_THREAD + 1);
    partition(vDiv, offset, NUM_THREAD, label_count_per_vertex, N);

    #pragma omp parallel for num_threads (NUM_THREAD)
    for (unsigned tid = 0; tid < NUM_THREAD; tid++)
    {
    	lCounts cnt = 0;
        lCounts wr_offset = 3*offset[tid];
    	Vertex start = (tid==0) ? 0 : vDiv[tid-1];
    	Vertex end = vDiv[tid];
    	for (Vertex v = start; v < end; ++v) {
    	    for (int side = 0; side < 2; ++side) {
                unsigned int nextOffset = local_labeling.label_v[v][side].size();
    	        for (unsigned int ii = loadOffset[2*v + side]; ii < local_labeling.label_v[v][side].size(); ++ii) {
    				Vertex i = local_labeling.label_v[v][side][ii];
                    if (i >= lastV)
                    {
                        //update offset for next load
                        nextOffset = ii;
                        break;
                    }
    				Distance d = local_labeling.label_d[v][side][ii];
                    label_list[wr_offset + cnt] = i*2 + side;
                    label_list[wr_offset + cnt + 1] = v;
                    label_list[wr_offset + cnt + 2] = d;
                    cnt += 3;
    	        }
                loadOffset[2*v + side] = nextOffset;
    	    }
    	}
    }
}

//<UPDATE> This function may have many mistakes
void loadFromRecvBuffer(const std::vector<int> &recv_buffer, const std::vector<unsigned int> &recv_mask, bool mask_apply, unsigned int start, unsigned int end, unsigned int local_start, unsigned int local_end, Vertex lastCV, Labeling &common_labeling, Labeling &labeling)
{
    size_t i = start;
    #pragma omp parallel for
    for (size_t i=start; i<end; i+=3)
    {
        bool remote_label = (i < local_start) || (i >= local_end);
        size_t idx = i/3;
        if (mask_apply) //reject labels only if they can be pruned
        {
            bool reject = ((recv_mask[idx>>5]&(1<<(idx&31)))==0);
            if (reject)
                continue;
        }
        Vertex v = recv_buffer[i]/2;
        if (v>=lastCV && remote_label)
            continue;
        bool forward = recv_buffer[i]&1;
        Vertex u = recv_buffer[i+1];
        Distance d = recv_buffer[i+2];
        if (v < lastCV)
            common_labeling.add(u, forward, v, d);
        else
            labeling.add(u, forward, v, d);
    }
}

void computeLocalMask(std::vector<unsigned int> &mask, const std::vector<int> &recv_buffer, unsigned int start, unsigned int end, std::vector<Vertex>&order, hl::Labeling &labelHP, hl::Labeling &labelLP, int world_rank)
{
    unsigned int label_global_sum = recv_buffer.size();
    #pragma omp parallel for schedule (dynamic, 512)
    for (unsigned int i=start; i<end; i+=3)
    {
        unsigned int vertex_order = recv_buffer[i]/2;
        Vertex v = order[vertex_order];

        bool forward = recv_buffer[i]&1;
        Vertex u = recv_buffer[i+1];
        Distance d = recv_buffer[i+2];
        bool mask_label = false;
        if (labelHP.clean_cover(u, v, forward, d, vertex_order))
            mask_label = true;
        else if (labelLP.clean_cover(u, v, forward, d, vertex_order))
            mask_label = true;
        if (mask_label)
        {
            unsigned int label_idx = i/3;
            unsigned int mask_idx = label_idx>>5;
            unsigned int bit_idx = label_idx & 31; //sizeof(int)-1, take 5 LSBs
            mask[mask_idx] = mask[mask_idx] & (~(1<<(bit_idx)));
        }
    }
    //if (world_rank==0)
    //    std::cout << "successfully computed local pruning mask" << std::endl;
}

void moveLocalToGlobal(Labeling &global_labeling, Labeling &local_labeling, std::vector<int> &local_load_offset)
{
    Vertex N = local_labeling.n;
    #pragma omp parallel for
    for (Vertex v=0; v<N; v++)
    {
        for (size_t ii=local_load_offset[2*v+0]; ii<local_labeling.label_v[v][0].size(); ii++)
            global_labeling.add_lockfree(v, 0, local_labeling.label_v[v][0][ii], local_labeling.label_d[v][0][ii]);

        for (size_t ii=local_load_offset[2*v+1]; ii<local_labeling.label_v[v][1].size(); ii++)
            global_labeling.add_lockfree(v, 1, local_labeling.label_v[v][1][ii], local_labeling.label_d[v][1][ii]);
    }
}

}
