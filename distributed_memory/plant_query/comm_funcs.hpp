//This file contains helper functions 
//for MPI communication (dividing the data, copmuting masks etc.)
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu

#ifndef COMM_FUNC_H
#define COMM_FUNC_H

#include "graph.hpp"
#include "labeling.hpp"
#include <mpi.h>
#include <vector>
#include <omp.h>
namespace hl {

int prefixSumExc (std::vector<int> &Out, std::vector<int> &In, unsigned int start, unsigned int end)
{
	Out[start] = 0;
	for (unsigned int i=start+1; i<end; i++)
	{
		Out[i] = Out[i-1] + In[i-1];
	}
    return Out[end-1] + In[end-1];
}

void parallelPrefixSum (std::vector<hl::lCounts> &Out, std::vector<unsigned> &In, unsigned int start, unsigned int end, unsigned NUM_THREAD)
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
    for (unsigned i=1; i<NUM_THREAD; i++)
        tVal[i] += tVal[i-1];
    #pragma omp parallel for num_threads(NUM_THREAD)
    for (unsigned tid=0; tid<NUM_THREAD; tid++)
    {
        unsigned first = start + tid*batch;
        unsigned last = std::min(start + (tid+1)*batch, end);
        for (unsigned i=first; i<last; i++)
            Out[i] += tVal[tid];
    }

}

void partition (std::vector<Vertex> &parts, std::vector<hl::lCounts> &offset, unsigned int num_parts, std::vector<unsigned> &In, unsigned int size)
{
	std::vector<hl::lCounts> Out (size);
	parallelPrefixSum(Out, In, 0, size, num_parts);
	hl::lCounts partSize = (Out[size-1]-1)/num_parts + 1;
	assert(partSize > 0 && num_parts>0);
	#pragma omp parallel for num_threads(num_parts)
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
}

std::vector<int> gather_size (hl::Labeling &labeling, Vertex start, Vertex end, int NUM_THREAD) {
    std::vector<int> local_size (end-start);
    std::vector<int> global_size (end-start);
    //std::vector<lCounts> prefix_sum (end - start + 1);
    #pragma omp parallel for num_threads (NUM_THREAD)
    for (Vertex v = start; v < end; v++) {
        local_size[v-start] = labeling.label_v[v][0].size() + labeling.label_v[v][1].size();
    }
    MPI_Allreduce(&local_size[0], &global_size[0], end-start, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    //parallelPrefixSum(prefix_sum, global_size, 0, end-start, NUM_THREADS);
    return global_size;
}
    
Vertex findLastSend(int MEM_BUDGET, std::vector<int> &global_size, Vertex single_start,Vertex start,  Vertex end, int NUM_THREAD) {
    lCounts count = 0;
    Vertex lastV = end;
    for (Vertex v = single_start; v < end; v++){ 
        count += global_size[v-start];
        if (count > MEM_BUDGET) {
            lastV = v;
            break;
        }
    }
    return lastV;
}


lCounts parallelLoad (std::vector<int> &label_list, hl::Labeling &local_labeling, Vertex part_start, Vertex part_end, unsigned int NUM_THREAD)
{
	unsigned num_vertices = part_end - part_start;
//	std::cout<<"num of vertices from "<<part_start<<" to "<<part_end<<std::endl;
    std::vector<unsigned> label_count_per_vertex(num_vertices);
    #pragma omp parallel for num_threads (NUM_THREAD)
    for (Vertex v=part_start; v<part_end; ++v)
    {
    	label_count_per_vertex[v - part_start] = local_labeling.label_v[v][0].size() + local_labeling.label_v[v][1].size();
    }
	lCounts total_labels = 0;
    #pragma omp parallel for reduction (+:total_labels)
    for (size_t ii=0; ii<num_vertices; ii++)
        total_labels += label_count_per_vertex[ii];

//	std::cout<<"local labels "<<total_labels<<std::endl;


    label_list.resize(total_labels * 3);

    //copy desired labels from local labeling to label_list
    std::vector<Vertex> vDiv (NUM_THREAD);
    std::vector<lCounts> offset (NUM_THREAD);
    partition(vDiv, offset, NUM_THREAD, label_count_per_vertex, num_vertices);

    for (unsigned tid = 0; tid < NUM_THREAD; tid++)
    #pragma omp parallel for num_threads (NUM_THREAD)
    for (unsigned tid = 0; tid < NUM_THREAD; tid++)
    {
    	lCounts cnt = 0;
        lCounts wr_offset = 3*offset[tid];
    	Vertex start = (tid==0) ? 0 : vDiv[tid-1];
    	Vertex end = vDiv[tid];
    	for (Vertex v = part_start + start; v < part_start + end; ++v) {
    	    for (int side = 0; side < 2; ++side) {
    	        for (unsigned int ii = 0; ii < local_labeling.label_v[v][side].size(); ++ii) {
    				Vertex i = local_labeling.label_v[v][side][ii];
    				Distance d = local_labeling.label_d[v][side][ii];
                    label_list[wr_offset + cnt] = i*2 + side;
                    label_list[wr_offset + cnt + 1] = v;
                    label_list[wr_offset + cnt + 2] = d;
                    cnt += 3;
    	        }
				//free the memory of local labeling promotly
				std::vector<Vertex>().swap(local_labeling.label_v[v][side]);
				std::vector<Distance>().swap(local_labeling.label_d[v][side]);
    	    }
    	}
    }
	return total_labels;
}
 
void loadFromRecvBuffer(const std::vector<int> &recv_buffer, Labeling &labeling)
{
    #pragma omp parallel for 
    for (size_t i=0; i<recv_buffer.size(); i+=3)
    {
        Vertex v = recv_buffer[i]/2;
        bool forward = recv_buffer[i]&1;
        Vertex u = recv_buffer[i+1];
        Distance d = recv_buffer[i+2];
        labeling.add(u, forward, v, d);
    }
}
//do an all gather given a local label list for every machine
std::vector<int> gatherAllLabels (const std::vector<int> &label_list, std::vector<int> &recv_buffer, int world_size, int world_rank)
{
    std::vector<int> world_node_offsets (world_size, 0);
    std::vector<int> world_node_list_sizes (world_size);
    unsigned int local_label_count = label_list.size();
    MPI_Allgather(&local_label_count, 1, MPI_UNSIGNED, &world_node_list_sizes[0], 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    int label_global_sum = prefixSumExc (world_node_offsets, world_node_list_sizes, 0, world_size);

    
    recv_buffer.resize(label_global_sum);

    MPI_Allgatherv(&label_list[0], world_node_list_sizes[world_rank], MPI_INT, &recv_buffer[0], &world_node_list_sizes[0], &world_node_offsets[0], MPI_INT, MPI_COMM_WORLD);
     
 //   if(world_rank==0) 
 //       std::cout << "gather successful" << std::endl;
    
    return world_node_offsets;

} 


}

#endif
