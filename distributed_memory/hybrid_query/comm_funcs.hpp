#include "graph.hpp"
#include <mpi.h>
#include <vector>
#include <omp.h>
namespace hl {

// store number of labels per hub //
std::vector<lCounts> accLabelCountsGen (std::vector<int> &labels_per_hub, int world_rank, unsigned int NUM_THREAD)
{
    Vertex spts_per_sync = labels_per_hub.size();
    //gather individual label counts from each machine
    std::vector<int> recv_labels_per_hub (labels_per_hub.size(), 0);
    MPI_Allreduce(&labels_per_hub[0], &recv_labels_per_hub[0], spts_per_sync, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    labels_per_hub.swap(recv_labels_per_hub); 

    //if (world_rank==0)
    //    std::cout << "successfully communicated per hub labels"<<std::endl;

    //prefix sum
    std::vector<lCounts> cumulative (spts_per_sync, 0);
    parallelPrefixSum (cumulative, labels_per_hub, 0, spts_per_sync, NUM_THREAD); 
    return cumulative;
}

// store number of labels per hub //
void computeGlobalMask (std::vector<unsigned int> &mask, int world_rank)
{
    unsigned int mask_size = mask.size();
    std::vector<unsigned int> global_mask (mask_size, 0);
    //gather and reduce mask with AND
    MPI_Allreduce(&mask[0], &global_mask[0], mask_size, MPI_UNSIGNED, MPI_BAND, MPI_COMM_WORLD);
    mask.swap(global_mask);
    //if (world_rank==0)
    //   std::cout << "successfully reduced and computed global prune mask" << std::endl;
}



Vertex findLastCommonSend (lCounts& COMMON_LABEL_BUDGET, std::vector<lCounts> &cumulative_label_counts, Vertex sync_thres, Vertex prev_sync_thres, unsigned int NUM_THREAD) {
    //special case - can't send even one vertex
    if ((cumulative_label_counts[0] > COMMON_LABEL_BUDGET) || (COMMON_LABEL_BUDGET==0))
    {
        COMMON_LABEL_BUDGET = 0;
        return prev_sync_thres;
    }
    Vertex spts_per_sync = sync_thres - prev_sync_thres;
    std::cout << " try  find last common send success "<<spts_per_sync<<std::endl;
    //find the least important vertex whose labels can still be counted
    Vertex lastV = sync_thres;
	volatile bool flag=false;
    #pragma omp parallel for num_threads(NUM_THREAD) shared(flag)
    for (unsigned int i=0; i<spts_per_sync-1; i++)
    {
        if (!flag && cumulative_label_counts[i]<=COMMON_LABEL_BUDGET && cumulative_label_counts[i+1]>COMMON_LABEL_BUDGET)
        {
            lastV = prev_sync_thres + i + 1;
			flag=true;
        }
    }
    std::cout << " flag success "<<std::endl;
    //recompute remaining budget
    if (cumulative_label_counts[spts_per_sync-1]>=COMMON_LABEL_BUDGET)
        COMMON_LABEL_BUDGET = 0;
    else
        COMMON_LABEL_BUDGET -= cumulative_label_counts[spts_per_sync-1];
    std::cout << " before return  "<<std::endl;

    return lastV;
}


Vertex findLastSend (lCounts& TOTAL_MEM_BUDGET, std::vector<lCounts> &cumulative_label_counts, Vertex sync_thres, Vertex prev_sync_thres, unsigned int NUM_THREAD) {
    Vertex spts_per_sync = sync_thres - prev_sync_thres;
    //find the least important vertex whose labels can still be counted
    unsigned int lastV = sync_thres;
	
	volatile bool flag=false;
    #pragma omp parallel for num_threads(NUM_THREAD) shared(flag)
    for (Vertex i=0; i<spts_per_sync-1; i++)
    {
        if (!flag && cumulative_label_counts[i]<=TOTAL_MEM_BUDGET && cumulative_label_counts[i+1]>TOTAL_MEM_BUDGET)
        {
            lastV = prev_sync_thres + i + 1;
            flag=true;
        }
    }
    return lastV;
}

void updateCumulativeCounts(std::vector<lCounts> &cumulative_label_counts, Vertex lastV, Vertex prev_sync_thres)
{
    assert(lastV > prev_sync_thres);
    Vertex last_sent_id = lastV - prev_sync_thres - 1;
    lCounts last_sent_num = cumulative_label_counts[last_sent_id];
    #pragma omp parallel for
    for (size_t i = 0; i < cumulative_label_counts.size(); i++)
    {
        if (i <= last_sent_id)
            cumulative_label_counts[i] = 0;
        else
            cumulative_label_counts[i] -= last_sent_num;
    } 
}


std::vector<int> gatherAllLabels (const std::vector<int> &label_list, std::vector<int> &recv_buffer, int world_size, int world_rank)
{
    std::vector<int> world_node_offsets (world_size, 0);
    std::vector<int> world_node_list_sizes (world_size);
    unsigned int local_label_count = label_list.size();
    MPI_Allgather(&local_label_count, 1, MPI_UNSIGNED, &world_node_list_sizes[0], 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    int label_global_sum = prefixSumExc (world_node_offsets, world_node_list_sizes, 0, world_size);

    //if(world_rank==0) 
    //    std::cout << "label global sum " << label_global_sum << " local_sum " << world_node_list_sizes[world_rank]  << std::endl;
    
    recv_buffer.resize(label_global_sum);

    //if(world_rank==0) 
    //    std::cout << "allocation successful" << std::endl;

    MPI_Allgatherv(&label_list[0], world_node_list_sizes[world_rank], MPI_INT, &recv_buffer[0], &world_node_list_sizes[0], &world_node_offsets[0], MPI_INT, MPI_COMM_WORLD);
     
    //if(world_rank==0) 
    //    std::cout << "gather successful" << std::endl;
    
    return world_node_offsets;

} 
}
