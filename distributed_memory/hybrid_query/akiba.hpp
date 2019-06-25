// This file contains implementation of Hybrida PLaNT + PLL algorithm
// for hub labeling on distributed machines.
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu

#pragma once

#include "graph.hpp"
#include "dijkstra.hpp"
#include "labeling.hpp"
#include "utils.hpp"
#include "comm_funcs.hpp"
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <map>
#include <assert.h>
#include <immintrin.h>
#include <mpi.h>
#include <memory.h>

namespace hl {
class Akiba : BasicDijkstra {
    private:
    unsigned int MASK_INIT_VAL;
    std::vector<Distance> label_tmp;
    std::vector<bool> is_dirty_tmp;
    std::vector<Vertex> dirty_tmp;

    public:

    void update_tmp(Vertex v, Distance d, Vertex p = none) {
        label_tmp[v] = d;
        if (!is_dirty_tmp[v]) { dirty_tmp.push_back(v); is_dirty_tmp[v] = true; }
    }

    // Clear internal structures
    void clear_tmp() {
       for(size_t i = 0; i < dirty_tmp.size(); ++i) {
           label_tmp[dirty_tmp[i]] = infty;
           is_dirty_tmp[dirty_tmp[i]] = false;
        }
        dirty_tmp.clear();
        dirty_tmp.reserve(g->get_n());
    }

    Vertex get_parent(Vertex v) { return parent[v]; }        // v's parent in the shortest path tree

    //
    std::pair<unsigned int, unsigned int> plant(size_t i, bool forward, std::vector<Vertex> &order, std::vector<unsigned> &rev_map, Labeling &labeling, Labeling &local_labeling, Labeling &common_labeling) {
        clear();
        clear_tmp();
        Vertex v = order[i];
        Vertex hub;
        update(v, 0, 1);
        //if not go through via a more important node (not pruned), the parent is 1.
        Distance d;
        unsigned int label_count = 0;
        unsigned int tree_size = 0;
        unsigned int queue_left = 1;
        //std::cout << " Start Tree:  "<<i<<std::endl;

        for (size_t la=0;la<common_labeling.label_v[v][forward].size();la++)
        //std::cout << " la  "<<la<<std::endl;
            update_tmp(common_labeling.label_v[v][forward][la],common_labeling.label_d[v][forward][la]);
        while (!queue.empty()) {
            Vertex u = queue.pop();
            tree_size++;
            bool via_hub = (get_parent(u)==0) || (rev_map[u]<i);
            if (!via_hub)
                queue_left--;
            d = distance[u];
            for (size_t la=0;la<common_labeling.label_v[u][!forward].size(); la++) {
                hub = common_labeling.label_v[u][!forward][la];
                if (hub>i) break;
                if (label_tmp[hub]!=infty &&
                    label_tmp[hub]+common_labeling.label_d[u][!forward][la]<= d) {  goto pruned;}
            }
            if (!via_hub)
            {
                local_labeling.add(u, !forward, i, d);
                label_count++;
            }
            for (Graph::arc_iterator a = g->begin(u, forward), end = g->end(u, forward); a < end; ++a) {
                Distance dd = d + a->length;
                assert(dd > d && dd < infty);
                Vertex parent = (via_hub || rev_map[a->head]<i) ? 0 : 1;
                if (dd <= distance[a->head]) {
                    queue_left += update_wo_prune(a->head, dd, parent);
                }
            }
            pruned: {}
            if (queue_left==0) break;
        }

        //<UPDATE> return both label count and tree size
        std::pair <unsigned int, unsigned int> sizes;
        sizes.first     = label_count;
        sizes.second    = tree_size;

        return sizes;
}


unsigned int iteration_buffer(size_t i, bool forward, std::vector<Vertex> &order, std::vector<unsigned> &rev_map, Labeling &labeling, Labeling &local_labeling, Labeling &common_labeling, bool* lck) {
    clear();
    clear_tmp();
    Vertex v = order[i];
    Vertex hub;
    int lcknum = ((forward)?1:0)+v*2;
    update(v, 0, 1);
    //if not go through via a more important node (not pruned), the parent is 1.
    Distance d;
    unsigned label_count = 0;
    //std::cout << " Start Normal Tree:  "<<i<<std::endl;

    //std::cout << " lck "<<lck[lcknum]<<std::endl;
    while (!__sync_bool_compare_and_swap(&lck[lcknum], false, true)) {}
    //std::cout << " aquire  success "<<lcknum<<std::endl;
    unsigned v_label_size=local_labeling.label_v[v][forward].size();
    //std::cout << " v_label_size  "<<v_label_size<<std::endl;
    for (size_t la=0;la<v_label_size;la++)  {
        if (local_labeling.label_v[v][forward][la]>i) continue;
        update_tmp(local_labeling.label_v[v][forward][la], local_labeling.label_d[v][forward][la]);
    }
    lck[lcknum]=false;
    //std::cout << " Copy local  "<<i<<std::endl;
    for (size_t la=0;la<labeling.label_v[v][forward].size();la++)
        update_tmp(labeling.label_v[v][forward][la],labeling.label_d[v][forward][la]);
    for (size_t la=0;la<common_labeling.label_v[v][forward].size();la++)
        update_tmp(common_labeling.label_v[v][forward][la],common_labeling.label_d[v][forward][la]);
    //std::cout << " Copy global  "<<i<<std::endl;
    while (!queue.empty()) {
        Vertex u = queue.pop();
        if (rev_map[u] < i)
            continue;
        d = distance[u];
        for (size_t la=0;la<common_labeling.label_v[u][!forward].size(); la++) {
            hub = common_labeling.label_v[u][!forward][la];
            if (hub>i) continue;
            if (label_tmp[hub]!=infty &&
                    label_tmp[hub]+common_labeling.label_d[u][!forward][la]<= d) {  goto pruned;}
        }
        for (size_t la=0;la<labeling.label_v[u][!forward].size(); la++) {
            hub = labeling.label_v[u][!forward][la];
            if (hub>i) continue;
            if (label_tmp[hub]!=infty &&
                    label_tmp[hub]+labeling.label_d[u][!forward][la]<= d) { goto pruned;}
        }
        lcknum=((forward)?0:1)+u*2;
        while (!__sync_bool_compare_and_swap(&lck[lcknum], false, true)) {}
        for (size_t la=0;la<local_labeling.label_v[u][!forward].size(); la++) {
            hub = local_labeling.label_v[u][!forward][la];
            if (hub>i) continue;
            if (label_tmp[hub]!=infty &&
                label_tmp[hub]+local_labeling.label_d[u][!forward][la]<= d) {
                    lck[lcknum]=false;
                    goto pruned;}
        }

        local_labeling.add_lockfree(u, !forward, i, d); label_count++;
        //local_labeling.add_lockfree(u, !forward, i, d); label_count++;
        lck[lcknum]=false;

        for (Graph::arc_iterator a = g->begin(u, forward), end = g->end(u, forward); a < end; ++a) {
            Distance dd = d + a->length;
            assert(dd > d && dd < infty);
            if (dd < distance[a->head]) {
                update(a->head, dd, 0);
            }

        }
        pruned: {}
   }

//std::cout << " END Normal Tree:  "<<i<<" Label: "<<label_count<<std::endl;
return label_count;
}


// initialize MASK_INIT_VAL
Akiba(Graph &g) : BasicDijkstra(g), label_tmp(g.get_n(), infty), is_dirty_tmp(g.get_n()){MASK_INIT_VAL = ~0;}


};

//void run_paraPLL(Graph* g, std::vector<Vertex> &order, Labeling &labeling, float sync, int NUM_THREAD) {
void run(Graph* g, std::vector<Vertex> &order, std::vector<unsigned> &rev_map, Labeling &labeling, float sync, float common_label_budget, float phase_switch, int NUM_THREAD) {
int N=order.size();
labeling.clear();
std::vector<int> label_list;
hl::Labeling local_labeling(N);
hl::Labeling common_labeling(N);

// set the number of threads//
//no need to pass it to each and every function//
//    std::cout << " labeling start   "<<std::endl;
omp_set_dynamic(0);
omp_set_num_threads(NUM_THREAD);

// set budgets on memory for recv_buffer and common_labels//
//set the threshold for switching from phase1 to phase2 - depends on graph/number of machines//
//The code assumes that MEM_BUDGET is greater than COMMON_LABEL_BUDGET//
   // MEMORYSTATUS MemStat;
   // memset(&MemStat, 0, sizeof(MemStat));
   // ::GlobalMemoryStatus(&MemStat);
   //  std::cout << "Length of structure: " << MemStat.dwLength
   //         << std::endl
   //         << "Memory usage: " << MemStat.dwMemoryLoad
   //         << " %" << std::endl
   //         << "Physical memory: " << MemStat.dwTotalPhys / 1024
   //         << " KB" << std::endli;
    lCounts MEM_BUDGET = 650000000; //<2 GB
    lCounts COMMON_LABEL_BUDGET = std::min( (unsigned) (N*2*16), (unsigned) (1000000 * common_label_budget));
    Vertex PHASE_SWITCH_THRESH = N;
    double PHASE_SWITCH_RATIO = phase_switch;

    //
    //First phase - no pruning
    bool do_plant               = true;
    //<UPDATE> flag to indicate state in phase1
    bool switch_compute     	= true; //haven't found the dynamic switch point yet
    //Initial labels into common labeling
    bool COMM_DONE = false;

    bool* lck = new bool[2*N+2]();
    std::vector<hl::Akiba> ak;
    for(int i = 0; i < NUM_THREAD; i++)
        ak.push_back(hl::Akiba(*g));

    int provided;
    //std::cout << " before MPI INIT   "<<std::endl;
    //MPI_Init_thread(NULL, NULL);
    MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, &provided);
    if(provided!=MPI_THREAD_MULTIPLE) {
        std::cout<<"NOT SUPPORTED"<<std::endl;
        return ;
    }
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //<UPDATE>create window for put//
    //std::cout << " before window   "<<world_size<<std::endl;

   	int *local_phase_sync = new int [1](); //a local thread crossed phase switch thresh
    int *global_phase_sync;                //a remote thread crossed phase switch thresh
    MPI_Alloc_mem(sizeof(int)*1, MPI_INFO_NULL, &global_phase_sync);
    local_phase_sync[0] = 0;
    global_phase_sync[0] = 0;
    int *global_check_tmp = new int[1]();
    global_check_tmp[0] = 0;

    MPI_Win win;
    MPI_Win_create((void*)global_phase_sync, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);

    if(world_rank==0) std::cout << " world size="<<world_size<<" Num of threads per node="<<NUM_THREAD<<std::endl;
    if(world_rank==0) std::cout << " switch_ratio="<<PHASE_SWITCH_RATIO<<" common label budget="<<COMMON_LABEL_BUDGET<<std::endl;
    unsigned label_local_sum    = 0;
    unsigned label_global_sum   = 0;
    int local_sum_array[world_size];
    int prefix_sum[world_size];

    // store previous synced value as well//
    Vertex sync_thres, prev_sync_thres;
    sync_thres = std::max((int)(N * sync), world_size*NUM_THREAD);
    sync_thres = std::min(sync_thres, PHASE_SWITCH_THRESH);
    sync_thres = world_size*NUM_THREAD;
    sync_thres = N;
    prev_sync_thres = 0;


    // last location till which global/common labeling is sorted//
    std::vector<int> labeling_last_loc(2*N, 0);
    std::vector<int> common_last_loc(2*N, 0);

    double start_time, end_time, total_time;
    double clean_start, clean_end, clean_time = 0;
    start_time = omp_get_wtime();

    int cnt = world_rank;

    while (cnt < N) {
        //if(world_rank==0) std::cout << " Threshold =  "<<sync_thres<<std::endl;
        // individual labels per hub//
        std::vector<int> labels_per_hub (sync_thres-prev_sync_thres,0);

        build_trees: {}
        #pragma omp parallel for num_threads(NUM_THREAD)
        for (int th = 0; th<NUM_THREAD; th++) {
            int tid = omp_get_thread_num();
            size_t local_cnt = __sync_fetch_and_add(&cnt, world_size);
            unsigned thread_sum = 0;
   			int *put_tmp = new int [1](); //avoid the data race by using local_phase_switch
			put_tmp[0] = 0;
			bool get_tmp = false;
            // store labels per hub individually//
            while (local_cnt<sync_thres && local_cnt<N) {
				//std::cout<<"Tree "<<local_cnt<<std::endl;
                if (do_plant) {
                    std::pair <unsigned int, unsigned int> sizes1   =   ak[tid].plant(local_cnt, true, order, rev_map, labeling, local_labeling, common_labeling);
                    std::pair <unsigned int, unsigned int> sizes2   =   ak[tid].plant(local_cnt, false, order, rev_map, labeling, local_labeling, common_labeling);
                    unsigned int label_size                         =   sizes1.first + sizes2.first;
                    unsigned int tree_size                          =   sizes1.second + sizes2.second;
                    labels_per_hub[local_cnt-prev_sync_thres]       +=  label_size;
                    //<UPDATE> compute ratio for dynamic switching
                    double size_ratio                               =   ((double)tree_size)/((double)label_size);
					//if(world_rank==0 && tid==0) std::cout<<"node 0, thread 0, Tree "<<local_cnt<<" ratio: "<<size_ratio<<std::endl;
                    if (switch_compute)
                    {
                        if (size_ratio > PHASE_SWITCH_RATIO)
                        {
                            local_phase_sync[0] = 1;
                        }
                        if (tid==0) {
				       	  bool global_check_res;
				   	   	  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, world_rank, 0, win);
				   	      global_check_res = (global_phase_sync[0] == 1);
				   	   	  MPI_Win_unlock(world_rank, win);
                          if (global_check_res) {
                            local_phase_sync[0] = 1;
                          }
                        
                          else if (local_phase_sync[0] == 1){
                                global_phase_sync[0] = 1;
								for (size_t node_id=0; node_id<world_size; node_id++) {
									if (node_id == world_rank) continue;
									MPI_Win_lock(MPI_LOCK_EXCLUSIVE, node_id, 0, win);
									//std::cout<<"node " <<world_rank<<" write to node "<<node_id<<", Tree "<<local_cnt<<", ratio: "<<size_ratio<<std::endl;
									MPI_Put(&global_phase_sync[0], 1, MPI_INT, node_id, 0, 1, MPI_INT, win);
									MPI_Win_unlock(node_id, win);
									//std::cout<<" Success! node " <<world_rank<<" write to node "<<node_id<<", Tree "<<local_cnt<<", ratio: "<<size_ratio<<std::endl;
								}
                           }
                        }
                       	if (local_phase_sync[0]==1 )
				   	 	{
                            //std::cout<<"node: "<<world_rank<<" local "<<local_phase_sync[0] <<" global "<< global_phase_sync[0]<<std::endl;
							//std::cout<<"node "<<world_rank<<" global_value is true. "<<std::endl;
                            thread_sum += label_size;
                            break;
                       }
                   	}
                }
                else {
                    labels_per_hub[local_cnt-prev_sync_thres]       +=  ak[tid].iteration_buffer(local_cnt, true, order, rev_map, labeling, local_labeling, common_labeling, lck);
                    labels_per_hub[local_cnt-prev_sync_thres]       +=  ak[tid].iteration_buffer(local_cnt, false, order, rev_map, labeling, local_labeling, common_labeling, lck);
                }
                thread_sum += labels_per_hub[local_cnt-prev_sync_thres]; //<BUG_FIX> offset not subtracted
                //if (local_cnt%25==0)
                    //std::cerr << "\r  "<<local_cnt<<"/"<<N;
                //std::cout <<local_cnt<<"/"<<N<<std::endl;
                local_cnt = __sync_fetch_and_add(&cnt, world_size);
            }
            __sync_fetch_and_add(&label_local_sum, thread_sum);
			
            //if(cnt >= sync_thres && sync_thres<N) __sync_fetch_and_add(&cnt, -world_size);
            if(cnt >= sync_thres ) __sync_fetch_and_add(&cnt, -world_size);
        }
    MPI_Win_fence(0, win);

		if (local_phase_sync[0]==1 && global_phase_sync[0]==0)
		{
									//std::cout<<"node " <<world_rank<<std::endl;
            global_phase_sync[0] = 1;
			for (size_t node_id=0; node_id<world_size; node_id++) {
				if (node_id == world_rank) continue;
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, node_id, 0, win);
				//std::cout<<"node " <<world_rank<<" write to node "<<node_id<<", Tree "<<local_cnt<<", ratio: "<<size_ratio<<std::endl;
				MPI_Put(&global_phase_sync[0], 1, MPI_INT, node_id, 0, 1, MPI_INT, win);
				MPI_Win_unlock(node_id, win);
				//std::cout<<" Success! node " <<world_rank<<" write to node "<<node_id<<", Tree "<<local_cnt<<", ratio: "<<size_ratio<<std::endl;
			}
		}
        //<UPDATE>// if phase switched, find the max tree done by any node
        if (do_plant && switch_compute && (global_phase_sync[0]==1))
        {
            //find the max tree to compute. allreduce
								//	std::cout<<"out of plant " <<world_rank<<std::endl;
            int max_cnt;
            MPI_Allreduce(&cnt, &max_cnt, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            //update sync_thres
            sync_thres          = max_cnt;
			std::cout<<"Max tree done before switch "<<max_cnt<<std::endl;
            //complete remaining trees and then switch phases
            PHASE_SWITCH_THRESH = max_cnt;
            //switch_compute =false
            switch_compute      = false;
            //goto just_before_for_loop
            goto build_trees;
        }

        clean_start = omp_get_wtime();
        local_labeling.sort(NUM_THREAD);

        // find out accumulated label counts generated in this sync//
        std::vector<lCounts> cumulative_label_counts = accLabelCountsGen(labels_per_hub, world_rank, NUM_THREAD);

        //std::cout << " cum label count success "<<std::endl;
        // find out last vertex that can be sent//
        Vertex lastV, lastCV;
        lastCV = findLastCommonSend(COMMON_LABEL_BUDGET, cumulative_label_counts, sync_thres, prev_sync_thres, NUM_THREAD);

        //std::cout << " successfully calculate last common send  "<<lastCV<<std::endl;

        // initialize offset vector to denote how many local labels have been loaded/processed //
        //local_load_offset[2*x] -> offset of x in backward (false) direction
        std::vector<int> local_load_offset (2*N, 0);
        std::vector<int> recv_buffer;



        // create label_list, all gather label_list, prune label_list <optional>, load into common/global labeling, load remaining vertices into global labeling <optional>
        if (do_plant) //no need to prune. Last hub to be used - lastCV
        {
            if (lastCV > prev_sync_thres) //at least one hub to load
            {
                do
                {
                    //std::cout << " try  find last send success "<<std::endl;
                    lastV = findLastSend(MEM_BUDGET, cumulative_label_counts, sync_thres, prev_sync_thres, NUM_THREAD);
                    //std::cout << " find last send success "<< lastV<<std::endl;
                    parallelLoadWOffset(label_list, local_labeling, local_load_offset, std::min(lastCV, lastV), NUM_THREAD);
                    //std::cout << " Load W offset success "<<std::endl;
                    int label_local_iter_sum = label_list.size();
                    //std::cout << "label_local_iter_sum "<<label_local_iter_sum<<std::endl;
                    std::vector<int> world_node_offsets = gatherAllLabels(label_list, recv_buffer, world_size, world_rank);
                    //std::cout << "Gather all labels success "<<std::endl;
                    int label_global_iter_sum = recv_buffer.size();
                    //std::cout << "label_global_iter_sum "<<label_global_iter_sum<<std::endl;
                    std::vector<unsigned int> dummy_mask (1);
                    loadFromRecvBuffer(recv_buffer, dummy_mask, false, 0, label_global_iter_sum, world_node_offsets[world_rank], world_node_offsets[world_rank]+label_local_iter_sum, lastCV, common_labeling, labeling);
                    //std::cout << "loadfrom recv buffer success"<<std::endl;
                    updateCumulativeCounts(cumulative_label_counts, std::min(lastCV, lastV), prev_sync_thres);
                    //std::cout << " common once success "<<std::endl;
                }
                while(lastV < lastCV);

                //std::cout << " common_labeling size "<<common_labeling.get_avg()<<std::endl;
                //std::cout << " label check "<<common_labeling.check_label()<<std::endl;
                common_labeling.sort_partial(NUM_THREAD, common_last_loc);
                //common_labeling.sort(NUM_THREAD);
                //std::cout << " sort common lableing success "<<std::endl;
            }
            //Move hubs that were not broadcasted to global labeling//
            if (lastCV < sync_thres)
            {
                 moveLocalToGlobal(labeling, local_labeling, local_load_offset); //<TODO> not implemented yet
                 COMM_DONE = true; //common labels have been stored
                 labeling.sort_partial(NUM_THREAD, labeling_last_loc);
                 //labeling.sort(NUM_THREAD);
            }
        }
        else //need pruning. Last hub to be used - lastV
        {
            do
            {
                lastV = findLastSend(MEM_BUDGET, cumulative_label_counts, sync_thres, prev_sync_thres, NUM_THREAD);
                //std::cout << " filtering: find last send "<< lastV<<std::endl;
                parallelLoadWOffset(label_list, local_labeling, local_load_offset, lastV, NUM_THREAD);
                //std::cout << " filtering: parallel load offset "<<std::endl;
                int label_local_iter_sum = label_list.size();
                //std::cout << " filtering: label_local_sum "<< label_local_iter_sum<<std::endl;
                std::vector<int> world_node_offsets = gatherAllLabels(label_list, recv_buffer, world_size, world_rank);
                int label_global_iter_sum = recv_buffer.size();
                //std::cout << " filtering: label_global_sum "<< label_global_iter_sum<<std::endl;
                int mask_size =(label_global_iter_sum/3-1)/32+1;
                std::vector<unsigned int> mask (mask_size, ~0);
                computeLocalMask(mask, recv_buffer, 0, label_global_iter_sum, order, labeling, local_labeling, world_rank);
                //std::cout << " filtering: compute local mask "<<std::endl;
                computeGlobalMask(mask, world_rank);
                //std::cout << " filtering: compute global mask "<<std::endl;

                if (COMM_DONE) //if common labels are done, only look at your own list
                    loadFromRecvBuffer(recv_buffer, mask, true, world_node_offsets[world_rank], world_node_offsets[world_rank]+label_local_iter_sum, world_node_offsets[world_rank], world_node_offsets[world_rank]+label_local_iter_sum, 0, common_labeling, labeling);
                else //look at all labels
                    loadFromRecvBuffer(recv_buffer, mask, true, 0, label_global_iter_sum, world_node_offsets[world_rank], world_node_offsets[world_rank]+label_local_iter_sum, lastCV, common_labeling, labeling);

                COMM_DONE = (lastV >= lastCV) && (lastCV < sync_thres);
                updateCumulativeCounts(cumulative_label_counts, lastV, prev_sync_thres);
            }
            while(lastV < sync_thres);
            if (lastCV < sync_thres) //something went into global labeling
                labeling.sort_partial(NUM_THREAD, labeling_last_loc);
            if (lastCV > prev_sync_thres) //something went into common labeling
                common_labeling.sort_partial(NUM_THREAD, common_last_loc);
        }
        if (!COMM_DONE)
            COMM_DONE = (COMMON_LABEL_BUDGET==0); //done if common label budget is exhausted
        ///// Reset threshold  //////////
        prev_sync_thres     = sync_thres;

        //<UPDATE> new logic to compute synchronization points//
       	do_plant         	= (do_plant && switch_compute); //trees upto dynamic switching point created
		//if (!do_plant) std::cout<<" turn to PLL "<<" cnt:"<<cnt<<std::endl;
        //do_plant         	= true; //trees upto dynamic switching point created
        //if(do_plant) sync_thres   = N;
        //else  sync_thres          = std::min(sync_thres*2, (unsigned)N);
        sync_thres          = std::min(sync_thres*2, (unsigned)N);

        label_local_sum = 0;
        label_global_sum = 0;
        local_labeling.clear(NUM_THREAD);
        label_list.clear();

        // print averge labeling size
       // if(world_rank==0) std::cout << "Average label size after sync is " << labeling.get_avg() << std::endl;
       // if(world_rank==0) std::cout << "Common label size after sync is " << common_labeling.get_avg() << std::endl;
        clean_end = omp_get_wtime();
        clean_time += clean_end - clean_start;
        if(world_rank==0) std::cout << "Clean time  " << clean_end - clean_start << std::endl;
    }
    end_time = omp_get_wtime();
    total_time = end_time - start_time;

    float avg_size = labeling.get_avg();
    float sum_size;
    MPI_Reduce(&avg_size, &sum_size, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(world_rank==0) std::cout << "Final common label size is " << common_labeling.get_avg() << std::endl;
    if(world_rank==0) std::cout << "Final total global label size is " <<common_labeling.get_avg()+sum_size << std::endl;
    if(world_rank==0) std::cout << "Time cost  " << total_time << std::endl;
    if(world_rank==0) std::cout << "Clean time  " << clean_time << std::endl;

}

}
