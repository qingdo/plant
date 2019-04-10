// Akiba et al. presented a 'pruned labeling' algorithm to build Hierarchical Hub Labels from a vertex order.
// This file contains Akiba et. al. algorithm implementation
//
// Copyright (c) 2014, 2015 savrus
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "graph.hpp"
#include "dijkstra.hpp"
#include "labeling.hpp"
#include "comm_funcs.hpp"
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <map>
#include <assert.h>
#include <immintrin.h>
#include <mpi.h>
#include <cmath>

namespace hl {
// Akiba et. al. 'pruned labeling' algorithm implementation
class paraPLL : BasicDijkstra {
	public:
	std::vector<Distance> label_tmp;
	std::vector<bool> is_dirty_tmp;
	std::vector<Vertex> dirty_tmp;

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


    unsigned int iteration_buffer(size_t i, bool forward, std::vector<Vertex> &order, Labeling &labeling, Labeling &local_labeling, bool* lck) {
		clear_tmp();
        Vertex v = order[i];
		Vertex hub;
		int lcknum = ((forward)?1:0)+v*2;	
        update(v, 0);
        Distance d;
		unsigned label_count = 0;

		while (!__sync_bool_compare_and_swap(&lck[lcknum], false, true)) {}
		unsigned v_label_size=local_labeling.label_v[v][forward].size();
		for (size_t la=0;la<v_label_size;la++)  {
			update_tmp(local_labeling.label_v[v][forward][la], local_labeling.label_d[v][forward][la]);
		}
		lck[lcknum]=false; 
		for (size_t la=0;la<labeling.label_v[v][forward].size();la++)  
		    update_tmp(labeling.label_v[v][forward][la],labeling.label_d[v][forward][la]);
        while (!queue.empty()) {
            Vertex u = queue.pop();
            d = distance[u];
            for (size_t la=0;la<labeling.label_v[u][!forward].size(); la++) {
		    	hub = labeling.label_v[u][!forward][la];
			 	if (label_tmp[hub]!=infty && 
						label_tmp[hub]+labeling.label_d[u][!forward][la]<= d) {	goto pruned;} 
			}
            lcknum=((forward)?0:1)+u*2;
            while (!__sync_bool_compare_and_swap(&lck[lcknum], false, true)) {}
            for (size_t la=0;la<local_labeling.label_v[u][!forward].size(); la++) {
		    	hub = local_labeling.label_v[u][!forward][la];
				if (label_tmp[hub]!=infty && 
					label_tmp[hub]+local_labeling.label_d[u][!forward][la]<= d) {
						lck[lcknum]=false;
						goto pruned;} 
			}

	    	local_labeling.add_lockfree(u, !forward, i, d); 
			lck[lcknum]=false;
			label_count++;

            for (Graph::arc_iterator a = g->begin(u, forward), end = g->end(u, forward); a < end; ++a) {
                Distance dd = d + a->length;
                assert(dd > d && dd < infty);
                if (dd < distance[a->head]) {
						update(a->head, dd);
				}
					
            }
            pruned: {}
       }
	
	return label_count;
    }


    //Akiba(Graph &g) : BasicDijkstra(g){}
	paraPLL(Graph &g) : BasicDijkstra(g), label_tmp(g.get_n(), infty), is_dirty_tmp(g.get_n()){}


};

void run_paraPLL(Graph* g, std::vector<Vertex> &order, Labeling &labeling, int NUM_THREAD) {
    unsigned N=order.size();
    labeling.clear(NUM_THREAD);
    hl::Labeling local_labeling(N);

	bool* lck = new bool[2*N+2]();
    std::vector<hl::paraPLL> ak;
	for(int i = 0; i < NUM_THREAD; i++)
		ak.push_back(hl::paraPLL(*g));
	
    //MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //std::cout << " world size =  "<<world_size<<std::endl;
    unsigned label_local_sum = 0;
	int local_sum_array[world_size];
	int prefix_sum[world_size];
    unsigned label_global_sum = 0;
	unsigned NUM_SYNCH =(unsigned) (log(N)/log(8));
	int sync_thres;
	sync_thres = (N-1)/NUM_SYNCH + 1;
	
	double start_time, end_time, total_time;
	double clean_start, clean_end, clean_time = 0;
	start_time = omp_get_wtime();

    int cnt = world_rank;

    while (cnt < N) {
    	//std::cout << " Threshold =  "<<sync_thres<<std::endl;
        hl::Labeling local_labeling(N);
		#pragma omp parallel for num_threads(NUM_THREAD)
	    for (int th = 0; th<NUM_THREAD; th++) {
			int tid = omp_get_thread_num();
			size_t local_cnt = __sync_fetch_and_add(&cnt, world_size);
			unsigned thread_sum = 0;
			while (local_cnt<sync_thres && local_cnt<N) {
				thread_sum += ak[tid].iteration_buffer(local_cnt, true, order, labeling, local_labeling, lck);
				thread_sum += ak[tid].iteration_buffer(local_cnt, false, order, labeling, local_labeling,lck);
				local_cnt = __sync_fetch_and_add(&cnt, world_size);
			}
			__sync_fetch_and_add(&label_local_sum, thread_sum);
			
			if(sync_thres<N) __sync_fetch_and_add(&cnt, -world_size);
		}

        Vertex start = 0, end = N; 
        //share labelig across all machines
        Vertex single_start = start;
        Vertex single_end = start;
        std::vector<int> size_per_vertex = gather_size(local_labeling, start, end, NUM_THREAD);
        int MPI_BUDGET = 650000000;
        do {
            single_end = findLastSend(MPI_BUDGET, size_per_vertex, single_start, start, end, NUM_THREAD);
	        //if(world_rank==0)	std::cout<<"one-time comm range: "<<single_start<<"-"<<single_end<<" partition range"<<start<<"-"<<end<<std::endl;
	        std::vector<int>label_list;
	        std::vector<int>recv_buffer;
	        parallelLoad (label_list, local_labeling, single_start, single_end, NUM_THREAD);
	        gatherAllLabels(label_list, recv_buffer, world_size, world_rank);
	      	loadFromRecvBuffer(recv_buffer, labeling);
            single_start = single_end;
        } while(single_end < end);
        //share finished 
        sync_thres = std::min(sync_thres + (N-1)/NUM_SYNCH + 1, N);
    }
	labeling.sort(NUM_THREAD);
	end_time = omp_get_wtime();
	total_time = end_time - start_time;
    
	if (world_rank==0)std::cout <<" "<< labeling.get_avg() << " "<< total_time;
		

  //  std::cout << "Final average label size is " << labeling.get_avg() << std::endl;
  //  std::cout << "Time cost  " << total_time << std::endl;
  //  std::cout << "Clean time  " << clean_time << std::endl;
    
}

}
