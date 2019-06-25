// This file contains implementation of Hybrida PLaNT + PLL algorithm
// for hub labeling on distributed machines.
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu


#pragma once

#include "graph.hpp"
#include "dijkstra.hpp"
#include "labeling.hpp"
//#include "utils.hpp"
//#include "comm_funcs.hpp"
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <map>
#include <assert.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>

namespace hl {
class Akiba : BasicDijkstra {
public:
    Vertex get_parent(Vertex v) { return parent[v]; }        // v's parent in the shortest path tree

    void plant(size_t i, bool forward, std::vector<Vertex> &order, std::vector<unsigned> &rev_map, Labeling &local_labeling) {
        clear();
        Vertex v = order[i];
        Vertex hub;
        update(v, 0, 1);
        Distance d;
        unsigned int queue_left = 1;

        while (!queue.empty()) {
            Vertex u = queue.pop();
            bool via_hub = (get_parent(u)==0) || (rev_map[u]<i);
            if (!via_hub)
                queue_left--;
            d = distance[u];
            if (!via_hub)
            {
                local_labeling.add(u, !forward, i, d);
            }
            for (Graph::arc_iterator a = g->begin(u, forward), end = g->end(u, forward); a < end; ++a) {
                Distance dd = d + a->length;
                assert(dd > d && dd < infty);
                Vertex parent = (via_hub || rev_map[a->head]<i) ? 0 : 1;
                if (dd <= distance[a->head]) {
                    queue_left += update_wo_prune(a->head, dd, parent);
                }
            }
            if (queue_left==0) break;
        }

}



Akiba(Graph &g) : BasicDijkstra(g){}


};

void run(Graph* g, std::vector<Vertex> &order, std::vector<unsigned> &rev_map, Labeling &labeling, int NUM_THREAD) {
    int N=order.size();
	//std::cout<<"NUM_THREAD"<<NUM_THREAD<<std::endl;

    labeling.clear(NUM_THREAD);

    bool* lck;
    lck = new bool[2*N]();
    std::vector<hl::Akiba> ak;
    for(int i = 0; i < NUM_THREAD; i++)
        ak.push_back(hl::Akiba(*g));

    //MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double start_plant, end_plant, plant_time;
    double end_sort, total_time = 0;
    start_plant = omp_get_wtime();
    int cnt = world_rank;

    #pragma omp parallel for num_threads(NUM_THREAD)
    for (int th = 0; th<NUM_THREAD; th++) {
        int tid = omp_get_thread_num();
        size_t local_cnt = __sync_fetch_and_add(&cnt, world_size);
        unsigned thread_sum = 0;
        while (local_cnt<N) {
            ak[tid].plant(local_cnt, true, order, rev_map, labeling);
            ak[tid].plant(local_cnt, false, order, rev_map, labeling);
            local_cnt = __sync_fetch_and_add(&cnt, world_size);
        }
    }
    
    float avg_size = labeling.get_avg();
    float sum_size;
    MPI_Reduce(&avg_size, &sum_size, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    labeling.sort(NUM_THREAD);

    end_sort = omp_get_wtime();
    total_time = end_sort - start_plant;
    if(world_rank==0) std::cout << sum_size <<" "<< total_time << std::endl;

}

}
