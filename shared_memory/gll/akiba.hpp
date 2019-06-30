// This file contains GLL algorithm for hub labeling that efficiently parallelizes
// PLL algorithm by Akiba et al.
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu
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
#include <vector>
#include <cassert>
#include <iostream>
#include <map>
#include <assert.h>
#include <immintrin.h>


namespace hl {
// Akiba et. al. 'pruned labeling' algorithm implementation
class Akiba : BasicDijkstra {

    // Add i'th vertex from the order into the labels of reachable vertices
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
    
    unsigned int iteration_gll(size_t i, bool forward, bool* lck, std::vector<Vertex> &order, std::vector<Vertex> &revMap, Labeling &labeling, Labeling &local_labeling) {
        clear();
        clear_tmp();
        Vertex v = order[i];
        int lcknum = ((forward)?1:0)+v*2;
        unsigned int label_count=0;
        update(v, 0);

        while (!__sync_bool_compare_and_swap(&lck[lcknum], false, true)) {}
        unsigned v_label_size=local_labeling.label_v[v][forward].size();
         
        for (size_t la=0;la<v_label_size;la++)  {
            if (local_labeling.label_v[v][forward][la]>i) continue;
            update_tmp(local_labeling.label_v[v][forward][la], local_labeling.label_d[v][forward][la]);
        }
        
        lck[lcknum]=false;
        
        for (size_t la=0;la<labeling.label_v[v][forward].size();la++)  
            update_tmp(labeling.label_v[v][forward][la],labeling.label_d[v][forward][la]);
        
        Vertex hub;
        Distance d;

        while (!queue.empty()) {
            Vertex u = queue.pop();
            d = distance[u];
            if (revMap[u]<i) continue;

            for (size_t la=0;la<labeling.label_v[u][!forward].size(); la++) {
                hub = labeling.label_v[u][!forward][la];
                if (hub>i) continue;
                if (label_tmp[hub]!=infty && 
                    label_tmp[hub]+labeling.label_d[u][!forward][la]<= d) {
                     goto pruned;} 
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

            local_labeling.add_lockfree(u, !forward, i, d);
            lck[lcknum]=false;
            label_count++;
            
            for (Graph::arc_iterator a = g->begin(u, forward), end = g->end(u, forward); a < end; ++a) {
                Distance dd = d + a->length;
                assert(dd > d && dd < infty);
                if (dd < distance[a->head]) 
                     update(a->head, dd);
            }
            pruned: {}
       }
       return label_count;
    }


    Akiba(Graph &g) : BasicDijkstra(g), label_tmp(g.get_n(), infty), is_dirty_tmp(g.get_n()){}
};

// Buld HHL from a vertex order
void run_paraPLL(Graph* g, std::vector<Vertex> &order, std::vector<Vertex> &revMap, Labeling &local_labeling, Labeling &labeling, size_t NUM_THREAD, size_t label_limit, bool isParaPLL) {
    int N=order.size();
    labeling.clear(NUM_THREAD);
    bool* lck = new bool[2*N+2]();

    std::vector<hl::Akiba> ak;
    for(int i = 0; i < NUM_THREAD; i++)
      ak.push_back(hl::Akiba(*g));

    ///////////////   Timing //////////////////////////
    double ls, le, se, ce; 
    double lt, st, ct;
    lt = 0;
    st = 0;
    ct = 0;


    unsigned int cnt = 0;

    ls = omp_get_wtime();
    while (cnt<2*N) {
        unsigned int label_sum = 0; 

        //clear the local labeling before start
        local_labeling.clear(NUM_THREAD);
        ls = omp_get_wtime();
        #pragma omp parallel for num_threads(NUM_THREAD) schedule (static,1) 
        for (int th = 0; th < NUM_THREAD; th++) {
            unsigned int tid = omp_get_thread_num();

            //count the number of labels generated by the thread
            unsigned int label_sum_thread=0;

            while (true) {
                unsigned int r = __sync_fetch_and_add(&cnt, 2);
                if (r>=2*N)
                {
                    __sync_fetch_and_add(&label_sum, label_sum_thread);
                    break;
                }
                size_t root = r/2;
                bool side = ((r % 2) == 1);
                unsigned int label_count = ak[th].iteration_gll(root, side, lck, order, revMap, labeling, local_labeling);
                label_sum_thread+=label_count;
                side = ((r % 2) == 0);
                label_count = ak[th].iteration_gll(root, side, lck, order, revMap, labeling, local_labeling);
                label_sum_thread+=label_count;

                if(label_sum_thread>N/32) {
                    if (__sync_fetch_and_add(&label_sum, label_sum_thread)>=label_limit-label_sum_thread)  break;
                    label_sum_thread=0;
                }
             }
        }
        le = omp_get_wtime();
        lt += le-ls;
         ////////////////// sort //////////////////////
        local_labeling.sort(NUM_THREAD);
        lt = le-ls;


        ///////////////// cleaning ///////////////////
        se = omp_get_wtime();
        st += se - le;
      //  printf("time for sorting %d trees = %lf \n", cnt/2, (se-le)*1000);

#pragma omp parallel for num_threads(NUM_THREAD) schedule (dynamic, NUM_THREAD)
        for (size_t vertex_i = 0; vertex_i < N; vertex_i++) {
             for (int side = 0; side < 2; ++side) {
                for (size_t i = 0; i < local_labeling.label_v[vertex_i][side].size(); ++i) {
                        size_t hub_order = local_labeling.label_v[vertex_i][side][i];
                        hl::Vertex hub = order[hub_order];
                        hl::Distance hub_dist = local_labeling.label_d[vertex_i][side][i];
                        if (!local_labeling.clean_cover(vertex_i, hub, side, hub_dist, hub_order))
                            labeling.add_lockfree(vertex_i, side, hub_order, hub_dist);
                }   
             }
        } 

        ce = omp_get_wtime();
        ct += ce - se;
        }

    std::cout << "Average label size after gll is " << labeling.get_avg() <<", labeling time = " << lt << ", sorting time = " << st << ", cleaning time = " << ct << ", total time = " << lt+st+ct << std::endl;
    std::cout << labeling.get_avg() <<" " << lt+st+ct << std::endl;
    
}

}
