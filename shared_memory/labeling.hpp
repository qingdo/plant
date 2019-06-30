// Hub Labels are lists of hubs and distances to them attached to every vertex in a graph.
// This file contains the class to store labels.
// Available methods include making a query, write labels to file, read labels from file,
// merge label tables, clean label tables.
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
#include <vector>
#include <limits>
#include <cassert>
#include <algorithm>
#include <utility>
#include <fstream>
#include <istream>
#include <omp.h>
namespace hl {

// Class to store labels
class Labeling {
public:
    std::vector< std::vector< std::vector<Vertex> > > label_v;     // Lists of forward/reverse hubs
    std::vector< std::vector< std::vector<Distance> > > label_d;   // Lists of distances to hubs
    Vertex n;
    bool *lck;

    Labeling(size_t n = 0) :
        label_v(n, std::vector< std::vector<Vertex> >(2)),
        label_d(n, std::vector< std::vector<Distance> >(2)),
        n(n),
        lck(new bool[n]()) {}
    // Find u-v distance
    Distance query(Vertex u, Vertex v, bool f = true) {
        Distance r = infty;
        for (size_t i=0, j=0; i < label_v[u][f].size() && j < label_v[v][!f].size();) {
            if (label_v[u][f][i] == label_v[v][!f][j]) {
                r = std::min(r, label_d[u][f][i++] + label_d[v][!f][j++]);
            } else if (label_v[u][f][i] < label_v[v][!f][j]) ++i;
            else ++j;
        }
        return r;
    }
    
    bool cover(Vertex u, Vertex v, bool f = true, Distance d=infty) {
        for (size_t i=0, j=0; i < label_v[u][f].size() && j < label_v[v][!f].size();) {
            if (label_v[u][f][i] == label_v[v][!f][j]) {
                if (d >= label_d[u][f][i++] + label_d[v][!f][j++]) return true;
            } else if (label_v[u][f][i] < label_v[v][!f][j]) ++i;
            else ++j;
        }
        return false;
    }

    
    inline bool clean_cover(Vertex u, Vertex v, unsigned int f, Distance d=infty, size_t hub_order=0) {
        for (size_t i=0, j=0; i < label_v[u][f].size() && j < label_v[v][!f].size();) {
            if (label_v[u][f][i] >= hub_order || label_v[v][!f][j] >= hub_order) return false;
            if (label_v[u][f][i] == label_v[v][!f][j]) {
                if (d >= label_d[u][f][i++] + label_d[v][!f][j++]) return true;
            } 
            else if (label_v[u][f][i] < label_v[v][!f][j]) ++i;
            else ++j;
        }
        return false;
    }

    inline void clean_roots(Vertex v, std::vector<Vertex> &order, unsigned int side)
    {
        std::vector<Vertex> temp_v;
        std::vector<Distance> temp_d;
        for (size_t i=0; i<label_v[v][side].size(); i++)
        {
            size_t hub_order = label_v[v][side][i];
            Vertex hub =  order[hub_order];
            Distance hub_dist = label_d[v][side][i];
            if (!clean_cover(hub, v, side, hub_dist, hub_order)); 
            {
                temp_v.push_back(hub_order);
                temp_d.push_back(hub_dist);
            }
        } 
        temp_v.swap(label_v[v][side]);
        temp_d.swap(label_d[v][side]);
    }

    // Add hub (v,d) to forward or reverse label of u
    inline void add(Vertex u, bool forward, Vertex v, Distance d) {
        
        while (!__sync_bool_compare_and_swap(&lck[u], false, true)) {} 
        label_v[u][forward].push_back(v);
        label_d[u][forward].push_back(d);
        lck[u]=false;
    }
    void add_lockfree(Vertex u, bool forward, Vertex v, Distance d) {
        label_v[u][forward].push_back(v);
        label_d[u][forward].push_back(d);
    }

    // Get labels
    std::vector< std::vector<Vertex> > &get_label_hubs(Vertex u) { return label_v[u]; }
    std::vector< std::vector<Distance> > &get_label_distances(Vertex u) { return label_d[u]; }

    // Get maximum label size
    size_t get_max() const {
        size_t max = 0;
        for (Vertex v = 0; v < n; ++v)
            for (int side = 0; side < 2; ++side)
                max = std::max(max, label_v[v][side].size());
        return max;

        //size_t maxVal = 0;
        ////#pragma omp parallel for num_threads (NUM_THREAD) reduction (max: maxVal)
        ////{
        //    for (Vertex v=0; v<n; v++)
        //    {
        //        for (unsigned side = 0; side < 2; side++)
        //        {
        //            if (label_v[v][side].size() < maxVal)
        //                maxVal = label_v[v][side].size;
        //        }
        //    }
        //}
        //return maxVal;



    }

    // Get average label size
    double get_avg() const {
        long long total = 0;
        for (Vertex v = 0; v < n; ++v)
            total += label_v[v][0].size() + label_v[v][1].size();
        return static_cast<double>(total)/n/2;
    }
    long long get_total() const {
        long long total = 0;
        for (Vertex v = 0; v < n; ++v)
            total += label_v[v][0].size() + label_v[v][1].size();
        return total;
    }

    // Write labels to file
    bool write(char *filename) {
        std::ofstream file;
        file.open(filename);
        file << n << std::endl;
        for (Vertex v = 0; v < n; ++v) {
            for (int side = 0; side < 2; ++side) {
                file << label_v[v][side].size();
                for (size_t i = 0; i < label_v[v][side].size(); ++i) {
                    file << " " << label_v[v][side][i];
                    file << " " << label_d[v][side][i];
                }
                file << std::endl;
            }
        }
        file.close();
        return file.good();
    }

    // Read labels from file
    bool read(char *filename, Vertex check_n = 0) {
        std::ifstream file;
        file.open(filename);
        file >> n;
        if (check_n && n != check_n) return false;
        label_v.resize(n, std::vector< std::vector<Vertex> >(2));
        label_d.resize(n, std::vector< std::vector<Distance> >(2));
        for (Vertex v = 0; v < n; ++v) {
            for (int side = 0; side < 2; ++side) {
                size_t s;
                file >> s;
                label_v[v][side].resize(s);
                label_d[v][side].resize(s);
                for (size_t i = 0; i < s; ++i) {
                    file >> label_v[v][side][i];
                    file >> label_d[v][side][i];
                }
            }
        }
        file >> std::ws;
        file.close();
        return file.eof() && !file.fail();
    }

    // Clear labels
    void clear(unsigned NUM_THREAD=72) { 
        #pragma omp parallel for num_threads(NUM_THREAD)
        for (Vertex v = 0; v < n; ++v) {
            for (int side = 0; side < 2; ++side) {
                label_v[v][side].clear();
                label_d[v][side].clear();
            }
        }
    }

    // Sort labels before making queries
    void sort(unsigned NUM_THREAD) {

        // std::vector<std::vector<std::pair<Vertex, Distance>>> label(NUM_THREAD)
        // maxSize = get_max();
        // for (int i=0; i<NUM_THREAD; i++)
        //     label[i] = std::vector<std::pair<Vertex, Distance>>.reserve(maxSize)
        //#pragma omp parallel for num_threads(NUM_THREAD) schedule (dynamic)
        //for (Vertex v = 0; v < n; ++v) {
        //    for (int side = 0; side < 2; ++side) {
        //        for (size_t i = 0; i < label_v[v][side].size(); ++i)
        //            label[i].push_back(std::make_pair(label_v[v][side][i], label_d[v][side][i]));
        //        std::sort(label.begin(),label.end());
        //        for (size_t i = 0; i < label_v[v][side].size(); ++i) {
        //            label_v[v][side][i] = label[i].first;
        //            label_d[v][side][i] = label[i].second;
        //        }
        //        label[i].clear();
        //    }
        //}


        #pragma omp parallel for num_threads(NUM_THREAD) schedule (dynamic, NUM_THREAD) 
        for (Vertex v = 0; v < n; ++v) {
            for (int side = 0; side < 2; ++side) {
                std::vector< std::pair<Vertex,Distance> > label(label_v[v][side].size());
                for (size_t i = 0; i < label_v[v][side].size(); ++i)
                    label[i] = std::make_pair(label_v[v][side][i], label_d[v][side][i]);
                std::sort(label.begin(),label.end());
                for (size_t i = 0; i < label_v[v][side].size(); ++i) {
                    label_v[v][side][i] = label[i].first;
                    label_d[v][side][i] = label[i].second;
                }
            }
        }
    }

    #if 0
    // Print labels
    void print() const {
        for (Vertex v = 0; v < n; ++v) {
            for (int side = 0; side < 2; ++side) {
                std::cout << "L(" << v << "," << side << ") =";
                for (size_t i = 0; i < label_v[v][side].size(); ++i) std::cout << " (" << label_v[v][side][i] << "," << label_d[v][side][i] << ")";
                std::cout << std::endl;
            }
        }
    }
    #endif

};

}
