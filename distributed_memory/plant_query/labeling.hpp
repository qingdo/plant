// Hub Labels are lists of hubs and distances to them attached to every vertex in a graph.
// This file contains the class to store labels.
// Available methods include making a query, read, write, filter, merge labels 
// and obtain label statistics.
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu
//


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
    //std::vector< bool > lck;   // locks
    bool *lck;

    Labeling(size_t n = 0) :
        label_v(n, std::vector< std::vector<Vertex> >(2)),
        label_d(n, std::vector< std::vector<Distance> >(2)),
        n(n),
        lck(new bool[2*n]()) {}

	Vertex get_n() {return n;}
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

    const bool cover(Vertex u, Vertex v, bool f = true, Distance d=infty) {
        for (size_t i=0, j=0; i < label_v[u][f].size() && j < label_v[v][!f].size();) {
            if (label_v[u][f][i] == label_v[v][!f][j]) {
                if (d >= label_d[u][f][i++] + label_d[v][!f][j++]) return true;
            } else if (label_v[u][f][i] < label_v[v][!f][j]) ++i;
            else ++j;
        }
        return false;
    }

    const bool clean_cover(Vertex u, Vertex v, unsigned int f, Distance d=infty, size_t hub_order=infty) {
        for (size_t i=0, j=0; i < label_v[u][f].size() && j < label_v[v][!f].size();) {
            if (label_v[u][f][i] >= hub_order || label_v[v][!f][j] >= hub_order) return false;
            if (label_v[u][f][i] == label_v[v][!f][j]) {
                if (d >= label_d[u][f][i++] + label_d[v][!f][j++]) return true;
            } else if (label_v[u][f][i] < label_v[v][!f][j]) ++i;
            else ++j;
        }
        return false;
    }

    // Add hub (v,d) to forward or reverse label of u
    inline void add(Vertex u, bool forward, Vertex v, Distance d) {
        while (!__sync_bool_compare_and_swap(&lck[u*2+forward], false, true)) {}
        label_v[u][forward].push_back(v);
        label_d[u][forward].push_back(d);
        lck[u*2+forward]=false;
    }
    inline void add_lockfree(Vertex u, bool forward, Vertex v, Distance d) {
        label_v[u][forward].push_back(v);
        label_d[u][forward].push_back(d);
    }

    void combine_buffer(int *recv_buffer, int recv_size) {
	    for (int i=0; i<recv_size; i++) {
		   Vertex v = recv_buffer[i*3]/2;
		   bool forward = recv_buffer[i*3]%2;
		   Vertex u = recv_buffer[i*3+1];
		   Distance d = recv_buffer[i*3+2];
		   label_v[u][forward].push_back(v);
		   label_d[u][forward].push_back(d);
	    }
    }

	unsigned* filter(int *recv_buffer, int recv_size, std::vector<Vertex>&order, unsigned *mask, unsigned NUM_THREAD) {
        #pragma omp parallel for num_threads (NUM_THREAD) schedule (dynamic, 512)
	    for (int i=0; i<recv_size; i++) {
			unsigned vertex_order = recv_buffer[i*3]/2;
		   	Vertex v = order[vertex_order];
		   	bool forward = recv_buffer[i*3]%2;
		   	Vertex u = recv_buffer[i*3+1];
		   	Distance d = recv_buffer[i*3+2];
			//if (clean_cover(v, u, !forward, d, vertex_order))
			//<UPDATE> order of vertices in clean cover -
			//make it same as the function declaration
			if (clean_cover(u, v, forward, d, vertex_order))
		   		mask[i/32]= mask[i/32] & (~(1<<(i%32)));
	    }
		return &mask[0];
	}


    void absorb(int *recv_buffer, unsigned *mask, int start_loc, int recv_size, unsigned NUM_THREAD, bool* lck) {
        #pragma omp parallel for num_threads (NUM_THREAD) schedule (dynamic, 512)
	    for (int i=start_loc; i<start_loc+recv_size; i++) {
		   	if (mask[i/32]&(1<<(i%32))) {
				//std::cout<<i<<std::endl;
				Vertex v = recv_buffer[i*3]/2;
				bool forward = recv_buffer[i*3]%2;
				Vertex u = recv_buffer[i*3+1];
				Distance d = recv_buffer[i*3+2];
				unsigned lcknum = u*2+forward;
				while (!__sync_bool_compare_and_swap(&lck[lcknum], false, true)) {}
				label_v[u][forward].push_back(v);
				label_d[u][forward].push_back(d);
				lck[lcknum]=false;
			}
	    }
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
        return static_cast<double>(total)/2/n;
    }
    long long get_total() const {
        long long total = 0;
        for (Vertex v = 0; v < n; ++v)
            total += label_v[v][0].size() + label_v[v][1].size();
        return total;
    }
    long long total_cap() const {
        long long total = 0;
        for (Vertex v = 0; v < n; ++v)
            total += label_v[v][0].capacity() + label_v[v][1].capacity();
        return total;
    }
    int check_label() const {
        for (Vertex v = 0; v < n; ++v) {
            if(label_d[v][0].size() != label_v[v][0].size())
				return v;
            if(label_d[v][1].size() != label_v[v][1].size())
				return v;
		}
        return -1;
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
    void clear(unsigned NUM_THREAD) {
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
        #pragma omp parallel for num_threads(NUM_THREAD)
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

    //<UPDATE> take vector input
    void sort_partial(unsigned NUM_THREAD, std::vector<int> &last_loc) {
        #pragma omp parallel for num_threads(NUM_THREAD)
        for (Vertex v = 0; v < n; ++v) {
            for (int side = 0; side < 2; ++side) {
				unsigned last_one = last_loc[2*v+side];
                unsigned current_size = label_v[v][side].size();
                std::vector< std::pair<Vertex,Distance> > label(current_size-last_one);
                for (size_t i = last_one; i < current_size; ++i)
                    label[i-last_one] = std::make_pair(label_v[v][side][i], label_d[v][side][i]);
                std::sort(label.begin(),label.end());
                for (size_t i = last_one; i < current_size; ++i) {
                    label_v[v][side][i] = label[i-last_one].first;
                    label_d[v][side][i] = label[i-last_one].second;
                }
				last_loc[2*v+side] = current_size;
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
