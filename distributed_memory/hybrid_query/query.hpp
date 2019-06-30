// This file implements querying using distributed machines.
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu


#include <vector>
#include <omp.h>
#include <cmath>
#include "graph.hpp"
#include "labeling.hpp"
#include "part_pair.hpp"
#include <stdlib.h>
#include <mpi.h>

namespace hl {

bool load_query(std::vector<Vertex> &queries, char* filename) {
    FILE *file;
    if ((file = fopen(filename, "r")) == NULL) return false;
    char buf[512];
    long long u,v;
    std::set<long> nodes;
    while (fgets(buf, sizeof(buf), file)) {
        if (buf[strlen(buf)-1] != '\n' && !feof(file)) return false;
        if (sscanf(buf, "%lld %lld", &u, &v) != 2) return false;
        queries.push_back(u);
        queries.push_back(v);
    }
	fclose(file);
    return true;
}

bool write_result(std::vector<Distance> &res, char* filename) {
    std::ofstream file;
    file.open(filename);
    for (int i = 0; i < res.size(); ++i) 
    	file << res[i]<<std::endl;
    file.close();
    return file.good();
}

bool generate_query(std::vector<Vertex> &queries, unsigned num_queries, unsigned range, int NUM_THREAD) {

	queries.resize(num_queries*2);
	for (unsigned i = 0; i < num_queries; i++)  {
        queries[i*2] = rand() % range;
        queries[i*2+1] = rand() % range;
    }
    return true;
}


bool query(std::vector<Distance> &dist, std::vector<Vertex> &queries, Labeling &labels, unsigned query_mode, int NUM_THREAD) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	dist.resize(queries.size()/2);	
	Vertex N = labels.n;
	double query_start = 0, query_end = 0, query_time = 0;
    if (query_mode == 0)    { //mode 0 Latency QLSN - full labeling is already in one machine
        Labeling all_labels(N);
	//	std::vector<int> label_list;
	//	std::vector<int> recv_buffer;
    //    parallelLoad(label_list, labels, 0, N, NUM_THREAD);
    //    gatherAllLabelsQuery(label_list, recv_buffer, world_size, world_rank);
    //    loadFromRecvBuffer(recv_buffer, all_labels);
    	allToAllLabels(labels, all_labels, world_size, world_rank, NUM_THREAD);
        all_labels.sort(NUM_THREAD);
        if(world_rank==0) std::cout<<"all label size "<<all_labels.get_avg()<<std::endl;
        std::vector<Vertex> single_queries(2);
        query_start = omp_get_wtime();
        for (int i=0;i<queries.size()/2;i++) {
            single_queries[0] = queries[i*2];
            single_queries[1] = queries[i*2+1];
            if(world_rank==0) singleLocalQuery(dist[i], single_queries, all_labels);
        }
        query_end = omp_get_wtime();
        if (world_rank==0) std::cout<<"QLSN latency: "<<query_end - query_start<<std::endl;
    }
    else if (query_mode == 1)    { //mode1 Throughput QLSN - gather labels in one machine and do query
        Labeling all_labels(N);
    	allToAllLabels(labels, all_labels, world_size, world_rank, NUM_THREAD);
        all_labels.sort(NUM_THREAD);
        if(world_rank==0) std::cout<<"all label size "<<all_labels.get_avg()<<std::endl;
        query_start = omp_get_wtime();
        if(world_rank==0) batchLocalQuery(dist, queries, all_labels, NUM_THREAD);
        query_end = omp_get_wtime();
        if (world_rank==0) std::cout<<"QLSN throughput: "<<query_end-query_start<<std::endl;
    }
    else if (query_mode == 2) { //mode2 Latency QDOL - partition pairs on every machine
        PartPair dist_labeling (labels, world_size, world_rank, NUM_THREAD);    
        Distance m2_res;
        std::vector<Vertex> single_queries(2);
        query_start = omp_get_wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        query_start = omp_get_wtime();
        int ret = 0;
        if (world_rank==0)
        {
            for (int i=0;i<queries.size()/2;i++) {
                std::vector<Vertex> single_queries(2);
                single_queries[0] = queries[i*2];
                single_queries[1] = queries[i*2+1];
                ret = dist_labeling.singleQuery(m2_res, single_queries);
				dist[i] = m2_res;
            }
            single_queries[0] = dist_labeling.n + 1;
            ret = dist_labeling.singleQuery(m2_res, single_queries);
        }
        else
        {
            while(ret >= 0)
            {
                ret = dist_labeling.singleQuery(m2_res, single_queries);
            } 
        }
        query_end = omp_get_wtime();
        if (world_rank==0) std::cout<<"QDOL Latency: "<<query_end - query_start<<std::endl;
        std::cout<<"world rank: "<<world_rank<<std::endl;
    }
    else if (query_mode == 3) { //mode3 Throughput QDOL - partition pairs on every machine
        PartPair dist_labeling (labels, world_size, world_rank, NUM_THREAD);    
        query_start = omp_get_wtime();
        dist_labeling.query(dist, queries);
        query_end = omp_get_wtime();
        if (world_rank==0) std::cout<<"QDOL throughput: "<<query_end-query_start<<std::endl;
    }
    else if (query_mode == 4) { //mode4 Latency QFDL - Querying with Fully Distributed Labels
        MPI_Barrier(MPI_COMM_WORLD);
        Distance m3_res;
        std::vector<Vertex> single_queries(2);
        query_start = omp_get_wtime();
        for (int i=0;i<queries.size()/2;i++) {
            single_queries[0] = queries[i*2];
            single_queries[1] = queries[i*2+1];
            singleDistQuery(m3_res, single_queries, labels, world_rank, world_size);
			dist[i] = m3_res;
        }
        query_end = omp_get_wtime();
        if (world_rank==0) std::cout<<"QFDL Latency: "<<query_end-query_start<<std::endl;
    }
    else if (query_mode == 5) { //mode5 Throughput QFDL - Querying with Fully Distributed Labels
        query_start = omp_get_wtime();
        batchDistQuery(dist, queries, labels, world_rank, world_size, NUM_THREAD);
        query_end = omp_get_wtime();
        if (world_rank==0) std::cout<<"QFDL throughput: "<<query_end-query_start<<std::endl;
    }
	return true;
}

}
