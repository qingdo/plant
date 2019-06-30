// Wrappers for batch querying on cluster nodes
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu
//
#include <vector>
#include <omp.h>
#include <cmath>
#include "graph.hpp"
#include "query_comm.hpp"
#include "labeling.hpp"
#include <stdlib.h>
#include <mpi.h>

namespace hl {

void singleLocalQuery(Distance &dist, std::vector<Vertex> &queries, Labeling &labels)
{
    for (size_t ii=0; ii<queries.size(); ii+=2)
        dist = labels.query(queries[ii], queries[ii+1]);
}

void batchLocalQuery(std::vector<Distance> &dist, std::vector<Vertex> &queries, Labeling &labels, int NUM_THREAD)
{
    dist.resize(queries.size()/2);
//    std::cout<<NUM_THREAD<<std::endl;
    #pragma omp parallel for num_threads(NUM_THREAD) schedule (dynamic, NUM_THREAD*2)
    for (size_t ii=0; ii<queries.size(); ii+=2)
        dist[ii>>1] = labels.query(queries[ii], queries[ii+1]);
}

 
void singleDistQuery(Distance &dist, std::vector<Vertex> &queries, Labeling &labels, int world_rank, int world_size)
{
    int num_queries = 2;
    Distance local_dist;
    if (world_rank != 0)
        queries.resize(num_queries);    


    MPI_Bcast(&queries[0], num_queries, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    singleLocalQuery(local_dist, queries, labels);

    MPI_Reduce(&local_dist, &dist, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
}

//mode3 - some labels on every machine
void batchDistQuery(std::vector<Distance> &dist, std::vector<Vertex> &queries, Labeling &labels, int world_rank, int world_size, int NUM_THREAD)
{
    int num_queries = queries.size();
    MPI_Bcast(&num_queries, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<Distance> local_dist(num_queries>>1);
    if (world_rank != 0)
        queries.resize(num_queries);    

    dist.resize(num_queries>>1);

    MPI_Bcast(&queries[0], num_queries, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    batchLocalQuery(local_dist, queries, labels, NUM_THREAD);

    MPI_Reduce(&local_dist[0], &dist[0], num_queries>>1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
}


}
