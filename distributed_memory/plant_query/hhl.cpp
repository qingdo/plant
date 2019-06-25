
// This file contains implementation of Hybrida PLaNT + PLL algorithm
// for hub labeling on distributed machines.
//
//  Author: Qing Dong, Kartik Lakhotia
//  Email id: qingdong@usc.edu, klakhoti@usc.edu

#pragma once
#include "labeling.hpp"
#include "ordering.hpp"
#include "akiba.hpp"
#include "query.hpp"
#include "paraPLL.hpp"
#include "dijkstra.hpp"
#include "omp.h"
#include <mpi.h>

int main(int argc, char** argv)
{
    unsigned int NUM_THREAD = 16;
    hl::Graph g;
    std::vector<unsigned int> order;
    std::vector<hl::Vertex> queries;
	bool query_from_file = false;
	unsigned int query_mode = 0;
    unsigned int i=0;
	unsigned int num_queries = 10000000;
	bool doParaPLL = false;
    for (unsigned int i=0; i<argc; i++)
    {
        if (strcmp(argv[i], "-g") == 0) // graph files
        {
            if (!g.read(argv[i+1])) {
                std::cerr << "Unable to read graph from raw " << argv[i+1] << std::endl;
                std::exit(1);
            }
        }
        if (strcmp(argv[i], "-o") == 0) // vertex ordering
            hl::Order::read(argv[i+1], order);
        if (strcmp(argv[i], "-p") == 0) // number of threads
            NUM_THREAD=std::stoi(argv[i+1]);
        if (strcmp(argv[i], "-q") == 0) { // get queries
			hl::load_query(queries, argv[i+1]);
			query_from_file = true;
		}
        if (strcmp(argv[i], "-qm") == 0) // number of threads
            query_mode=std::stoi(argv[i+1]);
        if (strcmp(argv[i], "-paraPLL") == 0) // number of threads
            doParaPLL = true;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREAD);

	MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    unsigned int N = order.size();
    hl::Labeling labeling(N) ;
	std::vector <unsigned int> rev_map (N);
    #pragma omp parallel for schedule (dynamic, NUM_THREAD)
    for (unsigned int i=0; i<N; i++)
        rev_map[order[i]] = i;
    if(doParaPLL) 
		hl::run_paraPLL(&g, order, labeling, NUM_THREAD);
	else 
		hl::run(&g, order, rev_map, labeling, NUM_THREAD);
	//std::cout<<"generate labels success"<<std::endl;
	if (!query_from_file && query_mode != 6) {
		queries.resize(num_queries*2);
		if(world_rank == 0) {
			hl::generate_query(queries, num_queries, N, NUM_THREAD);
			//std::cout<<"generate queries success"<<std::endl;
		}
	}
		
	std::vector<hl::Distance> dist(num_queries);
	hl::query(dist, queries, labeling, query_mode, NUM_THREAD); 
	//std::cout<<"answer queries success"<<std::endl;

    for (unsigned int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-l") == 0) //if write to a file
            labeling.write(argv[i+1]);
        if (strcmp(argv[i], "-qr") == 0) //if write query resutlt to a file
            hl::write_result(dist, argv[i+1]);
    }
    return 0;
}
