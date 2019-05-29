
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
    #pragma omp parallel for num_threads(NUM_THREAD)
	for (unsigned i = 0; i < num_queries; i++)  {
		//srand(i);
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
	
	double query_start = 0, query_end = 0, query_time = 0;
	if (query_mode == 0) 	{ //mode 0 - full labeling is already in one machine
		double total = 0;
		for (int i=0;i<queries.size()/2;i++) {
			std::vector<hl::Vertex> single_queries(2);
			single_queries[0] = queries[i*2];
			single_queries[1] = queries[i*2+1];
	   	 	query_start = omp_get_wtime();
			if(world_rank==0) batchLocalQuery(dist, single_queries, labels, NUM_THREAD);
	   	 	query_end = omp_get_wtime();
			total += query_end - query_start;
		}
		if (world_rank==0) std::cout<<"QLSN total: "<<total<<std::endl;
	}
	if (query_mode == 1)	{ //mode1 - gather labels in one machine and do query
		std::vector<int> label_list;
		std::vector<int> recv_buffer;
		int N = labels.n;
		Labeling all_labels(N);
		parallelLoad(label_list, labels, 0, N, NUM_THREAD);
		gatherAllLabelsQuery(label_list, recv_buffer, world_size, world_rank);
		loadFromRecvBuffer(recv_buffer, all_labels);
		all_labels.sort(NUM_THREAD);
		//std::cout<<"all label size "<<all_labels.get_avg()<<std::endl;
		if(world_rank==0) batchLocalQuery(dist, queries, all_labels, NUM_THREAD);
	}
	else if (query_mode == 2) { //mode2 - partition pairs on every machine
		PartPair dist_labeling (labels, world_size, world_rank, NUM_THREAD);	
		//std::cout<<"Distribute labels success"<<std::endl;
    	query_start = omp_get_wtime();
		dist_labeling.query(dist, queries);
    	query_end = omp_get_wtime();
		//std::cout<<"query success"<<std::endl;

	}
	else if (query_mode == 3) { //mode3 - Reduce results from all machines.
    	query_start = omp_get_wtime();
		batchDistQuery(dist, queries, labels, world_rank, world_size, NUM_THREAD);
    	query_end = omp_get_wtime();
	}
	else if (query_mode == 4) { //mode4 - verify mode2 result
		PartPair dist_labeling (labels, world_size, world_rank, NUM_THREAD);	
		//std::cout<<"Distribute labels success"<<std::endl;
    	query_start = omp_get_wtime();
		bool res = dist_labeling.verify(queries);
		if (world_rank==0)	
		//std::cout<<"verification res: "<<res<<std::endl;
    	query_end = omp_get_wtime();
		//std::cout<<"query success"<<std::endl;
///////////////////////////////////////////Verify with mode 1 ///////////
	}
	else if (query_mode == 5) { //mode5 - compare mode 2 and mode 3
      std::vector<Distance> m2_res;
      std::vector<Distance> m3_res;
    	query_start = omp_get_wtime();
		batchDistQuery(m3_res, queries, labels, world_rank, world_size, NUM_THREAD);
    	query_end = omp_get_wtime();
		if (world_rank==0) std::cout<<"QFDL: "<<query_end-query_start<<std::endl;
		// mode 3
		PartPair dist_labeling (labels, world_size, world_rank, NUM_THREAD);	
    	query_start = omp_get_wtime();
		dist_labeling.query(m2_res, queries);
    	query_end = omp_get_wtime();
		if (world_rank==0) std::cout<<"QDOL: "<<query_end-query_start<<std::endl;
		if (world_rank==0)	{
			for (int i = 0; i < queries.size()/10000 + 1; i++) {
				if (m2_res[i]!=m3_res[i]) {
					std::cout<<"Diff happenes in  checking "<<queries[i*2] <<" to "<<queries[i*2+1] <<" mode 2: "<<m2_res[i]<<" mode3: "<<m3_res[i]<<std::endl;
				
				}
			}
		}
	}
	else if (query_mode == 6) { //mode5 - compare mode 2 and mode 3 Latency
     	std::vector<Distance> m2_res;
     	std::vector<Distance> m3_res;
		double total = 0;
		for (int i=0;i<queries.size()/2;i++) {
			std::vector<hl::Vertex> single_queries(2);
			single_queries[0] = queries[i*2];
			single_queries[1] = queries[i*2+1];
	   	 	query_start = omp_get_wtime();
	   	 	batchDistQuery(m3_res, single_queries, labels, world_rank, world_size, NUM_THREAD);
	   	 	query_end = omp_get_wtime();
			total += query_end - query_start;
		}
		if (world_rank==0) std::cout<<"QFDL total: "<<total<<std::endl;
		// mode 3
		PartPair dist_labeling (labels, world_size, world_rank, NUM_THREAD);	
		total = 0;
		for (int i=0;i<queries.size()/2;i++) {
			std::vector<hl::Vertex> single_queries(2);
			single_queries[0] = queries[i*2];
			single_queries[1] = queries[i*2+1];
	   	 	query_start = omp_get_wtime();
	   	 	dist_labeling.query(m2_res, single_queries);
	   	 	query_end = omp_get_wtime();
			total += query_end - query_start;
		}
		if (world_rank==0) std::cout<<"QDOL total: "<<total<<std::endl;
	}
}
}
