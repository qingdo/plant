/**
 * Author: Qing Dong, Kartik Lakhotia
 * Email id: qingdong@usc.edu, klakhoti@usc.edu
 * Date: 04-May-2018
 *
 * This code implements the parallelization of the pruned landmark labeling algorithm which is the second phase of the hhl algorithm (Akiba et al. 2014). 
 */

#include "labeling.hpp"
#include "ordering.hpp"
#include "akiba.hpp"
#include "dijkstra.hpp"
#include "omp.h"


int main(int argc, char** argv)
{
    hl::Labeling *labeling;
    hl::Labeling *local_labeling;
    unsigned int NUM_THREAD;
    hl::Graph gDij;
    std::vector<unsigned int> order;
    unsigned int i=0;
    bool paraPLL=false;
    int sf = 32;

    for(unsigned int argi=0;argi<argc;argi++)
    {
        printf("%s ",argv[argi]);
    }
    std::cout<<std::endl;
    struct timespec labeling_start, labeling_end;
    float label_time_total;
    for (unsigned int i=0; i<argc; i++)
    {
        if (strcmp(argv[i], "-o") == 0) // vertex ordering
            hl::Order::read(argv[i+1], order);
        if (strcmp(argv[i], "-p") == 0) // number of threads
            NUM_THREAD=std::stoi(argv[i+1]);
        if (strcmp(argv[i], "-s") == 0) // number of threads
            sf=std::stoi(argv[i+1]);
        if (strcmp(argv[i], "-paraPLL") == 0) // paraPLL
            paraPLL = true;
        if (strcmp(argv[i], "-g") == 0) // graph files
        {
            if (!gDij.read(argv[i+1])) {
                std::cerr << "Unable to read graph from raw " << argv[i+1] << std::endl;
                std::exit(1);
            }
        }
    }
    
    unsigned int N = order.size();
    std::cerr<<"Order size: "<<N<<" |V|: "<<gDij.get_n()<<std::endl;
    labeling = new hl::Labeling(N);
    local_labeling=new hl::Labeling(N);
    unsigned int p = NUM_THREAD;

    std::vector <unsigned int> revMap (N);
    #pragma omp parallel for num_threads (NUM_THREAD) schedule (dynamic, NUM_THREAD)
    for (unsigned int i=0; i<N; i++)
        revMap[order[i]] = i;
    
    
    hl::run_paraPLL(&gDij, order, revMap, *local_labeling, *labeling, p, sf*N, paraPLL);

////////////////////////////////////////
/////////////// Labeling ///////////////
////u///////////////////////////////////
     
    for (unsigned int i=0; i<argc; i++)
    {
        if (strcmp(argv[i], "-l") == 0) //if write to a file
        {
            labeling->write(argv[i+1]);
        }
    }    
    return 0;
}
