# Hybrid implementation #
This is the implementation of the plant algorithm to construct cannonical hub labeling for shortest distance queries in distributed-memory environemnt. This code is contributed by Qing Dong and Kartik Lakhotia. 
## Getting started ##

#### Compile ####
```
make
```
#### Run ####
#### Regular environment ####
```
mpirun --np=<NUM_NODES> ./hhl -g <graphFile> -o -orderFile <orderFile> -p <numThreads>  -qm <qery mode>
```
#### Slurm environment ####
```
srun --ntasks=<NUM_NODES>  -c <CPUsPerTask> ./hhl -g <graphFile> -o -orderFile <orderFile> -p <numThreads>  -qm <query mode>
```
