# Hybrid implementation #
This is the implementation of hybird algorithm to construct cannonical hub labeling for shortest distance queries in distributed-memory environemnt. This code is contributed by Qing Dong and Kartik Lakhotia.  
## Getting started ##

#### Compile ####
```
make
```
#### Run ####
#### Regular environment ####
```
mpirun --np=<NUM_NODES> ./hhl -g <graphFile> -o -orderFile <orderFile> -p <numThreads> -cb <common label budget> -ps <phase swith threshlod> -qm <qery mode>
```
#### Slurm environment ####
```
srun --ntasks=<NUM_NODES>  -c <CPUsPerTask> ./hhl -g <graphFile> -o -orderFile <orderFile> -p <numThreads> -cb <common label budget> -ps <phase swith threshlod> -qm <query mode>
```

#### Paramemters ####
The common label budget, phase switch threshold, number of threads and query mode is optional. The default values are:
1. common label budget: 100M labels
2. phase switch threshold: 1024
3. Number of threads: 16
4. query mode: 5 (query on distributed labels with two different layouts: QFDL and QDOL and compare the time, the code generates 10M randoms queries)`

