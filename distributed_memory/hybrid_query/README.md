# Hybrid implementation #
This is the implementation of hybird algorithm to construct cannonical hub labeling for shortest distance queries in distributed-memory environemnt. 
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
