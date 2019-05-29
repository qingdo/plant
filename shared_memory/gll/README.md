# Hybrid implementation #
This is the implementation of GLL algorithm to construct cannonical hub labeling for shortest distance queries in shared-memory environemnt. This code is contributed by Qing Dong and Kartik Lakhotia. 
## Getting started ##

#### Compile ####
```
make
```
#### Run ####
```
./hhl -g <graphFile> -o -orderFile <orderFile> -p <numThreads> 
```
#### Paramemters ####
The number of threads is optional. The default values are:
1. Number of threads: 16


