# Introduction #
This is the implementation of paper submission "Planting Trees for scalable and efficient Canonical Hub Labeling" of SC'19. The codes are contributed by Qing Dong and Kartik Lakhotia. We provided the implementation of four algorithms as below. Please go to the corresponding folder to see detailed introduction and instructions for each algorithm. 
* In shared memory platform (shared\_memory)
	1.  Label Construction and Cleaning (lcc)
	2.  Global Local Labeling (gll)
* In distributed platform (distributed\_memory)
	1.  Prune Labels and Not Trees (plant)
	2.  Hybrid algorithm of plant and Distributed GLL (hybrid)
	
Some of the implementation (especially the shared memory code) are based on the [savrus's code] (https://github.com/savrus/hl). 
# Datasets #
We provided four datasets for testing. They are as following. Please go to corrsponding folder to see running instructions for each algorithm. All four algorithms takes two inputs, one is the graph file (graph topology, in DIMACS, METIS, SNAP or edgelist format) and the other one is the ordering files (the ordering of all vertices). 
* AUT (coAuther network)
* CAL (California road network)
* WND (University of Notre Dame webpages)
* YTB (Youtube social network)
## Ordering ##
We used [betweeness-based ordering] (http://degroup.cis.umac.mo/sspexp/) for road networks and [degree-based ordering] (https://github.com/savrus/hl) for scale-free networks. 

