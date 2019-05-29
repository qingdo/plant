# Introduction #
This is the implementation of paper submission "Planting Trees for scalable and efficient Canonical Hub Labeling" of SC'19. The codes are contributed by [Qing Dong](https://github.com/DongQing1996) and [Kartik Lakhotia](https://github.com/kartiklakhotia). We provided the implementation of four algorithms as below. Please go to the corresponding folder to see detailed introduction and instructions for each algorithm. 
* In shared memory platform (shared\_memory)
	1.  Label Construction and Cleaning (lcc)
	2.  Global Local Labeling (gll)
* In distributed platform (distributed\_memory)
	1.  Prune Labels and Not Trees (plant)
	2.  Hybrid algorithm of plant and Distributed GLL (hybrid)
	
Some of the implementation (especially the shared memory code) are based on the [savrus's code] (https://github.com/savrus/hl). 
# Datasets #
We provided four datasets for testing. They are as following. Please go to corrsponding folder to see running instructions for each algorithm. All four algorithms takes two inputs, one is the graph file (graph topology, in DIMACS, METIS, SNAP or edgelist format) and the other one is the ordering files (the ordering of all vertices). 
* [AUT](https://www.cc.gatech.edu/dimacs10/data/coauthor/) (coAuther network)
* [CAL](http://users.diag.uniroma1.it/challenge9/download.shtml) (California road network)
* [WND](https://snap.stanford.edu/data/web-NotreDame.html) (University of Notre Dame webpages)
* [YTB](https://snap.stanford.edu/data/com-Youtube.html) (Youtube social network)

We also used some other datasets for testing. We do not push them on github because file size limit. They can be found by clicking the links below.
* [EAS](http://users.diag.uniroma1.it/challenge9/download.shtml) (Eastern USA road network)
* [CTR](http://users.diag.uniroma1.it/challenge9/download.shtml) (Central USA road network)
* [USA](http://users.diag.uniroma1.it/challenge9/download.shtml) (Full USA road network)
* [SKIT](http://www.caida.org/data/active/skitter_aslinks_dataset.xml) (Skitter Autonomous Systems)
* [ACT](http://konect.uni-koblenz.de/networks/actor-collaboration) (Actor Collaboration Network)
* [BDU](http://konect.uni-koblenz.de/networks/zhishi-baidu-internallink) (Baidu HyperLink Network)
* [POK](https://snap.stanford.edu/data/soc-Pokec.html) (Social network Pokec)
* [LIJ](https://snap.stanford.edu/data/com-LiveJournal.html) (Live Journal Social Network)

## Ordering ##
We used [betweeness-based ordering] (http://degroup.cis.umac.mo/sspexp/) for road networks and [degree-based ordering] (https://github.com/savrus/hl) for scale-free networks. 

