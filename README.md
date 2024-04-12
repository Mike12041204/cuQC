# cuQC: Accelerating Maximal Quasi-Clique Mining using the GPU
This repository contains the code for the "cuQC: Accelerating Maximal Quasi-Clique Mining using the GPU" program, as well as related graphs and graph formatting tools. the cuQC program is a powerful maximal y-quasi-clique enumerator for the GPU.
## Obtaining the latest version of the program
Visit the [cuQC Github](https://github.com/Mike12041204/cuQC) to obtain the latest version of this software.
## Package Requirements
* CUDA(>=12.2.0)

We used `CUDA 12.2.0`
## Preparing Datasets
Our program runs off graphs represented in a custom serialized format, designed to prevent duplicate processing of graphs. We provide tools to convert graphs to this format.
### Software Requirements for Preparing Graphs
* g++
* python3

### Adjacency List to Serialized Format
Given a graph *input* in the format of an adjacency list, where line 1 in the graph's text file contains all the adjacencies of vertex 0 in the graph, and the adjacencies are represented as space-separated integers, for example:
```
1 2 6 9 15
0 2
0 1 7 9
.
.
.
```
We can convert *input* to the serialized format used by cuQC by using the adjToSer.cpp code.

This program should first be compiled using `g++` to an executable file, if this executable were to be named *AtoS* then the line would be as such:
```
g++ adjToSer.cpp -o AtoS
```

We could then use *input* and *AtoS* to generate our serialized graph representation, *output*, with the following line:
```
./AtoS input output
```
### Edge List to Adjacency List
Given a graph *input* provided in an edge list format where each line contains two numbers separated by whitespace representing an edge, with the first number being the source vertex and the second number being the destination vertex, for example:
```
0  1
1  0
2  7
.
.
.
```
We can convert *input* to an adjacency list format by using the edgeToAdj.py code.

We noticed that some unweighted undirected graphs represented in an edge list format had one line per undirected edge, while others had two lines, the second where the source and destination are reversed. To handle this we adjusted the code to have the option to duplicate all edges. This option is either `0 - no duplication` or `1 - duplication` and is taken on the command line when running the program as the second parameter.

The program also uses output redirection to write the generated graph into a file.

We could use this code and *input* to generate the adjacency list representation of the graph, *output*, without duplicating edges with the following line:
```
python3 edgeToAdj.py input 0 >output
```
## Build Instructions
We provide a script to build the program. Running `build.sh` will compile the program and produce the `cuQC` executable.

When using cuQC it should be noted that most data structure sizes and their related memory usage are determined statically at the start of the program through definitions, for example:
```
#define TASKS_SIZE 10000
```
We have set the program with default definitions which should work on a GPU with `40GB` of global memory for most graphs.

However, if a `segmentation fault` or `bus error` occurs during cuQC's run, these definitions may not be suitable for the graph. These definitions should be corrected by running the program again, this time in debug mode. This mode can be toggled on for cuQC by changing a definition within the program. This definition has the name of `DEBUG_MODE` and has two options: `0 - off` and `1 - on`. When debug mode is on, the program will display information indicating the size of the data within the data structures at each partial step and will provide information on which of these data structures might be causing the memory issue. This mode should allow the fine-tuning of the data structure definitions to allow cuQC to work on numerous graphs of considerable size. As explained in the paper tuning the definition `TASKS_PER_WARP` may also decrease the memory usage of the program. Of course, at some point, a graph will become too large to run, no matter what definitions are chosen.

It should also be noted that making the definitions for the data structure and thus their sizes as small as possible decreases the time spent by cuQC. This is because some of these data structures must be copied from the CPU to the GPU, and if they are smaller, it will take less time to do so. Thus, when timing cuQC, we would first run the program in debug mode to find some definitions that worked for the graph, then using the data structure size information given by DEBUG MODE, we would minimize the definitions to boost the speed of cuQC.
## Experiments
For running experiments with cuQC, the host should have at least `32GB` of memory, and the device should have at least `40GB` of global memory. If the machine doesn't have that much memory cuQC will still be able to run some experiment scenarios on smaller graphs. But it may run out of memory, throw an error, and terminate the program for other cases. If this scenario occurs refer to the `Build Instructions` sections for how to proceed.

The program takes 5 parameters:
1. graph_file, the file to find cliques in
2. gamma, the gamma of the cliques to be found, must be >= .5
3. min_size, the minimum size of the cliques to be found, must be > 1
4. output_file, the file to output the resulting cliques to
5. `scheduling_toggle`, the program task scheduling can run in two modes, `0 - dynamic` and `1 - static`

The program might be run as:
```
./cuQC GSE1730 .9 30 results.txt 0
```
Sample output:
```
>:PRE-PROCESSING
--->:LOADING TIME: 33 ms
>:INITIALIZING TASKS
>:BEGINNING EXPANSION
--->:ENUMERATION TIME: 171 ms
>:REMOVING NON-MAXIMAL CLIQUES
>:NUMBER OF MAXIMAL CLIQUES: 1602
--->:REMOVE NON-MAX TIME: 21 ms
--->:TOTAL TIME: 495 ms
>:PROGRAM END
```
Sample results.txt
```
33 0 1 2 3 4 5 8 9 10 11 12 18 19 20 21 23 24 27 30 33 38 40 44 47 48 50 53 56 57 58 73 105 157
33 0 1 2 3 4 5 8 9 10 11 12 18 19 20 21 23 24 27 30 33 38 40 44 47 48 50 53 56 57 58 73 105 304
33 0 1 2 3 4 5 8 9 10 11 12 18 19 20 21 23 24 27 30 33 38 40 44 47 48 50 53 56 57 58 73 157 304
33 0 1 2 3 4 5 8 9 10 11 12 18 19 20 21 23 24 27 30 33 38 40 44 47 48 50 53 56 57 58 105 157 304
32 0 1 2 3 4 5 8 9 10 11 12 18 19 20 21 23 24 27 30 33 38 40 44 47 48 50 53 57 58 61 73 157
.
.
.
```
The first number in each line represents the number of vertices within the clique, and the next numbers in the line represent their IDs.

Also note that if the program finds no cliques among completion, it will post an error and terminate.

Sample debug mode output (refer to `Build Instructions` section):
```
>:PRE-PROCESSING
--->:LOADING TIME: 4 ms
>:INITIALIZING TASKS
L: 0 T1: 1 55 T2: 0 0 B: 0 0 C: 0 0
L: 1 T1: 1 55 T2: 1 55 B: 0 0 C: 0 0
L: 2 T1: 24 1044 T2: 1 55 B: 0 0 C: 0 0
>:BEGINNING EXPANSION
WTasks( TC: 192 TS: 7212 MC: 12 MS: 515) WCliques ( TC: 0 TS: 0 MC: 0 MS: 0)
L: 3 T1: 24 1044 T2: 192 7212 B: 0 0 C: 0 0

WTasks( TC: 624 TS: 22060 MC: 12 MS: 452) WCliques ( TC: 21 TS: 640 MC: 3 MS: 91)
L: 4 T1: 624 22060 T2: 192 7212 B: 0 0 C: 21 640
.
.
.
```
Sample debug mode output continued:
```
.
.
.
WTasks( TC: 0 TS: 0 MC: 0 MS: 0) WCliques ( TC: 4 TS: 128 MC: 1 MS: 32)
L: 18 T1: 0 0 T2: 4 128 B: 0 0 C: 3163 98216

--->:ENUMERATION TIME: 1834 ms

TASKS SIZE: 53617
BUFFER SIZE: 0
BUFFER OFFSET SIZE: 0
CLIQUES SIZE: 98216
CLIQUES OFFSET SIZE: 3163
WCLIQUES SIZE: 124
WCLIQUES OFFSET SIZE: 4
WTASKS SIZE: 515
WTASKS OFFSET SIZE: 12
VERTICES SIZE: 55

>:REMOVING NON-MAXIMAL CLIQUES
>:NUMBER OF MAXIMAL CLIQUES: 1602
--->:REMOVE NON-MAX TIME: 22 ms
--->:TOTAL TIME: 2144 ms
>:PROGRAM END
```
# Benchmarking platform and Dataset
## Machine
* GPU: Nvidia Ampere A100 (108SMs, 80GB)
* OS: Red Hat Enterprise Linux Server release 7.9 (Maipo)
* CUDA: 12.2.0

## Dataset
* See the related paper for links to all the used data sets, and refer to the `Preparing Datasets` section on how to prepare them or other graphs for running by cuQC.
