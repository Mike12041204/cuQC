#ifndef DCUQC_COMMON_H
#define DCUQC_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <cstring>
#include <time.h>
#include <chrono>
#include <sys/timeb.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>
#include <device_atomic_functions.h>
#include <mpi.h>
using namespace std;

// GPU KERNEL LAUNCH
#define BLOCK_SIZE 1024
#define NUM_OF_BLOCKS 216
#define WARP_SIZE 32

// GPU INFORMATION
#define IDX ((blockIdx.x * blockDim.x) + threadIdx.x)
#define WARP_IDX (IDX / WARP_SIZE)
#define LANE_IDX (IDX % WARP_SIZE)
#define WIB_IDX (threadIdx.x / WARP_SIZE)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define NUMBER_OF_WARPS (NUM_OF_BLOCKS * WARPS_PER_BLOCK)
#define NUMBER_OF_THREADS (NUM_OF_BLOCKS * BLOCK_SIZE)

// PROGRAM RUN SETTINGS
// shared memory vertices
#define VERTICES_SIZE 70
// cpu settings
#define CPU_LEVELS 1
#define CPU_EXPAND_THRESHOLD 1
// mpi settings
#define NUMBER_OF_PROCESSESS 4
#define MAX_MESSAGE 1000000000
// must be atleast be 1
#define HELP_MULTIPLIER 1
#define HELP_PERCENT 50
#define HELP_THRESHOLD (NUMBER_OF_WARPS * HELP_MULTIPLIER)

// debug toggle 0-normal/1-debug
#define DEBUG_TOGGLE 0

// VERTEX DATA
struct Vertex
{
    uint32_t vertexid;
    int8_t label;               // labels: 0 -> candidate, 1 -> member, 2 -> covered vertex, 3 -> cover vertex, 4 -> critical adjacent vertex, -1 -> pruned vertex
    uint32_t indeg;
    uint32_t exdeg;
    uint32_t lvl2adj;
};

// CPU GRAPH / CONSTRUCTOR
class CPU_Graph
{
    public:

    int number_of_vertices;
    int number_of_edges;
    uint64_t number_of_lvl2adj;
    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    CPU_Graph(ifstream& graph_stream);
    ~CPU_Graph();
};

// CPU DATA
struct CPU_Data
{
    // structures for storing vertices
    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;
    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;
    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;
    // information about expansion
    uint64_t* current_level;
    bool* maximal_expansion;
    bool* dumping_cliques;
    // helpers in pruning
    int* vertex_order_map;
    int* remaining_candidates;
    int* removed_candidates;
    int* remaining_count;
    int* removed_count;
    int* candidate_indegs;
};

// CPU CLIQUES
struct CPU_Cliques
{
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;
};

// DEVICE DATA
struct GPU_Data
{
    // GPU DATA
    uint64_t* current_level;
    int* current_task;
    uint64_t* tasks_count;
    uint64_t* tasks_offset;
    Vertex* tasks_vertices;
    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;
    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    Vertex* wtasks_vertices;
    Vertex* global_vertices;
    int* removed_candidates;
    int* lane_removed_candidates;
    Vertex* remaining_candidates;
    int* lane_remaining_candidates;
    int* candidate_indegs;
    int* lane_candidate_indegs;
    int* adjacencies;
    int* total_tasks;
    double* minimum_degree_ratio;
    int* minimum_degrees;
    int* minimum_clique_size;
    uint64_t* buffer_offset_start;
    uint64_t* buffer_start;
    uint64_t* cliques_offset_start;
    uint64_t* cliques_start;
    // GPU GRAPH
    int* number_of_vertices;
    int* number_of_edges;
    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;
    // GPU CLIQUES
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;
    uint64_t* wcliques_count;
    uint64_t* wcliques_offset;
    int* wcliques_vertex;
    int* total_cliques;
    // DATA STRUCTURE SIZE
    uint64_t* tasks_size;
    uint64_t* tasks_per_warp;
    uint64_t* buffer_size;
    uint64_t* buffer_offset_size;
    uint64_t* cliques_size;
    uint64_t* cliques_offset_size;
    uint64_t* cliques_percent;
    uint64_t* wcliques_size;
    uint64_t* wcliques_offset_size;
    uint64_t* wtasks_size;
    uint64_t* wtasks_offset_size;
    uint64_t* wvertices_size;
    uint64_t* expand_threshold;
    uint64_t* cliques_dump;
};

// WARP DATA
struct Warp_Data
{
    // previous level
    uint64_t start[WARPS_PER_BLOCK];
    uint64_t end[WARPS_PER_BLOCK];
    int tot_vert[WARPS_PER_BLOCK];
    int num_mem[WARPS_PER_BLOCK];
    int num_cand[WARPS_PER_BLOCK];
    int expansions[WARPS_PER_BLOCK];
    // next level
    int number_of_members[WARPS_PER_BLOCK];
    int number_of_candidates[WARPS_PER_BLOCK];
    int total_vertices[WARPS_PER_BLOCK];
    Vertex shared_vertices[VERTICES_SIZE * WARPS_PER_BLOCK];
    // pruning helpers
    int removed_count[WARPS_PER_BLOCK];
    int remaining_count[WARPS_PER_BLOCK];
    int num_val_cands[WARPS_PER_BLOCK];
    int rw_counter[WARPS_PER_BLOCK];
    int min_ext_deg[WARPS_PER_BLOCK];
    int lower_bound[WARPS_PER_BLOCK];
    int upper_bound[WARPS_PER_BLOCK];
    int tightened_upper_bound[WARPS_PER_BLOCK];
    int min_clq_indeg[WARPS_PER_BLOCK];
    int min_indeg_exdeg[WARPS_PER_BLOCK];
    int min_clq_totaldeg[WARPS_PER_BLOCK];
    int sum_clq_indeg[WARPS_PER_BLOCK];
    int sum_candidate_indeg[WARPS_PER_BLOCK];
    bool success[WARPS_PER_BLOCK];
    int number_of_crit_adj[WARPS_PER_BLOCK];
};

// LOCAL DATA
struct Local_Data
{
    Vertex* vertices;
};

// DATA STRUCTURE SIZES
class DS_Sizes
{
    public:
    
    // DATA STRUCTURE SIZE
    uint64_t tasks_size;
    uint64_t tasks_per_warp;
    uint64_t buffer_size;
    uint64_t buffer_offset_size;
    uint64_t cliques_size;
    uint64_t cliques_offset_size;
    uint64_t cliques_percent;
    // per warp
    uint64_t wcliques_size;
    uint64_t wcliques_offset_size;
    uint64_t wtasks_size;
    uint64_t wtasks_offset_size;
    // global memory vertices, should be a multiple of 32 as to not waste space
    uint64_t wvertices_size;
    uint64_t expand_threshold;
    uint64_t cliques_dump;

    DS_Sizes(const string& filename);
};

// DEBUG - MAX TRACKER VARIABLES
extern uint64_t mts, mbs, mbo, mcs, mco, wts, wto, wcs, wco, mvs;
extern ofstream output_file;

// cuTS MPI VARIABLES
extern int wsize;
extern int grank;
extern char msg_buffer[NUMBER_OF_PROCESSESS][100];             // for every task there is a seperate message buffer and incoming/outgoing handle slot
extern MPI_Request rq_send_msg[NUMBER_OF_PROCESSESS];          // array of handles for messages with all other thread, allows for asynchronous messaging, handles say whether message is complete
extern MPI_Request rq_recv_msg[NUMBER_OF_PROCESSESS];
extern bool global_free_list[NUMBER_OF_PROCESSESS];

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        cout << cudaGetErrorString(code) << endl;
        exit(-1);
    }
}

#endif // DCUQC_COMMON_H