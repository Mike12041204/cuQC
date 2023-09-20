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
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sm_30_intrinsics.h>
#include <device_atomic_functions.h>
using namespace std;

// GPU: GTX 1660 Super
// SM's: 22
// Threads per SM: 1024
// Global Memory: 6 GB
// Shared Memory: 48 KB

// global memory size: 1.500.000.000 ints
#define TASKS_SIZE 15000000
#define EXPAND_THRESHOLD 286
#define BUFFER_SIZE 100000000
#define BUFFER_OFFSET_SIZE 1000000
#define CLIQUES_SIZE 50000000
#define CLIQUES_OFFSET_SIZE 500000
#define CLIQUES_PERCENT 50

// per warp
#define WCLIQUES_SIZE 50000
#define WCLIQUES_OFFSET_SIZE 500
#define WTASKS_SIZE 1000000
#define WTASKS_OFFSET_SIZE 5000
#define WVERTICES_SIZE 40000

// shared memory size: 12.300 ints
#define VERTICES_SIZE 110
 
#define BLOCK_SIZE 416
#define NUM_OF_BLOCKS 22
#define WARP_SIZE 32

// VERTEX DATA
struct Vertex
{
    int vertexid;
    int label;
    int indeg;
    int exdeg;
    int lvl2adj;
};

// CPU GRAPH / CONSTRUCTOR
class CPU_Graph
{
    public:

    int number_of_vertices;
    int number_of_edges;

    // one dimentional arrays of 1hop and 2hop neighbors and the offsets for each vertex
    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;

    uint64_t number_of_onehop_neighbors;
    uint64_t number_of_twohop_neighbors;

    CPU_Graph::CPU_Graph(ifstream& graph_stream)
    {
        // used to allocate memory for neighbor arrays
        number_of_onehop_neighbors = 0;
        number_of_twohop_neighbors = 0;

        // vectors of sets of 1hop and 2hop neighbors
        vector<set<int>> onehop_neighbors_vector;
        vector<set<int>> twohop_neighbors_vector;

        // generate 1hop neighbors vector
        string line;
        string neighbor;
        while (graph_stream.good()) {
            getline(graph_stream, line);
            if (line.length() != 0) {
                stringstream neighbor_stream(line);
                set<int> tempset;
                while (!neighbor_stream.eof()) {
                    getline(neighbor_stream, neighbor, ' ');
                    int neighbor_id;
                    try {
                        neighbor_id = stoi(neighbor);
                        tempset.insert(neighbor_id);
                        number_of_onehop_neighbors++;
                    }
                    catch (const std::invalid_argument& e) {}
                }
                onehop_neighbors_vector.push_back(tempset);
            }
            else {
                set<int> tempset;
                onehop_neighbors_vector.push_back(tempset);
            }
        }

        // set V and E
        number_of_vertices = onehop_neighbors_vector.size();
        number_of_edges = number_of_onehop_neighbors / 2;

        // generate 2hop neighbors vector
        int current_vertex = 0;
        for (set<int> vertex_neighbors : onehop_neighbors_vector) {
            set<int> tempset(vertex_neighbors);
            for (int neighbor : vertex_neighbors) {
                for (int twohop_neighbor : onehop_neighbors_vector.at(neighbor)) {
                    if (twohop_neighbor != current_vertex) {
                        tempset.insert(twohop_neighbor);
                    }
                }
            }
            twohop_neighbors_vector.push_back(tempset);
            number_of_twohop_neighbors += tempset.size();
            current_vertex++;
        }

        // convert onehop vector to arrays
        onehop_neighbors = new int[number_of_onehop_neighbors];
        onehop_offsets = new uint64_t[number_of_vertices + 1];
        if (onehop_neighbors == nullptr || onehop_offsets == nullptr) {
            cout << "ERROR: bad malloc" << endl;
        }
        onehop_offsets[0] = 0;
        int offset = 0;
        for (int i = 0; i < onehop_neighbors_vector.size(); i++) {
            offset += onehop_neighbors_vector.at(i).size();
            onehop_offsets[i + 1] = offset;
            int j = 0;
            for (int neighbor : onehop_neighbors_vector.at(i)) {
                onehop_neighbors[onehop_offsets[i] + j] = neighbor;
                j++;
            }
        }

        //convert twohop vector to arrays
        twohop_neighbors = new int[number_of_twohop_neighbors];
        twohop_offsets = new uint64_t[number_of_vertices + 1];
        if (twohop_neighbors == nullptr || twohop_offsets == nullptr) {
            cout << "ERROR: bad malloc" << endl;
        }
        twohop_offsets[0] = 0;
        offset = 0;
        for (int i = 0; i < twohop_neighbors_vector.size(); i++) {
            offset += twohop_neighbors_vector.at(i).size();
            twohop_offsets[i + 1] = offset;
            int j = 0;
            for (int neighbor : twohop_neighbors_vector.at(i)) {
                twohop_neighbors[twohop_offsets[i] + j] = neighbor;
                j++;
            }
        }
    }

    CPU_Graph::~CPU_Graph() {
        delete onehop_neighbors;
        delete onehop_offsets;
        delete twohop_neighbors;
        delete twohop_offsets;
    }
};

// CPU DATA
struct CPU_Data
{
    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    bool* maximal_expansion;
    bool* dumping_cliques;
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

    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    Vertex* tasks1_vertices;

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    Vertex* wtasks_vertices;

    Vertex* wvertices;

    int* total_tasks;

    bool* maximal_expansion;
    bool* dumping_cliques;

    double* minimum_degree_ratio;
    int* minimum_degrees;
    int* minimum_clique_size;

    uint64_t* buffer_offset_start;
    uint64_t* buffer_start;
    uint64_t* cliques_offset_start;
    uint64_t* cliques_start;

    // DEBUG
    bool* debug;
    int* idebug;

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
};

// WARP DATA
struct Warp_Data
{
    uint64_t start[(BLOCK_SIZE / WARP_SIZE)];
    uint64_t end[(BLOCK_SIZE / WARP_SIZE)];
    int tot_vert[(BLOCK_SIZE / WARP_SIZE)];
    int num_mem[(BLOCK_SIZE / WARP_SIZE)];
    int num_cand[(BLOCK_SIZE / WARP_SIZE)];
    int expansions[(BLOCK_SIZE / WARP_SIZE)];

    int number_of_members[(BLOCK_SIZE / WARP_SIZE)];
    int number_of_candidates[(BLOCK_SIZE / WARP_SIZE)];
    int total_vertices[(BLOCK_SIZE / WARP_SIZE)];

    Vertex shared_vertices[VERTICES_SIZE * (BLOCK_SIZE / WARP_SIZE)];

    int minimum_external_degree[(BLOCK_SIZE / WARP_SIZE)];
    int Lower_bound[(BLOCK_SIZE / WARP_SIZE)];
    int Upper_bound[(BLOCK_SIZE / WARP_SIZE)];

    int tightened_Upper_bound[(BLOCK_SIZE / WARP_SIZE)];
    int min_clq_indeg[(BLOCK_SIZE / WARP_SIZE)];
    int min_indeg_exdeg[(BLOCK_SIZE / WARP_SIZE)];
    int min_clq_totaldeg[(BLOCK_SIZE / WARP_SIZE)];
    int sum_clq_indeg[(BLOCK_SIZE / WARP_SIZE)];
    int sum_candidate_indeg[(BLOCK_SIZE / WARP_SIZE)];

    bool invalid_bounds[(BLOCK_SIZE / WARP_SIZE)];
    bool failed_found[(BLOCK_SIZE / WARP_SIZE)];
};

// LOCAL DATA
struct Local_Data
{
    Vertex* read_vertices;
    uint64_t* read_offsets;
    uint64_t* read_count;

    Vertex* vertices;
    int idx;
    int warp_in_block_idx;
};

// METHODS
void calculate_minimum_degrees(CPU_Graph& graph);
void search(CPU_Graph& input_graph, ofstream& temp_results);
void allocate_memory(CPU_Data& host_data, GPU_Data& dd, CPU_Cliques& host_cliques, CPU_Graph& input_graph);
void initialize_tasks(CPU_Graph& graph, CPU_Data& host_data);
void move_to_gpu(CPU_Data& host_data, GPU_Data& dd);
void dump_cliques(CPU_Cliques& host_cliques, GPU_Data& dd, ofstream& output_file);
void free_memory(CPU_Data& host_data, GPU_Data& dd, CPU_Cliques& host_cliques);
void RemoveNonMax(char* szset_filename, char* szoutput_filename);

int binary_search_array(int* search_array, int array_size, int search_number);
int sort_vertices(const void* a, const void* b);
inline int get_mindeg(int clique_size);
inline bool cand_isvalid(Vertex& vertex, int clique_size);
inline void chkerr(cudaError_t code);

void print_CPU_Data(CPU_Data& host_data);
void print_GPU_Data(GPU_Data& dd);
void print_CPU_Graph(CPU_Graph& host_graph);
void print_GPU_Graph(GPU_Data& dd, CPU_Graph& host_graph);
void print_WTask_Buffers(GPU_Data& dd);
void print_WClique_Buffers(GPU_Data& dd);
void print_GPU_Cliques(GPU_Data& dd);
void print_CPU_Cliques(CPU_Cliques& host_cliques);
void print_Data_Sizes(GPU_Data& dd);
void print_vertices(Vertex* vertices, int size);
void print_Data_Sizes_Every(GPU_Data& dd, int every);
void print_Warp_Data_Sizes(GPU_Data& dd);
void print_All_Warp_Data_Sizes(GPU_Data& dd);
void print_Warp_Data_Sizes_Every(GPU_Data& dd, int every);
void print_All_Warp_Data_Sizes_Every(GPU_Data& dd, int every);
void print_debug(GPU_Data& dd);
void print_idebug(GPU_Data& dd);
void print_idebug(GPU_Data& dd);

// KERNELS
__global__ void expand_level(GPU_Data dd);
__global__ void transfer_buffers(GPU_Data dd);
__global__ void fill_from_buffer(GPU_Data dd);
__device__ int lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid);
__device__ void degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, bool& failed_found);
__device__ void update_degrees(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_removed_candidates);
__device__ void calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);

__device__ void degree_pruning_nonLU(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, bool& failed_found);
__device__ void device_sort(Vertex* target, int size, int lane_idx);
__device__ __forceinline int sort_vert(Vertex& vertex1, Vertex& vertex2);
__device__ int device_bsearch_array(int* search_array, int array_size, int search_number);
__device__ __forceinline bool device_cand_isvalid(Vertex& vertex, int number_of_members, GPU_Data& dd);
__device__ __forceinline bool device_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ __forceinline bool device_vert_isextendable(Vertex& vertex, int number_of_members, GPU_Data& dd);
__device__ __forceinline bool device_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ __forceinline int device_get_mindeg(int number_of_members, GPU_Data& dd);

// TODO - make local
// INPUT SETTINGS
double minimum_degree_ratio;
int minimum_clique_size;
int* minimum_degrees;



// TODO - test program on larger graphs
// TODO - increase thread usage by monitoring and improving memory usage
// TODO - test if it would be beneficial to coalesce memory access in for loops throughout the program, check out cuts writing on this

// MAIN
int main(int argc, char* argv[])
{
    // ENSURE PROPER USAGE
    if (argc != 5) {
        printf("Usage: ./main <graph_file> <gamma> <min_size> <output_file.txt>\n");
        return 1;
    }
    ifstream graph_stream(argv[1], ios::in);
    if (!graph_stream.is_open()) {
        printf("invalid graph file\n");
        return 1;
    }
    minimum_degree_ratio = atof(argv[2]);
    if (minimum_degree_ratio < .5 || minimum_degree_ratio>1) {
        printf("minimum degree ratio must be between .5 and 1 inclusive\n");
        return 1;
    }
    minimum_clique_size = atoi(argv[3]);
    if (minimum_clique_size <= 1) {
        printf("minimum size must be greater than 1\n");
        return 1;
    }

    // TIME
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // GRAPH / MINDEGS
    cout << ">:PRE-PROCESSING" << endl;
    CPU_Graph input_graph(graph_stream);
    graph_stream.close();
    calculate_minimum_degrees(input_graph);
    ofstream temp_results("temp.txt");

    // DEBUG
    //print_CPU_Graph(input_graph);

    // SEARCH
    search(input_graph, temp_results);

    temp_results.close();

    // RM NON-MAX
    RemoveNonMax("temp.txt", argv[4]);

    // Record the stop event
    cudaEventRecord(stop);

    // TIME
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << ">:TIME: " << milliseconds << " ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << ">:PROGRAM END" << endl;
    return 0;
}



// --- HOST METHODS --- 

// initializes minimum degrees array
void calculate_minimum_degrees(CPU_Graph& graph)
{
    minimum_degrees = new int[graph.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 1; i <= graph.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}

void search(CPU_Graph& input_graph, ofstream& temp_results) 
{
    // DATA STRUCTURES
    CPU_Data host_data;
    CPU_Cliques host_cliques;
    GPU_Data dd;

    // HANDLE MEMORY
    allocate_memory(host_data, dd, host_cliques, input_graph);
    cudaDeviceSynchronize();

    // INITIALIZE TASKS
    cout << ">:INITIALIZING TASKS" << endl;
    initialize_tasks(input_graph, host_data);

    // TRANSFER TO GPU
    move_to_gpu(host_data, dd);
    cudaDeviceSynchronize();

    // DEBUG
    //print_GPU_Graph(dd, input_graph);
    //print_CPU_Data(host_data);
    //print_GPU_Data(dd);
    print_Data_Sizes(dd);

    // TODO - check cuts for cudaDeviceSynchronize
    // EXPAND LEVEL
    cout << ">:BEGINNING EXPANSION" << endl;
    while (!(*host_data.maximal_expansion))
    {
        // reset loop variables
        chkerr(cudaMemset(dd.maximal_expansion, true, sizeof(bool)));
        chkerr(cudaMemset(dd.dumping_cliques, false, sizeof(bool)));
        cudaDeviceSynchronize();

        // expand all tasks in 'tasks' array, each warp will write to their respective warp tasks buffer in global memory
        expand_level<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();

        // DEBUG
        //print_WClique_Buffers(dd);
        //print_WTask_Buffers(dd);
        print_Warp_Data_Sizes_Every(dd, 1);
        //print_All_Warp_Data_Sizes_Every(dd, 1);

        // consolidate all the warp tasks/cliques buffers into the next global tasks array, buffer, and cliques
        transfer_buffers<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();

        // if not enough tasks were generated when expanding the previous level to fill the next tasks array the program will attempt to fill the tasks array by popping tasks from the buffer
        fill_from_buffer<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();

        // update the loop variables
        chkerr(cudaMemcpy(host_data.maximal_expansion, dd.maximal_expansion, sizeof(bool), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(host_data.dumping_cliques, dd.dumping_cliques, sizeof(bool), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        if (*host_data.dumping_cliques) {
            dump_cliques(host_cliques, dd, temp_results);
        }

        // DEBUG
        //print_GPU_Data(dd);
        //print_GPU_Cliques(dd);
        print_Data_Sizes_Every(dd, 1);
        //print_debug(dd);
        //print_idebug(dd);
    }

    dump_cliques(host_cliques, dd, temp_results);

    // FREE MEMORY
    free_memory(host_data, dd, host_cliques);
}

// allocates memory for the data structures on the host and device
void allocate_memory(CPU_Data& host_data, GPU_Data& dd, CPU_Cliques& host_cliques, CPU_Graph& input_graph)
{
    int number_of_warps = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;

    // GPU GRAPH
    chkerr(cudaMalloc((void**)&dd.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.onehop_neighbors, sizeof(int) * input_graph.number_of_onehop_neighbors));
    chkerr(cudaMalloc((void**)&dd.onehop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&dd.twohop_neighbors, sizeof(int) * input_graph.number_of_twohop_neighbors));
    chkerr(cudaMalloc((void**)&dd.twohop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1)));

    chkerr(cudaMemcpy(dd.number_of_vertices, &(input_graph.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.number_of_edges, &(input_graph.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.onehop_neighbors, input_graph.onehop_neighbors, sizeof(int) * input_graph.number_of_onehop_neighbors, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.onehop_offsets, input_graph.onehop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.twohop_neighbors, input_graph.twohop_neighbors, sizeof(int) * input_graph.number_of_twohop_neighbors, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.twohop_offsets, input_graph.twohop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));

    // CPU DATA
    host_data.tasks1_count = new uint64_t;
    host_data.tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    host_data.tasks1_vertices = new Vertex[TASKS_SIZE];

    host_data.tasks1_offset[0] = 0;
    (*(host_data.tasks1_count)) = 0;

    host_data.buffer_count = new uint64_t;
    host_data.buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    host_data.buffer_vertices = new Vertex[BUFFER_SIZE];

    host_data.buffer_offset[0] = 0;
    (*(host_data.buffer_count)) = 0;

    host_data.maximal_expansion = new bool;
    host_data.dumping_cliques = new bool;

    (*host_data.maximal_expansion) = false;
    (*host_data.dumping_cliques) = false;

    // GPU DATA
    chkerr(cudaMalloc((void**)&dd.current_level, sizeof(uint64_t)));

    uint64_t temp = 1;
    chkerr(cudaMemcpy(dd.current_level, &temp, sizeof(uint64_t), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&dd.tasks1_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.tasks1_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&dd.tasks1_vertices, sizeof(Vertex) * TASKS_SIZE));

    chkerr(cudaMemset(dd.tasks1_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.tasks1_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.tasks2_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.tasks2_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&dd.tasks2_vertices, sizeof(Vertex) * TASKS_SIZE));

    chkerr(cudaMemset(dd.tasks2_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.tasks2_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.buffer_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.buffer_offset, sizeof(uint64_t) * BUFFER_OFFSET_SIZE));
    chkerr(cudaMalloc((void**)&dd.buffer_vertices, sizeof(Vertex) * BUFFER_SIZE));

    chkerr(cudaMemset(dd.buffer_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.buffer_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.wtasks_count, sizeof(uint64_t) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.wtasks_offset, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.wtasks_vertices, (sizeof(Vertex) * WTASKS_SIZE) * number_of_warps));

    chkerr(cudaMemset(dd.wtasks_offset, 0, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * number_of_warps));
    chkerr(cudaMemset(dd.wtasks_count, 0, sizeof(uint64_t) * number_of_warps));

    chkerr(cudaMalloc((void**)&dd.wvertices, (sizeof(Vertex) * WVERTICES_SIZE) * number_of_warps));

    chkerr(cudaMalloc((void**)&dd.maximal_expansion, sizeof(bool)));
    chkerr(cudaMalloc((void**)&dd.dumping_cliques, sizeof(bool)));

    chkerr(cudaMemset(dd.maximal_expansion, false, sizeof(bool)));
    chkerr(cudaMemset(dd.dumping_cliques, false, sizeof(bool)));

    chkerr(cudaMalloc((void**)&dd.minimum_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&dd.minimum_degrees, sizeof(int) * (input_graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&dd.minimum_clique_size, sizeof(int)));

    chkerr(cudaMemcpy(dd.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.minimum_degrees, minimum_degrees, sizeof(int) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&dd.total_tasks, sizeof(int)));

    chkerr(cudaMemset(dd.total_tasks, 0, sizeof(int)));

    // CPU CLIQUES
    host_cliques.cliques_count = new uint64_t;
    host_cliques.cliques_vertex = new int[CLIQUES_SIZE];
    host_cliques.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];

    host_cliques.cliques_offset[0] = 0;
    (*(host_cliques.cliques_count)) = 0;

    // GPU CLIQUES
    chkerr(cudaMalloc((void**)&dd.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE));
    chkerr(cudaMalloc((void**)&dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));

    chkerr(cudaMemset(dd.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(dd.cliques_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&dd.wcliques_count, sizeof(uint64_t) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.wcliques_offset, (sizeof(uint64_t)* WCLIQUES_OFFSET_SIZE)* number_of_warps));
    chkerr(cudaMalloc((void**)&dd.wcliques_vertex, (sizeof(int) * WCLIQUES_SIZE) * number_of_warps));

    chkerr(cudaMemset(dd.wcliques_offset, 0, (sizeof(uint64_t)* WCLIQUES_OFFSET_SIZE)* number_of_warps));
    chkerr(cudaMemset(dd.wcliques_count, 0, sizeof(uint64_t)* number_of_warps));

    chkerr(cudaMalloc((void**)&dd.total_cliques, sizeof(int)));

    chkerr(cudaMemset(dd.total_cliques, 0, sizeof(int)));

    chkerr(cudaMalloc((void**)&dd.buffer_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.buffer_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.cliques_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&dd.cliques_start, sizeof(uint64_t)));

    // DEBUG
    chkerr(cudaMalloc((void**)&dd.debug, sizeof(bool)));
    chkerr(cudaMalloc((void**)&dd.idebug, sizeof(int)));

    chkerr(cudaMemset(dd.debug, false, sizeof(bool)));
    chkerr(cudaMemset(dd.idebug, 0, sizeof(int)));
}

// processes 0th and 1st level of expansion
void initialize_tasks(CPU_Graph& graph, CPU_Data& host_data)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int pneighbors_count;
    int phelper1;
    int phelper2;

    // cover pruning
    int number_of_covered_vertices;
    int maximum_degree;
    int maximum_degree_index;

    // degree pruning
    int number_of_removed_candidates;

    // vertices information
    int expansions;
    int total_vertices;
    Vertex* old_vertices;
    int total_new_vertices;
    Vertex* new_vertices;



    // initialize vertices
    total_vertices = graph.number_of_vertices;
    old_vertices = new Vertex[total_vertices];
    for (int i = 0; i < total_vertices; i++) {
        old_vertices[i].vertexid = i;
        old_vertices[i].label = 0;
        old_vertices[i].indeg = 0;
        old_vertices[i].exdeg = graph.onehop_offsets[i + 1] - graph.onehop_offsets[i];
        old_vertices[i].lvl2adj = graph.twohop_offsets[i + 1] - graph.twohop_offsets[i];
    }



    // DEGREE-BASED PRUNING
    do {
        // remove cands that do not meet the deg requirement
        number_of_removed_candidates = 0;
        for (int i = 0; i < total_vertices; i++) {
            if (!cand_isvalid(old_vertices[i], 0)) {
                old_vertices[i].label = -1;
                number_of_removed_candidates++;
            }
        }
        qsort(old_vertices, total_vertices, sizeof(Vertex), sort_vertices);

        for (int i = 0; i < total_vertices - number_of_removed_candidates; i++) {
            pvertexid = old_vertices[i].vertexid;
            for (int j = total_vertices - number_of_removed_candidates; j < total_vertices; j++) {
                phelper1 = old_vertices[j].vertexid;
                pneighbors_start = graph.onehop_offsets[phelper1];
                pneighbors_end = graph.onehop_offsets[phelper1 + 1];
                pneighbors_count = pneighbors_end - pneighbors_start;
                phelper2 = binary_search_array(graph.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                if (phelper2 != -1) {
                    old_vertices[i].exdeg--;
                }

                pneighbors_start = graph.twohop_offsets[phelper1];
                pneighbors_end = graph.twohop_offsets[phelper1 + 1];
                pneighbors_count = pneighbors_end - pneighbors_start;
                phelper2 = binary_search_array(graph.twohop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                if (phelper2 != -1) {
                    old_vertices[i].lvl2adj--;
                }
            }
        }
        total_vertices -= number_of_removed_candidates;
    } while (number_of_removed_candidates > 0);
    


    // FIRST ROUND COVER PRUNING
    maximum_degree = 0;
    maximum_degree_index = 0;
    for (int i = 0; i < total_vertices; i++) {
        if (old_vertices[i].exdeg > maximum_degree) {
            maximum_degree = old_vertices[i].exdeg;
            maximum_degree_index = i;
        }
    }
    old_vertices[maximum_degree_index].label = 3;

    // set all neighbors of cover vertices as covered
    pvertexid = old_vertices[maximum_degree_index].vertexid;
    qsort(old_vertices, total_vertices, sizeof(Vertex), sort_vertices);
    number_of_covered_vertices = 0;
    for (int i = 0; i < total_vertices-1; i++) {
        phelper1 = old_vertices[i].vertexid;
        pneighbors_start = graph.onehop_offsets[phelper1];
        pneighbors_end = graph.onehop_offsets[phelper1 + 1];
        pneighbors_count = pneighbors_end - pneighbors_start;
        phelper2 = binary_search_array(graph.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
        if (phelper2 != -1) {
            old_vertices[i].label = 2;
            number_of_covered_vertices++;
        }
    }
    qsort(old_vertices, total_vertices, sizeof(Vertex), sort_vertices);



    // NEXT LEVEL
    expansions = total_vertices;
    for (int i = number_of_covered_vertices; i < expansions; i++)
    {



        // REMOVE CANDIDATE
        // only done after first iteration of for loop
        if (i > number_of_covered_vertices) {
            total_vertices--;

            // update info of vertices connected to removed cand
            pvertexid = old_vertices[total_vertices].vertexid;
            for (int j = 0; j < total_vertices; j++) {
                phelper1 = old_vertices[j].vertexid;
                pneighbors_start = graph.onehop_offsets[phelper1];
                pneighbors_end = graph.onehop_offsets[phelper1];
                pneighbors_count = pneighbors_end - pneighbors_start;
                phelper2 = binary_search_array(graph.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                if (phelper2 != -1) {
                    old_vertices[j].exdeg--;
                }

                pneighbors_start = graph.twohop_offsets[phelper1];
                pneighbors_end = graph.twohop_offsets[phelper1];
                pneighbors_count = pneighbors_end - pneighbors_start;
                phelper2 = binary_search_array(graph.twohop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                if (phelper2 != -1) {
                    old_vertices[j].lvl2adj--;
                }
            }
        }

        // break if not enough vertices as only less will be added in the next iteration
        if (total_vertices < minimum_clique_size) {
            break;
        }



        // NEW VERTICES
        new_vertices = new Vertex[total_vertices];
        total_new_vertices = total_vertices;
        for (int j = 0; j < total_new_vertices; j++) {
            new_vertices[j].vertexid = old_vertices[j].vertexid;
            new_vertices[j].label = old_vertices[j].label;
            new_vertices[j].indeg = old_vertices[j].indeg;
            new_vertices[j].exdeg = old_vertices[j].exdeg;
            new_vertices[j].lvl2adj = old_vertices[j].lvl2adj;
        }

        // set all covered vertices from previous level as candidates
        for (int j = 0; j < number_of_covered_vertices; j++) {
            new_vertices[j].label = 0;
        }
        


        // ADD ONE VERTEX
        new_vertices[total_new_vertices - 1].label = 1;
        pvertexid = new_vertices[total_new_vertices - 1].vertexid;
        for (int j = 0; j < total_vertices; j++) {
            phelper1 = new_vertices[j].vertexid;
            pneighbors_start = graph.onehop_offsets[phelper1];
            pneighbors_end = graph.onehop_offsets[phelper1 + 1];
            pneighbors_count = pneighbors_end - pneighbors_start;
            phelper2 = binary_search_array(graph.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
            if (phelper2 != -1) {
                new_vertices[j].exdeg--;
                new_vertices[j].indeg++;
            }
        }
        qsort(new_vertices, total_new_vertices, sizeof(Vertex), sort_vertices);



        // DIAMETER PRUNING
        number_of_removed_candidates = 0;
        pneighbors_start = graph.twohop_offsets[pvertexid];
        pneighbors_end = graph.twohop_offsets[pvertexid + 1];
        pneighbors_count = pneighbors_end - pneighbors_start;
        for (int j = 1; j < total_new_vertices; j++) {
            phelper1 = new_vertices[j].vertexid;
            phelper2 = binary_search_array(graph.twohop_neighbors + pneighbors_start, pneighbors_count, phelper1);
            if (phelper2 == -1) {
                new_vertices[j].label = -1;
                number_of_removed_candidates++;
            }
        }
        qsort(new_vertices, total_new_vertices, sizeof(Vertex), sort_vertices);

        // update exdeg of vertices connected to removed cands
        for (int i = 0; i < total_new_vertices - number_of_removed_candidates; i++) {
            pvertexid = new_vertices[i].vertexid;
            for (int j = total_new_vertices - number_of_removed_candidates; j < total_new_vertices; j++) {
                phelper1 = new_vertices[j].vertexid;
                pneighbors_start = graph.onehop_offsets[phelper1];
                pneighbors_end = graph.onehop_offsets[phelper1 + 1];
                pneighbors_count = pneighbors_end - pneighbors_start;
                phelper2 = binary_search_array(graph.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                if (phelper2 != -1) {
                    new_vertices[i].exdeg--;
                }

                pneighbors_start = graph.twohop_offsets[phelper1];
                pneighbors_end = graph.twohop_offsets[phelper1 + 1];
                pneighbors_count = pneighbors_end - pneighbors_start;
                phelper2 = binary_search_array(graph.twohop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                if (phelper2 != -1) {
                    new_vertices[i].lvl2adj--;
                }
            }
        }
        total_new_vertices -= number_of_removed_candidates;

        // continue if not enough vertices after pruning
        if (total_new_vertices < minimum_clique_size) {
            continue;
        }



        // DEGREE-BASED PRUNING
        do {
            // remove cands that do not meet the deg requirement
            number_of_removed_candidates = 0;
            for (int j = 1; j < total_new_vertices; j++) {
                if (!cand_isvalid(new_vertices[j], 1)) {
                    new_vertices[j].label = -1;
                    number_of_removed_candidates++;
                }
            }
            qsort(new_vertices, total_new_vertices, sizeof(Vertex), sort_vertices);

            // update exdeg of vertices connected to removed cands
            for (int i = 0; i < total_new_vertices - number_of_removed_candidates; i++) {
                pvertexid = new_vertices[i].vertexid;
                for (int j = total_new_vertices - number_of_removed_candidates; j < total_new_vertices; j++) {
                    phelper1 = new_vertices[j].vertexid;
                    pneighbors_start = graph.onehop_offsets[phelper1];
                    pneighbors_end = graph.onehop_offsets[phelper1 + 1];
                    pneighbors_count = pneighbors_end - pneighbors_start;
                    phelper2 = binary_search_array(graph.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                    if (phelper2 != -1) {
                        new_vertices[i].exdeg--;
                    }

                    pneighbors_start = graph.twohop_offsets[phelper1];
                    pneighbors_end = graph.twohop_offsets[phelper1 + 1];
                    pneighbors_count = pneighbors_end - pneighbors_start;
                    phelper2 = binary_search_array(graph.twohop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
                    if (phelper2 != -1) {
                        new_vertices[i].lvl2adj--;
                    }
                }
            }
            total_new_vertices -= number_of_removed_candidates;
        } while (number_of_removed_candidates > 0);

        // continue if not enough vertices after pruning
        if (total_new_vertices < minimum_clique_size) {
            continue;
        }



        // WRITE TO TASKS
        if (total_new_vertices - 1 > 0) 
        {
            if ((*(host_data.tasks1_count)) < EXPAND_THRESHOLD) {
                uint64_t start_write = host_data.tasks1_offset[(*(host_data.tasks1_count))];

                for (int j = 0; j < total_new_vertices; j++) {
                    host_data.tasks1_vertices[start_write + j].vertexid = new_vertices[j].vertexid;
                    host_data.tasks1_vertices[start_write + j].label = new_vertices[j].label;
                    host_data.tasks1_vertices[start_write + j].indeg = new_vertices[j].indeg;
                    host_data.tasks1_vertices[start_write + j].exdeg = new_vertices[j].exdeg;
                    host_data.tasks1_vertices[start_write + j].lvl2adj = new_vertices[j].lvl2adj;
                }
                (*(host_data.tasks1_count))++;
                host_data.tasks1_offset[(*(host_data.tasks1_count))] = start_write + total_new_vertices;
            }
            else {
                uint64_t start_write = host_data.buffer_offset[(*(host_data.buffer_count))];

                for (int j = 0; j < total_new_vertices; j++) {
                    host_data.buffer_vertices[start_write + j].vertexid = new_vertices[j].vertexid;
                    host_data.buffer_vertices[start_write + j].label = new_vertices[j].label;
                    host_data.buffer_vertices[start_write + j].indeg = new_vertices[j].indeg;
                    host_data.buffer_vertices[start_write + j].exdeg = new_vertices[j].exdeg;
                    host_data.buffer_vertices[start_write + j].lvl2adj = new_vertices[j].lvl2adj;
                }
                (*(host_data.buffer_count))++;
                host_data.buffer_offset[(*(host_data.buffer_count))] = start_write + total_new_vertices;
            }
        }
        delete new_vertices;
    }
    delete old_vertices;
}

void move_to_gpu(CPU_Data& host_data, GPU_Data& dd)
{
    cudaMemcpy(dd.tasks1_count, host_data.tasks1_count, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dd.tasks1_offset, host_data.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dd.tasks1_vertices, host_data.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyHostToDevice);

    cudaMemcpy(dd.buffer_count, host_data.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dd.buffer_offset, host_data.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dd.buffer_vertices, host_data.buffer_vertices, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
}

void dump_cliques(CPU_Cliques& host_cliques, GPU_Data& dd, ofstream& temp_results)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(host_cliques.cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(host_cliques.cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(host_cliques.cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // DEBUG
    //print_CPU_Cliques(host_cliques);

    for (int i = 0; i < ((*host_cliques.cliques_count)); i++) {
        uint64_t start = host_cliques.cliques_offset[i];
        uint64_t end = host_cliques.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << host_cliques.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    ((*host_cliques.cliques_count)) = 0;
    cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
}

void free_memory(CPU_Data& host_data, GPU_Data& dd, CPU_Cliques& host_cliques)
{
    // GPU GRAPH
    chkerr(cudaFree(dd.number_of_vertices));
    chkerr(cudaFree(dd.number_of_edges));
    chkerr(cudaFree(dd.onehop_neighbors));
    chkerr(cudaFree(dd.onehop_offsets));
    chkerr(cudaFree(dd.twohop_neighbors));
    chkerr(cudaFree(dd.twohop_offsets));

    // CPU DATA
    delete host_data.tasks1_count;
    delete host_data.tasks1_offset;
    delete host_data.tasks1_vertices;

    delete host_data.buffer_count;
    delete host_data.buffer_offset;
    delete host_data.buffer_vertices;

    delete host_data.maximal_expansion;
    delete host_data.dumping_cliques;

    // GPU DATA
    chkerr(cudaFree(dd.current_level));

    chkerr(cudaFree(dd.tasks1_count));
    chkerr(cudaFree(dd.tasks1_offset));
    chkerr(cudaFree(dd.tasks1_vertices));

    chkerr(cudaFree(dd.tasks2_count));
    chkerr(cudaFree(dd.tasks2_offset));
    chkerr(cudaFree(dd.tasks2_vertices));

    chkerr(cudaFree(dd.buffer_count));
    chkerr(cudaFree(dd.buffer_offset));
    chkerr(cudaFree(dd.buffer_vertices));

    chkerr(cudaFree(dd.wtasks_count));
    chkerr(cudaFree(dd.wtasks_offset));
    chkerr(cudaFree(dd.wtasks_vertices));

    chkerr(cudaFree(dd.wvertices));

    chkerr(cudaFree(dd.maximal_expansion));
    chkerr(cudaFree(dd.dumping_cliques));

    chkerr(cudaFree(dd.minimum_degree_ratio));
    chkerr(cudaFree(dd.minimum_degrees));
    chkerr(cudaFree(dd.minimum_clique_size));

    chkerr(cudaFree(dd.total_tasks));

    // CPU CLIQUES
    delete host_cliques.cliques_count;
    delete host_cliques.cliques_vertex;
    delete host_cliques.cliques_offset;

    // GPU CLIQUES
    chkerr(cudaFree(dd.cliques_count));
    chkerr(cudaFree(dd.cliques_vertex));
    chkerr(cudaFree(dd.cliques_offset));

    chkerr(cudaFree(dd.wcliques_count));
    chkerr(cudaFree(dd.wcliques_vertex));
    chkerr(cudaFree(dd.wcliques_offset));

    chkerr(cudaFree(dd.buffer_offset_start));
    chkerr(cudaFree(dd.buffer_start));
    chkerr(cudaFree(dd.cliques_offset_start));
    chkerr(cudaFree(dd.cliques_start));

    //DEBUG
    chkerr(cudaFree(dd.debug));
    chkerr(cudaFree(dd.idebug));
}



// --- HELPER METHODS ---

// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
int binary_search_array(int* search_array, int array_size, int search_number)
{
    // ALGO - binary
    // TYPE - serial
    // SPEED - 0(log(n))

    if (array_size <= 0) {
        return -1;
    }

    if (search_array[array_size / 2] == search_number) {
        // Base case: Center element matches search number
        return array_size / 2;
    }
    else if (search_array[array_size / 2] > search_number) {
        // Recursively search lower half
        return binary_search_array(search_array, array_size / 2, search_number);
    }
    else {
        // Recursively search upper half
        int upper_half_result = binary_search_array(search_array + array_size / 2 + 1, array_size - array_size / 2 - 1, search_number);
        return (upper_half_result != -1) ? (array_size / 2 + 1 + upper_half_result) : -1;
    }
}

int sort_vertices(const void* a, const void* b)
{
    // order is: in clique -> covered -> critical adj vertices -> cands -> cover -> pruned

    // in clique
    if ((*(Vertex*)a).label == 1 && (*(Vertex*)b).label != 1) {
        return -1;
    }
    else if ((*(Vertex*)a).label != 1 && (*(Vertex*)b).label == 1) {
        return 1;

        // covered candidate vertices
    }
    else if ((*(Vertex*)a).label == 2 && (*(Vertex*)b).label != 2) {
        return -1;
    }
    else if ((*(Vertex*)a).label != 2 && (*(Vertex*)b).label == 2) {
        return 1;

        // critical adjacent candidate vertices
    }
    else if ((*(Vertex*)a).label == 4 && (*(Vertex*)b).label != 4) {
        return -1;
    }
    else if ((*(Vertex*)a).label != 4 && (*(Vertex*)b).label == 4) {
        return 1;

        // candidate vertices
    }
    else if ((*(Vertex*)a).label == 0 && (*(Vertex*)b).label != 0) {
        return -1;
    }
    else if ((*(Vertex*)a).label != 0 && (*(Vertex*)b).label == 0) {
        return 1;

        // the cover vertex
    }
    else if ((*(Vertex*)a).label == 3 && (*(Vertex*)b).label != 3) {
        return -1;
    }
    else if ((*(Vertex*)a).label != 3 && (*(Vertex*)b).label == 3) {
        return 1;

        // vertices that have been pruned
    }
    else if ((*(Vertex*)a).label == -1 && (*(Vertex*)b).label != 1) {
        return 1;
    }
    else if ((*(Vertex*)a).label != -1 && (*(Vertex*)b).label == -1) {
        return -1;
    }

    // for ties: in clique low -> high, cand high -> low
    else if ((*(Vertex*)a).label == 1 && (*(Vertex*)b).label == 1) {
        if ((*(Vertex*)a).vertexid > (*(Vertex*)b).vertexid) {
            return 1;
        }
        else if ((*(Vertex*)a).vertexid < (*(Vertex*)b).vertexid) {
            return -1;
        }
        else {
            return 0;
        }
    }
    else if ((*(Vertex*)a).label == 0 && (*(Vertex*)b).label == 0) {
        if ((*(Vertex*)a).vertexid > (*(Vertex*)b).vertexid) {
            return -1;
        }
        else if ((*(Vertex*)a).vertexid < (*(Vertex*)b).vertexid) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else if ((*(Vertex*)a).label == 2 && (*(Vertex*)b).label == 2) {
        return 0;
    }
    else if ((*(Vertex*)a).label == -1 && (*(Vertex*)b).label == -1) {
        return 0;
    }
    return 0;
}

inline int get_mindeg(int clique_size) {
    if (clique_size < minimum_clique_size) {
        return minimum_degrees[minimum_clique_size];
    }
    else {
        return minimum_degrees[clique_size];
    }
}

inline bool cand_isvalid(Vertex& vertex, int clique_size) {
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.lvl2adj < minimum_clique_size - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < get_mindeg(clique_size + vertex.exdeg + 1)) {
        return false;
    }
    else {
        return true;
    }
}

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}



// --- DEBUG METHODS ---

void print_CPU_Graph(CPU_Graph& host_graph) {
    cout << endl << " --- (CPU_Graph)host_graph details --- " << endl;
    cout << endl << "|V|: " << host_graph.number_of_vertices << " |E|: " << host_graph.number_of_edges << endl;
    cout << endl << "Onehop Offsets:" << endl;
    for (uint64_t i = 0; i <= host_graph.number_of_vertices; i++) {
        cout << host_graph.onehop_offsets[i] << " ";
    }
    cout << endl << "Onehop Neighbors:" << endl;
    for (uint64_t i = 0; i < host_graph.number_of_onehop_neighbors; i++) {
        cout << host_graph.onehop_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= host_graph.number_of_vertices; i++) {
        cout << host_graph.twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < host_graph.number_of_twohop_neighbors; i++) {
        cout << host_graph.twohop_neighbors[i] << " ";
    }
    cout << endl << endl;
}

void print_GPU_Graph(GPU_Data& dd, CPU_Graph& host_graph)
{
    int* number_of_vertices = new int;
    int* number_of_edges = new int;

    int* onehop_neighbors = new int[host_graph.number_of_onehop_neighbors];
    uint64_t * onehop_offsets = new uint64_t[(host_graph.number_of_vertices)+1];
    int* twohop_neighbors = new int[host_graph.number_of_twohop_neighbors];
    uint64_t * twohop_offsets = new uint64_t[(host_graph.number_of_vertices)+1];

    chkerr(cudaMemcpy(number_of_vertices, dd.number_of_vertices, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(number_of_edges, dd.number_of_edges, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_neighbors, dd.onehop_neighbors, sizeof(int)*host_graph.number_of_onehop_neighbors, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_offsets, dd.onehop_offsets, sizeof(uint64_t)*(host_graph.number_of_vertices+1), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_neighbors, dd.twohop_neighbors, sizeof(int)*host_graph.number_of_twohop_neighbors, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_offsets, dd.twohop_offsets, sizeof(uint64_t)*(host_graph.number_of_vertices+1), cudaMemcpyDeviceToHost));

    cout << endl << " --- (GPU_Graph)device_graph details --- " << endl;
    cout << endl << "|V|: " << (*number_of_vertices) << " |E|: " << (*number_of_edges) << endl;
    cout << endl << "Onehop Offsets:" << endl;
    for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
        cout << onehop_offsets[i] << " ";
    }
    cout << endl << "Onehop Neighbors:" << endl;
    for (uint64_t i = 0; i < host_graph.number_of_onehop_neighbors; i++) {
        cout << onehop_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
        cout << twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < host_graph.number_of_twohop_neighbors; i++) {
        cout << twohop_neighbors[i] << " ";
    }
    cout << endl << endl;

    delete number_of_vertices;
    delete number_of_edges;

    delete onehop_offsets;
    delete onehop_neighbors;
    delete twohop_offsets;
    delete twohop_neighbors;
}

void print_CPU_Data(CPU_Data& host_data)
{
    cout << endl << " --- (CPU_Data)host_data details --- " << endl;
    cout << endl << "Tasks1: " << "Size: " << (*(host_data.tasks1_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(host_data.tasks1_count)); i++) {
        cout << host_data.tasks1_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Buffer: " << "Size: " << (*(host_data.buffer_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(host_data.buffer_count)); i++) {
        cout << host_data.buffer_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_vertices[i].lvl2adj << " ";
    }
    cout << endl << endl;
}

void print_GPU_Data(GPU_Data& dd)
{
    uint64_t* current_level = new uint64_t;

    uint64_t* tasks1_count = new uint64_t;
    uint64_t* tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    Vertex* tasks1_vertices = new Vertex[TASKS_SIZE];

    uint64_t* tasks2_count = new uint64_t;
    uint64_t* tasks2_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    Vertex* tasks2_vertices = new Vertex[TASKS_SIZE];


    uint64_t* buffer_count = new uint64_t;
    uint64_t* buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    Vertex* buffer_vertices = new Vertex[BUFFER_SIZE];


    chkerr(cudaMemcpy(current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks1_count, dd.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_offset, dd.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_vertices, dd.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks2_count, dd.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_offset, dd.tasks2_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_vertices, dd.tasks2_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_offset, dd.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_vertices, dd.buffer_vertices, (BUFFER_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    cout << " --- (GPU_Data)device_data details --- " << endl;
    cout << endl << "Tasks1: Level: " << (*current_level) << " Size: " << (*tasks1_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i <= (*tasks1_count); i++) {
        cout << tasks1_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;

    cout << endl << "Tasks2: " << "Size: " << (*tasks2_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i <= (*tasks2_count); i++) {
        cout << tasks2_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertices[i].lvl2adj << " " << flush;
    }
    cout << endl << endl;

    cout << endl << "Buffer: " << "Size: " << (*buffer_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i <= (*buffer_count); i++) {
        cout << buffer_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;

    delete current_level;

    delete tasks1_count;
    delete tasks1_offset;
    delete tasks1_vertices;

    delete tasks2_count;
    delete tasks2_offset;
    delete tasks2_vertices;

    delete buffer_count;
    delete buffer_offset;
    delete buffer_vertices;
}

// CURSOR - test this method, then run program on larger data sets
void print_Warp_Data_Sizes(GPU_Data& dd)
{
    int number_of_warps = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;

    uint64_t* tasks_counts = new uint64_t[number_of_warps];
    uint64_t* tasks_sizes = new uint64_t[number_of_warps];
    int tasks_tcount = 0;
    int tasks_tsize = 0;
    int tasks_mcount = 0;
    int tasks_msize = 0;
    uint64_t* cliques_counts = new uint64_t[number_of_warps];
    uint64_t* cliques_sizes = new uint64_t[number_of_warps];
    int cliques_tcount = 0;
    int cliques_tsize = 0;
    int cliques_mcount = 0;
    int cliques_msize = 0;

    chkerr(cudaMemcpy(tasks_counts, dd.wtasks_count, sizeof(uint64_t) * number_of_warps, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_counts, dd.wcliques_count, sizeof(uint64_t) * number_of_warps, cudaMemcpyDeviceToHost));
    for (int i = 0; i < number_of_warps; i++) {
        chkerr(cudaMemcpy(tasks_sizes + i, dd.wtasks_offset + (i * WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(cliques_sizes + i, dd.wcliques_offset + (i * WCLIQUES_OFFSET_SIZE) + cliques_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < number_of_warps; i++) {
        tasks_tcount += tasks_counts[i];
        if (tasks_counts[i] > tasks_mcount) {
            tasks_mcount = tasks_counts[i];
        }
        tasks_tsize += tasks_sizes[i];
        if (tasks_sizes[i] > tasks_msize) {
            tasks_msize = tasks_sizes[i];
        }
        cliques_tcount += cliques_counts[i];
        if (cliques_counts[i] > cliques_mcount) {
            cliques_mcount = cliques_counts[i];
        }
        cliques_tsize += cliques_sizes[i];
        if (cliques_sizes[i] > cliques_msize) {
            cliques_msize = cliques_sizes[i];
        }
    }

    cout << "WTasks( TC: " << tasks_tcount << " TS: " << tasks_tsize << " MC: " << tasks_mcount << " MS: " << tasks_msize << ") WCliques ( TC: " << cliques_tcount << " TS: " << cliques_tsize << " MC: " << cliques_mcount << " MS: " << cliques_msize << ")" << endl;

    if (tasks_mcount > WTASKS_OFFSET_SIZE || tasks_msize > WTASKS_OFFSET_SIZE || cliques_mcount > WCLIQUES_OFFSET_SIZE || cliques_msize > WCLIQUES_SIZE) {
        cout << "!!! WBUFFER SIZE ERROR !!!" << endl;
    }

    delete tasks_counts;
    delete tasks_sizes;
    delete cliques_counts;
    delete cliques_sizes;
}

// CURSOR - test this method, then run program on larger data sets
void print_All_Warp_Data_Sizes(GPU_Data& dd)
{
    int number_of_warps = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;

    uint64_t* tasks_counts = new uint64_t[number_of_warps];
    uint64_t* tasks_sizes = new uint64_t[number_of_warps];
    uint64_t* cliques_counts = new uint64_t[number_of_warps];
    uint64_t* cliques_sizes = new uint64_t[number_of_warps];

    chkerr(cudaMemcpy(tasks_counts, dd.wtasks_count, sizeof(uint64_t) * number_of_warps, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_counts, dd.wcliques_count, sizeof(uint64_t) * number_of_warps, cudaMemcpyDeviceToHost));
    for (int i = 0; i < number_of_warps; i++) {
        chkerr(cudaMemcpy(tasks_sizes + i, dd.wtasks_offset + (i * WTASKS_OFFSET_SIZE) + tasks_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(cliques_sizes + i, dd.wcliques_offset + (i * WCLIQUES_OFFSET_SIZE) + cliques_counts[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    cout << "WTasks Sizes: " << flush;
    for (int i = 0; i < number_of_warps; i++) {
        cout << i << ":" << tasks_counts[i] << " " << tasks_sizes[i] << " " << flush;
    }
    cout << "\nWCliques Sizez: " << flush;
    for (int i = 0; i < number_of_warps; i++) {
        cout << i << ":" << cliques_counts[i] << " " << cliques_sizes[i] << " " << flush;
    }

    delete tasks_counts;
    delete tasks_sizes;
    delete cliques_counts;
    delete cliques_sizes;
}

void print_Warp_Data_Sizes_Every(GPU_Data& dd, int every)
{
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        print_Warp_Data_Sizes(dd);
    }
}

void print_All_Warp_Data_Sizes_Every(GPU_Data& dd, int every)
{
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        print_All_Warp_Data_Sizes(dd);
    }
}

void print_debug(GPU_Data& dd)
{
    bool debug;
    chkerr(cudaMemcpy(&debug, dd.debug, sizeof(bool), cudaMemcpyDeviceToHost));
    if (debug) {
        cout << "!!!DEBUG!!! " << endl;
    }
    chkerr(cudaMemset(dd.debug, false, sizeof(bool)));
}

void print_idebug(GPU_Data& dd)
{
    int idebug;
    chkerr(cudaMemcpy(&idebug, dd.idebug, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "IDebug: " << idebug << flush;
    chkerr(cudaMemset(dd.idebug, 0, sizeof(int)));
}

void print_Data_Sizes_Every(GPU_Data& dd, int every)
{
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        print_Data_Sizes(dd);
    }
}

void print_Data_Sizes(GPU_Data& dd)
{
    uint64_t* current_level = new uint64_t;
    uint64_t* tasks1_count = new uint64_t;
    uint64_t* tasks2_count = new uint64_t;
    uint64_t* buffer_count = new uint64_t;
    uint64_t* cliques_count = new uint64_t;
    uint64_t* tasks1_size = new uint64_t;
    uint64_t* tasks2_size = new uint64_t;
    uint64_t* buffer_size = new uint64_t;
    uint64_t* cliques_size = new uint64_t;

    chkerr(cudaMemcpy(current_level, dd.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_count, dd.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_count, dd.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_count, dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_size, dd.tasks1_offset + (*tasks1_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_size, dd.tasks2_offset + (*tasks2_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_size, dd.buffer_offset + (*buffer_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_size, dd.cliques_offset + (*cliques_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cout << "L: " << (*current_level) << " T1: " << (*tasks1_count) << " " << (*tasks1_size) << " T2: " << (*tasks2_count) << " " << (*tasks2_size) << " B: " << (*buffer_count) << " " << (*buffer_size) << " C: " << 
        (*cliques_count) << " " << (*cliques_size) << endl;

    delete current_level;
    delete tasks1_count;
    delete tasks2_count;
    delete buffer_count;
    delete cliques_count;
    delete tasks1_size;
    delete tasks2_size;
    delete buffer_size;
    delete cliques_size;
}

void print_WTask_Buffers(GPU_Data& dd)
{
    int warp_count = (NUM_OF_BLOCKS * BLOCK_SIZE) / 32;
    uint64_t* wtasks_count = new uint64_t[warp_count];
    uint64_t* wtasks_offset = new uint64_t[warp_count*WTASKS_OFFSET_SIZE];
    Vertex* wtasks_vertices = new Vertex[warp_count*WTASKS_SIZE];

    chkerr(cudaMemcpy(wtasks_count, dd.wtasks_count, sizeof(uint64_t)*warp_count, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wtasks_offset, dd.wtasks_offset, sizeof(uint64_t) * (warp_count*WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wtasks_vertices, dd.wtasks_vertices, sizeof(Vertex) * (warp_count*WTASKS_SIZE), cudaMemcpyDeviceToHost));

    cout << endl << " --- Warp Task Buffers details --- " << endl;
    for (int i = 0; i < warp_count; i++) {
        int wtasks_offset_start = WTASKS_OFFSET_SIZE * i;
        int wtasks_start = WTASKS_SIZE * i;

        cout << endl << "Warp " << i << ": " << "Size : " << wtasks_count[i] << endl;
        if (wtasks_count[i] == 0) {
            continue;
        }
        cout << "Offsets:" << endl;
        for (int j = 0; j <= wtasks_count[i]; j++) {
            cout << wtasks_offset[wtasks_offset_start+j] << " ";
        }
        cout << endl << "Vertex:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start+wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start+j].vertexid << " ";
        }
        cout << endl << "Label:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].label << " ";
        }
        cout << endl << "Indeg:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].indeg << " ";
        }
        cout << endl << "Exdeg:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].exdeg << " ";
        }
        cout << endl << "Lvl2adj:" << endl;
        for (int j = 0; j < wtasks_offset[wtasks_offset_start + wtasks_count[i]]; j++) {
            cout << wtasks_vertices[wtasks_start + j].lvl2adj << " ";
        }
        cout << endl;
    }
    cout << endl << endl;

    delete wtasks_count;
    delete wtasks_offset;
    delete wtasks_vertices;
}

void print_WClique_Buffers(GPU_Data& dd)
{
    int warp_count = (NUM_OF_BLOCKS * BLOCK_SIZE) / 32;
    uint64_t* wcliques_count = new uint64_t[warp_count];
    uint64_t* wcliques_offset = new uint64_t[warp_count * WCLIQUES_OFFSET_SIZE];
    int* wcliques_vertex = new int[warp_count * WCLIQUES_SIZE];

    chkerr(cudaMemcpy(wcliques_count, dd.wcliques_count, sizeof(uint64_t) * warp_count, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wcliques_offset, dd.wcliques_offset, sizeof(uint64_t) * (warp_count * WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wcliques_vertex, dd.wcliques_vertex, sizeof(int) * (warp_count * WTASKS_SIZE), cudaMemcpyDeviceToHost));

    cout << endl << " --- Warp Clique Buffers details --- " << endl;
    for (int i = 0; i < warp_count; i++) {
        int wcliques_offset_start = WTASKS_OFFSET_SIZE * i;
        int wcliques_start = WTASKS_SIZE * i;

        cout << endl << "Warp " << i << ": " << "Size : " << wcliques_count[i] << endl;
        cout << "Offsets:" << endl;
        for (int j = 0; j <= wcliques_count[i]; j++) {
            cout << wcliques_offset[wcliques_offset_start + j] << " ";
        }
        cout << endl << "Vertex:" << endl;
        for (int j = 0; j < wcliques_offset[wcliques_offset_start + wcliques_count[i]]; j++) {
            cout << wcliques_vertex[wcliques_start + j] << " ";
        }
    }
    cout << endl << endl;

    delete wcliques_count;
    delete wcliques_offset;
    delete wcliques_vertex;
}

void print_GPU_Cliques(GPU_Data& dd)
{
    uint64_t* cliques_count = new uint64_t;
    uint64_t* cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];
    int* cliques_vertex = new int[CLIQUES_SIZE];

    chkerr(cudaMemcpy(cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));

    cout << endl << " --- (GPU_Cliques)device_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Size: " << (*cliques_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*cliques_count); i++) {
        cout << cliques_offset[i] << " ";
    }

    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < (*cliques_count); i++) {
        cout << i << " S: " << cliques_offset[i] << " E: " << cliques_offset[i+1] << " " << flush;
        for (uint64_t j = cliques_offset[i]; j < cliques_offset[i + 1]; j++) {
            cout << cliques_vertex[j] << " " << flush;
        }
        cout << endl;
    }

    delete cliques_count;
    delete cliques_offset;
    delete cliques_vertex;

    return;

    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < cliques_offset[(*cliques_count)]; i++) {
        cout << cliques_vertex[i] << " ";
    }
    cout << endl;
}

void print_CPU_Cliques(CPU_Cliques& host_cliques)
{
    cout << endl << " --- (CPU_Cliques)host_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Size: " << (*(host_cliques.cliques_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(host_cliques.cliques_count)); i++) {
        cout << host_cliques.cliques_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < host_cliques.cliques_offset[(*(host_cliques.cliques_count))]; i++) {
        cout << host_cliques.cliques_vertex[i] << " ";
    }
    cout << endl;
}

void print_vertices(Vertex* vertices, int size)
{
    cout << " --- level 0 details --- " << endl;
    cout << endl << "Tasks1: Level: " << 0 << " Size: " << size << endl;
    cout << endl << "Offsets:" << endl;
    cout << "0 " << size << flush;
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].vertexid << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].label << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].indeg << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].exdeg << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < size; i++) {
        cout << vertices[i].lvl2adj << " " << flush;
    }
    cout << endl;
}



// --- DEVICE KERNELS ---

__global__ void expand_level(GPU_Data dd)
{
    // data is stored in data structures to reduce the number of variables that need to be passed to methods
    __shared__ Warp_Data wd;
    Local_Data ld;

    // helper variables, not passed through to any methods
    int method_return;

    // initialize variables
    ld.idx = (blockIdx.x * blockDim.x + threadIdx.x);
    ld.warp_in_block_idx = ((ld.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE));

    /*
    * The program alternates between reading and writing between to 'tasks' arrays in device global memory. The program will read from one tasks, expand to the next level by generating and pruning, then it will write to the
    * other tasks array. It will write the first EXPAND_THRESHOLD to the tasks array and the rest to the top of the buffer. The buffers acts as a stack containing the excess data not being expanded from tasks. Since the 
    * buffer acts as a stack, in a last-in first-out manner, a subsection of the search space will be expanded until completion. This system allows the problem to essentially be divided into smaller problems and thus 
    * require less memory to handle.
    */
    if ((*(dd.current_level)) % 2 == 1) {
        ld.read_count = dd.tasks1_count;
        ld.read_offsets = dd.tasks1_offset;
        ld.read_vertices = dd.tasks1_vertices;
    } else {
        ld.read_count = dd.tasks2_count;
        ld.read_offsets = dd.tasks2_offset;
        ld.read_vertices = dd.tasks2_vertices;
    }



    // --- CURRENT LEVEL ---
    for (int i = (ld.idx / WARP_SIZE); i < (*(ld.read_count)); i += ((NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE))
    {
        // get information on vertices being handled within tasks
        if ((ld.idx % WARP_SIZE) == 0) {
            wd.start[ld.warp_in_block_idx] = ld.read_offsets[i];
            wd.end[ld.warp_in_block_idx] = ld.read_offsets[i + 1];
            wd.tot_vert[ld.warp_in_block_idx] = wd.end[ld.warp_in_block_idx] - wd.start[ld.warp_in_block_idx];
            wd.num_mem[ld.warp_in_block_idx] = 0;
            for (uint64_t j = wd.start[ld.warp_in_block_idx]; j < wd.end[ld.warp_in_block_idx]; j++) {
                if (ld.read_vertices[j].label == 1) {
                    wd.num_mem[ld.warp_in_block_idx]++;
                } else {
                    break;
                }
            }
            wd.num_cand[ld.warp_in_block_idx] = wd.tot_vert[ld.warp_in_block_idx] - wd.num_mem[ld.warp_in_block_idx];
            wd.expansions[ld.warp_in_block_idx] = wd.num_cand[ld.warp_in_block_idx];
        }
        __syncwarp();



        // LOOKAHEAD PRUNING
        method_return = lookahead_pruning(dd, wd, ld);
        if (method_return) {
            continue;
        }



        // --- NEXT LEVEL ---
        for (int j = 0; j < wd.expansions[ld.warp_in_block_idx]; j++)
        {


            // REMOVE ONE VERTEX
            if (j > 0) {
                method_return = remove_one_vertex(dd, wd, ld);
                if (method_return) {
                    continue;
                }
            }



            // INITIALIZE NEW VERTICES
            if ((ld.idx % WARP_SIZE) == 0) {
                wd.number_of_members[ld.warp_in_block_idx] = wd.num_mem[ld.warp_in_block_idx];
                wd.number_of_candidates[ld.warp_in_block_idx] = wd.num_cand[ld.warp_in_block_idx];
                wd.total_vertices[ld.warp_in_block_idx] = wd.tot_vert[ld.warp_in_block_idx];
            }
            __syncwarp();

            // select whether to store vertices in global or shared memory based on size
            if (wd.total_vertices[ld.warp_in_block_idx] <= VERTICES_SIZE) {
                ld.vertices = wd.shared_vertices + (VERTICES_SIZE * ld.warp_in_block_idx);
            } else {
                ld.vertices = dd.wvertices + (WVERTICES_SIZE * (ld.idx / WARP_SIZE));
            }

            for (int k = (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx]; k += WARP_SIZE) {
                ld.vertices[k] = ld.read_vertices[wd.start[ld.warp_in_block_idx] + k];
            }



            // ADD ONE VERTEX
            method_return = add_one_vertex(dd, wd, ld);
            // too many vertices pruned continue, no need to check as not enough vertices
            if (method_return == 1) {
                continue;
            }



            // HANDLE CLIQUES
            if (wd.number_of_members[ld.warp_in_block_idx] >= (*dd.minimum_clique_size)) {
                check_for_clique(dd, wd, ld);
            }

            // if vertex in x found as not extendable continue to next iteration
            if (method_return == 2) {
                continue;
            }



            // WRITE TASKS TO BUFFERS
            if (wd.number_of_candidates[ld.warp_in_block_idx] > 0) {
                write_to_tasks(dd, wd, ld);
            }
        }
    }



    if ((ld.idx % WARP_SIZE) == 0) {
        // sum to find tasks count
        atomicAdd(dd.total_tasks, dd.wtasks_count[(ld.idx / WARP_SIZE)]);
        atomicAdd(dd.total_cliques, dd.wcliques_count[(ld.idx / WARP_SIZE)]);
        // DEBUG
        atomicAdd(dd.idebug, dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + dd.wtasks_count[(ld.idx / WARP_SIZE)]]);
    }

    if (ld.idx == 0) {
        (*(dd.buffer_offset_start)) = (*(dd.buffer_count)) + 1;
        (*(dd.buffer_start)) = dd.buffer_offset[(*(dd.buffer_count))];
        (*(dd.cliques_offset_start)) = (*(dd.cliques_count)) + 1;
        (*(dd.cliques_start)) = dd.cliques_offset[(*(dd.cliques_count))];
    }
}

__global__ void transfer_buffers(GPU_Data dd)
{
    // THREAD INFO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_in_block_idx = ((idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE));

    __shared__ uint64_t tasks_write[(BLOCK_SIZE / WARP_SIZE)];
    __shared__ int tasks_offset_write[(BLOCK_SIZE / WARP_SIZE)];
    __shared__ uint64_t cliques_write[(BLOCK_SIZE / WARP_SIZE)];
    __shared__ int cliques_offset_write[(BLOCK_SIZE / WARP_SIZE)];

    __shared__ int twarp;
    __shared__ int toffsetwrite;
    __shared__ int twrite;
    __shared__ int tasks_end;
    
    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;

    if ((*(dd.current_level)) % 2 == 1) {
        write_count = dd.tasks2_count;
        write_offsets = dd.tasks2_offset;
        write_vertices = dd.tasks2_vertices;
    }
    else {
        write_count = dd.tasks1_count;
        write_offsets = dd.tasks1_offset;
        write_vertices = dd.tasks1_vertices;
    }

    // block level
    if (threadIdx.x == 0) {
        toffsetwrite = 0;
        twrite = 0;

        for (int i = 0; i < ((NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE); i++) {
            if (toffsetwrite + dd.wtasks_count[i] >= EXPAND_THRESHOLD) {
                twarp = i;
                break;
            }
            twrite += dd.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + dd.wtasks_count[i]];
            toffsetwrite += dd.wtasks_count[i];
        }
        tasks_end = twrite + dd.wtasks_offset[(WTASKS_OFFSET_SIZE * twarp) +
            (EXPAND_THRESHOLD - toffsetwrite)];
    }
    __syncthreads();

    // warp level
    if ((idx % WARP_SIZE) == 0)
    {
        tasks_write[warp_in_block_idx] = 0;
        tasks_offset_write[warp_in_block_idx] = 1;
        cliques_write[warp_in_block_idx] = 0;
        cliques_offset_write[warp_in_block_idx] = 1;

        for (int i = 0; i < (idx / WARP_SIZE); i++) {
            tasks_offset_write[warp_in_block_idx] += dd.wtasks_count[i];
            tasks_write[warp_in_block_idx] += dd.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + dd.wtasks_count[i]];

            cliques_offset_write[warp_in_block_idx] += dd.wcliques_count[i];
            cliques_write[warp_in_block_idx] += dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * i) + dd.wcliques_count[i]];
        }
    }
    __syncwarp();

    // TODO - for the next two blocks use two for loops rather than a conditional
    // move to tasks and buffer
    for (int i = (idx % WARP_SIZE) + 1; i <= dd.wtasks_count[(idx / WARP_SIZE)]; i += WARP_SIZE)
    {
        if (tasks_offset_write[warp_in_block_idx] + i - 1 <= EXPAND_THRESHOLD) {
            // to tasks
            write_offsets[tasks_offset_write[warp_in_block_idx] + i - 1] = dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + tasks_write[warp_in_block_idx];
        }
        else {
            // to buffer
            dd.buffer_offset[tasks_offset_write[warp_in_block_idx] + i - 2 - EXPAND_THRESHOLD + (*(dd.buffer_offset_start))] = dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + 
                tasks_write[warp_in_block_idx] - tasks_end + (*(dd.buffer_start));
        }
    }

    for (int i = (idx % WARP_SIZE); i < dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + dd.wtasks_count[(idx / WARP_SIZE)]]; i += WARP_SIZE) {
        if (tasks_write[warp_in_block_idx] + i < tasks_end) {
            // to tasks
            write_vertices[tasks_write[warp_in_block_idx] + i] = dd.wtasks_vertices[(WTASKS_SIZE * (idx / WARP_SIZE)) + i];
        }
        else {
            // to buffer
            dd.buffer_vertices[(*(dd.buffer_start)) + tasks_write[warp_in_block_idx] + i - tasks_end] = dd.wtasks_vertices[(WTASKS_SIZE * (idx / WARP_SIZE)) + i];
        }
    }

    //move to cliques
    for (int i = (idx % WARP_SIZE) + 1; i <= dd.wcliques_count[(idx / WARP_SIZE)]; i += WARP_SIZE) {
        dd.cliques_offset[(*(dd.cliques_offset_start)) + cliques_offset_write[warp_in_block_idx] + i - 2] = dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + (*(dd.cliques_start)) + 
            cliques_write[warp_in_block_idx];
    }
    for (int i = (idx % WARP_SIZE); i < dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (idx / WARP_SIZE)) + dd.wcliques_count[(idx / WARP_SIZE)]]; i += WARP_SIZE) {
        dd.cliques_vertex[(*(dd.cliques_start)) + cliques_write[warp_in_block_idx] + i] = dd.wcliques_vertex[(WCLIQUES_SIZE * (idx / WARP_SIZE)) + i];
    }

    if (idx == 0) {
        // handle tasks and buffer counts
        if ((*dd.total_tasks) <= EXPAND_THRESHOLD) {
            (*write_count) = (*(dd.total_tasks));
        }
        else {
            (*write_count) = EXPAND_THRESHOLD;
            (*(dd.buffer_count)) += ((*(dd.total_tasks)) - EXPAND_THRESHOLD);
        }
        (*(dd.cliques_count)) += (*(dd.total_cliques));

        (*(dd.total_tasks)) = 0;
        (*(dd.total_cliques)) = 0;
    }

    // HANDLE CLIQUES
    // only first thread for each warp
    if ((idx % WARP_SIZE) == 0 && cliques_write[warp_in_block_idx] > (CLIQUES_SIZE * (CLIQUES_PERCENT / 100.0))) {
        atomicExch((int*)dd.dumping_cliques, true);
    }
}

__global__ void fill_from_buffer(GPU_Data dd)
{
    // THREAD INFO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = (idx / 32);
    int lane_idx = (idx % 32);

    Vertex* write_vertices;
    uint64_t* write_offsets;
    uint64_t* write_count;

    if ((*(dd.current_level)) % 2 == 1) {
        write_count = dd.tasks2_count;
        write_offsets = dd.tasks2_offset;
        write_vertices = dd.tasks2_vertices;
    } else {
        write_count = dd.tasks1_count;
        write_offsets = dd.tasks1_offset;
        write_vertices = dd.tasks1_vertices;
    }

    if (lane_idx == 0) {
        dd.wtasks_count[warp_idx] = 0;
        dd.wcliques_count[warp_idx] = 0;
    }

    // FILL TASKS FROM BUFFER
    if ((*write_count) < EXPAND_THRESHOLD && (*(dd.buffer_count)) > 0)
    {
        // CRITICAL
        atomicExch((int*)dd.maximal_expansion, false);

        // get read and write locations
        int write_amount = ((*(dd.buffer_count)) >= (EXPAND_THRESHOLD - (*write_count))) ? EXPAND_THRESHOLD - (*write_count) : (*(dd.buffer_count));
        uint64_t start_buffer = dd.buffer_offset[(*(dd.buffer_count)) - write_amount];
        uint64_t end_buffer = dd.buffer_offset[(*(dd.buffer_count))];
        uint64_t size_buffer = end_buffer - start_buffer;
        uint64_t start_write = write_offsets[(*write_count)];

        // handle offsets
        for (int i = idx + 1; i <= write_amount; i += (NUM_OF_BLOCKS * BLOCK_SIZE)) {
            write_offsets[(*write_count) + i] = start_write + (dd.buffer_offset[(*(dd.buffer_count)) - write_amount + i] - start_buffer);
        }

        // handle data
        for (int i = idx; i < size_buffer; i += (NUM_OF_BLOCKS * BLOCK_SIZE)) {
            write_vertices[start_write + i] = dd.buffer_vertices[start_buffer + i];
        }

        if (idx == 0) {
            (*write_count) += write_amount;
            (*(dd.buffer_count)) -= write_amount;
        }
    }

    if (idx == 0) {
        (*dd.current_level)++;
    }
}

// returns 1 if lookahead succesful, 0 otherwise
__device__ int lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld) 
{
    bool lookahead_sucess = true;

    // compares all vertices to the lemmas from Quick
    for (int j = (ld.idx % WARP_SIZE); j < wd.tot_vert[ld.warp_in_block_idx]; j += WARP_SIZE) {
        if (ld.read_vertices[wd.start[ld.warp_in_block_idx] + j].lvl2adj != (wd.tot_vert[ld.warp_in_block_idx] - 1) || ld.read_vertices[wd.start[ld.warp_in_block_idx] + j].indeg + 
            ld.read_vertices[wd.start[ld.warp_in_block_idx] + j].exdeg < dd.minimum_degrees[wd.tot_vert[ld.warp_in_block_idx]]) {
            lookahead_sucess = false;
            break;
        }
    }
    lookahead_sucess = !(__any_sync(0xFFFFFFFF, !lookahead_sucess));

    if (lookahead_sucess) {
        // write to cliques
        uint64_t start_write = (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) + dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])];
        for (int j = (ld.idx % WARP_SIZE); j < wd.tot_vert[ld.warp_in_block_idx]; j += WARP_SIZE) {
            dd.wcliques_vertex[start_write + j] = ld.read_vertices[wd.start[ld.warp_in_block_idx] + j].vertexid;
        }
        if ((ld.idx % WARP_SIZE) == 0) {
            (dd.wcliques_count[(ld.idx / WARP_SIZE)])++;
            dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])] = start_write - (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) + wd.tot_vert[ld.warp_in_block_idx];
        }
        return 1;
    }
    return 0;
}

// returns 1 if failed found after removing, 0 otherwise
__device__ int remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld) 
{
    int pvertexid;
    bool failed_found;

    // remove the last candidate in vertices
    if ((ld.idx % WARP_SIZE) == 0) {
        wd.num_cand[ld.warp_in_block_idx]--;
        wd.tot_vert[ld.warp_in_block_idx]--;
    }
    __syncwarp();

    // get the id of the removed vertex and update the degrees of its adjacencies
    pvertexid = ld.read_vertices[wd.start[ld.warp_in_block_idx] + wd.tot_vert[ld.warp_in_block_idx]].vertexid;
    for (int k = (ld.idx % WARP_SIZE); k < wd.tot_vert[ld.warp_in_block_idx]; k += WARP_SIZE) {
        if (device_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], ld.read_vertices[wd.start[ld.warp_in_block_idx] + k].vertexid) != -1) {
            ld.read_vertices[wd.start[ld.warp_in_block_idx] + k].exdeg--;
        }

        if (device_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[pvertexid], dd.twohop_offsets[pvertexid + 1] - dd.twohop_offsets[pvertexid], ld.read_vertices[wd.start[ld.warp_in_block_idx] + k].vertexid) != -1) {
            ld.read_vertices[wd.start[ld.warp_in_block_idx] + k].lvl2adj--;
        }
    }
    __syncwarp();

    // check for failed vertices
    failed_found = false;
    for (int k = (ld.idx % WARP_SIZE); k < wd.num_mem[ld.warp_in_block_idx]; k += WARP_SIZE) {
        if (!device_vert_isextendable(ld.read_vertices[wd.start[ld.warp_in_block_idx] + k], wd.num_mem[ld.warp_in_block_idx], dd)) {
            failed_found = true;
            break;
        }

    }
    failed_found = __any_sync(0xFFFFFFFF, failed_found);
    if (failed_found) {
        return 1;
    }
    return 0;
}

// returns 2, if too many vertices pruned to be considered, 1 if failed found or invalid bound, 0 otherwise
__device__ int add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld) 
{
    int pvertexid;
    bool failed_found;

    if ((ld.idx % WARP_SIZE) == 0) {
        ld.vertices[wd.total_vertices[ld.warp_in_block_idx] - 1].label = 1;
        wd.number_of_members[ld.warp_in_block_idx]++;
        wd.number_of_candidates[ld.warp_in_block_idx]--;
    }
    __syncwarp();

    // update the exdeg and indeg of all vertices adj to the vertex just added to the vertex set
    pvertexid = ld.vertices[wd.total_vertices[ld.warp_in_block_idx] - 1].vertexid;
    for (int k = (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx]; k += WARP_SIZE) {
        if (device_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[ld.vertices[k].vertexid], dd.onehop_offsets[ld.vertices[k].vertexid + 1] - dd.onehop_offsets[ld.vertices[k].vertexid], pvertexid) != -1) {
            ld.vertices[k].exdeg--;
            ld.vertices[k].indeg++;
        }
    }
    __syncwarp();

    // TODO - this might be able to be hard coded rather than sorted
    // sort new vertices putting just added vertex at end of all vertices in x
    device_sort(ld.vertices + wd.number_of_members[ld.warp_in_block_idx] - 1, wd.number_of_candidates[ld.warp_in_block_idx] + 1, (ld.idx % WARP_SIZE));



    // --- DIAMETER PRUNING ---
    diameter_pruning(dd, wd, ld, pvertexid);

    // continue if not enough vertices after pruning
    if (wd.total_vertices[ld.warp_in_block_idx] < (*(dd.minimum_clique_size))) {
        return 1;
    }



    // DEGREE BASED PRUNING
    degree_pruning(dd, wd, ld, failed_found);

    // continue if not enough vertices after pruning
    if (wd.total_vertices[ld.warp_in_block_idx] < (*(dd.minimum_clique_size))) {
        return 1;
    }

    // TODO - test if we need to check vertex sets that have invalid bounds, dont think so
    // if vertex in x found as not extendable continue to next iteration
    if (failed_found || wd.invalid_bounds[ld.warp_in_block_idx]) {
        return 2;
    }
    
    return 0;
}

__device__ void check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    bool clique = true;

    for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.warp_in_block_idx]; k += WARP_SIZE) {
        if (ld.vertices[k].indeg < dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx]]) {
            clique = false;
            break;
        }
    }
    // set to false if any threads in warp do not meet degree requirement
    clique = !(__any_sync(0xFFFFFFFF, !clique));

    // if clique write to warp buffer for cliques
    if (clique) {
        uint64_t start_write = (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) + dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])];
        for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.warp_in_block_idx]; k += WARP_SIZE) {
            dd.wcliques_vertex[start_write + k] = ld.vertices[k].vertexid;
        }
        if ((ld.idx % WARP_SIZE) == 0) {
            (dd.wcliques_count[(ld.idx / WARP_SIZE)])++;
            dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])] = start_write - (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) +
                wd.number_of_members[ld.warp_in_block_idx];
        }
    }
}

__device__ void write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    // CRITICAL
    atomicExch((int*)dd.maximal_expansion, false);

    uint64_t start_write = (WTASKS_SIZE * (ld.idx / WARP_SIZE)) + dd.wtasks_offset[WTASKS_OFFSET_SIZE * (ld.idx / WARP_SIZE) + (dd.wtasks_count[(ld.idx / WARP_SIZE)])];

    for (int k = (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx]; k += WARP_SIZE) {
        dd.wtasks_vertices[start_write + k].vertexid = ld.vertices[k].vertexid;
        dd.wtasks_vertices[start_write + k].label = ld.vertices[k].label;
        dd.wtasks_vertices[start_write + k].indeg = ld.vertices[k].indeg;
        dd.wtasks_vertices[start_write + k].exdeg = ld.vertices[k].exdeg;
        dd.wtasks_vertices[start_write + k].lvl2adj = ld.vertices[k].lvl2adj;
    }
    if ((ld.idx % WARP_SIZE) == 0) {
        (dd.wtasks_count[(ld.idx / WARP_SIZE)])++;
        dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wtasks_count[(ld.idx / WARP_SIZE)])] = start_write - (WTASKS_SIZE * (ld.idx / WARP_SIZE)) + wd.total_vertices[ld.warp_in_block_idx];
    }
}

__device__ void diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid)
{
    int number_of_removed_candidates;

    number_of_removed_candidates = 0;
    for (int k = wd.number_of_members[ld.warp_in_block_idx] + (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx]; k += WARP_SIZE) {
        if (device_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[pvertexid], dd.twohop_offsets[pvertexid + 1] - dd.twohop_offsets[pvertexid], ld.vertices[k].vertexid) == -1) {
            ld.vertices[k].label = -1;
            number_of_removed_candidates++;
        }
    }
    for (int k = 1; k < 32; k *= 2) {
        number_of_removed_candidates += __shfl_xor_sync(0xFFFFFFFF, number_of_removed_candidates, k);
    }
    device_sort(ld.vertices + wd.number_of_members[ld.warp_in_block_idx], wd.number_of_candidates[ld.warp_in_block_idx], (ld.idx % WARP_SIZE));

    // update exdeg of vertices connected to removed cands
    update_degrees(dd, wd, ld, number_of_removed_candidates);

    if ((ld.idx % WARP_SIZE) == 0) {
        wd.total_vertices[ld.warp_in_block_idx] -= number_of_removed_candidates;
        wd.number_of_candidates[ld.warp_in_block_idx] -= number_of_removed_candidates;
    }
    __syncwarp();
}

// TODO - return failed_found rather than use a reference
__device__ void degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, bool& failed_found)
{
    int number_of_removed_candidates;

    do
    {
        // calculate lower and upper bounds for vertices
        calculate_LU_bounds(dd, wd, ld);

        if (wd.invalid_bounds[ld.warp_in_block_idx]) {
            break;
        }

        // check for failed vertices
        failed_found = false;
        for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.warp_in_block_idx]; k += WARP_SIZE) {
            if (!device_vert_isextendable_LU(ld.vertices[k], dd, wd, ld)) {
                failed_found = true;
                break;
            }

        }
        failed_found = __any_sync(0xFFFFFFFF, failed_found);
        if (failed_found) {
            break;
        }

        // remove cands that do not meet the deg requirement
        number_of_removed_candidates = 0;
        for (int k = wd.number_of_members[ld.warp_in_block_idx] + (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx]; k += WARP_SIZE) {
            if (!device_cand_isvalid_LU(ld.vertices[k], dd, wd, ld)) {
                ld.vertices[k].label = -1;
                number_of_removed_candidates++;
            }
        }
        for (int k = 1; k < 32; k *= 2) {
            number_of_removed_candidates += __shfl_xor_sync(0xFFFFFFFF, number_of_removed_candidates, k);
        }
        device_sort(ld.vertices + wd.number_of_members[ld.warp_in_block_idx], wd.number_of_candidates[ld.warp_in_block_idx], (ld.idx % WARP_SIZE));

        // update exdeg of vertices connected to removed cands
        update_degrees(dd, wd, ld, number_of_removed_candidates);

        if ((ld.idx % WARP_SIZE) == 0) {
            wd.total_vertices[ld.warp_in_block_idx] -= number_of_removed_candidates;
            wd.number_of_candidates[ld.warp_in_block_idx] -= number_of_removed_candidates;
        }
        __syncwarp();
    } while (number_of_removed_candidates > 0);
}

__device__ void update_degrees(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_removed_candidates)
{
    int pvertexid;

    /*
    * Program updates degrees by: for each vertex, for each removed vertex, binary search neighbors of removed vertex for (non-removed)vertex. This is an improvement from the Quick algorithm because it uses binary search.
    * Additonally the program also dyanmically selects which for loop to parallelize based on which one is larger, this is the pupose of the if statement.
    */
    if (wd.total_vertices[ld.warp_in_block_idx] - number_of_removed_candidates > number_of_removed_candidates) {
        for (int k = (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx] - number_of_removed_candidates; k += WARP_SIZE) {
            pvertexid = ld.vertices[k].vertexid;
            for (int l = wd.total_vertices[ld.warp_in_block_idx] - number_of_removed_candidates; l < wd.total_vertices[ld.warp_in_block_idx]; l++) {
                if (device_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[ld.vertices[l].vertexid], dd.onehop_offsets[ld.vertices[l].vertexid + 1] - dd.onehop_offsets[ld.vertices[l].vertexid], pvertexid) != -1) {
                    ld.vertices[k].exdeg--;
                }

                if (device_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[ld.vertices[l].vertexid], dd.twohop_offsets[ld.vertices[l].vertexid + 1] - dd.twohop_offsets[ld.vertices[l].vertexid], pvertexid) != -1) {
                    ld.vertices[k].lvl2adj--;
                }
            }
        }
        __syncwarp();
    }
    else {
        for (int k = 0; k < wd.total_vertices[ld.warp_in_block_idx] - number_of_removed_candidates; k++) {
            pvertexid = ld.vertices[k].vertexid;
            for (int l = wd.total_vertices[ld.warp_in_block_idx] - number_of_removed_candidates + (ld.idx % WARP_SIZE); l < wd.total_vertices[ld.warp_in_block_idx]; l += WARP_SIZE) {
                if (device_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[ld.vertices[l].vertexid], dd.onehop_offsets[ld.vertices[l].vertexid + 1] - dd.onehop_offsets[ld.vertices[l].vertexid], pvertexid) != -1) {
                    ld.vertices[k].exdeg--;
                }

                if (device_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[ld.vertices[l].vertexid], dd.twohop_offsets[ld.vertices[l].vertexid + 1] - dd.twohop_offsets[ld.vertices[l].vertexid], pvertexid) != -1) {
                    ld.vertices[k].lvl2adj--;
                }
            }
            __syncwarp();
        }
    }
}

// TODO - try to parallelize as much calculation as possible
__device__ void calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    int index;

    int min_clq_indeg;
    int min_indeg_exdeg;
    int min_clq_totaldeg;
    int sum_clq_indeg;

    // initialize the values of the LU calculation variables to the first vertices values so they can be compared to other vertices without error
    min_clq_indeg = ld.vertices[0].indeg;
    min_indeg_exdeg = ld.vertices[0].exdeg;
    min_clq_totaldeg = ld.vertices[0].indeg + ld.vertices[0].exdeg;
    sum_clq_indeg = 0;

    // each warp also has a copy of these variables to allow for intra-warp comparison of these variables.
    if ((ld.idx % WARP_SIZE) == 0) {
        wd.invalid_bounds[ld.warp_in_block_idx] = false;

        wd.sum_candidate_indeg[ld.warp_in_block_idx] = 0;
        wd.tightened_Upper_bound[ld.warp_in_block_idx] = 0;

        wd.min_clq_indeg[ld.warp_in_block_idx] = ld.vertices[0].indeg;
        wd.min_indeg_exdeg[ld.warp_in_block_idx] = ld.vertices[0].exdeg;
        wd.min_clq_totaldeg[ld.warp_in_block_idx] = ld.vertices[0].indeg + ld.vertices[0].exdeg;
        wd.sum_clq_indeg[ld.warp_in_block_idx] = ld.vertices[0].indeg;

        wd.minimum_external_degree[ld.warp_in_block_idx] = device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + 1,
            dd);
    }
    __syncwarp();

    // each warp finds these values on their subsection of vertices
    for (index = 1 + (ld.idx % WARP_SIZE); index < wd.number_of_members[ld.warp_in_block_idx]; index += WARP_SIZE) {
        sum_clq_indeg += ld.vertices[index].indeg;

        if (ld.vertices[index].indeg < min_clq_indeg) {
            min_clq_indeg = ld.vertices[index].indeg;
            min_indeg_exdeg = ld.vertices[index].exdeg;
        }
        else if (ld.vertices[index].indeg == min_clq_indeg) {
            if (ld.vertices[index].exdeg < min_indeg_exdeg) {
                min_indeg_exdeg = ld.vertices[index].exdeg;
            }
        }

        if (ld.vertices[index].indeg + ld.vertices[index].exdeg < min_clq_totaldeg) {
            min_clq_totaldeg = ld.vertices[index].indeg + ld.vertices[index].exdeg;
        }
    }

    // get sum
    for (int i = 1; i < 32; i *= 2) {
        sum_clq_indeg += __shfl_xor_sync(0xFFFFFFFF, sum_clq_indeg, i);
    }
    if ((ld.idx % WARP_SIZE) == 0) {
        // add to shared memory sum
        wd.sum_clq_indeg[ld.warp_in_block_idx] += sum_clq_indeg;
    }
    __syncwarp();

    // CRITICAL SECTION - each lane then compares their values to the next to get a warp level value
    for (int i = 0; i < WARP_SIZE; i++) {
        if ((ld.idx % WARP_SIZE) == i) {
            if (min_clq_indeg < wd.min_clq_indeg[ld.warp_in_block_idx]) {
                wd.min_clq_indeg[ld.warp_in_block_idx] = min_clq_indeg;
                wd.min_indeg_exdeg[ld.warp_in_block_idx] = min_indeg_exdeg;
            }
            else if (min_clq_indeg == wd.min_clq_indeg[ld.warp_in_block_idx]) {
                if (min_indeg_exdeg < wd.min_indeg_exdeg[ld.warp_in_block_idx]) {
                    wd.min_indeg_exdeg[ld.warp_in_block_idx] = min_indeg_exdeg;
                }
            }

            if (min_clq_totaldeg < wd.min_clq_totaldeg[ld.warp_in_block_idx]) {
                wd.min_clq_totaldeg[ld.warp_in_block_idx] = min_clq_totaldeg;
            }
        }
        __syncwarp();
    }

    // TODO - CRITICAL SECTION - unsure how to parallelize this, very complex, determine whether this section is worth having at all
    if ((ld.idx % WARP_SIZE) == 0) {
        if (wd.min_clq_indeg[ld.warp_in_block_idx] < dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx]])
        {
            // lower
            wd.Lower_bound[ld.warp_in_block_idx] = device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx], dd) - min_clq_indeg;

            while (wd.Lower_bound[ld.warp_in_block_idx] <= wd.min_indeg_exdeg[ld.warp_in_block_idx] && wd.min_clq_indeg[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx] <
                dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx]]) {
                wd.Lower_bound[ld.warp_in_block_idx]++;
            }

            if (wd.min_clq_indeg[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx] < dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx]]) {
                wd.invalid_bounds[ld.warp_in_block_idx] = true;
            }

            // upper
            wd.Upper_bound[ld.warp_in_block_idx] = floor(wd.min_clq_totaldeg[ld.warp_in_block_idx] / (*(dd.minimum_degree_ratio))) + 1 - wd.number_of_members[ld.warp_in_block_idx];

            if (wd.Upper_bound[ld.warp_in_block_idx] > wd.number_of_candidates[ld.warp_in_block_idx]) {
                wd.Upper_bound[ld.warp_in_block_idx] = wd.number_of_candidates[ld.warp_in_block_idx];
            }

            // tighten
            if (wd.Lower_bound[ld.warp_in_block_idx] < wd.Upper_bound[ld.warp_in_block_idx]) {
                // tighten lower
                for (index = 0; index < wd.Lower_bound[ld.warp_in_block_idx]; index++) {
                    wd.sum_candidate_indeg[ld.warp_in_block_idx] += ld.vertices[wd.number_of_members[ld.warp_in_block_idx] + index].indeg;
                }

                while (index < wd.Upper_bound[ld.warp_in_block_idx] && wd.sum_clq_indeg[ld.warp_in_block_idx] + wd.sum_candidate_indeg[ld.warp_in_block_idx] < wd.number_of_members[ld.warp_in_block_idx] *
                    dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + index]) {
                    wd.sum_candidate_indeg[ld.warp_in_block_idx] += ld.vertices[wd.number_of_members[ld.warp_in_block_idx] + index].indeg;
                    index++;
                }

                if (wd.sum_clq_indeg[ld.warp_in_block_idx] + wd.sum_candidate_indeg[ld.warp_in_block_idx] < wd.number_of_members[ld.warp_in_block_idx] * dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + index]) {
                    wd.invalid_bounds[ld.warp_in_block_idx] = true;
                }
                else {
                    wd.Lower_bound[ld.warp_in_block_idx] = index;

                    wd.tightened_Upper_bound[ld.warp_in_block_idx] = index;

                    while (index < wd.Upper_bound[ld.warp_in_block_idx]) {
                        wd.sum_candidate_indeg[ld.warp_in_block_idx] += ld.vertices[wd.number_of_members[ld.warp_in_block_idx] + index].indeg;

                        index++;

                        if (wd.sum_clq_indeg[ld.warp_in_block_idx] + wd.sum_candidate_indeg[ld.warp_in_block_idx] >= wd.number_of_members[ld.warp_in_block_idx] * 
                            dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + index]) {
                            wd.tightened_Upper_bound[ld.warp_in_block_idx] = index;
                        }
                    }

                    if (wd.Upper_bound[ld.warp_in_block_idx] > wd.tightened_Upper_bound[ld.warp_in_block_idx]) {
                        wd.Upper_bound[ld.warp_in_block_idx] = wd.tightened_Upper_bound[ld.warp_in_block_idx];
                    }

                    if (wd.Lower_bound[ld.warp_in_block_idx] > 1) {
                        wd.minimum_external_degree[ld.warp_in_block_idx] = device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx], dd);
                    }
                }
            }
        }
        else {
            wd.minimum_external_degree[ld.warp_in_block_idx] = device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + 1,
                dd);

            wd.Upper_bound[ld.warp_in_block_idx] = wd.number_of_candidates[ld.warp_in_block_idx];

            if (wd.number_of_members[ld.warp_in_block_idx] < (*(dd.minimum_clique_size))) {
                wd.Lower_bound[ld.warp_in_block_idx] = (*(dd.minimum_clique_size)) - wd.number_of_members[ld.warp_in_block_idx];
            }
            else {
                wd.Lower_bound[ld.warp_in_block_idx] = 0;
            }
        }

        if (wd.number_of_members[ld.warp_in_block_idx] + wd.Upper_bound[ld.warp_in_block_idx] < (*(dd.minimum_clique_size))) {
            wd.invalid_bounds[ld.warp_in_block_idx] = true;
        }

        if (wd.Upper_bound[ld.warp_in_block_idx] < 0 || wd.Upper_bound[ld.warp_in_block_idx] < wd.Lower_bound[ld.warp_in_block_idx]) {
            wd.invalid_bounds[ld.warp_in_block_idx] = true;
        }
    }
    __syncwarp();
}



// --- HELPER KERNELS ---

// DEBUG
__device__ void degree_pruning_nonLU(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, bool& failed_found)
{
    int number_of_removed_candidates;

    do
    {
        // check for failed vertices
        failed_found = false;
        for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.warp_in_block_idx]; k += WARP_SIZE) {
            if (!device_vert_isextendable(ld.vertices[k], wd.number_of_members[ld.warp_in_block_idx], dd)) {
                failed_found = true;
                break;
            }

        }
        failed_found = __any_sync(0xFFFFFFFF, failed_found);
        if (failed_found) {
            break;
        }

        // remove cands that do not meet the deg requirement
        number_of_removed_candidates = 0;
        for (int k = wd.number_of_members[ld.warp_in_block_idx] + (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.warp_in_block_idx]; k += WARP_SIZE) {
            if (!device_cand_isvalid(ld.vertices[k], wd.number_of_members[ld.warp_in_block_idx], dd)) {
                ld.vertices[k].label = -1;
                number_of_removed_candidates++;
            }
        }
        for (int k = 1; k < 32; k *= 2) {
            number_of_removed_candidates += __shfl_xor_sync(0xFFFFFFFF, number_of_removed_candidates, k);
        }
        device_sort(ld.vertices + wd.number_of_members[ld.warp_in_block_idx], wd.number_of_candidates[ld.warp_in_block_idx], (ld.idx % WARP_SIZE));

        // update exdeg of vertices connected to removed cands
        update_degrees(dd, wd, ld, number_of_removed_candidates);

        if ((ld.idx % WARP_SIZE) == 0) {
            wd.total_vertices[ld.warp_in_block_idx] -= number_of_removed_candidates;
            wd.number_of_candidates[ld.warp_in_block_idx] -= number_of_removed_candidates;
        }
        __syncwarp();
    } while (number_of_removed_candidates > 0);
}

// TODO - convert to merge or radix sort, merge is recursive
__device__ void device_sort(Vertex* target, int size, int lane_idx)
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (lane_idx * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            Vertex vertex1 = target[j];
            Vertex vertex2 = target[j + 1];

            if (sort_vert(vertex1, vertex2) == 1) {
                target[j] = target[j + 1];
                target[j + 1] = vertex1;
            }
        }
        __syncwarp();
    }
}

// TODO - clean up method
__device__ __forceinline int sort_vert(Vertex& vertex1, Vertex& vertex2)
{
    // order is: in clique -> covered -> critical adj vertices -> cands -> cover -> pruned

    // in clique
    if (vertex1.label == 1 && vertex2.label != 1) {
        return -1;
    }
    else if (vertex1.label != 1 && vertex2.label == 1) {
        return 1;

    // covered candidate vertices
    }
    else if (vertex1.label == 2 && vertex2.label != 2) {
        return -1;
    }
    else if (vertex1.label != 2 && vertex2.label == 2) {
        return 1;

    // critical adjacent candidate vertices
    }
    else if (vertex1.label == 4 && vertex2.label != 4) {
        return -1;
    }
    else if (vertex1.label != 4 && vertex2.label == 4) {
        return 1;

    // candidate vertices
    }
    else if (vertex1.label == 0 && vertex2.label != 0) {
        return -1;
    }
    else if (vertex1.label != 0 && vertex2.label == 0) {
        return 1;

    // the cover vertex
    }
    else if (vertex1.label == 3 && vertex2.label != 3) {
        return -1;
    }
    else if (vertex1.label != 3 && vertex2.label == 3) {
        return 1;

    // vertices that have been pruned
    }
    else if (vertex1.label == -1 && vertex2.label != 1) {
        return 1;
    }
    else if (vertex1.label != -1 && vertex2.label == -1) {
        return -1;
    }

    // for ties: in clique low -> high, cand high -> low
    else if (vertex1.label == 1 && vertex2.label == 1) {
        if (vertex1.vertexid > vertex2.vertexid) {
            return 1;
        }
        else if (vertex1.vertexid < vertex2.vertexid) {
            return -1;
        }
        else {
            return 0;
        }
    }
    else if (vertex1.label == 0 && vertex2.label == 0) {
        if (vertex1.vertexid > vertex2.vertexid) {
            return -1;
        }
        else if (vertex1.vertexid < vertex2.vertexid) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else if (vertex1.label == 2 && vertex2.label == 2) {
        return 0;
    }
    else if (vertex1.label == -1 && vertex2.label == -1) {
        return 0;
    }
    return 0;
}

// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
__device__ int device_bsearch_array(int* search_array, int array_size, int search_number)
{
    // ALGO - binary
    // TYPE - serial
    // SPEED - 0(log(n))

    if (array_size <= 0) {
        return -1;
    }

    if (search_array[array_size / 2] == search_number) {
        // Base case: Center element matches search number
        return array_size / 2;
    }
    else if (search_array[array_size / 2] > search_number) {
        // Recursively search lower half
        return device_bsearch_array(search_array, array_size / 2, search_number);
    }
    else {
        // Recursively search upper half
        int upper_half_result = device_bsearch_array(search_array + array_size / 2 + 1, array_size - array_size / 2 - 1, search_number);
        return (upper_half_result != -1) ? (array_size / 2 + 1 + upper_half_result) : -1;
    }
}

__device__ __forceinline bool device_vert_isextendable(Vertex& vertex, int number_of_members, GPU_Data& dd)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(dd.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(number_of_members+vertex.exdeg, dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ __forceinline bool device_cand_isvalid(Vertex& vertex, int number_of_members, GPU_Data& dd)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(dd.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(number_of_members + vertex.exdeg + 1, dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ __forceinline bool device_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(dd.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + vertex.exdeg + 1, dd)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < wd.minimum_external_degree[ld.warp_in_block_idx]) {
        return false;
    }
    else if (vertex.indeg + wd.Upper_bound[ld.warp_in_block_idx] - 1 < dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx], dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ __forceinline bool device_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(dd.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + vertex.exdeg, dd)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < wd.minimum_external_degree[ld.warp_in_block_idx]) {
        return false;
    }
    // TODO - I think this else if is useless
    else if (vertex.exdeg == 0 && vertex.indeg < device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + vertex.exdeg, dd)) {
        return false;
    }
    else if (vertex.indeg + wd.Upper_bound[ld.warp_in_block_idx] < dd.minimum_degrees[wd.number_of_members[ld.warp_in_block_idx] + wd.Upper_bound[ld.warp_in_block_idx]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(wd.number_of_members[ld.warp_in_block_idx] + wd.Lower_bound[ld.warp_in_block_idx], dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ __forceinline int device_get_mindeg(int number_of_members, GPU_Data& dd)
{
    if (number_of_members < (*(dd.minimum_clique_size))) {
        return dd.minimum_degrees[(*(dd.minimum_clique_size))];
    }
    else {
        return dd.minimum_degrees[number_of_members];
    }
}



// --- RM NON-MAX (from Quick) ---

int comp_int(const void* e1, const void* e2)
{
    int n1, n2;
    n1 = *(int*)e1;
    n2 = *(int*)e2;

    if (n1 > n2)
        return 1;
    else if (n1 < n2)
        return -1;
    else
        return 0;
}

extern int gntotal_max_cliques;

struct TREE_NODE
{
    int nid;
    TREE_NODE* pchild;
    TREE_NODE* pright_sib;
    bool bis_max;
};

#define TNODE_PAGE_SIZE (1<<10)

struct TNODE_PAGE
{
    TREE_NODE ptree_nodes[TNODE_PAGE_SIZE];
    TNODE_PAGE* pnext;
};

struct TNODE_BUF
{
    TNODE_PAGE* phead;
    TNODE_PAGE* pcur_page;
    int ncur_pos;
    int ntotal_pages;
};

extern TNODE_BUF gotreenode_buf;

inline TREE_NODE* NewTreeNode()
{
    TREE_NODE* ptnode;
    TNODE_PAGE* pnew_page;

    if (gotreenode_buf.ncur_pos == TNODE_PAGE_SIZE)
    {
        if (gotreenode_buf.pcur_page->pnext == NULL)
        {
            pnew_page = new TNODE_PAGE;
            pnew_page->pnext = NULL;
            gotreenode_buf.pcur_page->pnext = pnew_page;
            gotreenode_buf.pcur_page = pnew_page;
            gotreenode_buf.ntotal_pages++;
        }
        else
            gotreenode_buf.pcur_page = gotreenode_buf.pcur_page->pnext;
        gotreenode_buf.ncur_pos = 0;
    }

    ptnode = &(gotreenode_buf.pcur_page->ptree_nodes[gotreenode_buf.ncur_pos]);
    gotreenode_buf.ncur_pos++;

    ptnode->bis_max = true;

    return ptnode;
}

inline void OutputOneSet(FILE* fp, int* pset, int nlen)
{
    int i;

    gntotal_max_cliques++;

    fprintf(fp, "%d ", nlen);
    for (i = 0; i < nlen; i++)
        fprintf(fp, "%d ", pset[i]);
    fprintf(fp, "\n");

}

#include <stdio.h>
#include <time.h>
#include <sys/timeb.h>

int gntotal_max_cliques;

TNODE_BUF gotreenode_buf;

void DelTNodeBuf()
{
    TNODE_PAGE* ppage;

    ppage = gotreenode_buf.phead;
    while (ppage != NULL)
    {
        gotreenode_buf.phead = gotreenode_buf.phead->pnext;
        delete ppage;
        gotreenode_buf.ntotal_pages--;
        ppage = gotreenode_buf.phead;
    }
    if (gotreenode_buf.ntotal_pages != 0)
        printf("Error: inconsistent number of pages\n");
}

void InsertOneSet(int* pset, int nlen, TREE_NODE*& proot)
{
    TREE_NODE* pnode, * pparent, * pleftsib, * pnew_node;
    int i, j;

    i = 0;
    pparent = NULL;
    pnode = proot;
    pleftsib = NULL;

    while (i < nlen)
    {
        while (pnode != NULL && pnode->nid < pset[i])
        {
            pleftsib = pnode;
            pnode = pnode->pright_sib;
        }

        if (pnode == NULL || pnode->nid > pset[i])
        {
            pnew_node = NewTreeNode();
            pnew_node->nid = pset[i];
            pnew_node->pchild = NULL;
            pnew_node->pright_sib = pnode;
            if (pleftsib != NULL)
                pleftsib->pright_sib = pnew_node;
            else if (pparent != NULL)
                pparent->pchild = pnew_node;
            if (i == 0 && pleftsib == NULL)
                proot = pnew_node;
            pparent = pnew_node;
            for (j = i + 1; j < nlen; j++)
            {
                pnew_node = NewTreeNode();
                pnew_node->nid = pset[j];
                pnew_node->pchild = NULL;
                pnew_node->pright_sib = NULL;
                pparent->pchild = pnew_node;
                pparent = pnew_node;
            }
            break;
        }
        else
        {
            pparent = pnode;
            pnode = pnode->pchild;
            pleftsib = NULL;
        }
        i++;
    }
}

int BuildTree(char* szset_filename, TREE_NODE*& proot)
{
    FILE* fp;
    int nlen, * pset, nset_size, i, nmax_len, num_of_sets;

    fp = fopen(szset_filename, "rt");
    if (fp == NULL)
    {
        printf("Error: cannot open file %s for read\n", szset_filename);
        return 0;
    }

    gotreenode_buf.phead = new TNODE_PAGE;
    gotreenode_buf.phead->pnext = NULL;
    gotreenode_buf.pcur_page = gotreenode_buf.phead;
    gotreenode_buf.ntotal_pages = 1;
    gotreenode_buf.ncur_pos = 0;

    proot = NULL;

    num_of_sets = 0;

    nset_size = 100;
    pset = new int[nset_size];

    nmax_len = 0;
    fscanf(fp, "%d", &nlen);
    while (!feof(fp))
    {
        if (nmax_len < nlen)
            nmax_len = nlen;
        if (nlen > nset_size)
        {
            delete[]pset;
            nset_size *= 2;
            if (nset_size < nlen)
                nset_size = nlen;
            pset = new int[nset_size];
        }
        for (i = 0; i < nlen; i++)
            fscanf(fp, "%d", &pset[i]);
        qsort(pset, nlen, sizeof(int), comp_int);
        InsertOneSet(pset, nlen, proot);

        num_of_sets++;
        fscanf(fp, "%d", &nlen);
    }
    fclose(fp);

    delete[]pset;

    return nmax_len;
}

void SearchSubset(int* pset, int nset_len, TREE_NODE* proot, TREE_NODE** pstack, int* ppos)
{
    TREE_NODE* pnode;
    int ntop, npos;

    if (proot == NULL)
        return;
    ntop = 0;
    npos = 0;
    pnode = proot;

    while (ntop >= 0)
    {
        while (pnode != NULL && npos < nset_len && pnode->nid != pset[npos])
        {
            if (pnode->nid < pset[npos])
                pnode = pnode->pright_sib;
            else
                npos++;
        }
        if (pnode != NULL && npos < nset_len)
        {
            if (pnode->pchild == NULL && pnode->bis_max)
                pnode->bis_max = false;
            pstack[ntop] = pnode;
            ppos[ntop] = npos;
            ntop++;
            pnode = pnode->pchild;
            npos++;
        }
        else
        {
            ntop--;
            if (ntop >= 0)
            {
                pnode = pstack[ntop]->pright_sib;
                npos = ppos[ntop] + 1;
            }
        }
    }

}

void RmNonMax(TREE_NODE* proot, int nmax_len)
{
    TREE_NODE* pnode, ** pstack, ** psearch_stack;
    int* pset, ntop, i, * ppos;

    pset = new int[nmax_len];
    pstack = new TREE_NODE * [nmax_len];
    psearch_stack = new TREE_NODE * [nmax_len];
    ppos = new int[nmax_len];

    pstack[0] = proot;
    pset[0] = proot->nid;
    ntop = 1;
    pnode = proot;

    while (ntop > 0)
    {
        if (pnode->pchild != NULL)
        {
            pnode = pnode->pchild;
            pstack[ntop] = pnode;
            pset[ntop] = pnode->nid;
            ntop++;
        }
        else
        {
            if (ntop >= 2 && pnode->bis_max)
            {
                for (i = ntop - 1; i >= 1; i--)
                {
                    if (pstack[i - 1]->pright_sib != NULL)
                        SearchSubset(&pset[i], ntop - i, pstack[i - 1]->pright_sib, psearch_stack, ppos);
                }
            }

            while (ntop > 0 && pnode->pright_sib == NULL)
            {
                ntop--;
                if (ntop > 0)
                    pnode = pstack[ntop - 1];
            }
            if (ntop == 0)
                break;
            else //if(pnode->pright_sib!=NULL)
            {
                pnode = pnode->pright_sib;
                pstack[ntop - 1] = pnode;
                pset[ntop - 1] = pnode->nid;
            }
        }
    }

    delete[]pset;
    delete[]pstack;
    delete[]psearch_stack;
    delete[]ppos;
}

void OutputMaxSet(TREE_NODE* proot, int nmax_len, char* szoutput_filename)
{
    FILE* fp;
    TREE_NODE** pstack, * pnode;
    int* pset, ntop;

    fp = fopen(szoutput_filename, "wt");
    if (fp == NULL)
    {
        printf("Error: cannot open file %s for write\n", szoutput_filename);
        return;
    }

    pstack = new TREE_NODE * [nmax_len];
    pset = new int[nmax_len];

    pstack[0] = proot;
    pset[0] = proot->nid;
    ntop = 1;
    pnode = proot;

    while (ntop > 0)
    {
        if (pnode->pchild != NULL)
        {
            pnode = pnode->pchild;
            pstack[ntop] = pnode;
            pset[ntop] = pnode->nid;
            ntop++;
        }
        else
        {
            if (pnode->bis_max)
                OutputOneSet(fp, pset, ntop);

            while (ntop > 0 && pnode->pright_sib == NULL)
            {
                ntop--;
                if (ntop > 0)
                    pnode = pstack[ntop - 1];
            }
            if (ntop == 0)
                break;
            else //if(pnode->pright_sib!=NULL)
            {
                pnode = pnode->pright_sib;
                pstack[ntop - 1] = pnode;
                pset[ntop - 1] = pnode->nid;
            }
        }
    }

    delete[]pstack;
    delete[]pset;

    fclose(fp);
}

void RemoveNonMax(char* szset_filename, char* szoutput_filename)
{
    cout << ">:REMOVING NON-MAXIMAL CLIQUES" << endl;

    TREE_NODE* proot;
    int nmax_len;
    struct timeb start, end;

    ftime(&start);

    gntotal_max_cliques = 0;

    nmax_len = BuildTree(szset_filename, proot);
    RmNonMax(proot, nmax_len);
    OutputMaxSet(proot, nmax_len, szoutput_filename);

    DelTNodeBuf();

    ftime(&end);


    printf(">:NUMBER OF MAXIMAL CLIQUES: %d\n", gntotal_max_cliques);
}