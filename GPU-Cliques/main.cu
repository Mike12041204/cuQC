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
#define WTASKS_SIZE 50000
#define WTASKS_OFFSET_SIZE 500
#define WVERTICES_SIZE 20000

// shared memory size: 12.300 ints
#define VERTICES_SIZE 120
 
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
void allocate_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques, CPU_Graph& input_graph);
void initialize_tasks(CPU_Graph& graph, CPU_Data& host_data);
void move_to_gpu(CPU_Data& host_data, GPU_Data& device_data);
void dump_cliques(CPU_Cliques& host_cliques, GPU_Data& device_data, ofstream& output_file);
void free_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques);
void RemoveNonMax(char* szset_filename, char* szoutput_filename);

int binary_search_array(int* search_array, int array_size, int search_number);
int sort_vertices(const void* a, const void* b);
int get_mindeg(int clique_size);
bool cand_isvalid(Vertex& vertex, int clique_size);
inline void chkerr(cudaError_t code);

void print_CPU_Data(CPU_Data& host_data);
void print_GPU_Data(GPU_Data& device_data);
void print_CPU_Graph(CPU_Graph& host_graph);
void print_GPU_Graph(GPU_Data& device_data, CPU_Graph& host_graph);
void print_WTask_Buffers(GPU_Data& device_data);
void print_WClique_Buffers(GPU_Data& device_data);
void print_GPU_Cliques(GPU_Data& device_data);
void print_CPU_Cliques(CPU_Cliques& host_cliques);
void print_Data_Sizes(GPU_Data& device_data);
void print_vertices(Vertex* vertices, int size);

// KERNELS
__global__ void expand_level(GPU_Data device_data);
__global__ void transfer_buffers(GPU_Data device_data);
__global__ void fill_from_buffer(GPU_Data device_data);
__device__ void check_for_clique(GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data);
__device__ void write_to_tasks(GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data);
__device__ void calculate_LU_bounds(GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data);

__device__ void device_sort(Vertex* target, int size, int lane_idx);
__device__ void sort_vert(Vertex& vertex1, Vertex& vertex2, int& result);
__device__ int device_bsearch_array(int* search_array, int array_size, int search_number);
__device__ bool device_cand_isvalid(Vertex& vertex, int number_of_members, GPU_Data& device_data);
__device__ bool device_cand_isvalid_LU(Vertex& vertex, GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data);
__device__ bool device_vert_isextendable(Vertex& vertex, int number_of_members, GPU_Data& device_data);
__device__ bool device_vert_isextendable_LU(Vertex& vertex, GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data);
__device__ int device_get_mindeg(int number_of_members, GPU_Data& device_data);

// INPUT SETTINGS
double minimum_degree_ratio;
int minimum_clique_size;
int* minimum_degrees;



// TODO - verify dumping cliques works
// TODO - test program on larger graphs
// TODO - increase thread usage by monitoring and improving memory usage
// UNSURE - inline smaller methods?

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
    GPU_Data device_data;

    // HANDLE MEMORY
    allocate_memory(host_data, device_data, host_cliques, input_graph);
    cudaDeviceSynchronize();

    // INITIALIZE TASKS
    cout << ">:INITIALIZING TASKS" << endl;
    initialize_tasks(input_graph, host_data);

    // TRANSFER TO GPU
    move_to_gpu(host_data, device_data);
    cudaDeviceSynchronize();

    // DEBUG
    //print_GPU_Graph(device_data, input_graph);
    //print_CPU_Data(host_data);
    //print_GPU_Data(device_data);
    print_Data_Sizes(device_data);

    // UNSURE - are all device syncs are necessary? how does chkerr effect this
    // EXPAND LEVEL
    cout << ">:BEGINNING EXPANSION" << endl;
    while (!(*host_data.maximal_expansion))
    {
        chkerr(cudaMemset(device_data.maximal_expansion, true, sizeof(bool)));
        chkerr(cudaMemset(device_data.dumping_cliques, false, sizeof(bool)));
        cudaDeviceSynchronize();

        expand_level<<<NUM_OF_BLOCKS, BLOCK_SIZE >>>(device_data);
        cudaDeviceSynchronize();

        // DEBUG
        //print_WClique_Buffers(device_cliques);
        //print_WTask_Buffers(device_data);

        transfer_buffers<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(device_data);
        cudaDeviceSynchronize();

        fill_from_buffer<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(device_data);
        cudaDeviceSynchronize();

        chkerr(cudaMemcpy(host_data.maximal_expansion, device_data.maximal_expansion, sizeof(bool), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(host_data.dumping_cliques, device_data.dumping_cliques, sizeof(bool), cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize();

        if (*host_data.dumping_cliques) {
            dump_cliques(host_cliques, device_data, temp_results);
        }

        // DEBUG
        //print_GPU_Data(device_data);
        //print_GPU_Cliques(device_cliques);
        print_Data_Sizes(device_data);
        //bool debug;
        //chkerr(cudaMemcpy(&debug, device_data.debug, sizeof(bool), cudaMemcpyDeviceToHost));
        //if (debug) {
        //    cout << "!!!DEBUG!!! " << endl;
        //}
        //chkerr(cudaMemset(device_data.debug, false, sizeof(bool)));
        //int idebug;
        //chkerr(cudaMemcpy(&idebug, device_data.idebug, sizeof(int), cudaMemcpyDeviceToHost));
        //cout << "LU-PRUNING REMOVED: " << idebug << endl;
        //chkerr(cudaMemset(device_data.idebug, 0, sizeof(int)));
    }

    dump_cliques(host_cliques, device_data, temp_results);

    // FREE MEMORY
    free_memory(host_data, device_data, host_cliques);
}

// allocates memory for the data structures on the host and device
void allocate_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques, CPU_Graph& input_graph)
{
    int number_of_warps = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;

    // GPU GRAPH
    chkerr(cudaMalloc((void**)&device_data.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_data.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_data.onehop_neighbors, sizeof(int) * input_graph.number_of_onehop_neighbors));
    chkerr(cudaMalloc((void**)&device_data.onehop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&device_data.twohop_neighbors, sizeof(int) * input_graph.number_of_twohop_neighbors));
    chkerr(cudaMalloc((void**)&device_data.twohop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1)));

    chkerr(cudaMemcpy(device_data.number_of_vertices, &(input_graph.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.number_of_edges, &(input_graph.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.onehop_neighbors, input_graph.onehop_neighbors, sizeof(int) * input_graph.number_of_onehop_neighbors, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.onehop_offsets, input_graph.onehop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.twohop_neighbors, input_graph.twohop_neighbors, sizeof(int) * input_graph.number_of_twohop_neighbors, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.twohop_offsets, input_graph.twohop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));

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
    chkerr(cudaMalloc((void**)&device_data.current_level, sizeof(uint64_t)));

    uint64_t temp = 1;
    chkerr(cudaMemcpy(device_data.current_level, &temp, sizeof(uint64_t), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&device_data.tasks1_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.tasks1_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&device_data.tasks1_vertices, sizeof(Vertex) * TASKS_SIZE));

    chkerr(cudaMemset(device_data.tasks1_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.tasks1_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.tasks2_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.tasks2_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&device_data.tasks2_vertices, sizeof(Vertex) * TASKS_SIZE));

    chkerr(cudaMemset(device_data.tasks2_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.tasks2_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.buffer_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.buffer_offset, sizeof(uint64_t) * BUFFER_OFFSET_SIZE));
    chkerr(cudaMalloc((void**)&device_data.buffer_vertices, sizeof(Vertex) * BUFFER_SIZE));

    chkerr(cudaMemset(device_data.buffer_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.buffer_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.wtasks_count, sizeof(uint64_t) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_offset, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_vertices, (sizeof(Vertex) * WTASKS_SIZE) * number_of_warps));

    chkerr(cudaMemset(device_data.wtasks_offset, 0, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * number_of_warps));
    chkerr(cudaMemset(device_data.wtasks_count, 0, sizeof(uint64_t) * number_of_warps));

    chkerr(cudaMalloc((void**)&device_data.wvertices, (sizeof(Vertex) * WVERTICES_SIZE) * number_of_warps));

    chkerr(cudaMalloc((void**)&device_data.maximal_expansion, sizeof(bool)));
    chkerr(cudaMalloc((void**)&device_data.dumping_cliques, sizeof(bool)));

    chkerr(cudaMemset(device_data.maximal_expansion, false, sizeof(bool)));
    chkerr(cudaMemset(device_data.dumping_cliques, false, sizeof(bool)));

    chkerr(cudaMalloc((void**)&device_data.minimum_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&device_data.minimum_degrees, sizeof(int) * (input_graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&device_data.minimum_clique_size, sizeof(int)));

    chkerr(cudaMemcpy(device_data.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.minimum_degrees, minimum_degrees, sizeof(int) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&device_data.total_tasks, sizeof(int)));

    chkerr(cudaMemset(device_data.total_tasks, 0, sizeof(int)));

    // CPU CLIQUES
    host_cliques.cliques_count = new uint64_t;
    host_cliques.cliques_vertex = new int[CLIQUES_SIZE];
    host_cliques.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];

    host_cliques.cliques_offset[0] = 0;
    (*(host_cliques.cliques_count)) = 0;

    // GPU CLIQUES
    chkerr(cudaMalloc((void**)&device_data.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.cliques_vertex, sizeof(int) * CLIQUES_SIZE));
    chkerr(cudaMalloc((void**)&device_data.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));

    chkerr(cudaMemset(device_data.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.cliques_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.wcliques_count, sizeof(uint64_t) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wcliques_offset, (sizeof(uint64_t)* WCLIQUES_OFFSET_SIZE)* number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wcliques_vertex, (sizeof(int) * WCLIQUES_SIZE) * number_of_warps));

    chkerr(cudaMemset(device_data.wcliques_offset, 0, (sizeof(uint64_t)* WCLIQUES_OFFSET_SIZE)* number_of_warps));
    chkerr(cudaMemset(device_data.wcliques_count, 0, sizeof(uint64_t)* number_of_warps));

    chkerr(cudaMalloc((void**)&device_data.total_cliques, sizeof(int)));

    chkerr(cudaMemset(device_data.total_cliques, 0, sizeof(int)));

    chkerr(cudaMalloc((void**)&device_data.buffer_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.buffer_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.cliques_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.cliques_start, sizeof(uint64_t)));

    // DEBUG
    chkerr(cudaMalloc((void**)&device_data.debug, sizeof(bool)));
    chkerr(cudaMalloc((void**)&device_data.idebug, sizeof(int)));

    chkerr(cudaMemset(device_data.debug, false, sizeof(bool)));
    chkerr(cudaMemset(device_data.idebug, 0, sizeof(int)));
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

void move_to_gpu(CPU_Data& host_data, GPU_Data& device_data)
{
    cudaMemcpy(device_data.tasks1_count, host_data.tasks1_count, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.tasks1_offset, host_data.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.tasks1_vertices, host_data.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyHostToDevice);

    cudaMemcpy(device_data.buffer_count, host_data.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_offset, host_data.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_vertices, host_data.buffer_vertices, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
}

void dump_cliques(CPU_Cliques& host_cliques, GPU_Data& device_data, ofstream& temp_results)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(host_cliques.cliques_count, device_data.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(host_cliques.cliques_offset, device_data.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(host_cliques.cliques_vertex, device_data.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));
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
    cudaMemset(device_data.cliques_count, 0, sizeof(uint64_t));
}

void free_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques)
{
    // GPU GRAPH
    chkerr(cudaFree(device_data.number_of_vertices));
    chkerr(cudaFree(device_data.number_of_edges));
    chkerr(cudaFree(device_data.onehop_neighbors));
    chkerr(cudaFree(device_data.onehop_offsets));
    chkerr(cudaFree(device_data.twohop_neighbors));
    chkerr(cudaFree(device_data.twohop_offsets));

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
    chkerr(cudaFree(device_data.current_level));

    chkerr(cudaFree(device_data.tasks1_count));
    chkerr(cudaFree(device_data.tasks1_offset));
    chkerr(cudaFree(device_data.tasks1_vertices));

    chkerr(cudaFree(device_data.tasks2_count));
    chkerr(cudaFree(device_data.tasks2_offset));
    chkerr(cudaFree(device_data.tasks2_vertices));

    chkerr(cudaFree(device_data.buffer_count));
    chkerr(cudaFree(device_data.buffer_offset));
    chkerr(cudaFree(device_data.buffer_vertices));

    chkerr(cudaFree(device_data.wtasks_count));
    chkerr(cudaFree(device_data.wtasks_offset));
    chkerr(cudaFree(device_data.wtasks_vertices));

    chkerr(cudaFree(device_data.wvertices));

    chkerr(cudaFree(device_data.maximal_expansion));
    chkerr(cudaFree(device_data.dumping_cliques));

    chkerr(cudaFree(device_data.minimum_degree_ratio));
    chkerr(cudaFree(device_data.minimum_degrees));
    chkerr(cudaFree(device_data.minimum_clique_size));

    chkerr(cudaFree(device_data.total_tasks));

    // CPU CLIQUES
    delete host_cliques.cliques_count;
    delete host_cliques.cliques_vertex;
    delete host_cliques.cliques_offset;

    // GPU CLIQUES
    chkerr(cudaFree(device_data.cliques_count));
    chkerr(cudaFree(device_data.cliques_vertex));
    chkerr(cudaFree(device_data.cliques_offset));

    chkerr(cudaFree(device_data.wcliques_count));
    chkerr(cudaFree(device_data.wcliques_vertex));
    chkerr(cudaFree(device_data.wcliques_offset));

    chkerr(cudaFree(device_data.buffer_offset_start));
    chkerr(cudaFree(device_data.buffer_start));
    chkerr(cudaFree(device_data.cliques_offset_start));
    chkerr(cudaFree(device_data.cliques_start));

    //DEBUG
    chkerr(cudaFree(device_data.debug));
    chkerr(cudaFree(device_data.idebug));
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

int get_mindeg(int clique_size) {
    if (clique_size < minimum_clique_size) {
        return minimum_degrees[minimum_clique_size];
    }
    else {
        return minimum_degrees[clique_size];
    }
}

bool cand_isvalid(Vertex& vertex, int clique_size) {
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

void print_GPU_Graph(GPU_Data& device_data, CPU_Graph& host_graph)
{
    int* number_of_vertices = new int;
    int* number_of_edges = new int;

    int* onehop_neighbors = new int[host_graph.number_of_onehop_neighbors];
    uint64_t * onehop_offsets = new uint64_t[(host_graph.number_of_vertices)+1];
    int* twohop_neighbors = new int[host_graph.number_of_twohop_neighbors];
    uint64_t * twohop_offsets = new uint64_t[(host_graph.number_of_vertices)+1];

    chkerr(cudaMemcpy(number_of_vertices, device_data.number_of_vertices, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(number_of_edges, device_data.number_of_edges, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_neighbors, device_data.onehop_neighbors, sizeof(int)*host_graph.number_of_onehop_neighbors, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_offsets, device_data.onehop_offsets, sizeof(uint64_t)*(host_graph.number_of_vertices+1), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_neighbors, device_data.twohop_neighbors, sizeof(int)*host_graph.number_of_twohop_neighbors, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_offsets, device_data.twohop_offsets, sizeof(uint64_t)*(host_graph.number_of_vertices+1), cudaMemcpyDeviceToHost));

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

void print_GPU_Data(GPU_Data& device_data)
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


    chkerr(cudaMemcpy(current_level, device_data.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks1_count, device_data.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_offset, device_data.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_vertices, device_data.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks2_count, device_data.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_offset, device_data.tasks2_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_vertices, device_data.tasks2_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(buffer_count, device_data.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_offset, device_data.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_vertices, device_data.buffer_vertices, (BUFFER_SIZE) * sizeof(Vertex), cudaMemcpyDeviceToHost));

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

void print_Data_Sizes(GPU_Data& device_data)
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

    chkerr(cudaMemcpy(current_level, device_data.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_count, device_data.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_count, device_data.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_count, device_data.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_count, device_data.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_size, device_data.tasks1_offset + (*tasks1_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_size, device_data.tasks2_offset + (*tasks2_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_size, device_data.buffer_offset + (*buffer_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_size, device_data.cliques_offset + (*cliques_count), sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cout << "L: " << (*current_level) << " T1: " << (*tasks1_count) << " " << (*tasks1_size) << " T2: " << (*tasks2_count) << " " << (*tasks2_size) << " B: " << (*buffer_count) << " " << (*buffer_size) << " C: " << (*cliques_count) << " " << (*cliques_size) << endl;

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

void print_WTask_Buffers(GPU_Data& device_data)
{
    int warp_count = (NUM_OF_BLOCKS * BLOCK_SIZE) / 32;
    uint64_t* wtasks_count = new uint64_t[warp_count];
    uint64_t* wtasks_offset = new uint64_t[warp_count*WTASKS_OFFSET_SIZE];
    Vertex* wtasks_vertices = new Vertex[warp_count*WTASKS_SIZE];

    chkerr(cudaMemcpy(wtasks_count, device_data.wtasks_count, sizeof(uint64_t)*warp_count, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wtasks_offset, device_data.wtasks_offset, sizeof(uint64_t) * (warp_count*WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wtasks_vertices, device_data.wtasks_vertices, sizeof(Vertex) * (warp_count*WTASKS_SIZE), cudaMemcpyDeviceToHost));

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

void print_WClique_Buffers(GPU_Data& device_data)
{
    int warp_count = (NUM_OF_BLOCKS * BLOCK_SIZE) / 32;
    uint64_t* wcliques_count = new uint64_t[warp_count];
    uint64_t* wcliques_offset = new uint64_t[warp_count * WCLIQUES_OFFSET_SIZE];
    int* wcliques_vertex = new int[warp_count * WCLIQUES_SIZE];

    chkerr(cudaMemcpy(wcliques_count, device_data.wcliques_count, sizeof(uint64_t) * warp_count, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wcliques_offset, device_data.wcliques_offset, sizeof(uint64_t) * (warp_count * WTASKS_OFFSET_SIZE), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(wcliques_vertex, device_data.wcliques_vertex, sizeof(int) * (warp_count * WTASKS_SIZE), cudaMemcpyDeviceToHost));

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

void print_GPU_Cliques(GPU_Data& device_data)
{
    uint64_t* cliques_count = new uint64_t;
    uint64_t* cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];
    int* cliques_vertex = new int[CLIQUES_SIZE];

    chkerr(cudaMemcpy(cliques_count, device_data.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_offset, device_data.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(cliques_vertex, device_data.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));

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

__global__ void expand_level(GPU_Data device_data)
{
    // data is stored in data structures to reduce the number of variables that need to be passed to methods
    __shared__ Warp_Data warp_data;
    Local_Data local_data;

    // helper variables, not passed through to any methods
    int pvertexid;
    bool failed_found;
    bool lookahead_sucess;
    int number_of_removed_candidates;

    // initialize variables
    local_data.idx = (blockIdx.x * blockDim.x + threadIdx.x);
    local_data.warp_in_block_idx = ((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE));

    if ((*(device_data.current_level)) % 2 == 1) {
        local_data.read_count = device_data.tasks1_count;
        local_data.read_offsets = device_data.tasks1_offset;
        local_data.read_vertices = device_data.tasks1_vertices;
    } else {
        local_data.read_count = device_data.tasks2_count;
        local_data.read_offsets = device_data.tasks2_offset;
        local_data.read_vertices = device_data.tasks2_vertices;
    }



    // --- CURRENT LEVEL ---
    for (int i = (local_data.idx / WARP_SIZE); i < (*(local_data.read_count)); i += ((NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE))
    {
        // get information of vertices being handled within tasks
        if ((local_data.idx % WARP_SIZE) == 0) {
            warp_data.start[local_data.warp_in_block_idx] = local_data.read_offsets[i];
            warp_data.end[local_data.warp_in_block_idx] = local_data.read_offsets[i + 1];
            warp_data.tot_vert[local_data.warp_in_block_idx] = warp_data.end[local_data.warp_in_block_idx] - warp_data.start[local_data.warp_in_block_idx];
            warp_data.num_mem[local_data.warp_in_block_idx] = 0;
            for (uint64_t j = warp_data.start[local_data.warp_in_block_idx]; j < warp_data.end[local_data.warp_in_block_idx]; j++) {
                if (local_data.read_vertices[j].label == 1) {
                    warp_data.num_mem[local_data.warp_in_block_idx]++;
                } else {
                    break;
                }
            }
            warp_data.num_cand[local_data.warp_in_block_idx] = warp_data.tot_vert[local_data.warp_in_block_idx] - warp_data.num_mem[local_data.warp_in_block_idx];
            warp_data.expansions[local_data.warp_in_block_idx] = warp_data.num_cand[local_data.warp_in_block_idx];
        }
        __syncwarp();



        // LOOKAHEAD PRUNING
        lookahead_sucess = true;
        for (int j = (local_data.idx % WARP_SIZE); j < warp_data.tot_vert[local_data.warp_in_block_idx]; j += WARP_SIZE) {
            if (local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + j].lvl2adj != (warp_data.tot_vert[local_data.warp_in_block_idx] - 1) ||
                local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + j].indeg + local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + j].exdeg <
                device_data.minimum_degrees[warp_data.tot_vert[local_data.warp_in_block_idx]]) {
                lookahead_sucess = false;
                break;
            }
        }
        lookahead_sucess = !(__any_sync(0xFFFFFFFF, !lookahead_sucess));

        if (lookahead_sucess) {
            // write to cliques
            uint64_t start_write = (WCLIQUES_SIZE * (local_data.idx / WARP_SIZE)) + device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (local_data.idx / WARP_SIZE)) +
                (device_data.wcliques_count[(local_data.idx / WARP_SIZE)])];
            for (int j = (local_data.idx % WARP_SIZE); j < warp_data.tot_vert[local_data.warp_in_block_idx]; j += WARP_SIZE) {
                device_data.wcliques_vertex[start_write + j] = local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + j].vertexid;
            }
            if ((local_data.idx % WARP_SIZE) == 0) {
                (device_data.wcliques_count[(local_data.idx / WARP_SIZE)])++;
                device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (local_data.idx / WARP_SIZE)) + (device_data.wcliques_count[(local_data.idx / WARP_SIZE)])] = start_write - (WCLIQUES_SIZE * (local_data.idx / WARP_SIZE)) +
                    warp_data.tot_vert[local_data.warp_in_block_idx];
            }
            continue;
        }



        // --- NEXT LEVEL ---
        for (int j = 0; j < warp_data.expansions[local_data.warp_in_block_idx]; j++)
        {



            // REMOVE ONE VERTEX
            if (j > 0) {
                if ((local_data.idx % WARP_SIZE) == 0) {
                    warp_data.num_cand[local_data.warp_in_block_idx]--;
                    warp_data.tot_vert[local_data.warp_in_block_idx]--;
                }
                __syncwarp();

                pvertexid = local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + warp_data.tot_vert[local_data.warp_in_block_idx]].vertexid;
                for (int k = (local_data.idx % WARP_SIZE); k < warp_data.tot_vert[local_data.warp_in_block_idx]; k += WARP_SIZE) {
                    if (device_bsearch_array(device_data.onehop_neighbors + device_data.onehop_offsets[pvertexid], device_data.onehop_offsets[pvertexid + 1] - device_data.onehop_offsets[pvertexid],
                        local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + k].vertexid) != -1) {
                        local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + k].exdeg--;
                    }

                    if (device_bsearch_array(device_data.twohop_neighbors + device_data.twohop_offsets[pvertexid], device_data.twohop_offsets[pvertexid + 1] - device_data.twohop_offsets[pvertexid],
                        local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + k].vertexid) != -1) {
                        local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + k].lvl2adj--;
                    }
                }
                __syncwarp();
            }

            // check for failed vertices
            failed_found = false;
            for (int k = (local_data.idx % WARP_SIZE); k < warp_data.num_mem[local_data.warp_in_block_idx]; k += WARP_SIZE) {
                if (!device_vert_isextendable(local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + k], 
                    warp_data.num_mem[local_data.warp_in_block_idx], device_data)) {
                    failed_found = true;
                    break;
                }

            }
            failed_found = __any_sync(0xFFFFFFFF, failed_found);
            if (failed_found) {
                continue;
            }



            // INITIALIZE NEW VERTICES
            if ((local_data.idx % WARP_SIZE) == 0) {
                warp_data.number_of_members[local_data.warp_in_block_idx] = warp_data.num_mem[local_data.warp_in_block_idx];
                warp_data.number_of_candidates[local_data.warp_in_block_idx] = warp_data.num_cand[local_data.warp_in_block_idx];
                warp_data.total_vertices[local_data.warp_in_block_idx] = warp_data.tot_vert[local_data.warp_in_block_idx];
            }
            __syncwarp();

            // select whether to store vertices in global or shared memory based on size
            if (warp_data.total_vertices[local_data.warp_in_block_idx] <= VERTICES_SIZE) {
                local_data.vertices = warp_data.shared_vertices + (VERTICES_SIZE * local_data.warp_in_block_idx);
            } else {
                local_data.vertices = device_data.wvertices + (WVERTICES_SIZE * (local_data.idx / WARP_SIZE));
            }

            for (int k = (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[local_data.warp_in_block_idx]; k += WARP_SIZE) {
                local_data.vertices[k] = local_data.read_vertices[warp_data.start[local_data.warp_in_block_idx] + k];
            }



            // ADD ONE VERTEX
            if ((local_data.idx % WARP_SIZE) == 0) {
                local_data.vertices[warp_data.total_vertices[local_data.warp_in_block_idx] - 1].label = 1;
                warp_data.number_of_members[local_data.warp_in_block_idx]++;
                warp_data.number_of_candidates[local_data.warp_in_block_idx]--;
            }
            __syncwarp();

            // update the exdeg and indeg of all vertices adj to the vertex just added to the vertex set
            pvertexid = local_data.vertices[warp_data.total_vertices[local_data.warp_in_block_idx] - 1].vertexid;
            for (int k = (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[local_data.warp_in_block_idx]; k += WARP_SIZE) {
                if (device_bsearch_array(device_data.onehop_neighbors + device_data.onehop_offsets[local_data.vertices[k].vertexid], device_data.onehop_offsets[local_data.vertices[k].vertexid + 1] -
                    device_data.onehop_offsets[local_data.vertices[k].vertexid], pvertexid) != -1) {
                    local_data.vertices[k].exdeg--;
                    local_data.vertices[k].indeg++;
                }
            }
            __syncwarp();

            // sort new vertices putting just added vertex at end of all vertices in x
            device_sort(local_data.vertices + warp_data.number_of_members[local_data.warp_in_block_idx] - 1, warp_data.number_of_candidates[local_data.warp_in_block_idx] + 1,
                (local_data.idx% WARP_SIZE));



            // --- DIAMETER PRUNING ---
            number_of_removed_candidates = 0;
            for (int k = warp_data.number_of_members[local_data.warp_in_block_idx] + (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[local_data.warp_in_block_idx];
                k += WARP_SIZE) {
                if (device_bsearch_array(device_data.twohop_neighbors + device_data.twohop_offsets[pvertexid], device_data.twohop_offsets[pvertexid + 1] - device_data.twohop_offsets[pvertexid], local_data.vertices[k].vertexid) == -1) {
                    local_data.vertices[k].label = -1;
                    number_of_removed_candidates++;
                }
            }
            for (int k = 1; k < 32; k *= 2) {
                number_of_removed_candidates += __shfl_xor_sync(0xFFFFFFFF, number_of_removed_candidates, k);
            }
            device_sort(local_data.vertices + warp_data.number_of_members[local_data.warp_in_block_idx], warp_data.number_of_candidates[local_data.warp_in_block_idx],
                (local_data.idx % WARP_SIZE));

            // update exdeg of vertices connected to removed cands
            if (warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates > number_of_removed_candidates) {
                for (int k = (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates; k += WARP_SIZE) {
                    pvertexid = local_data.vertices[k].vertexid;
                    for (int l = warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates; l <
                        warp_data.total_vertices[local_data.warp_in_block_idx]; l++) {
                        if (device_bsearch_array(device_data.onehop_neighbors + device_data.onehop_offsets[local_data.vertices[l].vertexid], device_data.onehop_offsets[local_data.vertices[l].vertexid + 1] -
                            device_data.onehop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                            local_data.vertices[k].exdeg--;
                        }

                        if (device_bsearch_array(device_data.twohop_neighbors + device_data.twohop_offsets[local_data.vertices[l].vertexid], device_data.twohop_offsets[local_data.vertices[l].vertexid + 1] -
                            device_data.twohop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                            local_data.vertices[k].lvl2adj--;
                        }
                    }
                }
                __syncwarp();
            }
            else {
                for (int k = 0; k < warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates; k++) {
                    pvertexid = local_data.vertices[k].vertexid;
                    for (int l = warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates + (local_data.idx % WARP_SIZE); l < warp_data.total_vertices[((local_data.idx / WARP_SIZE) %
                        (BLOCK_SIZE / WARP_SIZE))]; l += WARP_SIZE) {
                        if (device_bsearch_array(device_data.onehop_neighbors + device_data.onehop_offsets[local_data.vertices[l].vertexid], device_data.onehop_offsets[local_data.vertices[l].vertexid + 1] -
                            device_data.onehop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                            local_data.vertices[k].exdeg--;
                        }

                        if (device_bsearch_array(device_data.twohop_neighbors + device_data.twohop_offsets[local_data.vertices[l].vertexid], device_data.twohop_offsets[local_data.vertices[l].vertexid + 1] -
                            device_data.twohop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                            local_data.vertices[k].lvl2adj--;
                        }
                    }
                    __syncwarp();
                }
            }
            if ((local_data.idx % WARP_SIZE) == 0) {
                warp_data.total_vertices[local_data.warp_in_block_idx] -= number_of_removed_candidates;
                warp_data.number_of_candidates[local_data.warp_in_block_idx] -= number_of_removed_candidates;
            }
            __syncwarp();

            // continue if not enough vertices after pruning
            if (warp_data.total_vertices[local_data.warp_in_block_idx] < (*(device_data.minimum_clique_size))) {
                return;
            }



            // DEGREE BASED PRUNING
            do
            {
                // calculate lower and upper bounds for vertices
                //calculate_LU_bounds(device_data, warp_data, local_data);

                if (warp_data.invalid_bounds[local_data.warp_in_block_idx]) {
                    break;
                }

                // check for failed vertices
                failed_found = false;
                for (int k = (local_data.idx % WARP_SIZE); k < warp_data.number_of_members[local_data.warp_in_block_idx]; k += WARP_SIZE) {
                    if (!device_vert_isextendable(local_data.vertices[k], warp_data.number_of_members[local_data.warp_in_block_idx], device_data)) {
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
                for (int k = warp_data.number_of_members[local_data.warp_in_block_idx] + (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[local_data.warp_in_block_idx];
                    k += WARP_SIZE) {
                    if (!device_cand_isvalid(local_data.vertices[k], warp_data.number_of_members[local_data.warp_in_block_idx], device_data)) {
                        local_data.vertices[k].label = -1;
                        number_of_removed_candidates++;
                    }
                }
                for (int k = 1; k < 32; k *= 2) {
                    number_of_removed_candidates += __shfl_xor_sync(0xFFFFFFFF, number_of_removed_candidates, k);
                }
                // DEBUG
                if (local_data.idx % WARP_SIZE == 0) {
                    atomicAdd(device_data.idebug, number_of_removed_candidates);
                }
                device_sort(local_data.vertices + warp_data.number_of_members[local_data.warp_in_block_idx], warp_data.number_of_candidates[local_data.warp_in_block_idx],
                    (local_data.idx % WARP_SIZE));

                // update exdeg of vertices connected to removed cands
                if (warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates > number_of_removed_candidates) {
                    for (int k = (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates; k += WARP_SIZE) {
                        pvertexid = local_data.vertices[k].vertexid;
                        for (int l = warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates; l < warp_data.total_vertices[((local_data.idx / WARP_SIZE) %
                            (BLOCK_SIZE / WARP_SIZE))]; l++) {
                            if (device_bsearch_array(device_data.onehop_neighbors + device_data.onehop_offsets[local_data.vertices[l].vertexid], device_data.onehop_offsets[local_data.vertices[l].vertexid + 1] -
                                device_data.onehop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                                local_data.vertices[k].exdeg--;
                            }

                            if (device_bsearch_array(device_data.twohop_neighbors + device_data.twohop_offsets[local_data.vertices[l].vertexid], device_data.twohop_offsets[local_data.vertices[l].vertexid + 1] -
                                device_data.twohop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                                local_data.vertices[k].lvl2adj--;
                            }
                        }
                    }
                    __syncwarp();
                }
                else {
                    for (int k = 0; k < warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates; k++) {
                        pvertexid = local_data.vertices[k].vertexid;
                        for (int l = warp_data.total_vertices[local_data.warp_in_block_idx] - number_of_removed_candidates + (local_data.idx % WARP_SIZE); l <
                            warp_data.total_vertices[local_data.warp_in_block_idx]; l += WARP_SIZE) {
                            if (device_bsearch_array(device_data.onehop_neighbors + device_data.onehop_offsets[local_data.vertices[l].vertexid], device_data.onehop_offsets[local_data.vertices[l].vertexid + 1] -
                                device_data.onehop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                                local_data.vertices[k].exdeg--;
                            }

                            if (device_bsearch_array(device_data.twohop_neighbors + device_data.twohop_offsets[local_data.vertices[l].vertexid], device_data.twohop_offsets[local_data.vertices[l].vertexid + 1] -
                                device_data.twohop_offsets[local_data.vertices[l].vertexid], pvertexid) != -1) {
                                local_data.vertices[k].lvl2adj--;
                            }
                        }
                        __syncwarp();
                    }
                }
                if ((local_data.idx % WARP_SIZE) == 0) {
                    warp_data.total_vertices[local_data.warp_in_block_idx] -= number_of_removed_candidates;
                    warp_data.number_of_candidates[local_data.warp_in_block_idx] -= number_of_removed_candidates;
                }
                __syncwarp();
            } while (number_of_removed_candidates > 0);

            // continue if not enough vertices after pruning
            if (warp_data.total_vertices[local_data.warp_in_block_idx] < (*(device_data.minimum_clique_size))) {
                continue;
            }



            // HANDLE CLIQUES
            if (warp_data.number_of_members[local_data.warp_in_block_idx] >= (*device_data.minimum_clique_size)) {
                check_for_clique(device_data, warp_data, local_data);
            }

            // TODO - test if we need to check vertex sets that have invalid bounds, dont think so
            // if vertex in x found as not extendable continue to next iteration
            if (failed_found || warp_data.invalid_bounds[local_data.warp_in_block_idx]) {
                continue;
            }



            // WRITE TASKS TO BUFFERS
            if (warp_data.number_of_candidates[local_data.warp_in_block_idx] > 0) {
                write_to_tasks(device_data, warp_data, local_data);
            }
        }
    }



    if ((local_data.idx % WARP_SIZE) == 0) {
        // sum to find tasks count
        atomicAdd(device_data.total_tasks, device_data.wtasks_count[(local_data.idx / WARP_SIZE)]);
        atomicAdd(device_data.total_cliques, device_data.wcliques_count[(local_data.idx / WARP_SIZE)]);
    }

    if (local_data.idx == 0) {
        (*(device_data.buffer_offset_start)) = (*(device_data.buffer_count)) + 1;
        (*(device_data.buffer_start)) = device_data.buffer_offset[(*(device_data.buffer_count))];
        (*(device_data.cliques_offset_start)) = (*(device_data.cliques_count)) + 1;
        (*(device_data.cliques_start)) = device_data.cliques_offset[(*(device_data.cliques_count))];
    }
}

__global__ void transfer_buffers(GPU_Data device_data)
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

    if ((*(device_data.current_level)) % 2 == 1) {
        write_count = device_data.tasks2_count;
        write_offsets = device_data.tasks2_offset;
        write_vertices = device_data.tasks2_vertices;
    }
    else {
        write_count = device_data.tasks1_count;
        write_offsets = device_data.tasks1_offset;
        write_vertices = device_data.tasks1_vertices;
    }

    // block level
    if (threadIdx.x == 0) {
        toffsetwrite = 0;
        twrite = 0;

        for (int i = 0; i < ((NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE); i++) {
            if (toffsetwrite + device_data.wtasks_count[i] >= EXPAND_THRESHOLD) {
                twarp = i;
                break;
            }
            twrite += device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + device_data.wtasks_count[i]];
            toffsetwrite += device_data.wtasks_count[i];
        }
        tasks_end = twrite + device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * twarp) +
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
            tasks_offset_write[warp_in_block_idx] += device_data.wtasks_count[i];
            tasks_write[warp_in_block_idx] += device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + device_data.wtasks_count[i]];

            cliques_offset_write[warp_in_block_idx] += device_data.wcliques_count[i];
            cliques_write[warp_in_block_idx] += device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * i) + device_data.wcliques_count[i]];
        }
    }
    __syncwarp();

    // move to tasks and buffer
    for (int i = (idx % WARP_SIZE) + 1; i <= device_data.wtasks_count[(idx / WARP_SIZE)]; i += WARP_SIZE)
    {
        if (tasks_offset_write[warp_in_block_idx] + i - 1 <= EXPAND_THRESHOLD) {
            // to tasks
            write_offsets[tasks_offset_write[warp_in_block_idx] + i - 1] = device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + i] 
                + tasks_write[warp_in_block_idx];
        }
        else {
            // to buffer
            device_data.buffer_offset[tasks_offset_write[warp_in_block_idx] + i - 2 - EXPAND_THRESHOLD + (*(device_data.buffer_offset_start))] = 
                device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + tasks_write[warp_in_block_idx] - tasks_end + 
                (*(device_data.buffer_start));
        }
    }
    for (int i = (idx % WARP_SIZE); i < device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + device_data.wtasks_count[(idx / WARP_SIZE)]]; i += WARP_SIZE) {
        if (tasks_write[warp_in_block_idx] + i < tasks_end) {
            // to tasks
            write_vertices[tasks_write[warp_in_block_idx] + i] = device_data.wtasks_vertices[(WTASKS_SIZE * (idx / WARP_SIZE)) + i];
        }
        else {
            // to buffer
            device_data.buffer_vertices[(*(device_data.buffer_start)) + tasks_write[warp_in_block_idx] + i - tasks_end] = 
                device_data.wtasks_vertices[(WTASKS_SIZE * (idx / WARP_SIZE)) + i];
        }
    }

    //move to cliques
    for (int i = (idx % WARP_SIZE) + 1; i <= device_data.wcliques_count[(idx / WARP_SIZE)]; i += WARP_SIZE) {
        device_data.cliques_offset[(*(device_data.cliques_offset_start)) + cliques_offset_write[warp_in_block_idx] + i - 2] = 
            device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + (*(device_data.cliques_start)) + cliques_write[warp_in_block_idx];
    }
    for (int i = (idx % WARP_SIZE); i < device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (idx / WARP_SIZE)) + device_data.wcliques_count[(idx / WARP_SIZE)]]; i += WARP_SIZE) {
        device_data.cliques_vertex[(*(device_data.cliques_start)) + cliques_write[warp_in_block_idx] + i] = device_data.wcliques_vertex[(WCLIQUES_SIZE * 
            (idx / WARP_SIZE)) + i];
    }

    if (idx == 0) {
        // handle tasks and buffer counts
        if ((*device_data.total_tasks) <= EXPAND_THRESHOLD) {
            (*write_count) = (*(device_data.total_tasks));
        }
        else {
            (*write_count) = EXPAND_THRESHOLD;
            (*(device_data.buffer_count)) += ((*(device_data.total_tasks)) - EXPAND_THRESHOLD);
        }
        (*(device_data.cliques_count)) += (*(device_data.total_cliques));

        (*(device_data.total_tasks)) = 0;
        (*(device_data.total_cliques)) = 0;
    }

    // HANDLE CLIQUES
    // only first thread for each warp
    if ((idx % WARP_SIZE) == 0 && cliques_write[warp_in_block_idx] > (CLIQUES_SIZE * (CLIQUES_PERCENT / 100.0))) {
        atomicExch((int*)device_data.dumping_cliques, true);
    }
}

__global__ void fill_from_buffer(GPU_Data device_data)
{
    // THREAD INFO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = (idx / 32);
    int lane_idx = (idx % 32);

    Vertex* write_vertices;
    uint64_t* write_offsets;
    uint64_t* write_count;

    if ((*(device_data.current_level)) % 2 == 1) {
        write_count = device_data.tasks2_count;
        write_offsets = device_data.tasks2_offset;
        write_vertices = device_data.tasks2_vertices;
    } else {
        write_count = device_data.tasks1_count;
        write_offsets = device_data.tasks1_offset;
        write_vertices = device_data.tasks1_vertices;
    }

    if (lane_idx == 0) {
        device_data.wtasks_count[warp_idx] = 0;
        device_data.wcliques_count[warp_idx] = 0;
    }

    // FILL TASKS FROM BUFFER
    if ((*write_count) < EXPAND_THRESHOLD && (*(device_data.buffer_count)) > 0)
    {
        // CRITICAL
        atomicExch((int*)device_data.maximal_expansion, false);

        // get read and write locations
        int write_amount = ((*(device_data.buffer_count)) >= (EXPAND_THRESHOLD - (*write_count))) ? EXPAND_THRESHOLD - (*write_count) : (*(device_data.buffer_count));
        uint64_t start_buffer = device_data.buffer_offset[(*(device_data.buffer_count)) - write_amount];
        uint64_t end_buffer = device_data.buffer_offset[(*(device_data.buffer_count))];
        uint64_t size_buffer = end_buffer - start_buffer;
        uint64_t start_write = write_offsets[(*write_count)];

        // handle offsets
        for (int i = idx + 1; i <= write_amount; i += (NUM_OF_BLOCKS * BLOCK_SIZE)) {
            write_offsets[(*write_count) + i] = start_write + (device_data.buffer_offset[(*(device_data.buffer_count)) - write_amount + i] - start_buffer);
        }

        // handle data
        for (int i = idx; i < size_buffer; i += (NUM_OF_BLOCKS * BLOCK_SIZE)) {
            write_vertices[start_write + i] = device_data.buffer_vertices[start_buffer + i];
        }

        if (idx == 0) {
            (*write_count) += write_amount;
            (*(device_data.buffer_count)) -= write_amount;
        }
    }

    if (idx == 0) {
        (*device_data.current_level)++;
    }
}

__device__ void check_for_clique(GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data)
{
    bool clique = true;

    for (int k = (local_data.idx % WARP_SIZE); k < warp_data.number_of_members[((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE))]; k += WARP_SIZE) {
        if (local_data.vertices[k].indeg < device_data.minimum_degrees[warp_data.number_of_members[((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE))]]) {
            clique = false;
            break;
        }
    }
    // set to false if any threads in warp do not meet degree requirement
    clique = !(__any_sync(0xFFFFFFFF, !clique));

    // if clique write to warp buffer for cliques
    if (clique) {
        uint64_t start_write = (WCLIQUES_SIZE * (local_data.idx / WARP_SIZE)) + device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (local_data.idx / WARP_SIZE)) + (device_data.wcliques_count[(local_data.idx / WARP_SIZE)])];
        for (int k = (local_data.idx % WARP_SIZE); k < warp_data.number_of_members[((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE))]; k += WARP_SIZE) {
            device_data.wcliques_vertex[start_write + k] = local_data.vertices[k].vertexid;
        }
        if ((local_data.idx % WARP_SIZE) == 0) {
            (device_data.wcliques_count[(local_data.idx / WARP_SIZE)])++;
            device_data.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (local_data.idx / WARP_SIZE)) + (device_data.wcliques_count[(local_data.idx / WARP_SIZE)])] = start_write - (WCLIQUES_SIZE * (local_data.idx / WARP_SIZE)) +
                warp_data.number_of_members[((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE))];
        }
    }
}

__device__ void write_to_tasks(GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data)
{
    // CRITICAL
    atomicExch((int*)device_data.maximal_expansion, false);

    uint64_t start_write = (WTASKS_SIZE * (local_data.idx / WARP_SIZE)) + device_data.wtasks_offset[WTASKS_OFFSET_SIZE * (local_data.idx / WARP_SIZE) + (device_data.wtasks_count[(local_data.idx / WARP_SIZE)])];

    for (int k = (local_data.idx % WARP_SIZE); k < warp_data.total_vertices[((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE))]; k += WARP_SIZE) {
        device_data.wtasks_vertices[start_write + k].vertexid = local_data.vertices[k].vertexid;
        device_data.wtasks_vertices[start_write + k].label = local_data.vertices[k].label;
        device_data.wtasks_vertices[start_write + k].indeg = local_data.vertices[k].indeg;
        device_data.wtasks_vertices[start_write + k].exdeg = local_data.vertices[k].exdeg;
        device_data.wtasks_vertices[start_write + k].lvl2adj = local_data.vertices[k].lvl2adj;
    }
    if ((local_data.idx % WARP_SIZE) == 0) {
        (device_data.wtasks_count[(local_data.idx / WARP_SIZE)])++;
        device_data.wtasks_offset[(WTASKS_OFFSET_SIZE * (local_data.idx / WARP_SIZE)) + (device_data.wtasks_count[(local_data.idx / WARP_SIZE)])] = start_write - (WTASKS_SIZE * (local_data.idx / WARP_SIZE)) +
            warp_data.total_vertices[((local_data.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE))];
    }
}

// TODO - try to parallelize as much calculation as possible
__device__ void calculate_LU_bounds(GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data)
{
    int index;

    int min_clq_indeg;
    int min_indeg_exdeg;
    int min_clq_totaldeg;
    int sum_clq_indeg;

    // CALCULATE LOWER UPPER BOUNDS
    min_clq_indeg = local_data.vertices[0].indeg;
    min_indeg_exdeg = local_data.vertices[0].exdeg;
    min_clq_totaldeg = local_data.vertices[0].indeg + local_data.vertices[0].exdeg;
    sum_clq_indeg = 0;

    if ((local_data.idx % WARP_SIZE) == 0) {
        warp_data.invalid_bounds[local_data.warp_in_block_idx] = false;

        warp_data.sum_candidate_indeg[local_data.warp_in_block_idx] = 0;
        warp_data.tightened_Upper_bound[local_data.warp_in_block_idx] = 0;

        warp_data.min_clq_indeg[local_data.warp_in_block_idx] = local_data.vertices[0].indeg;
        warp_data.min_indeg_exdeg[local_data.warp_in_block_idx] = local_data.vertices[0].exdeg;
        warp_data.min_clq_totaldeg[local_data.warp_in_block_idx] = local_data.vertices[0].indeg + local_data.vertices[0].exdeg;
        warp_data.sum_clq_indeg[local_data.warp_in_block_idx] = local_data.vertices[0].indeg;

        warp_data.minimum_external_degree[local_data.warp_in_block_idx] = device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + 1,
            device_data);
    }
    __syncwarp();

    // each warp finds these values on their subsection of vertices
    for (index = 1 + (local_data.idx % WARP_SIZE); index < warp_data.number_of_members[local_data.warp_in_block_idx]; index += WARP_SIZE) {
        sum_clq_indeg += local_data.vertices[index].indeg;

        if (local_data.vertices[index].indeg < min_clq_indeg) {
            min_clq_indeg = local_data.vertices[index].indeg;
            min_indeg_exdeg = local_data.vertices[index].exdeg;
        }
        else if (local_data.vertices[index].indeg == min_clq_indeg) {
            if (local_data.vertices[index].exdeg < min_indeg_exdeg) {
                min_indeg_exdeg = local_data.vertices[index].exdeg;
            }
        }

        if (local_data.vertices[index].indeg + local_data.vertices[index].exdeg < min_clq_totaldeg) {
            min_clq_totaldeg = local_data.vertices[index].indeg + local_data.vertices[index].exdeg;
        }
    }

    // get sum
    for (int i = 1; i < 32; i *= 2) {
        sum_clq_indeg += __shfl_xor_sync(0xFFFFFFFF, sum_clq_indeg, i);
    }
    if ((local_data.idx % WARP_SIZE) == 0) {
        // add to shared memory sum
        warp_data.sum_clq_indeg[local_data.warp_in_block_idx] += sum_clq_indeg;
    }
    __syncwarp();

    // CRITICAL SECTION - each lane then compares their values to the next to get a warp level value
    for (int i = 0; i < WARP_SIZE; i++) {
        if ((local_data.idx % WARP_SIZE) == i) {
            if (min_clq_indeg < warp_data.min_clq_indeg[local_data.warp_in_block_idx]) {
                warp_data.min_clq_indeg[local_data.warp_in_block_idx] = min_clq_indeg;
                warp_data.min_indeg_exdeg[local_data.warp_in_block_idx] = min_indeg_exdeg;
            }
            else if (min_clq_indeg == warp_data.min_clq_indeg[local_data.warp_in_block_idx]) {
                if (min_indeg_exdeg < warp_data.min_indeg_exdeg[local_data.warp_in_block_idx]) {
                    warp_data.min_indeg_exdeg[local_data.warp_in_block_idx] = min_indeg_exdeg;
                }
            }

            if (min_clq_totaldeg < warp_data.min_clq_totaldeg[local_data.warp_in_block_idx]) {
                warp_data.min_clq_totaldeg[local_data.warp_in_block_idx] = min_clq_totaldeg;
            }
        }
        __syncwarp();
    }

    // CRITICAL SECTION - unsure how to parallelize this, very complex
    if ((local_data.idx % WARP_SIZE) == 0) {
        if (warp_data.min_clq_indeg[local_data.warp_in_block_idx] < device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx]])
        {
            // lower
            warp_data.Lower_bound[local_data.warp_in_block_idx] = device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx], device_data) -
                min_clq_indeg;

            while (warp_data.Lower_bound[local_data.warp_in_block_idx] <= warp_data.min_indeg_exdeg[local_data.warp_in_block_idx] &&
                warp_data.min_clq_indeg[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx] <
                device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx]]) {
                warp_data.Lower_bound[local_data.warp_in_block_idx]++;
            }

            if (warp_data.min_clq_indeg[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx] <
                device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx]]) {
                warp_data.invalid_bounds[local_data.warp_in_block_idx] = true;
            }

            // upper
            warp_data.Upper_bound[local_data.warp_in_block_idx] = floor(warp_data.min_clq_totaldeg[local_data.warp_in_block_idx] /
                (*(device_data.minimum_degree_ratio))) + 1 - warp_data.number_of_members[local_data.warp_in_block_idx];

            if (warp_data.Upper_bound[local_data.warp_in_block_idx] > warp_data.number_of_candidates[local_data.warp_in_block_idx]) {
                warp_data.Upper_bound[local_data.warp_in_block_idx] = warp_data.number_of_candidates[local_data.warp_in_block_idx];
            }

            // tighten
            if (warp_data.Lower_bound[local_data.warp_in_block_idx] < warp_data.Upper_bound[local_data.warp_in_block_idx]) {
                // tighten lower
                for (index = 0; index < warp_data.Lower_bound[local_data.warp_in_block_idx]; index++) {
                    warp_data.sum_candidate_indeg[local_data.warp_in_block_idx] += local_data.vertices[warp_data.number_of_members[local_data.warp_in_block_idx] + index].indeg;
                }

                while (index < warp_data.Upper_bound[local_data.warp_in_block_idx] && warp_data.sum_clq_indeg[local_data.warp_in_block_idx] + warp_data.sum_candidate_indeg[local_data.warp_in_block_idx]
                    < warp_data.number_of_members[local_data.warp_in_block_idx] *
                    device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + index]) {
                    warp_data.sum_candidate_indeg[local_data.warp_in_block_idx] += local_data.vertices[warp_data.number_of_members[local_data.warp_in_block_idx] + index].indeg;
                    index++;
                }

                if (warp_data.sum_clq_indeg[local_data.warp_in_block_idx] + warp_data.sum_candidate_indeg[local_data.warp_in_block_idx] < warp_data.number_of_members[local_data.warp_in_block_idx] *
                    device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + index]) {
                    warp_data.invalid_bounds[local_data.warp_in_block_idx] = true;
                }
                else {
                    warp_data.Lower_bound[local_data.warp_in_block_idx] = index;

                    warp_data.tightened_Upper_bound[local_data.warp_in_block_idx] = index;

                    while (index < warp_data.Upper_bound[local_data.warp_in_block_idx]) {
                        warp_data.sum_candidate_indeg[local_data.warp_in_block_idx] += local_data.vertices[warp_data.number_of_members[local_data.warp_in_block_idx] + index].indeg;

                        index++;

                        if (warp_data.sum_clq_indeg[local_data.warp_in_block_idx] + warp_data.sum_candidate_indeg[local_data.warp_in_block_idx] >= warp_data.number_of_members[local_data.warp_in_block_idx]
                            * device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + index]) {
                            warp_data.tightened_Upper_bound[local_data.warp_in_block_idx] = index;
                        }
                    }

                    if (warp_data.Upper_bound[local_data.warp_in_block_idx] > warp_data.tightened_Upper_bound[local_data.warp_in_block_idx]) {
                        warp_data.Upper_bound[local_data.warp_in_block_idx] = warp_data.tightened_Upper_bound[local_data.warp_in_block_idx];
                    }

                    if (warp_data.Lower_bound[local_data.warp_in_block_idx] > 1) {
                        warp_data.minimum_external_degree[local_data.warp_in_block_idx] = device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx]
                            + warp_data.Lower_bound[local_data.warp_in_block_idx], device_data);
                    }
                }
            }
        }
        else {
            warp_data.minimum_external_degree[local_data.warp_in_block_idx] = device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + 1,
                device_data);

            warp_data.Upper_bound[local_data.warp_in_block_idx] = warp_data.number_of_candidates[local_data.warp_in_block_idx];

            if (warp_data.number_of_members[local_data.warp_in_block_idx] < (*(device_data.minimum_clique_size))) {
                warp_data.Lower_bound[local_data.warp_in_block_idx] = (*(device_data.minimum_clique_size)) - warp_data.number_of_members[local_data.warp_in_block_idx];
            }
            else {
                warp_data.Lower_bound[local_data.warp_in_block_idx] = 0;
            }
        }

        if (warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Upper_bound[local_data.warp_in_block_idx] < (*(device_data.minimum_clique_size))) {
            warp_data.invalid_bounds[local_data.warp_in_block_idx] = true;
        }

        if (warp_data.Upper_bound[local_data.warp_in_block_idx] < 0 || warp_data.Upper_bound[local_data.warp_in_block_idx] <
            warp_data.Lower_bound[local_data.warp_in_block_idx]) {
            warp_data.invalid_bounds[local_data.warp_in_block_idx] = true;
        }
    }
    __syncwarp();
}


// --- HELPER KERNELS ---

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

            int result;
            sort_vert(vertex1, vertex2, result);

            if (result == 1) {
                target[j] = target[j + 1];
                target[j + 1] = vertex1;
            }
        }
        __syncwarp();
    }
}

// TODO - use return rather than result reference
__device__ void sort_vert(Vertex& vertex1, Vertex& vertex2, int& result)
{
    // order is: in clique -> covered -> critical adj vertices -> cands -> cover -> pruned

    // in clique
    if (vertex1.label == 1 && vertex2.label != 1) {
        result = -1;
        return;
    }
    else if (vertex1.label != 1 && vertex2.label == 1) {
        result = 1;
        return;

        // covered candidate vertices
    }
    else if (vertex1.label == 2 && vertex2.label != 2) {
        result = -1;
        return;
    }
    else if (vertex1.label != 2 && vertex2.label == 2) {
        result = 1;
        return;

        // critical adjacent candidate vertices
    }
    else if (vertex1.label == 4 && vertex2.label != 4) {
        result = -1;
        return;
    }
    else if (vertex1.label != 4 && vertex2.label == 4) {
        result = 1;
        return;

        // candidate vertices
    }
    else if (vertex1.label == 0 && vertex2.label != 0) {
        result = -1;
        return;
    }
    else if (vertex1.label != 0 && vertex2.label == 0) {
        result = 1;
        return;

        // the cover vertex
    }
    else if (vertex1.label == 3 && vertex2.label != 3) {
        result = -1;
        return;
    }
    else if (vertex1.label != 3 && vertex2.label == 3) {
        result = 1;
        return;

        // vertices that have been pruned
    }
    else if (vertex1.label == -1 && vertex2.label != 1) {
        result = 1;
        return;
    }
    else if (vertex1.label != -1 && vertex2.label == -1) {
        result = -1;
        return;
    }

    // for ties: in clique low -> high, cand high -> low
    else if (vertex1.label == 1 && vertex2.label == 1) {
        if (vertex1.vertexid > vertex2.vertexid) {
            result = 1;
            return;
        }
        else if (vertex1.vertexid < vertex2.vertexid) {
            result = -1;
            return;
        }
        else {
            result = 0;
            return;
        }
    }
    else if (vertex1.label == 0 && vertex2.label == 0) {
        if (vertex1.vertexid > vertex2.vertexid) {
            result = -1;
            return;
        }
        else if (vertex1.vertexid < vertex2.vertexid) {
            result = 1;
            return;
        }
        else {
            result = 0;
            return;
        }
    }
    else if (vertex1.label == 2 && vertex2.label == 2) {
        result = 0;
        return;
    }
    else if (vertex1.label == -1 && vertex2.label == -1) {
        result = 0;
        return;
    }
    result = 0;
    return;
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

__device__ bool device_vert_isextendable(Vertex& vertex, int number_of_members, GPU_Data& device_data)
{
    if (vertex.indeg + vertex.exdeg < device_data.minimum_degrees[(*(device_data.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(device_data.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(number_of_members+vertex.exdeg, device_data)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool device_cand_isvalid(Vertex& vertex, int number_of_members, GPU_Data& device_data)
{
    if (vertex.indeg + vertex.exdeg < device_data.minimum_degrees[(*(device_data.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(device_data.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(number_of_members + vertex.exdeg + 1, device_data)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool device_cand_isvalid_LU(Vertex& vertex, GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data)
{
    if (vertex.indeg + vertex.exdeg < device_data.minimum_degrees[(*(device_data.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(device_data.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + vertex.exdeg + 1, device_data)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < warp_data.minimum_external_degree[local_data.warp_in_block_idx]) {
        return false;
    }
    else if (vertex.indeg + warp_data.Upper_bound[local_data.warp_in_block_idx] - 1 < device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx], device_data)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool device_vert_isextendable_LU(Vertex& vertex, GPU_Data& device_data, Warp_Data& warp_data, Local_Data& local_data)
{
    if (vertex.indeg + vertex.exdeg < device_data.minimum_degrees[(*(device_data.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.lvl2adj < (*(device_data.minimum_clique_size)) - 1) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + vertex.exdeg, device_data)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < warp_data.minimum_external_degree[local_data.warp_in_block_idx]) {
        return false;
    }
    // TODO - I think this else if is useless
    else if (vertex.exdeg == 0 && vertex.indeg < device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + vertex.exdeg, device_data)) {
        return false;
    }
    else if (vertex.indeg + warp_data.Upper_bound[local_data.warp_in_block_idx] < device_data.minimum_degrees[warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Upper_bound[local_data.warp_in_block_idx]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < device_get_mindeg(warp_data.number_of_members[local_data.warp_in_block_idx] + warp_data.Lower_bound[local_data.warp_in_block_idx], device_data)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ int device_get_mindeg(int number_of_members, GPU_Data& device_data)
{
    if (number_of_members < (*(device_data.minimum_clique_size))) {
        return device_data.minimum_degrees[(*(device_data.minimum_clique_size))];
    }
    else {
        return device_data.minimum_degrees[number_of_members];
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