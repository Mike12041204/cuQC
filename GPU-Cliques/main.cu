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
using namespace std;

// TODO - ensure proper cudaMemset usage, cannot memset multiple uint64, unless setting to 0

// GPU: GTX 1660 Super
// SM's: 23
// Threads per SM: 1024
// Global Memory: 6 GB
// Shared Memory: 48 KB

// global memory size: 1.500.000.000 ints (75% used)
#define TASKS_SIZE 10000000
#define EXPAND_THRESHOLD 32
#define BUFFER_SIZE 50000000
#define BUFFER_OFFSET_SIZE 500000
#define CLIQUES_SIZE 50000000
#define CLIQUES_OFFSET_SIZE 500000
#define CLIQUES_PERCENT 70

// per warp
#define WCLIQUES_SIZE 500000
#define WCLIQUES_OFFSET_SIZE 5000
#define WTASKS_SIZE 500000
#define WTASKS_OFFSET_SIZE 5000

// TODO - store vertices in global memory to allow the usage of more threads per sm
// shared memory size: 12.300 ints (75% used)
#define VERTICES_SIZE 2000

#define BLOCK_SIZE 32
#define NUM_OF_BLOCKS 32
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

// GPU GRAPH
struct GPU_Graph
{
    int* number_of_vertices;
    int* number_of_edges;

    int* onehop_neighbors;
    uint64_t* onehop_offsets;
    int* twohop_neighbors;
    uint64_t* twohop_offsets;
};

// CPU DATA
struct CPU_Data
{
    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    int* tasks1_vertex;
    int* tasks1_label;
    int* tasks1_indeg;
    int* tasks1_exdeg;
    int* tasks1_lvl2adj;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    int* buffer_vertex;
    int* buffer_label;
    int* buffer_indeg;
    int* buffer_exdeg;
    int* buffer_lvl2adj;

    bool* maximal_expansion;
    bool* dumping_cliques;
};

// GPU DATA
struct GPU_Data
{
    uint64_t* current_level;

    uint64_t* tasks1_count;
    uint64_t* tasks1_offset;
    int* tasks1_vertex;
    int* tasks1_label;
    int* tasks1_indeg;
    int* tasks1_exdeg;
    int* tasks1_lvl2adj;

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    int* tasks2_vertex;
    int* tasks2_label;
    int* tasks2_indeg;
    int* tasks2_exdeg;
    int* tasks2_lvl2adj;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    int* buffer_vertex;
    int* buffer_label;
    int* buffer_indeg;
    int* buffer_exdeg;
    int* buffer_lvl2adj;

    uint64_t* wtasks_count;
    uint64_t* wtasks_offset;
    int* wtasks_vertex;
    int* wtasks_label;
    int* wtasks_indeg;
    int* wtasks_exdeg;
    int* wtasks_lvl2adj;

    int* helper_int1;
    int* helper_int2;
    int* helper_int3;
    int* helper_int4;
    int* total_tasks;

    bool* maximal_expansion;
    bool* dumping_cliques;

    int* minimum_degrees;
    int* minimum_clique_size;
};

// CPU CLIQUES
struct CPU_Cliques
{
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;
};

// GPU CLIQUES
struct GPU_Cliques
{
    uint64_t* cliques_count;
    uint64_t* cliques_offset;
    int* cliques_vertex;

    uint64_t* wcliques_count;
    uint64_t* wcliques_offset;
    int* wcliques_vertex;
};

// METHODS
void allocate_graph(GPU_Graph& device_graph, CPU_Graph& input_graph);
void calculate_minimum_degrees(CPU_Graph& graph);
void search(CPU_Graph& input_graph, ofstream& temp_results, GPU_Graph& device_graph);
void allocate_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques, GPU_Cliques& device_cliques, CPU_Graph& input_graph);
void initialize_tasks(CPU_Graph& graph, CPU_Data& host_data);
void move_to_gpu(CPU_Data& host_data, GPU_Data& device_data);
void dump_cliques(CPU_Cliques& host_cliques, GPU_Cliques& device_cliques, ofstream& output_file);
void free_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques, GPU_Cliques& device_cliques);
void RemoveNonMax(char* szset_filename, char* szoutput_filename);

int linear_search_vertices(Vertex* search_array, int array_size, int search_vertexid);
int linear_search_array(int* search_array, int array_size, int search_number);
int sort_vertices(const void* a, const void* b);
inline void chkerr(cudaError_t code);
void print_CPU_Data(CPU_Data& host_data);
void print_GPU_Data(GPU_Data& device_data);

// KERNELS
__global__ void expand_level(GPU_Data device_data, GPU_Cliques device_cliques, GPU_Graph graph);
__global__ void write_level(GPU_Data device_data, GPU_Cliques device_cliques);
__device__ void remove_one_vertex(int& num_cand, int& tot_vert, int* read_vertices, uint64_t& start, GPU_Graph& graph, int& lane_idx, int* read_exdeg, int* read_lvl2adj);
__device__ void add_one_vertex(int& lane_idx, Vertex* vertices, int& total_vertices, int& number_of_members, int& number_of_candidates, GPU_Graph& graph);
__device__ void check_for_clique(int& number_of_members, int& lane_idx, int& warp_idx, Vertex* vertices, int& wcliques_write, int& wcliques_offset_write,
    GPU_Cliques& device_cliques, GPU_Data& device_data);
__device__ void write_to_tasks(GPU_Data& device_data, int& wtasks_write, int& warp_idx, int& lane_idx, int& total_vertices, Vertex* vertices, int& wtasks_offset_write);
__device__ void transfer_buffers(GPU_Data& device_data, int& warp_idx, int& wtasks_offset_write, GPU_Cliques& device_cliques, int& wcliques_offset_write, int& lane_idx,
    uint64_t* write_count, int& tasks_offset_write, uint64_t& tasks_write, int& cliques_offset_write, uint64_t& cliques_write, int& warp_count,
    uint64_t* write_offsets, uint64_t& buffer_offset_start, uint64_t& buffer_start, int& wtasks_write, int* write_vertices, int* write_labels,
    int* write_indeg, int* write_exdeg, int* write_lvl2adj, uint64_t& cliques_offset_start, uint64_t& cliques_start, int& wcliques_write);
__device__ void fill_from_buffer(GPU_Data& device_data, uint64_t* write_count, uint64_t* write_offsets, int* write_vertices, int* write_labels, int* write_indeg,
    int* write_exdeg, int* write_lvl2adj, int& idx, int& thread_count);

__device__ void device_sort(Vertex* target, int& size, int& lane_idx);
__device__ void sort_vert(Vertex& vertex1, Vertex& vertex2, int& result);
__device__ void device_search_vertices(Vertex* search_array, int& array_size, int& search_vertexid, int& lane_idx, int& result);
__device__ void device_search_array(int* search_array, int& array_size, int& search_number, int& lane_idx, int& result);

// INPUT SETTINGS
double minimum_degree_ratio;
int minimum_clique_size;
int* minimum_degrees;

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

    // GRAPH / MINDEGS
    CPU_Graph input_graph(graph_stream);
    graph_stream.close();
    // TODO - free graph memory
    GPU_Graph device_graph;
    allocate_graph(device_graph, input_graph);
    calculate_minimum_degrees(input_graph);
    ofstream temp_results("temp.txt");

    // SEARCH
    search(input_graph, temp_results, device_graph);

    // RM NON-MAX
    RemoveNonMax("misc/temp.txt", argv[4]);

    cout << ">:PROGRAM END" << endl;
    return 0;
}



// --- HOST METHODS --- 

void allocate_graph(GPU_Graph& device_graph, CPU_Graph& input_graph)
{
    chkerr(cudaMalloc((void**)&device_graph.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_graph.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_graph.onehop_neighbors, sizeof(int) * input_graph.number_of_onehop_neighbors));
    chkerr(cudaMalloc((void**)&device_graph.onehop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&device_graph.twohop_neighbors, sizeof(int) * input_graph.number_of_onehop_neighbors));
    chkerr(cudaMalloc((void**)&device_graph.twohop_offsets, sizeof(uint64_t) * (input_graph.number_of_vertices + 1)));

    chkerr(cudaMemcpy(device_graph.number_of_vertices, &(input_graph.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_graph.number_of_edges, &(input_graph.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_graph.onehop_neighbors, &(input_graph.onehop_neighbors), sizeof(int) * input_graph.number_of_onehop_neighbors, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_graph.onehop_offsets, &(input_graph.onehop_offsets), sizeof(int) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_graph.twohop_neighbors, &(input_graph.twohop_neighbors), sizeof(int) * input_graph.number_of_onehop_neighbors, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_graph.twohop_offsets, &(input_graph.twohop_offsets), sizeof(int) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
}

// initializes minimum degrees array
void calculate_minimum_degrees(CPU_Graph& graph)
{
    minimum_degrees = new int[graph.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 0; i <= graph.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}

void search(CPU_Graph& input_graph, ofstream& temp_results, GPU_Graph& device_graph)
{
    // DATA STRUCTURES
    CPU_Data host_data;
    GPU_Data device_data;
    CPU_Cliques host_cliques;
    GPU_Cliques device_cliques;

    // HANDLE MEMORY
    allocate_memory(host_data, device_data, host_cliques, device_cliques, input_graph);

    // INITIALIZE TASKS
    initialize_tasks(input_graph, host_data);

    // TRANSFER TO GPU
    move_to_gpu(host_data, device_data);

    // DEBUG - PRINT
    print_CPU_Data(host_data);
    print_GPU_Data(device_data);

    // EXPAND LEVEL
    while (!(*host_data.maximal_expansion))
    {
        expand_level<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(device_data, device_cliques, device_graph);

        cudaDeviceSynchronize();

        write_level<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(device_data, device_cliques);

        cudaDeviceSynchronize();

        chkerr(cudaMemcpy(host_data.maximal_expansion, device_data.maximal_expansion, sizeof(bool), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(host_data.dumping_cliques, device_data.dumping_cliques, sizeof(bool), cudaMemcpyDeviceToHost));

        if (*host_data.dumping_cliques) {
            dump_cliques(host_cliques, device_cliques, temp_results);
        }

        // DEBUG
        print_GPU_Data(device_data);
    }
    dump_cliques(host_cliques, device_cliques, temp_results);

    // FREE MEMORY
    free_memory(host_data, device_data, host_cliques, device_cliques);
}

// allocates mnemory for the data sturtcures on the host and device
void allocate_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques, GPU_Cliques& device_cliques, CPU_Graph& input_graph)
{
    int number_of_warps = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;

    // CPU DATA
    host_data.tasks1_count = new uint64_t;
    host_data.tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    host_data.tasks1_vertex = new int[TASKS_SIZE];
    host_data.tasks1_label = new int[TASKS_SIZE];
    host_data.tasks1_indeg = new int[TASKS_SIZE];
    host_data.tasks1_exdeg = new int[TASKS_SIZE];
    host_data.tasks1_lvl2adj = new int[TASKS_SIZE];

    host_data.tasks1_offset[0] = 0;
    (*(host_data.tasks1_count)) = 0;

    host_data.buffer_count = new uint64_t;
    host_data.buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    host_data.buffer_vertex = new int[BUFFER_SIZE];
    host_data.buffer_label = new int[BUFFER_SIZE];
    host_data.buffer_indeg = new int[BUFFER_SIZE];
    host_data.buffer_exdeg = new int[BUFFER_SIZE];
    host_data.buffer_lvl2adj = new int[BUFFER_SIZE];

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
    chkerr(cudaMalloc((void**)&device_data.tasks1_vertex, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks1_label, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks1_indeg, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks1_exdeg, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks1_lvl2adj, sizeof(int) * TASKS_SIZE));

    chkerr(cudaMemset(device_data.tasks1_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.tasks1_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.tasks2_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.tasks2_offset, sizeof(uint64_t) * (EXPAND_THRESHOLD + 1)));
    chkerr(cudaMalloc((void**)&device_data.tasks2_vertex, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks2_label, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks2_indeg, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks2_exdeg, sizeof(int) * TASKS_SIZE));
    chkerr(cudaMalloc((void**)&device_data.tasks2_lvl2adj, sizeof(int) * TASKS_SIZE));

    chkerr(cudaMemset(device_data.tasks2_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.tasks2_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.buffer_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_data.buffer_offset, sizeof(uint64_t) * BUFFER_OFFSET_SIZE));
    chkerr(cudaMalloc((void**)&device_data.buffer_vertex, sizeof(int) * BUFFER_SIZE));
    chkerr(cudaMalloc((void**)&device_data.buffer_label, sizeof(int) * BUFFER_SIZE));
    chkerr(cudaMalloc((void**)&device_data.buffer_indeg, sizeof(int) * BUFFER_SIZE));
    chkerr(cudaMalloc((void**)&device_data.buffer_exdeg, sizeof(int) * BUFFER_SIZE));
    chkerr(cudaMalloc((void**)&device_data.buffer_lvl2adj, sizeof(int) * BUFFER_SIZE));

    chkerr(cudaMemset(device_data.buffer_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.buffer_count, 0, sizeof(uint64_t)));

    chkerr(cudaMalloc((void**)&device_data.maximal_expansion, sizeof(bool)));
    chkerr(cudaMalloc((void**)&device_data.dumping_cliques, sizeof(bool)));

    chkerr(cudaMemset(device_data.maximal_expansion, false, sizeof(bool)));
    chkerr(cudaMemset(device_data.dumping_cliques, false, sizeof(bool)));

    // CPU CLIQUES
    host_cliques.cliques_count = new uint64_t;
    host_cliques.cliques_vertex = new int[CLIQUES_SIZE];
    host_cliques.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];

    host_cliques.cliques_offset[0] = 0;
    (*(host_cliques.cliques_count)) = 0;

    // GPU CLIQUES
    chkerr(cudaMalloc((void**)&device_cliques.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&device_cliques.cliques_vertex, sizeof(int) * CLIQUES_SIZE));
    chkerr(cudaMalloc((void**)&device_cliques.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE));

    chkerr(cudaMemset(device_cliques.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_cliques.cliques_count, 0, sizeof(uint64_t)));

    // WARP BUFFER TASKS
    chkerr(cudaMalloc((void**)&device_data.wtasks_count, sizeof(uint64_t) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_offset, (sizeof(uint64_t) * WTASKS_OFFSET_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_vertex, (sizeof(int) * WTASKS_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_label, (sizeof(int) * WTASKS_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_indeg, (sizeof(int) * WTASKS_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_exdeg, (sizeof(int) * WTASKS_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_data.wtasks_lvl2adj, (sizeof(int) * WTASKS_SIZE) * number_of_warps));

    chkerr(cudaMemset(device_data.wtasks_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_data.wtasks_count, 0, sizeof(uint64_t)));

    // WARP BUFFER CLIQUES
    chkerr(cudaMalloc((void**)&device_cliques.wcliques_count, sizeof(uint64_t) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_cliques.wcliques_vertex, (sizeof(int) * WCLIQUES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&device_cliques.wcliques_offset, (sizeof(int) * WCLIQUES_OFFSET_SIZE) * number_of_warps));

    chkerr(cudaMemset(device_cliques.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(device_cliques.cliques_count, 0, sizeof(uint64_t)));

    // HELPER VARIABLES
    chkerr(cudaMalloc((void**)&device_data.helper_int1, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_data.helper_int2, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_data.helper_int3, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_data.helper_int4, sizeof(int)));
    chkerr(cudaMalloc((void**)&device_data.total_tasks, sizeof(int)));

    // INPUT DATA
    chkerr(cudaMalloc((void**)&device_data.minimum_degrees, sizeof(int) * (input_graph.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&device_data.minimum_clique_size, sizeof(int)));

    chkerr(cudaMemcpy(device_data.minimum_degrees, minimum_degrees, sizeof(int) * (input_graph.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(device_data.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));
}

// TODO - ensure searches are only conducted over relevant sections of memeory (only candidates rather than all vertices)
void initialize_tasks(CPU_Graph& graph, CPU_Data& host_data)
{
    // variables for pruning techniques
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    int phelper2;
    int pneighbors_count;

    // initialize vertices array and size
    int total_vertices = graph.number_of_vertices;
    int expansions = total_vertices;
    Vertex* old_vertices = new Vertex[total_vertices];
    if (old_vertices == nullptr) {
        cout << "ERROR: bad malloc" << endl;
    }
    for (int i = 0; i < total_vertices; i++) {
        old_vertices[i].vertexid = i;
        old_vertices[i].label = 0;
        old_vertices[i].indeg = 0;
        old_vertices[i].exdeg = graph.onehop_offsets[i + 1] - graph.onehop_offsets[i];
        old_vertices[i].lvl2adj = graph.twohop_offsets[i + 1] - graph.twohop_offsets[i];
    }

    // NEXT LEVEL
    for (int i = 0; i < expansions; i++)
    {
        // REMOVE CANDIDATE
        if (i > 0) {
            total_vertices--;

            // update exdeg of vertices connected to removed cand
            pvertexid = old_vertices[total_vertices].vertexid;
            pneighbors_start = graph.onehop_offsets[pvertexid];
            pneighbors_end = graph.onehop_offsets[pvertexid + 1];
            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++) {
                phelper1 = graph.onehop_neighbors[j];
                phelper2 = linear_search_vertices(old_vertices, total_vertices, phelper1);
                if (phelper2 != -1) {
                    old_vertices[phelper2].exdeg--;
                }
            }

            // update lvl2adj
            pneighbors_start = graph.twohop_offsets[pvertexid];
            pneighbors_end = graph.twohop_offsets[pvertexid + 1];
            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++) {
                phelper1 = graph.twohop_neighbors[j];
                phelper2 = linear_search_vertices(old_vertices, total_vertices, phelper1);
                if (phelper2 != -1) {
                    old_vertices[phelper2].lvl2adj--;
                }
            }
        }

        // NEW VERTICES
        Vertex* new_vertices = new Vertex[total_vertices];
        if (new_vertices == nullptr) {
            cout << "ERROR: bad malloc" << endl;
        }
        int total_new_vertices = total_vertices;
        for (int j = 0; j < total_new_vertices; j++) {
            new_vertices[j].vertexid = old_vertices[j].vertexid;
            new_vertices[j].label = old_vertices[j].label;
            new_vertices[j].indeg = old_vertices[j].indeg;
            new_vertices[j].exdeg = old_vertices[j].exdeg;
            new_vertices[j].lvl2adj = old_vertices[j].lvl2adj;
        }

        // TODO - choose between iterating through all vertices or neighbors to update adj when pruning
        // ADD ONE VERTEX
        new_vertices[total_new_vertices - 1].label = 1;

        // update the exdeg and indeg of all vertices adj to the vertex just added to the vertex set
        pvertexid = new_vertices[total_new_vertices - 1].vertexid;
        pneighbors_start = graph.onehop_offsets[pvertexid];
        pneighbors_end = graph.onehop_offsets[pvertexid + 1];
        for (uint64_t j = pneighbors_start; j < pneighbors_end; j++) {
            phelper1 = graph.onehop_neighbors[j];
            phelper2 = linear_search_vertices(new_vertices, total_new_vertices, phelper1);
            if (phelper2 != -1) {
                new_vertices[phelper2].exdeg--;
                new_vertices[phelper2].indeg++;
            }
        }

        // sort new vertices putting just added vertex at end of all vertices in x
        qsort(new_vertices, total_new_vertices, sizeof(Vertex), sort_vertices);

        // DIAMETER PRUNING
        int number_of_removed_candidates = 0;
        pneighbors_start = graph.twohop_offsets[pvertexid];
        pneighbors_end = graph.twohop_offsets[pvertexid + 1];
        pneighbors_count = pneighbors_end - pneighbors_start;
        for (int j = 1; j < total_new_vertices; j++) {
            phelper1 = new_vertices[j].vertexid;
            phelper2 = linear_search_array(graph.twohop_neighbors + pneighbors_start, pneighbors_count, phelper1);
            if (phelper2 == -1) {
                new_vertices[j].label = -1;
                number_of_removed_candidates++;
            }
        }
        qsort(new_vertices, total_new_vertices, sizeof(Vertex), sort_vertices);

        // update exdeg of vertices connected to removed cands
        for (int i = total_new_vertices - number_of_removed_candidates; i < total_new_vertices; i++) {
            pvertexid = new_vertices[i].vertexid;
            pneighbors_start = graph.onehop_offsets[pvertexid];
            pneighbors_end = graph.onehop_offsets[pvertexid + 1];
            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++) {
                phelper1 = graph.onehop_neighbors[j];
                phelper2 = linear_search_vertices(new_vertices, total_new_vertices, phelper1);
                if (phelper2 != -1) {
                    new_vertices[phelper2].exdeg--;
                }
            }

            // update lvl2adj
            pneighbors_start = graph.twohop_offsets[pvertexid];
            pneighbors_end = graph.twohop_offsets[pvertexid + 1];
            for (uint64_t j = pneighbors_start; j < pneighbors_end; j++) {
                phelper1 = graph.twohop_neighbors[j];
                phelper2 = linear_search_vertices(new_vertices, total_new_vertices, phelper1);
                if (phelper2 != -1) {
                    new_vertices[phelper2].lvl2adj--;
                }
            }
        }
        total_new_vertices -= number_of_removed_candidates;

        // WRITE TO TASKS
        if (total_new_vertices - 1 > 0) {

            if ((*(host_data.tasks1_count)) < EXPAND_THRESHOLD) {
                uint64_t start_write = host_data.tasks1_offset[(*(host_data.tasks1_count))];

                for (int j = 0; j < total_new_vertices; j++) {
                    host_data.tasks1_vertex[start_write + j] = new_vertices[j].vertexid;
                    host_data.tasks1_label[start_write + j] = new_vertices[j].label;
                    host_data.tasks1_indeg[start_write + j] = new_vertices[j].indeg;
                    host_data.tasks1_exdeg[start_write + j] = new_vertices[j].exdeg;
                    host_data.tasks1_lvl2adj[start_write + j] = new_vertices[j].lvl2adj;
                }
                (*(host_data.tasks1_count))++;
                host_data.tasks1_offset[(*(host_data.tasks1_count))] = start_write + total_new_vertices;
            }
            else {
                uint64_t start_write = host_data.buffer_offset[(*(host_data.buffer_count))];

                for (int j = 0; j < total_new_vertices; j++) {
                    host_data.buffer_vertex[start_write + j] = new_vertices[j].vertexid;
                    host_data.buffer_label[start_write + j] = new_vertices[j].label;
                    host_data.buffer_indeg[start_write + j] = new_vertices[j].indeg;
                    host_data.buffer_exdeg[start_write + j] = new_vertices[j].exdeg;
                    host_data.buffer_lvl2adj[start_write + j] = new_vertices[j].lvl2adj;
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
    cudaMemcpy(device_data.tasks1_vertex, host_data.tasks1_vertex, (TASKS_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.tasks1_label, host_data.tasks1_label, (TASKS_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.tasks1_indeg, host_data.tasks1_indeg, (TASKS_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.tasks1_exdeg, host_data.tasks1_exdeg, (TASKS_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.tasks1_lvl2adj, host_data.tasks1_lvl2adj, (TASKS_SIZE) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(device_data.buffer_count, host_data.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_offset, host_data.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_vertex, host_data.buffer_vertex, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_label, host_data.buffer_label, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_indeg, host_data.buffer_indeg, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_exdeg, host_data.buffer_exdeg, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data.buffer_lvl2adj, host_data.buffer_lvl2adj, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
}

void dump_cliques(CPU_Cliques& host_cliques, GPU_Cliques& device_cliques, ofstream& temp_results)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(host_cliques.cliques_count, device_cliques.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(host_cliques.cliques_offset, device_cliques.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(host_cliques.cliques_vertex, device_cliques.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));

    for (int i = 0; i < (*host_cliques.cliques_count); i++) {
        uint64_t start = host_cliques.cliques_offset[i];
        uint64_t end = host_cliques.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << host_cliques.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    (*host_cliques.cliques_count) = 0;
    cudaMemset(device_cliques.cliques_count, 0, sizeof(uint64_t));
}

void free_memory(CPU_Data& host_data, GPU_Data& device_data, CPU_Cliques& host_cliques, GPU_Cliques& device_cliques)
{
    // CPU DATA
    delete host_data.tasks1_count;
    delete host_data.tasks1_offset;
    delete host_data.tasks1_vertex;
    delete host_data.tasks1_label;
    delete host_data.tasks1_indeg;
    delete host_data.tasks1_exdeg;
    delete host_data.tasks1_lvl2adj;

    delete host_data.buffer_count;
    delete host_data.buffer_offset;
    delete host_data.buffer_vertex;
    delete host_data.buffer_label;
    delete host_data.buffer_indeg;
    delete host_data.buffer_exdeg;
    delete host_data.buffer_lvl2adj;

    delete host_data.maximal_expansion;
    delete host_data.dumping_cliques;

    // GPU DATA
    chkerr(cudaFree(device_data.current_level));

    chkerr(cudaFree(device_data.tasks1_count));
    chkerr(cudaFree(device_data.tasks1_offset));
    chkerr(cudaFree(device_data.tasks1_vertex));
    chkerr(cudaFree(device_data.tasks1_label));
    chkerr(cudaFree(device_data.tasks1_indeg));
    chkerr(cudaFree(device_data.tasks1_exdeg));
    chkerr(cudaFree(device_data.tasks1_lvl2adj));

    chkerr(cudaFree(device_data.tasks2_count));
    chkerr(cudaFree(device_data.tasks2_offset));
    chkerr(cudaFree(device_data.tasks2_vertex));
    chkerr(cudaFree(device_data.tasks2_label));
    chkerr(cudaFree(device_data.tasks2_indeg));
    chkerr(cudaFree(device_data.tasks2_exdeg));
    chkerr(cudaFree(device_data.tasks2_lvl2adj));

    chkerr(cudaFree(device_data.buffer_count));
    chkerr(cudaFree(device_data.buffer_offset));
    chkerr(cudaFree(device_data.buffer_vertex));
    chkerr(cudaFree(device_data.buffer_label));
    chkerr(cudaFree(device_data.buffer_indeg));
    chkerr(cudaFree(device_data.buffer_exdeg));
    chkerr(cudaFree(device_data.buffer_lvl2adj));

    chkerr(cudaFree(device_data.maximal_expansion));
    chkerr(cudaFree(device_data.dumping_cliques));

    // CPU CLIQUES

    delete host_cliques.cliques_count;
    delete host_cliques.cliques_vertex;
    delete host_cliques.cliques_offset;

    // GPU CLIQUES
    chkerr(cudaFree(device_cliques.cliques_count));
    chkerr(cudaFree(device_cliques.cliques_vertex));
    chkerr(cudaFree(device_cliques.cliques_offset));

    // WARP BUFFER TASKS
    chkerr(cudaFree(device_data.wtasks_count));
    chkerr(cudaFree(device_data.wtasks_offset));
    chkerr(cudaFree(device_data.wtasks_vertex));
    chkerr(cudaFree(device_data.wtasks_label));
    chkerr(cudaFree(device_data.wtasks_indeg));
    chkerr(cudaFree(device_data.wtasks_exdeg));
    chkerr(cudaFree(device_data.wtasks_lvl2adj));

    // WARP BUFFER CLIQUES
    chkerr(cudaFree(device_cliques.wcliques_count));
    chkerr(cudaFree(device_cliques.wcliques_vertex));
    chkerr(cudaFree(device_cliques.wcliques_offset));

    // HELPER VARIABLES
    chkerr(cudaFree(device_data.helper_int1));
    chkerr(cudaFree(device_data.helper_int2));
    chkerr(cudaFree(device_data.helper_int3));
    chkerr(cudaFree(device_data.helper_int4));
    chkerr(cudaFree(device_data.total_tasks));

    // INPUT DATA
    chkerr(cudaFree(device_data.minimum_degrees));
    chkerr(cudaFree(device_data.minimum_clique_size));
}



// --- HELPER METHODS ---

void print_GPU_Data(GPU_Data& device_data)
{
    uint64_t* current_level = new uint64_t;

    uint64_t* tasks1_count = new uint64_t;
    uint64_t* tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    int* tasks1_vertex = new int[TASKS_SIZE];
    int* tasks1_label = new int[TASKS_SIZE];
    int* tasks1_indeg = new int[TASKS_SIZE];
    int* tasks1_exdeg = new int[TASKS_SIZE];
    int* tasks1_lvl2adj = new int[TASKS_SIZE];

    uint64_t* tasks2_count = new uint64_t;
    uint64_t* tasks2_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    int* tasks2_vertex = new int[TASKS_SIZE];
    int* tasks2_label = new int[TASKS_SIZE];
    int* tasks2_indeg = new int[TASKS_SIZE];
    int* tasks2_exdeg = new int[TASKS_SIZE];
    int* tasks2_lvl2adj = new int[TASKS_SIZE];

    uint64_t* buffer_count = new uint64_t;
    uint64_t* buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    int* buffer_vertex = new int[BUFFER_SIZE];
    int* buffer_label = new int[BUFFER_SIZE];
    int* buffer_indeg = new int[BUFFER_SIZE];
    int* buffer_exdeg = new int[BUFFER_SIZE];
    int* buffer_lvl2adj = new int[BUFFER_SIZE];

    chkerr(cudaMemcpy(current_level, device_data.current_level, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks1_count, device_data.tasks1_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_offset, device_data.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_vertex, device_data.tasks1_vertex, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_label, device_data.tasks1_label, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_indeg, device_data.tasks1_indeg, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_exdeg, device_data.tasks1_exdeg, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks1_lvl2adj, device_data.tasks1_lvl2adj, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(tasks2_count, device_data.tasks2_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_offset, device_data.tasks2_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_vertex, device_data.tasks2_vertex, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_label, device_data.tasks2_label, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_indeg, device_data.tasks2_indeg, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_exdeg, device_data.tasks2_exdeg, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(tasks2_lvl2adj, device_data.tasks2_lvl2adj, (TASKS_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));

    chkerr(cudaMemcpy(buffer_count, device_data.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_offset, device_data.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_vertex, device_data.buffer_vertex, (BUFFER_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_label, device_data.buffer_label, (BUFFER_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_indeg, device_data.buffer_indeg, (BUFFER_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_exdeg, device_data.buffer_exdeg, (BUFFER_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(buffer_lvl2adj, device_data.buffer_lvl2adj, (BUFFER_SIZE) * sizeof(int), cudaMemcpyDeviceToHost));

    cout << " --- (GPU_Data)device_data details --- " << endl;
    cout << endl << "Tasks1: Level: " << (*current_level) << " Size: " << (*tasks1_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i < (*tasks1_count); i++) {
        cout << tasks1_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_vertex[i] << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_label[i] << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_indeg[i] << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_exdeg[i] << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < tasks1_offset[*tasks1_count]; i++) {
        cout << tasks1_lvl2adj[i] << " " << flush;
    }
    cout << endl;

    cout << endl << "Tasks2: " << "Size: " << (*tasks2_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i < (*tasks2_count); i++) {
        cout << tasks2_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_vertex[i] << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_label[i] << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_indeg[i] << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_exdeg[i] << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < tasks2_offset[*tasks2_count]; i++) {
        cout << tasks2_lvl2adj[i] << " " << flush;
    }
    cout << endl << endl;

    cout << endl << "Buffer: " << "Size: " << (*buffer_count) << endl;
    cout << endl << "Offsets:" << endl;
    for (int i = 0; i < (*buffer_count); i++) {
        cout << buffer_offset[i] << " " << flush;
    }
    cout << endl << "Vertex:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_vertex[i] << " " << flush;
    }
    cout << endl << "Label:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_label[i] << " " << flush;
    }
    cout << endl << "Indeg:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_indeg[i] << " " << flush;
    }
    cout << endl << "Exdeg:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_exdeg[i] << " " << flush;
    }
    cout << endl << "Lvl2adj:" << endl;
    for (int i = 0; i < buffer_offset[*buffer_count]; i++) {
        cout << buffer_lvl2adj[i] << " " << flush;
    }
    cout << endl;
}

// TODO - convert to binary search as adj lists are sorted
// searches an Vertex array for a vertex of a certain label, returns the position in the array that item was found, or -1 if not found
int linear_search_vertices(Vertex* search_array, int array_size, int search_vertexid)
{
    // ALGO - linear
    // TYPE - serial
    // SPEED - O(n)

    for (int i = 0; i < array_size; i++) {
        if (search_array[i].vertexid == search_vertexid) {
            return i;
        }
    }
    return -1;
}

// TODO - convert to binary search
// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
int linear_search_array(int* search_array, int array_size, int search_number)
{
    // ALGO - linear
    // TYPE - serial
    // SPEED - O(n)

    for (int i = 0; i < array_size; i++) {
        if (search_array[i] == search_number) {
            return i;
        }
    }
    return -1;
}

void print_CPU_Data(CPU_Data& host_data)
{
    cout << endl << " --- (CPU_Data)host_data details --- " << endl;
    cout << endl << "Tasks1: " << "Size: " << (*(host_data.tasks1_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i < (*(host_data.tasks1_count)); i++) {
        cout << host_data.tasks1_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_vertex[i] << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_label[i] << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_indeg[i] << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_exdeg[i] << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < host_data.tasks1_offset[(*(host_data.tasks1_count))]; i++) {
        cout << host_data.tasks1_lvl2adj[i] << " ";
    }

    cout << endl << endl << "Buffer: " << "Size: " << (*(host_data.buffer_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i < (*(host_data.buffer_count)); i++) {
        cout << host_data.buffer_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_vertex[i] << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_label[i] << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_indeg[i] << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_exdeg[i] << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < host_data.buffer_offset[(*(host_data.buffer_count))]; i++) {
        cout << host_data.buffer_lvl2adj[i] << " ";
    }
    cout << endl << endl;
}

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}

// TODO - implement -> CURSOR <-
void print_CPU_Graph()
{

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



// --- DEVICE KERNELS ---

// TODO - ensure searches are only conducted over relevant sections of memeory (only candidates rather than all vertices)
__global__ void expand_level(GPU_Data device_data, GPU_Cliques device_cliques, GPU_Graph graph)
{
    // THREAD INFO
    int warp_count = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = idx / 32;
    int lane_idx = idx % 32;

    // warp buffer write starts
    int wtasks_write = WTASKS_SIZE * warp_idx;
    int wtasks_offset_write = WTASKS_OFFSET_SIZE * warp_idx;
    int wcliques_write = WCLIQUES_SIZE * warp_idx;
    int wcliques_offset_write = WCLIQUES_OFFSET_SIZE * warp_idx;

    // INITIATE VARIABLES
    int* read_vertices, * read_labels, * read_indeg, * read_exdeg, * read_lvl2adj;
    uint64_t* read_offsets, * read_count;
    if ((*(device_data.current_level)) % 2 == 1) {
        read_offsets = device_data.tasks1_offset;
        read_vertices = device_data.tasks1_vertex;
        read_labels = device_data.tasks1_label;
        read_indeg = device_data.tasks1_indeg;
        read_exdeg = device_data.tasks1_exdeg;
        read_lvl2adj = device_data.tasks1_lvl2adj;
        read_count = device_data.tasks1_count;
    }
    else {
        read_offsets = device_data.tasks2_offset;
        read_vertices = device_data.tasks2_vertex;
        read_labels = device_data.tasks2_label;
        read_indeg = device_data.tasks2_indeg;
        read_exdeg = device_data.tasks2_exdeg;
        read_lvl2adj = device_data.tasks2_lvl2adj;
        read_count = device_data.tasks2_count;
    }

    // vertices array is in shared memory
    __shared__ Vertex vertices[VERTICES_SIZE];

    // --- CURRENT LEVEL ---
    for (int i = warp_idx; i < *read_count; i += warp_count)
    {
        // TODO - can cut counting members short if it is ensured that theyre at the start of vertices
        // get information of vertices being handled within tasks
        uint64_t start = read_offsets[i];
        uint64_t end = read_offsets[i + 1];
        int tot_vert = end - start;
        int num_mem = 0;
        for (uint64_t j = start; j < end; j++) {
            if (read_labels[j] == 1) {
                num_mem++;
            }
        }
        int num_cand = tot_vert - num_mem;
        int expansions = num_cand;

        // --- NEXT LEVEL ---
        for (int j = 0; j < expansions; j++)
        {
            // REMOVE ONE VERTEX
            if (j > 0) {
                remove_one_vertex(num_cand, tot_vert, read_vertices, start, graph, lane_idx, read_exdeg, read_lvl2adj);

                // break if not enough vertices as only less will be added in the next iteration
                if (tot_vert < (*device_data.minimum_clique_size)) {
                    break;
                }
            }
            __syncwarp();

            // INITIALIZE NEW VERTICES
            int number_of_members = num_mem;
            int number_of_candidates = num_cand;
            int total_vertices = tot_vert;
            for (int k = lane_idx; k < total_vertices; k += WARP_SIZE) {
                vertices[k].vertexid = read_vertices[start + k];
                vertices[k].label = read_labels[start + k];
                vertices[k].indeg = read_indeg[start + k];
                vertices[k].exdeg = read_exdeg[start + k];
                vertices[k].lvl2adj = read_lvl2adj[start + k];
            }

            // ADD ONE VERTEX
            add_one_vertex(lane_idx, vertices, total_vertices, number_of_members, number_of_candidates, graph);

            // continue if not enough vertices after pruning
            if (total_vertices < (*device_data.minimum_clique_size)) {
                continue;
            }

            // HANDLE CLIQUES
            if (number_of_members >= (*device_data.minimum_clique_size)) {
                check_for_clique(number_of_members, lane_idx, warp_idx, vertices, wcliques_write, wcliques_offset_write, device_cliques, device_data);
            }

            // WRITE TASKS TO BUFFERS
            if (number_of_candidates > 0) {
                write_to_tasks(device_data, wtasks_write, warp_idx, lane_idx, total_vertices, vertices, wtasks_offset_write);
            }
        }
    }
}

__global__ void write_level(GPU_Data device_data, GPU_Cliques device_cliques)
{
    // THREAD INFO
    int thread_count = NUM_OF_BLOCKS * BLOCK_SIZE;
    int warp_count = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = idx / 32;
    int lane_idx = idx % 32;

    // warp buffer write starts
    int wtasks_write = WTASKS_SIZE * warp_idx;
    int wtasks_offset_write = WTASKS_OFFSET_SIZE * warp_idx;
    int wcliques_write = WCLIQUES_SIZE * warp_idx;
    int wcliques_offset_write = WCLIQUES_OFFSET_SIZE * warp_idx;

    // memory write starts
    uint64_t tasks_write;
    int tasks_offset_write;
    uint64_t cliques_write;
    int cliques_offset_write;

    // start of empty memory for data structures
    uint64_t buffer_offset_start = (*device_data.buffer_count) + 1;
    uint64_t buffer_start = device_data.buffer_offset[(*device_data.buffer_count)];
    uint64_t cliques_offset_start = (*device_cliques.cliques_count) + 1;
    uint64_t cliques_start = device_cliques.cliques_offset[(*device_cliques.cliques_count)];

    // first thread only
    if (idx == 0) {
        (*device_data.maximal_expansion) = true;
        (*device_data.dumping_cliques) = true;
    }

    // INITIATE VARIABLES
    int* write_vertices, * write_labels, * write_indeg, * write_exdeg, * write_lvl2adj;
    uint64_t* write_offsets, * write_count;
    if ((*(device_data.current_level)) % 2 == 1) {
        write_offsets = device_data.tasks2_offset;
        write_vertices = device_data.tasks2_vertex;
        write_labels = device_data.tasks2_label;
        write_indeg = device_data.tasks2_indeg;
        write_exdeg = device_data.tasks2_exdeg;
        write_lvl2adj = device_data.tasks2_lvl2adj;
        write_count = device_data.tasks2_count;
    }
    else {
        write_offsets = device_data.tasks1_offset;
        write_vertices = device_data.tasks1_vertex;
        write_labels = device_data.tasks1_label;
        write_indeg = device_data.tasks1_indeg;
        write_exdeg = device_data.tasks1_exdeg;
        write_lvl2adj = device_data.tasks1_lvl2adj;
        write_count = device_data.tasks1_count;
    }
    (*write_count) = 0;
    write_offsets[0] = 0;

    // TRANSFER BUFFERS TO DATA
    transfer_buffers(device_data, warp_idx, wtasks_offset_write, device_cliques, wcliques_offset_write, lane_idx, write_count, tasks_offset_write, tasks_write,
        cliques_offset_write, cliques_write, warp_count, write_offsets, buffer_offset_start, buffer_start, wtasks_write, write_vertices, write_labels,
        write_indeg, write_exdeg, write_lvl2adj, cliques_offset_start, cliques_start, wcliques_write);

    // FILL TASKS FROM BUFFER
    if ((*write_count) < EXPAND_THRESHOLD && (*(device_data.buffer_count)) > 0)
    {
        fill_from_buffer(device_data, write_count, write_offsets, write_vertices, write_labels, write_indeg, write_exdeg, write_lvl2adj, idx, thread_count);
    }

    // HANDLE CLIQUES
    // only first thread for each warp
    if (lane_idx == 0 && cliques_write > (CLIQUES_SIZE * (CLIQUES_PERCENT / 100))) {
        atomicExch((int*)device_data.dumping_cliques, true);
    }

    // only the first thread
    if (idx == 0) {
        (*device_data.current_level)++;
    }
}

__device__ void remove_one_vertex(int& num_cand, int& tot_vert, int* read_vertices, uint64_t& start, GPU_Graph& graph, int& lane_idx, int* read_exdeg, int* read_lvl2adj)
{
    // pruning helper variables
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    int phelper2;

    num_cand--;
    tot_vert--;

    // update exdeg of vertices connected to removed cand
    pvertexid = read_vertices[start + tot_vert];
    pneighbors_start = graph.onehop_offsets[pvertexid];
    pneighbors_end = graph.onehop_offsets[pvertexid + 1];
    for (uint64_t k = pneighbors_start + lane_idx; k < pneighbors_end; k += WARP_SIZE) {
        phelper1 = graph.onehop_neighbors[k];
        device_search_array(read_vertices + start, tot_vert, phelper1, lane_idx, phelper2);
        if (phelper2 != -1) {
            read_exdeg[phelper2]--;
        }
    }

    // update lvl2adj
    pneighbors_start = graph.twohop_offsets[pvertexid];
    pneighbors_end = graph.twohop_offsets[pvertexid + 1];
    for (uint64_t k = pneighbors_start + lane_idx; k < pneighbors_end; k += WARP_SIZE) {
        phelper1 = graph.twohop_neighbors[k];
        device_search_array(read_vertices + start, tot_vert, phelper1, lane_idx, phelper2);
        if (phelper2 != -1) {
            read_lvl2adj[phelper2]--;
        }
    }
}

__device__ void add_one_vertex(int& lane_idx, Vertex* vertices, int& total_vertices, int& number_of_members, int& number_of_candidates, GPU_Graph& graph)
{
    // pruning helper variables
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    int phelper2;

    if (lane_idx == 0) {
        vertices[total_vertices - 1].label = 1;
    }

    number_of_members++;
    number_of_candidates--;

    pvertexid = vertices[total_vertices - 1].vertexid;
    pneighbors_start = graph.onehop_offsets[pvertexid];
    pneighbors_end = graph.onehop_offsets[pvertexid + 1];

    // update the exdeg and indeg of all vertices adj to the vertice just added to the vertex set
    for (uint64_t k = pneighbors_start + lane_idx; k < pneighbors_end; k += WARP_SIZE) {
        phelper1 = graph.onehop_neighbors[k];
        device_search_vertices(vertices, total_vertices, phelper1, lane_idx, phelper2);
        if (phelper2 != -1) {
            vertices[phelper2].exdeg--;
            vertices[phelper2].indeg++;
        }
    }

    // sort new vertices putting just added vertex at end of all vertices in x
    __syncwarp();
    device_sort(vertices, total_vertices, lane_idx);

    // --- DIAMETER PRUNING ---
    int number_of_removed_candidates = 0;
    pneighbors_start = graph.twohop_offsets[pvertexid];
    pneighbors_end = graph.twohop_offsets[pvertexid + 1];
    int added_twohop_neighbors_count = pneighbors_end - pneighbors_start;
    for (int k = number_of_members + lane_idx; k < total_vertices; k += WARP_SIZE) {
        phelper1 = vertices[k].vertexid;
        device_search_array(graph.twohop_neighbors + pneighbors_start, added_twohop_neighbors_count, phelper1, lane_idx, phelper2);
        if (phelper2 == -1) {
            vertices[k].label = -1;
            number_of_removed_candidates++;
        }
    }

    __syncwarp();
    device_sort(vertices + number_of_members, number_of_candidates, lane_idx);

    // update exdeg of vertices connected to removed cands
    for (int k = total_vertices - number_of_removed_candidates; k < total_vertices; k++) {
        pvertexid = vertices[k].vertexid;
        pneighbors_start = graph.onehop_offsets[pvertexid];
        pneighbors_end = graph.onehop_offsets[pvertexid + 1];
        for (uint64_t l = pneighbors_start + lane_idx; l < pneighbors_end; l += WARP_SIZE) {
            phelper1 = graph.onehop_neighbors[l];
            device_search_vertices(vertices, total_vertices, phelper1, lane_idx, phelper2);
            if (phelper2 != -1) {
                vertices[phelper2].exdeg--;
            }
        }

        // update lvl2adj
        pneighbors_start = graph.twohop_offsets[pvertexid];
        pneighbors_end = graph.twohop_offsets[pvertexid + 1];
        for (uint64_t l = pneighbors_start + lane_idx; l < pneighbors_end; l += WARP_SIZE) {
            phelper1 = graph.twohop_neighbors[l];
            device_search_vertices(vertices, total_vertices, phelper1, lane_idx, phelper2);
            if (phelper2 != -1) {
                vertices[phelper2].lvl2adj--;
            }
        }
    }
    total_vertices -= number_of_removed_candidates;
    number_of_candidates -= number_of_removed_candidates;
}

__device__ void check_for_clique(int& number_of_members, int& lane_idx, int& warp_idx, Vertex* vertices, int& wcliques_write, int& wcliques_offset_write,
    GPU_Cliques& device_cliques, GPU_Data& device_data)
{
    bool meet_requirement = true;
    int degree_requirement = device_data.minimum_degrees[number_of_members];
    for (int k = lane_idx; k < number_of_members; k += WARP_SIZE) {
        if (vertices[k].indeg < degree_requirement) {
            meet_requirement = false;
            break;
        }
    }

    // set to false if any threads in warp do not meet degree requirement
    __syncwarp();
    bool clique = !(__any_sync(0xFFFFFFFF, !meet_requirement));

    // if clique write to warp buffer for cliques
    if (clique) {
        uint64_t start_write = wcliques_write + device_cliques.wcliques_offset[(device_cliques.wcliques_count[warp_idx])];
        for (int k = lane_idx; k < number_of_members; k += WARP_SIZE) {
            device_cliques.wcliques_vertex[start_write + k] = vertices[k].vertexid;
        }
        (device_cliques.wcliques_count[warp_idx])++;
        device_cliques.wcliques_offset[wcliques_offset_write + (device_cliques.wcliques_count[warp_idx])] = start_write + number_of_members;
    }
}

__device__ void write_to_tasks(GPU_Data& device_data, int& wtasks_write, int& warp_idx, int& lane_idx, int& total_vertices, Vertex* vertices, int& wtasks_offset_write)
{
    // CRITICAL
    atomicExch((int*)device_data.maximal_expansion, false);

    uint64_t start_write = wtasks_write + device_data.wtasks_offset[(device_data.wtasks_count[warp_idx])];

    for (int k = lane_idx; k < total_vertices; k += WARP_SIZE) {
        device_data.wtasks_vertex[start_write + k] = vertices[k].vertexid;
        device_data.wtasks_label[start_write + k] = vertices[k].label;
        device_data.wtasks_indeg[start_write + k] = vertices[k].indeg;
        device_data.wtasks_exdeg[start_write + k] = vertices[k].exdeg;
        device_data.wtasks_lvl2adj[start_write + k] = vertices[k].lvl2adj;
    }
    (device_data.wtasks_count[warp_idx])++;
    device_data.wtasks_offset[wtasks_offset_write + (device_data.wtasks_count[warp_idx])] = start_write - wtasks_write + total_vertices;
}

__device__ void transfer_buffers(GPU_Data& device_data, int& warp_idx, int& wtasks_offset_write, GPU_Cliques& device_cliques, int& wcliques_offset_write, int& lane_idx,
    uint64_t* write_count, int& tasks_offset_write, uint64_t& tasks_write, int& cliques_offset_write, uint64_t& cliques_write, int& warp_count,
    uint64_t* write_offsets, uint64_t& buffer_offset_start, uint64_t& buffer_start, int& wtasks_write, int* write_vertices, int* write_labels,
    int* write_indeg, int* write_exdeg, int* write_lvl2adj, uint64_t& cliques_offset_start, uint64_t& cliques_start, int& wcliques_write)
{
    int tasks_count = device_data.wtasks_count[warp_idx];
    int tasks_size = device_data.wtasks_offset[wtasks_offset_write + tasks_count];

    int cliques_count = device_cliques.wcliques_count[warp_idx];
    int cliques_size = device_cliques.wcliques_offset[wcliques_offset_write + cliques_count];

    // !!! CRITICAL SECTION !!!
    if (lane_idx == 0) {
        // TODO - is sum can be optimized
        // sum to find tasks count
        atomicAdd(device_data.total_tasks, tasks_count);
        atomicAdd(device_cliques.cliques_count, cliques_count);

        // TODO - is scan can be optimized
        // sum for tasks offset
        if (warp_idx == 0)
        {
            // handle tasks and buffer counts
            if ((*device_data.total_tasks) <= EXPAND_THRESHOLD) {
                (*write_count) += (*device_data.total_tasks);
            }
            else {
                (*write_count) += EXPAND_THRESHOLD;
                (*device_data.buffer_count) += ((*device_data.total_tasks) - EXPAND_THRESHOLD);
            }

            // handle 1st warp write starts
            tasks_offset_write = 0;
            (*(device_data.helper_int1)) = tasks_count + 1;
            tasks_write = 0;
            (*(device_data.helper_int2)) = tasks_size;
            cliques_offset_write = 0;
            (*(device_data.helper_int3)) = cliques_count + 1;
            cliques_write = 0;
            (*(device_data.helper_int4)) = cliques_size;
        }
        for (int i = 1; i < warp_count; i++) {
            if (warp_idx == i) {
                // handle next warp write starts
                tasks_offset_write = (*(device_data.helper_int1));
                (*(device_data.helper_int1)) += tasks_count;
                tasks_write = (*(device_data.helper_int2));
                (*(device_data.helper_int2)) += tasks_size;
                cliques_offset_write = (*(device_data.helper_int3));
                (*(device_data.helper_int3)) += cliques_count;
                cliques_write = (*(device_data.helper_int4));
                (*(device_data.helper_int4)) += cliques_size;
            }
        }
    }
    // !!! END SECTION !!!

    // distribute amongst waprs
    tasks_offset_write = __shfl_sync(0xFFFFFFFF, tasks_offset_write, 0);
    tasks_write = __shfl_sync(0xFFFFFFFF, tasks_offset_write, 0);
    cliques_offset_write = __shfl_sync(0xFFFFFFFF, tasks_offset_write, 0);
    cliques_write = __shfl_sync(0xFFFFFFFF, tasks_offset_write, 0);

    // move to tasks and buffer
    for (int i = lane_idx; i < tasks_count; i += WARP_SIZE)
    {
        if (tasks_offset_write + i > EXPAND_THRESHOLD) {
            // to tasks
            write_offsets[tasks_offset_write + i] = tasks_write + device_data.wtasks_offset[wtasks_offset_write + i];
        }
        else if (tasks_offset_write + i == EXPAND_THRESHOLD) {
            // to end tasks
            write_offsets[tasks_offset_write + i] = tasks_write + device_data.wtasks_offset[wtasks_offset_write + i];
            (*(device_data.helper_int1)) = tasks_write + device_data.wtasks_offset[wtasks_offset_write + i];
        }
        else if (tasks_offset_write + i == (*device_data.total_tasks) - 1) {
            // to end buffer
            device_data.buffer_offset[buffer_offset_start + tasks_offset_write + i - EXPAND_THRESHOLD] = buffer_start + tasks_write + device_data.wtasks_offset[wtasks_offset_write + i] - (*(device_data.helper_int1));
            device_data.buffer_offset[buffer_offset_start + tasks_offset_write + i - EXPAND_THRESHOLD + 1] = buffer_start + tasks_write + device_data.wtasks_offset[wtasks_offset_write + i + 1] - (*(device_data.helper_int1));
        }
        else {
            // to buffer
            device_data.buffer_offset[buffer_offset_start + tasks_offset_write + i - EXPAND_THRESHOLD] = buffer_start + tasks_write + device_data.wtasks_offset[wtasks_offset_write + i] - (*(device_data.helper_int1));
        }
    }
    for (int i = lane_idx; i < tasks_size; i += WARP_SIZE) {
        if (tasks_write + lane_idx > (*(device_data.helper_int1))) {
            // to tasks
            write_vertices[tasks_write + i] = device_data.wtasks_vertex[wtasks_write + i];
            write_labels[tasks_write + i] = device_data.wtasks_label[wtasks_write + i];
            write_indeg[tasks_write + i] = device_data.wtasks_indeg[wtasks_write + i];
            write_exdeg[tasks_write + i] = device_data.wtasks_exdeg[wtasks_write + i];
            write_lvl2adj[tasks_write + i] = device_data.wtasks_lvl2adj[wtasks_write + i];
        }
        else {
            // to buffer
            device_data.buffer_vertex[buffer_start + tasks_write + i - (*(device_data.helper_int1))] = device_data.wtasks_vertex[wtasks_write + i];
            device_data.buffer_label[buffer_start + tasks_write + i - (*(device_data.helper_int1))] = device_data.wtasks_label[wtasks_write + i];
            device_data.buffer_indeg[buffer_start + tasks_write + i - (*(device_data.helper_int1))] = device_data.wtasks_indeg[wtasks_write + i];
            device_data.buffer_exdeg[buffer_start + tasks_write + i - (*(device_data.helper_int1))] = device_data.wtasks_exdeg[wtasks_write + i];
            device_data.buffer_lvl2adj[buffer_start + tasks_write + i - (*(device_data.helper_int1))] = device_data.wtasks_lvl2adj[wtasks_write + i];
        }
    }

    //move to cliques
    for (int i = lane_idx; i < tasks_count; i += WARP_SIZE) {
        device_cliques.cliques_offset[cliques_offset_start + cliques_offset_write + i] = cliques_start + cliques_write + device_cliques.wcliques_offset[wcliques_offset_write + i];
    }
    for (int i = lane_idx; i < tasks_size; i += WARP_SIZE) {
        device_cliques.cliques_vertex[cliques_start + cliques_write + i] = device_cliques.wcliques_vertex[wcliques_write + i];
    }
}

__device__ void fill_from_buffer(GPU_Data& device_data, uint64_t* write_count, uint64_t* write_offsets, int* write_vertices, int* write_labels, int* write_indeg,
    int* write_exdeg, int* write_lvl2adj, int& idx, int& thread_count)
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
    for (int i = idx + 1; i <= write_amount; i += thread_count) {
        write_offsets[(*write_count) + i] = start_write + (device_data.buffer_offset[(*(device_data.buffer_count)) - write_amount + i] - start_buffer);
    }

    // handle data
    for (int i = idx; i < size_buffer; i += thread_count) {
        write_vertices[start_write + i] = device_data.buffer_vertex[start_buffer + i];
        write_labels[start_write + i] = device_data.buffer_label[start_buffer + i];
        write_indeg[start_write + i] = device_data.buffer_indeg[start_buffer + i];
        write_exdeg[start_write + i] = device_data.buffer_exdeg[start_buffer + i];
        write_lvl2adj[start_write + i] = device_data.buffer_lvl2adj[start_buffer + i];
    }

    // CRITICAL
    if (idx == 0) {
        // update counts
        (*write_count) += write_amount;
        (*(device_data.buffer_count)) -= write_amount;
    }
}



// --- HELPER KERNELS ---

// TODO - convert to merge or radix sort, merge is recursive
__device__ void device_sort(Vertex* target, int& size, int& lane_idx)
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    for (int i = 0; i < size - 1; i++) {
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
    }
}

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

// TODO - convert to binary search as adj lists are sorted
// searches an Vertex array for a vertex of a certain label, returns the position in the array that item was found, or -1 if not found
__device__ void device_search_vertices(Vertex* search_array, int& array_size, int& search_vertexid, int& lane_idx, int& result)
{
    // ALGO - linear
    // TYPE - parallel
    // SPEED - O(n)

    for (int i = lane_idx; i < array_size; i += WARP_SIZE) {
        if (search_array[i].vertexid == search_vertexid) {
            result = i;
        }
    }
    result = -1;
}

// TODO - convert to binary search
// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
__device__ void device_search_array(int* search_array, int& array_size, int& search_number, int& lane_idx, int& result)
{
    // ALGO - linear
    // TYPE - parallel
    // SPEED - O(n)

    for (int i = lane_idx; i < array_size; i += WARP_SIZE) {
        if (search_array[i] == search_number) {
            result = i;
        }
    }
    result = -1;
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
    cout << endl << "Beginning removal of non-maximal cliques" << endl;

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

    cout << endl << "Program successfully completed! Results:" << endl;

    printf("\nNumber of Maximal Cliques: %d\n\n", gntotal_max_cliques);
}