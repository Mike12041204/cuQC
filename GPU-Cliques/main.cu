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
#include <time.h>
#include <chrono>
#include <sys/timeb.h>
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
#define TASKS_SIZE 2000000
#define EXPAND_THRESHOLD 704
#define BUFFER_SIZE 100000000
#define BUFFER_OFFSET_SIZE 1000000
#define CLIQUES_SIZE 2000000
#define CLIQUES_OFFSET_SIZE 20000
#define CLIQUES_PERCENT 50

// buffer size for CPU onehop and twohop adjacency array and offsets, ensure these are large enough
#define OFFSETS_SIZE 1000000
#define LVL1ADJ_SIZE 10000000
#define LVL2ADJ_SIZE 100000000

// per warp
#define WCLIQUES_SIZE 5000
#define WCLIQUES_OFFSET_SIZE 500
#define WTASKS_SIZE 50000
#define WTASKS_OFFSET_SIZE 5000
#define WVERTICES_SIZE 3200
#define WADJACENCIES_SIZE 320

// shared memory size: 12.300 ints
#define VERTICES_SIZE 50
 
// threads info
#define BLOCK_SIZE 1024
#define NUM_OF_BLOCKS 22
#define WARP_SIZE 32

// run settings
#define CPU_LEVELS_x2 0

// debug toggle
#define DEBUG_TOGGLE 0

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
int h_sort_asce(const void* a, const void* b);
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

    CPU_Graph(ifstream& graph_stream)
    {
        graph_stream.seekg(0, graph_stream.end);
        string graph_text(graph_stream.tellg(), 0);
        graph_stream.seekg(0);
        graph_stream.read(const_cast<char*>(graph_text.data()), graph_text.size());

        onehop_offsets = new uint64_t[OFFSETS_SIZE];
        onehop_neighbors = new int[LVL1ADJ_SIZE];
        twohop_neighbors = new int[LVL2ADJ_SIZE];

        onehop_offsets[0] = 0;
        number_of_lvl2adj = 0;

        // DEBUG
        //cout << "|V| = " << number_of_vertices << " |E| = " << number_of_edges << " #bytes: " << text.size() << endl;

        int vertex_count = 0;
        int number_count = 0;
        int current_number = 0;
        bool empty = true;

        // TODO - way to detect and handle these cases without changing code?
        // TWO FORMATS SO FAR
        // 1 -  VSCode \r\n between lines, no ending character
        // 2 - Visual Studio \n between lines, numerous \0 ending characters
        
        // parse graph file assume adj are seperated by spaces ' ' and vertices are seperated by newlines "\r\n"
        for (int i = 0; i < graph_text.size(); i++) {
            char character = graph_text[i];

            // line depends on whether newline is "\r\n" or '\n'
            if (character == '\n') {
                if (!empty) {
                    onehop_neighbors[number_count++] = current_number;
                }
                onehop_offsets[++vertex_count] = number_count;
                current_number = 0;
                // line depends on whether newline is "\r\n" or '\n'
                //i++;
                empty = true;
            }
            else if (character == ' ') {
                onehop_neighbors[number_count++] = current_number;
                current_number = 0;
            }
            else if (character == '\0') {
                // line depends on whether newline is "\r\n" or '\n'
                break;
            }
            else {
                current_number = current_number * 10 + (graph_text[i] - '0');
                empty = false;
            }
        }

        // line depends on whether newline is "\r\n" or '\n'
        // handle last element
        if (!empty) {
            onehop_neighbors[number_count++] = current_number;
        }
        onehop_offsets[++vertex_count] = number_count;

        // set variables an initialize twohop arrays
        number_of_vertices = vertex_count;
        number_of_edges = number_count / 2;

        twohop_offsets = new uint64_t[number_of_vertices + 1];

        twohop_offsets[0] = 0;

        bool* twohop_flag_DIA;
        twohop_flag_DIA = new bool[number_of_vertices];
        memset(twohop_flag_DIA, true, number_of_vertices * sizeof(bool));

        // handle lvl2 adj
        for (int i = 0; i < vertex_count; i++) {
            for (int j = onehop_offsets[i]; j < onehop_offsets[i + 1]; j++) {
                int lvl1adj = onehop_neighbors[j];
                if (twohop_flag_DIA[lvl1adj]) {
                    twohop_neighbors[number_of_lvl2adj++] = lvl1adj;
                    twohop_flag_DIA[lvl1adj] = false;
                }

                for (int k = onehop_offsets[lvl1adj]; k < onehop_offsets[lvl1adj + 1]; k++) {
                    int lvl2adj = onehop_neighbors[k];
                    if (twohop_flag_DIA[lvl2adj] && lvl2adj != i) {
                        twohop_neighbors[number_of_lvl2adj++] = lvl2adj;
                        twohop_flag_DIA[lvl2adj] = false;
                    }
                }
            }

            twohop_offsets[i + 1] = number_of_lvl2adj;
            memset(twohop_flag_DIA, true, number_of_vertices * sizeof(bool));
        }

        // sort adjacencies
        for (int i = 0; i < vertex_count; i++) {
            qsort(onehop_neighbors + onehop_offsets[i], onehop_offsets[i + 1] - onehop_offsets[i], sizeof(int), h_sort_asce);
            qsort(twohop_neighbors + twohop_offsets[i], twohop_offsets[i + 1] - twohop_offsets[i], sizeof(int), h_sort_asce);
        }

        // DEBUG
        if (DEBUG_TOGGLE) {
            cout << "|V| = " << vertex_count << " |E| = " << number_count / 2 << " lvl1adj: " << number_count << " lvl2adj: " << number_of_lvl2adj << endl;
        }
        if (false) {
            cout << graph_text << "\n!!!" << endl;
            for (int i = 0; i < vertex_count; i++) {
                cout << i << ": " << flush;
                for (int j = onehop_offsets[i]; j < onehop_offsets[i + 1]; j++) {
                    cout << onehop_neighbors[j] << " " << flush;
                }
                cout << endl;
            }
            cout << "!!!" << endl;
            for (int i = 0; i < vertex_count; i++) {
                cout << i << ": " << flush;
                for (int j = twohop_offsets[i]; j < twohop_offsets[i + 1]; j++) {
                    cout << twohop_neighbors[j] << " " << flush;
                }
                cout << endl;
            }
        }

        delete twohop_flag_DIA;
    }

    ~CPU_Graph() 
    {
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

    uint64_t* tasks2_count;
    uint64_t* tasks2_offset;
    Vertex* tasks2_vertices;

    uint64_t* buffer_count;
    uint64_t* buffer_offset;
    Vertex* buffer_vertices;

    uint64_t* current_level;
    bool* maximal_expansion;
    bool* dumping_cliques;

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

    Vertex* global_vertices;
    int* removed_candidates;
    int* lane_removed_candidates;
    Vertex* remaining_candidates;
    int* lane_remaining_candidates;
    int* candidate_indegs;
    int* lane_candidate_indegs;

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

    int removed_count[(BLOCK_SIZE / WARP_SIZE)];
    int remaining_count[(BLOCK_SIZE / WARP_SIZE)];
    int num_val_cands[(BLOCK_SIZE / WARP_SIZE)];
    int rw_counter[(BLOCK_SIZE / WARP_SIZE)];

    int min_ext_deg[(BLOCK_SIZE / WARP_SIZE)];
    int lower_bound[(BLOCK_SIZE / WARP_SIZE)];
    int upper_bound[(BLOCK_SIZE / WARP_SIZE)];

    int tightened_upper_bound[(BLOCK_SIZE / WARP_SIZE)];
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
    int wib_idx;
};

// METHODS
void calculate_minimum_degrees(CPU_Graph& hg);
void search(CPU_Graph& hg, ofstream& temp_results);
void allocate_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc, CPU_Graph& hg);
void initialize_tasks(CPU_Graph& hg, CPU_Data& hd);
void move_to_gpu(CPU_Data& hd, GPU_Data& dd);
void dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& output_file);
void flush_cliques(CPU_Cliques& hc, ofstream& temp_results);
void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc);
void RemoveNonMax(char* szset_filename, char* szoutput_filename);

void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc);
int h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, int tot_vert, int num_mem, int num_cand, uint64_t start);
int h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, int& num_cand, int& num_vert, uint64_t start);
int h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg);
void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, int& total_vertices, int& number_of_candidates, int number_of_members);
bool h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg);
bool h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates);
void h_update_degrees(CPU_Graph& hg, Vertex* vertices, int total_vertices, int number_of_removed);
void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members);
int h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg);
void h_write_to_tasks(CPU_Data& hd, Vertex* vertices, int total_vertices, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count);
void h_fill_from_buffer(CPU_Data& hd, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count);

bool h_calculate_LU_bounds_old(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates);

int h_bsearch_array(int* search_array, int array_size, int search_number);
int h_lsearch_vert(Vertex* search_array, int array_size, int search_vertexid);
int h_sort_vert(const void* a, const void* b);
int h_sort_vert_Q(const void* a, const void* b);
int h_sort_vert_LU(const void* a, const void* b);
int h_sort_desc(const void* a, const void* b);
inline int h_get_mindeg(int clique_size);
inline bool h_cand_isvalid(Vertex vertex, int clique_size);
inline bool h_cand_isvalid_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg);
inline bool  h_vert_isextendable(Vertex vertex, int clique_size);
inline bool  h_vert_isextendable_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg);
inline void chkerr(cudaError_t code);

// DEBUG
void print_CPU_Data(CPU_Data& hd);
void print_GPU_Data(GPU_Data& dd);
void print_CPU_Graph(CPU_Graph& hg);
void print_GPU_Graph(GPU_Data& dd, CPU_Graph& hg);
void print_WTask_Buffers(GPU_Data& dd);
void print_WClique_Buffers(GPU_Data& dd);
void print_GPU_Cliques(GPU_Data& dd);
void print_CPU_Cliques(CPU_Cliques& hc);
void print_Data_Sizes(GPU_Data& dd);
void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc);
void print_vertices(Vertex* vertices, int size);
void print_Data_Sizes_Every(GPU_Data& dd, int every);
bool print_Warp_Data_Sizes(GPU_Data& dd);
void print_All_Warp_Data_Sizes(GPU_Data& dd);
bool print_Warp_Data_Sizes_Every(GPU_Data& dd, int every);
void print_All_Warp_Data_Sizes_Every(GPU_Data& dd, int every);
void print_debug(GPU_Data& dd);
void print_idebug(GPU_Data& dd);
void print_idebug(GPU_Data& dd);

// KERNELS
__global__ void d_expand_level(GPU_Data dd);
__global__ void transfer_buffers(GPU_Data dd);
__global__ void fill_from_buffer(GPU_Data dd);
__device__ int d_lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid);
__device__ void d_update_degrees(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_removed);
__device__ void d_calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates);
__device__ bool d_degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld);

__device__ void d_sort(Vertex* target, int size, int lane_idx, int (*func)(Vertex&, Vertex&));
__device__ void d_sort_i(int* target, int size, int lane_idx, int (*func)(int, int));
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2);
__device__ int d_sort_degs(int n1, int n2);
__device__ int d_sort_vert_cp(Vertex& vertex1, Vertex& vertex2);
__device__ int d_sort_vert_cc(Vertex& vertex1, Vertex& vertex2);
__device__ int d_sort_vert_lu(Vertex& vertex1, Vertex& vertex2);
__device__ int d_sort_vert_ex(Vertex& vertex1, Vertex& vertex2);
__device__ int d_bsearch_array(int* search_array, int array_size, int search_number);
__device__ bool d_cand_isvalid(Vertex& vertex, int number_of_members, GPU_Data& dd);
__device__ bool d_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ bool d_vert_isextendable(Vertex& vertex, int number_of_members, GPU_Data& dd);
__device__ bool d_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_get_mindeg(int number_of_members, GPU_Data& dd);

__device__ void d_print_vertices(Vertex* vertices, int size);



// - bus error debug; 
// - test code when working
// - analyze graph loading step why it is slow, 
// - implement the critcal pruning on CPU and GPU
// - improve GPU sorting algorithm

// TODO GENERALLY
// - local memory usage is right around 100% cant enable exact LU pruning while being able to use all threads
// - test program on larger graphs

// TODO (HIGH PRIORITY)
// - fill tasks kernel does not always need to launch can check outside of kernel to determine so
// - critical vertex on cpu and gpu
// - decrease variables on gpu
// - ensure no unecessary syncs on the gpu

// TODO (LOW PRIORITY)
// - reevaluate and change where uint64_t's are used
// - label for vertices can be a byte rather than int
// - cpu hybrid dfs-bfs expansion
// - cover pruning on cpu
// - dont need lvl2adj in all places anymore
// - optimize generating graph, see what Quick does
// - way to skip early levels for smaller graphs to decrease CPU time



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

    // TIME
    auto start = std::chrono::high_resolution_clock::now();

    // GRAPH / MINDEGS
    cout << ">:PRE-PROCESSING" << endl;
    CPU_Graph hg(graph_stream);
    graph_stream.close();
    calculate_minimum_degrees(hg);
    ofstream temp_results("temp.txt");

    // DEBUG
    //print_CPU_Graph(hg);

    // SEARCH
    search(hg, temp_results);

    temp_results.close();

    // RM NON-MAX
    RemoveNonMax("temp.txt", argv[4]);

    // TIME
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout << ">:TIME: " << duration.count() << " ms" << endl;

    cout << ">:PROGRAM END" << endl;
    return 0;
}



// --- HOST METHODS --- 

// initializes minimum degrees array
void calculate_minimum_degrees(CPU_Graph& hg)
{
    minimum_degrees = new int[hg.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 1; i <= hg.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}

void search(CPU_Graph& hg, ofstream& temp_results) 
{
    // DATA STRUCTURES
    CPU_Data hd;
    CPU_Cliques hc;
    GPU_Data dd;

    // HANDLE MEMORY
    allocate_memory(hd, dd, hc, hg);
    cudaDeviceSynchronize();

    // INITIALIZE TASKS
    cout << ">:INITIALIZING TASKS" << endl;
    initialize_tasks(hg, hd);

    // DEBUG
    if (DEBUG_TOGGLE) {
        h_print_Data_Sizes(hd, hc);
    }
    //print_CPU_Data(hd);

    // CPU EXPANSION
    // cpu levels is multiplied by two to ensure that data ends up in tasks1, this allows us to always copy tasks1 without worry like before hybrid cpu approach
    // cpu expand must be called atleast one time to handle first round cover pruning as the gpu code cannot do this
    for (int i = 0; i < 2 * (CPU_LEVELS_x2 + 1) && !(*hd.maximal_expansion); i++) {
        h_expand_level(hg, hd, hc);
    
        // if cliques is more than half full, flush to file
        if (hc.cliques_offset[(*hc.cliques_count)] > CLIQUES_SIZE / 2) {
            flush_cliques(hc, temp_results);
        }

        // DEBUG
        if (DEBUG_TOGGLE) {
            h_print_Data_Sizes(hd, hc);
        }
        //print_CPU_Data(hd);
    }

    flush_cliques(hc, temp_results);

    // TRANSFER TO GPU
    move_to_gpu(hd, dd);
    cudaDeviceSynchronize();

    // DEBUG
    //print_GPU_Graph(dd, hg);
    //print_CPU_Data(hd);
    //print_GPU_Data(dd);
    //print_Data_Sizes(dd);

    // EXPAND LEVEL
    cout << ">:BEGINNING EXPANSION" << endl;
    while (!(*hd.maximal_expansion))
    {
        // reset loop variables
        chkerr(cudaMemset(dd.maximal_expansion, true, sizeof(bool)));
        chkerr(cudaMemset(dd.dumping_cliques, false, sizeof(bool)));
        cudaDeviceSynchronize();

        // expand all tasks in 'tasks' array, each warp will write to their respective warp tasks buffer in global memory
        d_expand_level<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();

        // DEBUG
        //print_WClique_Buffers(dd);
        //print_WTask_Buffers(dd);
        if (DEBUG_TOGGLE) {
            if (print_Warp_Data_Sizes_Every(dd, 1)) { break; }
        }
        //print_All_Warp_Data_Sizes_Every(dd, 1);

        // consolidate all the warp tasks/cliques buffers into the next global tasks array, buffer, and cliques
        transfer_buffers<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();

        // if not enough tasks were generated when expanding the previous level to fill the next tasks array the program will attempt to fill the tasks array by popping tasks from the buffer
        fill_from_buffer<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
        cudaDeviceSynchronize();

        // update the loop variables
        chkerr(cudaMemcpy(hd.maximal_expansion, dd.maximal_expansion, sizeof(bool), cudaMemcpyDeviceToHost));
        chkerr(cudaMemcpy(hd.dumping_cliques, dd.dumping_cliques, sizeof(bool), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        if (*hd.dumping_cliques) {
            dump_cliques(hc, dd, temp_results);
        }

        // DEBUG
        //print_GPU_Data(dd);
        //print_GPU_Cliques(dd);
        if (DEBUG_TOGGLE) {
            print_Data_Sizes_Every(dd, 1); printf("\n");
        }
        //print_debug(dd);
        //print_idebug(dd);
        //break;
    }

    dump_cliques(hc, dd, temp_results);

    // FREE MEMORY
    free_memory(hd, dd, hc);
}

// allocates memory for the data structures on the host and device
void allocate_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc, CPU_Graph& hg)
{
    int number_of_warps = (NUM_OF_BLOCKS * BLOCK_SIZE) / WARP_SIZE;

    // GPU GRAPH
    chkerr(cudaMalloc((void**)&dd.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&dd.onehop_neighbors, sizeof(int) * hg.number_of_edges * 2));
    chkerr(cudaMalloc((void**)&dd.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&dd.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj));
    chkerr(cudaMalloc((void**)&dd.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));

    chkerr(cudaMemcpy(dd.number_of_vertices, &(hg.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.number_of_edges, &(hg.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.onehop_neighbors, hg.onehop_neighbors, sizeof(int) * hg.number_of_edges * 2, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.onehop_offsets, hg.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.twohop_neighbors, hg.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.twohop_offsets, hg.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));

    // CPU DATA
    hd.tasks1_count = new uint64_t;
    hd.tasks1_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    hd.tasks1_vertices = new Vertex[TASKS_SIZE];

    hd.tasks1_offset[0] = 0;
    (*(hd.tasks1_count)) = 0;

    hd.tasks2_count = new uint64_t;
    hd.tasks2_offset = new uint64_t[EXPAND_THRESHOLD + 1];
    hd.tasks2_vertices = new Vertex[TASKS_SIZE];

    hd.tasks2_offset[0] = 0;
    (*(hd.tasks2_count)) = 0;

    hd.buffer_count = new uint64_t;
    hd.buffer_offset = new uint64_t[BUFFER_OFFSET_SIZE];
    hd.buffer_vertices = new Vertex[BUFFER_SIZE];

    hd.buffer_offset[0] = 0;
    (*(hd.buffer_count)) = 0;

    hd.current_level = new uint64_t;
    hd.maximal_expansion = new bool;
    hd.dumping_cliques = new bool;

    (*hd.current_level) = 0;
    (*hd.maximal_expansion) = false;
    (*hd.dumping_cliques) = false;

    hd.vertex_order_map = new int[hg.number_of_vertices];
    hd.remaining_candidates = new int[hg.number_of_vertices];
    hd.removed_candidates = new int[hg.number_of_vertices];
    hd.remaining_count = new int;
    hd.removed_count = new int;
    hd.candidate_indegs = new int[hg.number_of_vertices];

    memset(hd.vertex_order_map, -1, sizeof(int) * hg.number_of_vertices);

    // GPU DATA
    chkerr(cudaMalloc((void**)&dd.current_level, sizeof(uint64_t)));

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

    chkerr(cudaMalloc((void**)&dd.global_vertices, (sizeof(Vertex) * WVERTICES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.removed_candidates, (sizeof(int) * WVERTICES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.lane_removed_candidates, (sizeof(int) * WVERTICES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.remaining_candidates, (sizeof(Vertex) * WVERTICES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.lane_remaining_candidates, (sizeof(int) * WVERTICES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * number_of_warps));
    chkerr(cudaMalloc((void**)&dd.lane_candidate_indegs, (sizeof(int) * WVERTICES_SIZE) * number_of_warps));

    chkerr(cudaMalloc((void**)&dd.maximal_expansion, sizeof(bool)));
    chkerr(cudaMalloc((void**)&dd.dumping_cliques, sizeof(bool)));

    chkerr(cudaMemset(dd.dumping_cliques, false, sizeof(bool)));

    chkerr(cudaMalloc((void**)&dd.minimum_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&dd.minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&dd.minimum_clique_size, sizeof(int)));

    chkerr(cudaMemcpy(dd.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.minimum_degrees, minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMalloc((void**)&dd.total_tasks, sizeof(int)));

    chkerr(cudaMemset(dd.total_tasks, 0, sizeof(int)));

    // CPU CLIQUES
    hc.cliques_count = new uint64_t;
    hc.cliques_vertex = new int[CLIQUES_SIZE];
    hc.cliques_offset = new uint64_t[CLIQUES_OFFSET_SIZE];

    hc.cliques_offset[0] = 0;
    (*(hc.cliques_count)) = 0;

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

// processes 0th level of expansion
void initialize_tasks(CPU_Graph& hg, CPU_Data& hd)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    // cover pruning
    int maximum_degree;
    int maximum_degree_index;

    // vertices information
    int total_vertices;
    int number_of_candidates;
    Vertex* vertices;



    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // initialize vertices
    total_vertices = hg.number_of_vertices;
    vertices = new Vertex[total_vertices];
    number_of_candidates = total_vertices;
    for (int i = 0; i < total_vertices; i++) {
        vertices[i].vertexid = i;
        vertices[i].indeg = 0;
        vertices[i].exdeg = hg.onehop_offsets[i + 1] - hg.onehop_offsets[i];
        vertices[i].lvl2adj = hg.twohop_offsets[i + 1] - hg.twohop_offsets[i];
        if (vertices[i].exdeg >= minimum_degrees[minimum_clique_size] && vertices[i].lvl2adj >= minimum_clique_size - 1) {
            vertices[i].label = 0;
            hd.remaining_candidates[(*hd.remaining_count)++] = i;
        }
        else {
            vertices[i].label = -1;
            hd.removed_candidates[(*hd.removed_count)++] = i;
        }
    }

    

    // DEGREE-BASED PRUNING
    // update while half of vertices have been removed
    while ((*hd.remaining_count) < number_of_candidates / 2) {
        number_of_candidates = (*hd.remaining_count);
        
        for (int i = 0; i < number_of_candidates; i++) {
            vertices[hd.remaining_candidates[i]].exdeg = 0;
        }

        for (int i = 0; i < number_of_candidates; i++) {
            // in 0th level id is same as position in vertices as all vertices are in vertices, see last block
            pvertexid = hd.remaining_candidates[i];
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (int j = pneighbors_start; j < pneighbors_end; j++) {
                phelper1 = hg.onehop_neighbors[j];
                if (vertices[phelper1].label == 0) {
                    vertices[phelper1].exdeg++;
                }
            }
        }

        (*hd.remaining_count) = 0;
        (*hd.removed_count) = 0;

        // remove more vertices based on updated degrees
        for (int i = 0; i < number_of_candidates; i++) {
            phelper1 = hd.remaining_candidates[i];
            if (vertices[phelper1].exdeg >= minimum_degrees[minimum_clique_size]) {
                hd.remaining_candidates[(*hd.remaining_count)++] = phelper1;
            }
            else {
                vertices[phelper1].label = -1;
                hd.removed_candidates[(*hd.removed_count)++] = phelper1;
            }
        }
    }
    number_of_candidates = (*hd.remaining_count);

    // update degrees based on last round of removed vertices
    int removed_start = 0;
    while((*hd.removed_count) > removed_start) {
        pvertexid = hd.removed_candidates[removed_start];
        pneighbors_start = hg.onehop_offsets[pvertexid];
        pneighbors_end = hg.onehop_offsets[pvertexid + 1];

        for (int j = pneighbors_start; j < pneighbors_end; j++) {
            phelper1 = hg.onehop_neighbors[j];

            if (vertices[phelper1].label == 0) {
                vertices[phelper1].exdeg--;

                if (vertices[phelper1].exdeg < minimum_degrees[minimum_clique_size]) {
                    vertices[phelper1].label = -1;
                    number_of_candidates--;
                    hd.removed_candidates[(*hd.removed_count)++] = phelper1;
                }
            }
        }
        removed_start++;
    }


    
    // FIRST ROUND COVER PRUNING
        // find cover vertex
    maximum_degree = 0;
    maximum_degree_index = 0;
    for (int i = 0; i < total_vertices; i++) {
        if (vertices[i].label == 0) {
            if (vertices[i].exdeg > maximum_degree) {
                maximum_degree = vertices[i].exdeg;
                maximum_degree_index = i;
            }
        }
    }
    vertices[maximum_degree_index].label = 3;

    // find all covered vertices
    pneighbors_start = hg.onehop_offsets[maximum_degree_index];
    pneighbors_end = hg.onehop_offsets[maximum_degree_index + 1];
    for (int i = pneighbors_start; i < pneighbors_end; i++) {
        pvertexid = hg.onehop_neighbors[i];
        if (vertices[pvertexid].label == 0) {
            vertices[pvertexid].label = 2;
        }
    }

    // sort enumeration order before writing to tasks
    qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert_Q);
    total_vertices = number_of_candidates;



    // WRITE TO TASKS
    if (total_vertices > 0)
    {
        for (int j = 0; j < total_vertices; j++) {
            hd.tasks1_vertices[j].vertexid = vertices[j].vertexid;
            hd.tasks1_vertices[j].label = vertices[j].label;
            hd.tasks1_vertices[j].indeg = vertices[j].indeg;
            hd.tasks1_vertices[j].exdeg = vertices[j].exdeg;
            hd.tasks1_vertices[j].lvl2adj = 0;
        }
        (*(hd.tasks1_count))++;
        hd.tasks1_offset[(*(hd.tasks1_count))] = total_vertices;
    }

    delete vertices;
}

void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc)
{
    // initiate the variables containing the location of the read and write task vectors, done in an alternating, odd-even manner like the c-intersection of cuTS
    uint64_t* read_count;
    uint64_t* read_offsets;
    Vertex* read_vertices;
    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;

    // old vertices information
    uint64_t start;
    uint64_t end;
    int tot_vert;
    int num_mem;
    int num_cand;
    int expansions;
    int number_of_covered;

    // new vertices information
    Vertex* vertices;
    int number_of_members;
    int number_of_candidates;
    int total_vertices;

    // calculate lower-upper bounds
    int min_ext_deg;
    int lower_bound;
    int upper_bound;

    int method_return;
    int index;



    if ((*hd.current_level) % 2 == 0) {
        read_count = hd.tasks1_count;
        read_offsets = hd.tasks1_offset;
        read_vertices = hd.tasks1_vertices;
        write_count = hd.tasks2_count;
        write_offsets = hd.tasks2_offset;
        write_vertices = hd.tasks2_vertices;
    }
    else {
        read_count = hd.tasks2_count;
        read_offsets = hd.tasks2_offset;
        read_vertices = hd.tasks2_vertices;
        write_count = hd.tasks1_count;
        write_offsets = hd.tasks1_offset;
        write_vertices = hd.tasks1_vertices;
    }
    *write_count = 0;
    write_offsets[0] = 0;

    // set to false later if task is generated indicating non-maximal expansion
    (*hd.maximal_expansion) = true;



    // CURRENT LEVEL
    for (int i = 0; i < *read_count; i++)
    {
        // get information of vertices being handled within tasks
        start = read_offsets[i];
        end = read_offsets[i + 1];
        tot_vert = end - start;
        num_mem = 0;
        for (uint64_t j = start; j < end; j++) {
            if (read_vertices[j].label != 1) {
                break;
            }
            num_mem++;
        }
        number_of_covered = 0;
        for (uint64_t j = start + num_mem; j < end; j++) {
            if (read_vertices[j].label != 2) {
                break;
            }
            number_of_covered++;
        }
        num_cand = tot_vert - num_mem;
        expansions = num_cand;



        // LOOKAHEAD PRUNING
        method_return = h_lookahead_pruning(hg, hc, hd, read_vertices, tot_vert, num_mem, num_cand, start);
        if (method_return) {
            continue;
        }



        // NEXT LEVEL
        for (int j = number_of_covered; j < expansions; j++) {



            // REMOVE ONE VERTEX
            if (j != number_of_covered) {
                method_return = h_remove_one_vertex(hg, hd, read_vertices, tot_vert, num_cand, num_mem, start);
                if (method_return) {
                    break;
                }
            }



            // NEW VERTICES
            vertices = new Vertex[tot_vert];
            number_of_members = num_mem;
            number_of_candidates = num_cand;
            total_vertices = tot_vert;
            for (index = 0; index < number_of_members; index++) {
                vertices[index] = read_vertices[start + index];
            }
            vertices[number_of_members] = read_vertices[start + total_vertices - 1];
            for (; index < total_vertices - 1; index++) {
                vertices[index + 1] = read_vertices[start + index];
            }

            if (number_of_covered > 0) {
                // set all covered vertices from previous level as candidates
                for (int j = num_mem + 1; j <= num_mem + number_of_covered; j++) {
                    vertices[j].label = 0;
                }
            }



            // ADD ONE VERTEX
            method_return = h_add_one_vertex(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

            // continue if not enough vertices after pruning
            if (method_return == 2) {
                delete vertices;
                continue;
            }

            // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
            if (method_return == 1) {
                if (number_of_members >= minimum_clique_size) {
                    h_check_for_clique(hc, vertices, number_of_members);
                }

                delete vertices;
                continue;
            }



            // CRITICAL VERTEX PRUNING
            //method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

            // continue if not enough vertices after pruning
            if (method_return == 2) {
                delete vertices;
                continue;
            }





            // CHECK FOR CLIQUE
            if (number_of_members >= minimum_clique_size) {
                h_check_for_clique(hc, vertices, number_of_members);
            }

            // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
            if (method_return == 1) {
                delete vertices;
                continue;
            }



            // WRITE TO TASKS
            //sort vertices so that lowest degree vertices are first in enumeration order before writing to tasks
            qsort(vertices + number_of_members, number_of_candidates, sizeof(Vertex), h_sort_vert_Q);

            // TODO - do we need this if?
            if (number_of_candidates > 0) {
                h_write_to_tasks(hd, vertices, total_vertices, write_vertices, write_offsets, write_count);
            }



            delete vertices;
        }
    }



    // FILL TASKS FROM BUFFER
    if (*write_count < EXPAND_THRESHOLD && (*hd.buffer_count) > 0)
    {
        h_fill_from_buffer(hd, write_vertices, write_offsets, write_count);
    }

    (*hd.current_level)++;
}

void move_to_gpu(CPU_Data& hd, GPU_Data& dd)
{
    chkerr(cudaMemcpy(dd.tasks1_count, hd.tasks1_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.tasks1_offset, hd.tasks1_offset, (EXPAND_THRESHOLD + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.tasks1_vertices, hd.tasks1_vertices, (TASKS_SIZE) * sizeof(Vertex), cudaMemcpyHostToDevice));

    chkerr(cudaMemcpy(dd.buffer_count, hd.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.buffer_offset, hd.buffer_offset, (BUFFER_OFFSET_SIZE) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.buffer_vertices, hd.buffer_vertices, (BUFFER_SIZE) * sizeof(int), cudaMemcpyHostToDevice));

    chkerr(cudaMemcpy(dd.maximal_expansion, hd.maximal_expansion, sizeof(bool), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dd.current_level, hd.current_level, sizeof(uint64_t), cudaMemcpyHostToDevice));
}

void dump_cliques(CPU_Cliques& hc, GPU_Data& dd, ofstream& temp_results)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(hc.cliques_count, dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_offset, dd.cliques_offset, sizeof(uint64_t) * CLIQUES_OFFSET_SIZE, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_vertex, dd.cliques_vertex, sizeof(int) * CLIQUES_SIZE, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // DEBUG
    //print_CPU_Cliques(hc);

    flush_cliques(hc, temp_results);

    cudaMemset(dd.cliques_count, 0, sizeof(uint64_t));
}

void flush_cliques(CPU_Cliques& hc, ofstream& temp_results) 
{
    for (int i = 0; i < ((*hc.cliques_count)); i++) {
        uint64_t start = hc.cliques_offset[i];
        uint64_t end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << hc.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    ((*hc.cliques_count)) = 0;
}

void free_memory(CPU_Data& hd, GPU_Data& dd, CPU_Cliques& hc)
{
    // GPU GRAPH
    chkerr(cudaFree(dd.number_of_vertices));
    chkerr(cudaFree(dd.number_of_edges));
    chkerr(cudaFree(dd.onehop_neighbors));
    chkerr(cudaFree(dd.onehop_offsets));
    chkerr(cudaFree(dd.twohop_neighbors));
    chkerr(cudaFree(dd.twohop_offsets));

    // CPU DATA
    delete hd.tasks1_count;
    delete hd.tasks1_offset;
    delete hd.tasks1_vertices;

    delete hd.tasks2_count;
    delete hd.tasks2_offset;
    delete hd.tasks2_vertices;

    delete hd.buffer_count;
    delete hd.buffer_offset;
    delete hd.buffer_vertices;

    delete hd.current_level;
    delete hd.maximal_expansion;
    delete hd.dumping_cliques;

    delete hd.vertex_order_map;
    delete hd.remaining_candidates;
    delete hd.remaining_count;
    delete hd.removed_candidates;
    delete hd.removed_count;
    delete hd.candidate_indegs;

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

    chkerr(cudaFree(dd.global_vertices));
    chkerr(cudaFree(dd.remaining_candidates));
    chkerr(cudaFree(dd.lane_remaining_candidates));
    chkerr(cudaFree(dd.removed_candidates));
    chkerr(cudaFree(dd.lane_removed_candidates));
    chkerr(cudaFree(dd.candidate_indegs));
    chkerr(cudaFree(dd.lane_candidate_indegs));

    chkerr(cudaFree(dd.maximal_expansion));
    chkerr(cudaFree(dd.dumping_cliques));

    chkerr(cudaFree(dd.minimum_degree_ratio));
    chkerr(cudaFree(dd.minimum_degrees));
    chkerr(cudaFree(dd.minimum_clique_size));

    chkerr(cudaFree(dd.total_tasks));

    // CPU CLIQUES
    delete hc.cliques_count;
    delete hc.cliques_vertex;
    delete hc.cliques_offset;

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



// --- HOST EXPANSION METHODS ---

// returns 1 if lookahead was a success, else 0
int h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, int tot_vert, int num_mem, int num_cand, uint64_t start)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;


    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning guarentees all members will be within 2hops of eveything
    for (int i = 0; i < num_mem; i++) {
        if (read_vertices[start + i].indeg + read_vertices[start + i].exdeg < minimum_degrees[tot_vert]) {
            return 0;
        }
    }

    // initialize vertex order map
    for (int i = num_mem; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    // update lvl2adj to candidates for all vertices
    for (int i = num_mem; i < tot_vert; i++) {
        pvertexid = read_vertices[start + i].vertexid;
        pneighbors_start = hg.twohop_offsets[pvertexid];
        pneighbors_end = hg.twohop_offsets[pvertexid + 1];
        for (int j = pneighbors_start; j < pneighbors_end; j++) {
            phelper1 = hd.vertex_order_map[hg.twohop_neighbors[j]];

            if (phelper1 >= num_mem) {
                read_vertices[start + phelper1].lvl2adj++;
            }
        }
    }

    // reset vertex order map
    for (int i = num_mem; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
    }

    // check for lookahead
    for (int j = num_mem; j < tot_vert; j++) {
        if (read_vertices[start + j].lvl2adj < num_cand - 1 || read_vertices[start + j].indeg + read_vertices[start + j].exdeg < minimum_degrees[tot_vert]) {
            return 0;
        }
    }

    // write to cliques
    uint64_t start_write = hc.cliques_offset[(*hc.cliques_count)];
    for (int j = 0; j < tot_vert; j++) {
        hc.cliques_vertex[start_write + j] = read_vertices[start + j].vertexid;
    }
    (*hc.cliques_count)++;
    hc.cliques_offset[(*hc.cliques_count)] = start_write + tot_vert;

    return 1;
}

// returns 1 is failed found or not enough vertices, else 0
int h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, int& num_cand, int& num_mem, uint64_t start)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    int mindeg;

    bool failed_found = false;

    // TODO - change vert is extendable to Quick check using mindeg
    mindeg = h_get_mindeg(num_mem);

    // remove one vertex
    num_cand--;
    tot_vert--;

    // initialize vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    // update info of vertices connected to removed cand
    pvertexid = read_vertices[start + tot_vert].vertexid;
    pneighbors_start = hg.onehop_offsets[pvertexid];
    pneighbors_end = hg.onehop_offsets[pvertexid + 1];
    for (int i = pneighbors_start; i < pneighbors_end; i++) {
        phelper1 = hd.vertex_order_map[hg.onehop_neighbors[i]];

        if (phelper1 > -1) {
            read_vertices[start + phelper1].exdeg--;

            if (phelper1 < num_mem && read_vertices[start + phelper1].indeg + read_vertices[start + phelper1].exdeg < mindeg) {
                failed_found = true;
                break;
            }
        }
    }

    // reset vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = -1;
    }

    if (failed_found) {
        return 1;
    }

    return 0;
}

// returns 2 if too many vertices pruned to be considered, 1 if failed found or invalid bound, 0 otherwise
int h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg)
{
    // helper variables
    bool method_return;

    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int pneighbors_count;
    int phelper1;



    // ADD ONE VERTEX
    pvertexid = vertices[number_of_members].vertexid;

    vertices[number_of_members].label = 1;
    number_of_members++;
    number_of_candidates--;

    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
    }

    pneighbors_start = hg.onehop_offsets[pvertexid];
    pneighbors_end = hg.onehop_offsets[pvertexid + 1];
    pneighbors_count = pneighbors_end - pneighbors_start;
    for (int i = 0; i < pneighbors_count; i++) {
        phelper1 = hd.vertex_order_map[hg.onehop_neighbors[pneighbors_start + i]];

        if (phelper1 > -1) {
            vertices[phelper1].indeg++;
            vertices[phelper1].exdeg--;
        }
    }



    // DIAMETER PRUNING
    h_diameter_pruning(hg, hd, vertices, pvertexid, total_vertices, number_of_candidates, number_of_members);



    // DEGREE-BASED PRUNING
    method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

    for (int i = 0; i < hg.number_of_vertices; i++) {
        hd.vertex_order_map[i] = -1;
    }

    // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
    if (method_return) {
        return 1;
    }

    return 0;
}

// returns 2 if too many vertices pruned or a critical vertex fail, returns 1 if failed found or invalid bounds, else 0
int h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int pneighbors_count;
    int phelper1;
    int phelper2;

    bool critical_fail = false;
    int number_of_removed;

    bool method_return;

    // CRITICAL VERTEX PRUNING
    set<int> critical_vertex_neighbors;
    // adj_counter[0] = 10, means that the vertex at position 0 in new_vertices has 10 critical vertices neighbors within 2 hops
    int* adj_counters = new int[total_vertices];
    memset(adj_counters, 0, sizeof(int) * total_vertices);

    // iterate through all vertices in clique
    for (int k = 0; k < number_of_members; k++)
    {
        // if they are a critical vertex
        if (vertices[k].indeg + vertices[k].exdeg == minimum_degrees[number_of_members + lower_bound] && vertices[k].exdeg > 0) {

            // iterate through all neighbors
            pvertexid = vertices[k].vertexid;
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (uint64_t l = pneighbors_start; l < pneighbors_end; l++) {
                int neighbor_id = hg.onehop_neighbors[l];
                int position_of_neighbor = h_lsearch_vert(vertices, total_vertices, neighbor_id);

                // if neighbor is cand
                if (position_of_neighbor != -1) {
                    if (vertices[position_of_neighbor].label == 0) {
                        critical_vertex_neighbors.insert(neighbor_id);
                        vertices[position_of_neighbor].label = 4;
                    }
                }
            }
        }
    }

    // if there were any neighbors of critical vertices
    if (critical_vertex_neighbors.size() > 0)
    {
        // sort vertices so that critical vertex adjacent candidates are immediately after vertices within the clique
        qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert);

        // iterate through all neighbors
        for (int neighbor : critical_vertex_neighbors) {
            // update 1hop adj
            pneighbors_start = hg.onehop_offsets[neighbor];
            pneighbors_end = hg.onehop_offsets[neighbor + 1];
            for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {
                int neighbor_of_added_vertex = hg.onehop_neighbors[k];
                int position_of_neighbor = h_lsearch_vert(vertices, total_vertices, neighbor_of_added_vertex);
                if (position_of_neighbor != -1) {
                    vertices[position_of_neighbor].indeg++;
                    vertices[position_of_neighbor].exdeg--;
                }
            }

            // track 2hop adj
            pneighbors_start = hg.twohop_offsets[neighbor];
            pneighbors_end = hg.twohop_offsets[neighbor + 1];
            for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {
                int neighbor_of_added_vertex = hg.twohop_neighbors[k];
                int position_of_neighbor = h_lsearch_vert(vertices, total_vertices, neighbor_of_added_vertex);
                if (position_of_neighbor != -1) {
                    adj_counters[position_of_neighbor]++;
                }
            }
        }

        critical_fail = false;

        // all vertices within the clique must be within 2hops of the newly added critical vertex adj vertices
        for (int k = 0; k < number_of_members; k++) {
            if (adj_counters[k] != critical_vertex_neighbors.size()) {
                critical_fail = true;
            }
        }

        // all critical adj vertices must all be within 2 hops of each other
        for (int k = number_of_members; k < number_of_members + critical_vertex_neighbors.size(); k++) {
            if (adj_counters[k] < critical_vertex_neighbors.size() - 1) {
                critical_fail = true;
            }
        }

        if (critical_fail) {
            delete adj_counters;
            return 2;
        }

        // no failed vertices found so add all critical vertex adj candidates to clique
        for (int k = number_of_members; k < number_of_members + critical_vertex_neighbors.size(); k++) {
            vertices[k].label = 1;
        }
        number_of_members += critical_vertex_neighbors.size();
        number_of_candidates -= critical_vertex_neighbors.size();
        qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert);
    }



    // DIAMTER PRUNING
    number_of_removed = 0;
    // remove all cands who are not within 2hops of all newly added cands
    for (int k = number_of_members; k < total_vertices; k++) {
        if (adj_counters[k] != critical_vertex_neighbors.size()) {
            vertices[k].label = -1;
            number_of_removed++;
        }
    }
    qsort(vertices, total_vertices, sizeof(Vertex), h_sort_vert);

    delete adj_counters;

    // update exdeg of vertices connected to removed cands
    h_update_degrees(hg, vertices, total_vertices, number_of_removed);

    total_vertices -= number_of_removed;
    number_of_candidates -= number_of_removed;

    // continue if not enough vertices after pruning
    if (total_vertices < minimum_clique_size) {
        return 2;
    }



    // DEGREE-BASED PRUNING
    method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg);

    // continue if not enough vertices after pruning
    if (total_vertices < minimum_clique_size) {
        return 2;
    }

    // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
    if (method_return) {
        return 1;
    }

    return 0;
}

void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, int& total_vertices, int& number_of_candidates, int number_of_members)
{
    // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    (*hd.remaining_count) = 0;

    for (int i = number_of_members; i < total_vertices; i++) {
        vertices[i].label = -1;
    }

    pneighbors_start = hg.twohop_offsets[pvertexid];
    pneighbors_end = hg.twohop_offsets[pvertexid + 1];
    for (int i = pneighbors_start; i < pneighbors_end; i++) {
        phelper1 = hd.vertex_order_map[hg.twohop_neighbors[i]];

        if (phelper1 >= number_of_members) {
            vertices[phelper1].label = 0;
            hd.candidate_indegs[(*hd.remaining_count)++] = vertices[phelper1].indeg;
        }
    }
}

// returns true is invalid bounds calculated or a failed vertex was found, else false
bool h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;

    // helper variables
    int num_val_cands;

    qsort(hd.candidate_indegs, (*hd.remaining_count), sizeof(int), h_sort_desc);

    // if invalid bounds found while calculating lower and upper bounds
    if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, (*hd.remaining_count))) {
        return true;
    }

    // check for failed vertices
    for (int k = 0; k < number_of_members; k++) {
        if (!h_vert_isextendable_LU(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
            return true;
        }
    }

    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // check for invalid candidates
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 0 && h_cand_isvalid_LU(vertices[i], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
            hd.remaining_candidates[(*hd.remaining_count)++] = i;
        }
        else {
            hd.removed_candidates[(*hd.removed_count)++] = i;
        }
    }

    while ((*hd.remaining_count) > 0 && (*hd.removed_count) > 0) {
        // update degrees
        if ((*hd.remaining_count) < (*hd.removed_count)) {
            // reset exdegs
            for (int i = 0; i < total_vertices; i++) {
                vertices[i].exdeg = 0;
            }

            for (int i = 0; i < (*hd.remaining_count); i++) {
                pvertexid = vertices[hd.remaining_candidates[i]].vertexid;
                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (int j = pneighbors_start; j < pneighbors_end; j++) {
                    phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].exdeg++;
                    }
                }
            }
        }
        else {
            for (int i = 0; i < (*hd.removed_count); i++) {
                pvertexid = vertices[hd.removed_candidates[i]].vertexid;
                pneighbors_start = hg.onehop_offsets[pvertexid];
                pneighbors_end = hg.onehop_offsets[pvertexid + 1];
                for (int j = pneighbors_start; j < pneighbors_end; j++) {
                    phelper1 = hd.vertex_order_map[hg.onehop_neighbors[j]];

                    if (phelper1 > -1) {
                        vertices[phelper1].exdeg--;
                    }
                }
            }
        }

        num_val_cands = 0;

        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid_LU(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
                hd.candidate_indegs[num_val_cands++] = vertices[hd.remaining_candidates[k]].indeg;
            }
        }

        qsort(hd.candidate_indegs, num_val_cands, sizeof(int), h_sort_desc);

        // if invalid bounds found while calculating lower and upper bounds
        if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, num_val_cands)) {
            return true;
        }

        // check for failed vertices
        for (int k = 0; k < number_of_members; k++) {
            if (!h_vert_isextendable_LU(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
                return true;
            }
        }

        num_val_cands = 0;
        (*hd.removed_count) = 0;

        // check for invalid candidates
        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid_LU(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg)) {
                hd.remaining_candidates[num_val_cands++] = hd.remaining_candidates[k];
            }
            else {
                hd.removed_candidates[(*hd.removed_count)++] = hd.remaining_candidates[k];
            }
        }

        (*hd.remaining_count) = num_val_cands;
    }

    for (int i = 0; i < (*hd.remaining_count); i++) {
        vertices[number_of_members + i] = vertices[hd.remaining_candidates[i]];
    }

    total_vertices = total_vertices - number_of_candidates + (*hd.remaining_count);
    number_of_candidates = (*hd.remaining_count);

    return false;
}

bool h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates)
{
    bool invalid_bounds = false;
    int index;

    int sum_candidate_indeg = 0;
    int tightened_upper_bound = 0;

    int min_clq_indeg = vertices[0].indeg;
    int min_indeg_exdeg = vertices[0].exdeg;
    int min_clq_totaldeg = vertices[0].indeg + vertices[0].exdeg;
    int sum_clq_indeg = vertices[0].indeg;

    for (index = 1; index < number_of_members; index++) {
        sum_clq_indeg += vertices[index].indeg;

        if (vertices[index].indeg < min_clq_indeg) {
            min_clq_indeg = vertices[index].indeg;
            min_indeg_exdeg = vertices[index].exdeg;
        }
        else if (vertices[index].indeg == min_clq_indeg) {
            if (vertices[index].exdeg < min_indeg_exdeg) {
                min_indeg_exdeg = vertices[index].exdeg;
            }
        }

        if (vertices[index].indeg + vertices[index].exdeg < min_clq_totaldeg) {
            min_clq_totaldeg = vertices[index].indeg + vertices[index].exdeg;
        }
    }

    min_ext_deg = h_get_mindeg(number_of_members + 1);

    if (min_clq_indeg < minimum_degrees[number_of_members])
    {
        // lower
        lower_bound = h_get_mindeg(number_of_members) - min_clq_indeg;

        while (lower_bound <= min_indeg_exdeg && min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound]) {
            lower_bound++;
        }

        if (min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound]) {
            lower_bound = number_of_candidates + 1;
            invalid_bounds = true;
        }

        // upper
        upper_bound = floor(min_clq_totaldeg / minimum_degree_ratio) + 1 - number_of_members;

        if (upper_bound > number_of_candidates) {
            upper_bound = number_of_candidates;
        }

        // tighten
        if (lower_bound < upper_bound) {
            // tighten lower
            for (index = 0; index < lower_bound; index++) {
                sum_candidate_indeg += hd.candidate_indegs[index];
            }

            while (index < upper_bound && sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index]) {
                sum_candidate_indeg += hd.candidate_indegs[index];
                index++;
            }

            if (sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index]) {
                lower_bound = upper_bound + 1;
                invalid_bounds = true;
            }
            else {
                lower_bound = index;

                tightened_upper_bound = index;

                while (index < upper_bound) {
                    sum_candidate_indeg += hd.candidate_indegs[index];

                    index++;

                    if (sum_clq_indeg + sum_candidate_indeg >= number_of_members * minimum_degrees[number_of_members + index]) {
                        tightened_upper_bound = index;
                    }
                }

                if (upper_bound > tightened_upper_bound) {
                    upper_bound = tightened_upper_bound;
                }

                if (lower_bound > 1) {
                    min_ext_deg = h_get_mindeg(number_of_members + lower_bound);
                }
            }
        }
    }
    else {
        upper_bound = number_of_candidates;

        if (number_of_members < minimum_clique_size) {
            lower_bound = minimum_clique_size - number_of_members;
        }
        else {
            lower_bound = 0;
        }
    }

    if (number_of_members + upper_bound < minimum_clique_size) {
        invalid_bounds = true;
    }

    if (upper_bound < 0 || upper_bound < lower_bound) {
        invalid_bounds = true;
    }

    return invalid_bounds;
}

bool h_calculate_LU_bounds_old(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates)
{
    bool invalid_bounds = false;
    int index;

    int sum_candidate_indeg = 0;
    int tightened_upper_bound = 0;

    int min_clq_indeg = vertices[0].indeg;
    int min_indeg_exdeg = vertices[0].exdeg;
    int min_clq_totaldeg = vertices[0].indeg + vertices[0].exdeg;
    int sum_clq_indeg = vertices[0].indeg;

    for (index = 1; index < number_of_members; index++) {
        sum_clq_indeg += vertices[index].indeg;

        if (vertices[index].indeg < min_clq_indeg) {
            min_clq_indeg = vertices[index].indeg;
            min_indeg_exdeg = vertices[index].exdeg;
        }
        else if (vertices[index].indeg == min_clq_indeg) {
            if (vertices[index].exdeg < min_indeg_exdeg) {
                min_indeg_exdeg = vertices[index].exdeg;
            }
        }

        if (vertices[index].indeg + vertices[index].exdeg < min_clq_totaldeg) {
            min_clq_totaldeg = vertices[index].indeg + vertices[index].exdeg;
        }
    }

    min_ext_deg = h_get_mindeg(number_of_members + 1);

    if (min_clq_indeg < minimum_degrees[number_of_members])
    {
        // lower
        lower_bound = h_get_mindeg(number_of_members) - min_clq_indeg;

        while (lower_bound <= min_indeg_exdeg && min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound]) {
            lower_bound++;
        }

        if (min_clq_indeg + lower_bound < minimum_degrees[number_of_members + lower_bound]) {
            lower_bound = number_of_candidates + 1;
            invalid_bounds = true;
        }

        // upper
        upper_bound = floor(min_clq_totaldeg / minimum_degree_ratio) + 1 - number_of_members;

        if (upper_bound > number_of_candidates) {
            upper_bound = number_of_candidates;
        }

        // tighten
        if (lower_bound < upper_bound) {
            // tighten lower
            for (index = 0; index < lower_bound; index++) {
                sum_candidate_indeg += vertices[number_of_members + index].indeg;
            }

            while (index < upper_bound && sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index]) {
                sum_candidate_indeg += vertices[number_of_members + index].indeg;
                index++;
            }

            if (sum_clq_indeg + sum_candidate_indeg < number_of_members * minimum_degrees[number_of_members + index]) {
                lower_bound = upper_bound + 1;
                invalid_bounds = true;
            }
            else {
                lower_bound = index;

                tightened_upper_bound = index;

                while (index < upper_bound) {
                    sum_candidate_indeg += vertices[number_of_members + index].indeg;

                    index++;

                    if (sum_clq_indeg + sum_candidate_indeg >= number_of_members * minimum_degrees[number_of_members + index]) {
                        tightened_upper_bound = index;
                    }
                }

                if (upper_bound > tightened_upper_bound) {
                    upper_bound = tightened_upper_bound;
                }

                if (lower_bound > 1) {
                    min_ext_deg = h_get_mindeg(number_of_members + lower_bound);
                }
            }
        }
    }
    else {
        upper_bound = number_of_candidates;

        if (number_of_members < minimum_clique_size) {
            lower_bound = minimum_clique_size - number_of_members;
        }
        else {
            lower_bound = 0;
        }
    }

    if (number_of_members + upper_bound < minimum_clique_size) {
        invalid_bounds = true;
    }

    if (upper_bound < 0 || upper_bound < lower_bound) {
        invalid_bounds = true;
    }

    return invalid_bounds;
}

void h_update_degrees(CPU_Graph& hg, Vertex* vertices, int total_vertices, int number_of_removed)
{
    // intersection
    int pvertexid;
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int pneighbors_count;
    int phelper1;
    int phelper2;

    // update exdeg of vertices connected to removed cands
    for (int i = 0; i < total_vertices - number_of_removed; i++) {
        pvertexid = vertices[i].vertexid;
        for (int j = total_vertices - number_of_removed; j < total_vertices; j++) {
            phelper1 = vertices[j].vertexid;
            pneighbors_start = hg.onehop_offsets[phelper1];
            pneighbors_end = hg.onehop_offsets[phelper1 + 1];
            pneighbors_count = pneighbors_end - pneighbors_start;
            phelper2 = h_bsearch_array(hg.onehop_neighbors + pneighbors_start, pneighbors_count, pvertexid);
            if (phelper2 != -1) {
                vertices[i].exdeg--;
            }
        }
    }
}

void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members)
{
    bool clique = true;

    int degree_requirement = minimum_degrees[number_of_members];
    for (int k = 0; k < number_of_members; k++) {
        if (vertices[k].indeg < degree_requirement) {
            clique = false;
            break;
        }
    }

    // if clique write to cliques array
    if (clique) {
        uint64_t start_write = hc.cliques_offset[(*hc.cliques_count)];
        for (int k = 0; k < number_of_members; k++) {
            hc.cliques_vertex[start_write + k] = vertices[k].vertexid;
        }
        (*hc.cliques_count)++;
        hc.cliques_offset[(*hc.cliques_count)] = start_write + number_of_members;
    }
}

void h_write_to_tasks(CPU_Data& hd, Vertex* vertices, int total_vertices, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count)
{
    (*hd.maximal_expansion) = false;

    if ((*write_count) < EXPAND_THRESHOLD) {
        uint64_t start_write = write_offsets[*write_count];

        for (int k = 0; k < total_vertices; k++) {
            write_vertices[start_write + k].vertexid = vertices[k].vertexid;
            write_vertices[start_write + k].label = vertices[k].label;
            write_vertices[start_write + k].indeg = vertices[k].indeg;
            write_vertices[start_write + k].exdeg = vertices[k].exdeg;
            write_vertices[start_write + k].lvl2adj = 0;
        }
        (*write_count)++;
        write_offsets[*write_count] = start_write + total_vertices;
    }
    else {
        uint64_t start_write = hd.buffer_offset[(*hd.buffer_count)];

        for (int k = 0; k < total_vertices; k++) {
            hd.buffer_vertices[start_write + k].vertexid = vertices[k].vertexid;
            hd.buffer_vertices[start_write + k].label = vertices[k].label;
            hd.buffer_vertices[start_write + k].indeg = vertices[k].indeg;
            hd.buffer_vertices[start_write + k].exdeg = vertices[k].exdeg;
            hd.buffer_vertices[start_write + k].lvl2adj = 0;
        }
        (*hd.buffer_count)++;
        hd.buffer_offset[(*hd.buffer_count)] = start_write + total_vertices;
    }
}

void h_fill_from_buffer(CPU_Data& hd, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count)
{
    // read from end of buffer, write to end of tasks, decrement buffer
    (*hd.maximal_expansion) = false;

    // get read and write locations
    int write_amount = ((*hd.buffer_count) >= (EXPAND_THRESHOLD - *write_count)) ? EXPAND_THRESHOLD - *write_count : (*hd.buffer_count);
    uint64_t start_buffer = hd.buffer_offset[(*hd.buffer_count) - write_amount];
    uint64_t end_buffer = hd.buffer_offset[(*hd.buffer_count)];
    uint64_t size_buffer = end_buffer - start_buffer;
    uint64_t start_write = write_offsets[*write_count];

    // copy tasks data from end of buffer to end of tasks
    memcpy(&write_vertices[start_write], &hd.buffer_vertices[start_buffer], sizeof(Vertex) * size_buffer);

    // handle offsets
    for (int j = 1; j <= write_amount; j++) {
        write_offsets[*write_count + j] = start_write + (hd.buffer_offset[(*hd.buffer_count) - write_amount + j] - start_buffer);
    }

    // update counts
    (*write_count) += write_amount;
    (*hd.buffer_count) -= write_amount;
}



// --- HELPER METHODS ---

// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
int h_bsearch_array(int* search_array, int array_size, int search_number)
{
    // ALGO - binary
    // TYPE - serial
    // SPEED - 0(log(n))

    int low = 0;
    int high = array_size - 1;

    while (low <= high) {
        int mid = (low + high) / 2;

        if (search_array[mid] == search_number) {
            return mid;
        }
        else if (search_array[mid] > search_number) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }

    return -1;
}

int h_lsearch_vert(Vertex* search_array, int array_size, int search_vertexid) {
    for (int i = 0; i < array_size; i++) {
        if (search_array[i].vertexid == search_vertexid) {
            return i;
        }
    }
    return -1;
}

int h_sort_vert(const void* a, const void* b)
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

    // for ties: in cand low -> high

    else if ((*(Vertex*)a).label == 0 && (*(Vertex*)b).label == 0) {
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
    else if ((*(Vertex*)a).label == 2 && (*(Vertex*)b).label == 2) {
        return 0;
    }
    else if ((*(Vertex*)a).label == -1 && (*(Vertex*)b).label == -1) {
        return 0;
    }
    return 0;
}

// update how this method looks
int h_sort_vert_Q(const void* a, const void* b)
{
    // order is: covered -> cands -> cover
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid
    
    Vertex* v1;
    Vertex* v2;

    v1 = (Vertex*)a;
    v2 = (Vertex*)b;

    if (v1->label == 2 && v2->label != 2)
        return -1;
    else if (v1->label != 2 && v2->label == 2)
        return 1;
    else if (v1->label == 0 && v2->label != 0)
        return -1;
    else if (v1->label != 0 && v2->label == 0)
        return 1;
    else if (v1->label == 3 && v2->label != 3)
        return -1;
    else if (v1->label != 3 && v2->label == 3)
        return 1;
    else if (v1->label == 0 && v2->label != 0)
        return -1;
    else if (v1->indeg > v2->indeg)
        return -1;
    else if (v1->indeg < v2->indeg)
        return 1;
    else if (v1->exdeg > v2->exdeg)
        return -1;
    else if (v1->exdeg < v2->exdeg)
        return 1;
    else if (v1->lvl2adj > v2->lvl2adj)
        return -1;
    else if (v1->lvl2adj < v2->lvl2adj)
        return 1;
    else if (v1->vertexid > v2->vertexid)
        return -1;
    else if (v1->vertexid < v2->vertexid)
        return 1;
    else
        return 0;
}

int h_sort_vert_LU(const void* a, const void* b)
{
    // order is: in clique -> covered -> crtical adjacent -> cands -> cover -> pruned

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

    // for ties: in clique low -> high, cand high indeg -> low indeg
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
        if ((*(Vertex*)a).indeg > (*(Vertex*)b).indeg) {
            return -1;
        }
        else if ((*(Vertex*)a).indeg < (*(Vertex*)b).indeg) {
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

// sorts degrees in descending order
int h_sort_desc(const void* a, const void* b) 
{
    int n1;
    int n2;

    n1 = *(int*)a;
    n2 = *(int*)b;

    if (n1 > n2) {
        return -1;
    }
    else if (n1 < n2) {
        return 1;
    }
    else {
        return 0;
    }
}

// sorts degrees in ascending order
int h_sort_asce(const void* a, const void* b)
{
    int n1;
    int n2;

    n1 = *(int*)a;
    n2 = *(int*)b;

    if (n1 < n2) {
        return -1;
    }
    else if (n1 > n2) {
        return 1;
    }
    else {
        return 0;
    }
}

inline int h_get_mindeg(int clique_size) {
    if (clique_size < minimum_clique_size) {
        return minimum_degrees[minimum_clique_size];
    }
    else {
        return minimum_degrees[clique_size];
    }
}

inline bool h_cand_isvalid(Vertex vertex, int clique_size) {
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + vertex.exdeg + 1)) {
        return false;
    }
    else {
        return true;
    }
}

inline bool h_cand_isvalid_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg) 
{
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + vertex.exdeg + 1)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < min_ext_deg) {
        return false;
    }
    else if (vertex.indeg + upper_bound - 1 < minimum_degrees[clique_size + lower_bound]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + lower_bound)) {
        return false;
    }
    else {
        return true;
    }
}

inline bool h_vert_isextendable(Vertex vertex, int clique_size) 
{
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + vertex.exdeg)) {
        return false;
    }
    else {
        return true;
    }
}

inline bool h_vert_isextendable_LU(Vertex vertex, int clique_size, int upper_bound, int lower_bound, int min_ext_deg)
{
    if (vertex.indeg + vertex.exdeg < minimum_degrees[minimum_clique_size]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + vertex.exdeg)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < min_ext_deg) {
        return false;
    }
    else if (vertex.exdeg == 0 && vertex.indeg < h_get_mindeg(clique_size + vertex.exdeg)) {
        return false;
    }
    else if (vertex.indeg + upper_bound < minimum_degrees[clique_size + upper_bound]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < h_get_mindeg(clique_size + lower_bound)) {
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

void print_CPU_Graph(CPU_Graph& hg) {
    cout << endl << " --- (CPU_Graph)host_graph details --- " << endl;
    cout << endl << "|V|: " << hg.number_of_vertices << " |E|: " << hg.number_of_edges << endl;
    cout << endl << "Onehop Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.onehop_offsets[i] << " ";
    }
    cout << endl << "Onehop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_edges * 2; i++) {
        cout << hg.onehop_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= hg.number_of_vertices; i++) {
        cout << hg.twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
        cout << hg.twohop_neighbors[i] << " ";
    }
    cout << endl << endl;
}

void print_GPU_Graph(GPU_Data& dd, CPU_Graph& hg)
{
    int* number_of_vertices = new int;
    int* number_of_edges = new int;

    int* onehop_neighbors = new int[hg.number_of_edges * 2];
    uint64_t * onehop_offsets = new uint64_t[(hg.number_of_vertices)+1];
    int* twohop_neighbors = new int[hg.number_of_lvl2adj];
    uint64_t * twohop_offsets = new uint64_t[(hg.number_of_vertices)+1];

    chkerr(cudaMemcpy(number_of_vertices, dd.number_of_vertices, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(number_of_edges, dd.number_of_edges, sizeof(int), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_neighbors, dd.onehop_neighbors, sizeof(int)*hg.number_of_edges * 2, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(onehop_offsets, dd.onehop_offsets, sizeof(uint64_t)*(hg.number_of_vertices+1), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_neighbors, dd.twohop_neighbors, sizeof(int)*hg.number_of_lvl2adj, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(twohop_offsets, dd.twohop_offsets, sizeof(uint64_t)*(hg.number_of_vertices+1), cudaMemcpyDeviceToHost));

    cout << endl << " --- (GPU_Graph)device_graph details --- " << endl;
    cout << endl << "|V|: " << (*number_of_vertices) << " |E|: " << (*number_of_edges) << endl;
    cout << endl << "Onehop Offsets:" << endl;
    for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
        cout << onehop_offsets[i] << " ";
    }
    cout << endl << "Onehop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_edges * 2; i++) {
        cout << onehop_neighbors[i] << " ";
    }
    cout << endl << "Twohop Offsets:" << endl;
    for (uint64_t i = 0; i <= (*number_of_vertices); i++) {
        cout << twohop_offsets[i] << " ";
    }
    cout << endl << "Twohop Neighbors:" << endl;
    for (uint64_t i = 0; i < hg.number_of_lvl2adj; i++) {
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

void print_CPU_Data(CPU_Data& hd)
{
    cout << endl << " --- (CPU_Data)host_data details --- " << endl;
    cout << endl << "Tasks1: " << "Size: " << (*(hd.tasks1_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.tasks1_count)); i++) {
        cout << hd.tasks1_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.tasks1_offset[(*(hd.tasks1_count))]; i++) {
        cout << hd.tasks1_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Tasks2: " << "Size: " << (*(hd.tasks2_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.tasks2_count)); i++) {
        cout << hd.tasks2_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.tasks2_offset[(*(hd.tasks2_count))]; i++) {
        cout << hd.tasks2_vertices[i].lvl2adj << " ";
    }

    cout << endl << endl << "Buffer: " << "Size: " << (*(hd.buffer_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hd.buffer_count)); i++) {
        cout << hd.buffer_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].vertexid << " ";
    }
    cout << endl << "Label:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].label << " ";
    }
    cout << endl << "Indeg:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].indeg << " ";
    }
    cout << endl << "Exdeg:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].exdeg << " ";
    }
    cout << endl << "Lvl2adj:" << endl;
    for (uint64_t i = 0; i < hd.buffer_offset[(*(hd.buffer_count))]; i++) {
        cout << hd.buffer_vertices[i].lvl2adj << " ";
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

// returns true if warp buffer was too small causing error
bool print_Warp_Data_Sizes(GPU_Data& dd)
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

    if (tasks_mcount > WTASKS_OFFSET_SIZE || tasks_msize > WTASKS_SIZE || cliques_mcount > WCLIQUES_OFFSET_SIZE || cliques_msize > WCLIQUES_SIZE) {
        cout << "!!! WBUFFER SIZE ERROR !!!" << endl;
        return true;
    }

    delete tasks_counts;
    delete tasks_sizes;
    delete cliques_counts;
    delete cliques_sizes;

    return false;
}

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

bool print_Warp_Data_Sizes_Every(GPU_Data& dd, int every)
{
    bool result;
    int level;
    chkerr(cudaMemcpy(&level, dd.current_level, sizeof(int), cudaMemcpyDeviceToHost));
    if (level % every == 0) {
        result = print_Warp_Data_Sizes(dd);
    }
    return result;
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

void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc)
{
    cout << "L: " << (*hd.current_level) << " T1: " << (*hd.tasks1_count) << " " << (*(hd.tasks1_offset + (*hd.tasks1_count))) << " T2: " << (*hd.tasks2_count) << " " << 
        (*(hd.tasks2_offset + (*hd.tasks2_count))) << " B: " << (*hd.buffer_count) << " " << (*(hd.buffer_offset + (*hd.buffer_count))) << " C: " << 
        (*hc.cliques_count) << " " << (*(hc.cliques_offset + (*hc.cliques_count))) << endl;
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

void print_CPU_Cliques(CPU_Cliques& hc)
{
    cout << endl << " --- (CPU_Cliques)host_cliques details --- " << endl;
    cout << endl << "Cliques: " << "Size: " << (*(hc.cliques_count)) << endl;
    cout << endl << "Offsets:" << endl;
    for (uint64_t i = 0; i <= (*(hc.cliques_count)); i++) {
        cout << hc.cliques_offset[i] << " ";
    }
    cout << endl << "Vertex:" << endl;
    for (uint64_t i = 0; i < hc.cliques_offset[(*(hc.cliques_count))]; i++) {
        cout << hc.cliques_vertex[i] << " ";
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

__global__ void d_expand_level(GPU_Data dd)
{
    // data is stored in data structures to reduce the number of variables that need to be passed to methods
    __shared__ Warp_Data wd;
    Local_Data ld;

    // helper variables, not passed through to any methods
    int method_return;
    int index;

    // initialize variables
    ld.idx = (blockIdx.x * blockDim.x + threadIdx.x);
    ld.wib_idx = ((ld.idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE));

    /*
    * The program alternates between reading and writing between to 'tasks' arrays in device global memory. The program will read from one tasks, expand to the next level by generating and pruning, then it will write to the
    * other tasks array. It will write the first EXPAND_THRESHOLD to the tasks array and the rest to the top of the buffer. The buffers acts as a stack containing the excess data not being expanded from tasks. Since the 
    * buffer acts as a stack, in a last-in first-out manner, a subsection of the search space will be expanded until completion. This system allows the problem to essentially be divided into smaller problems and thus 
    * require less memory to handle.
    */
    if ((*(dd.current_level)) % 2 == 0) {
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
            wd.start[ld.wib_idx] = ld.read_offsets[i];
            wd.end[ld.wib_idx] = ld.read_offsets[i + 1];
            wd.tot_vert[ld.wib_idx] = wd.end[ld.wib_idx] - wd.start[ld.wib_idx];
            wd.num_mem[ld.wib_idx] = 0;
            for (uint64_t j = wd.start[ld.wib_idx]; j < wd.end[ld.wib_idx]; j++) {
                if (ld.read_vertices[j].label == 1) {
                    wd.num_mem[ld.wib_idx]++;
                } else {
                    break;
                }
            }
            wd.num_cand[ld.wib_idx] = wd.tot_vert[ld.wib_idx] - wd.num_mem[ld.wib_idx];
            wd.expansions[ld.wib_idx] = wd.num_cand[ld.wib_idx];
        }
        __syncwarp();



        // LOOKAHEAD PRUNING
        method_return = d_lookahead_pruning(dd, wd, ld);
        if (method_return) {
            continue;
        }



        // --- NEXT LEVEL ---
        for (int j = 0; j < wd.expansions[ld.wib_idx]; j++)
        {



            // REMOVE ONE VERTEX
            if (j > 0) {
                method_return = d_remove_one_vertex(dd, wd, ld);
                if (method_return) {
                    break;
                }
            }



            // INITIALIZE NEW VERTICES
            if ((ld.idx % WARP_SIZE) == 0) {
                wd.number_of_members[ld.wib_idx] = wd.num_mem[ld.wib_idx];
                wd.number_of_candidates[ld.wib_idx] = wd.num_cand[ld.wib_idx];
                wd.total_vertices[ld.wib_idx] = wd.tot_vert[ld.wib_idx];
            }
            __syncwarp();

            // select whether to store vertices in global or shared memory based on size
            if (wd.total_vertices[ld.wib_idx] <= VERTICES_SIZE) {
                ld.vertices = wd.shared_vertices + (VERTICES_SIZE * ld.wib_idx);
            } else {
                ld.vertices = dd.global_vertices + (WVERTICES_SIZE * (ld.idx / WARP_SIZE));
            }

            for (index = (ld.idx % WARP_SIZE); index < wd.number_of_members[ld.wib_idx]; index += WARP_SIZE) {
                ld.vertices[index] = ld.read_vertices[wd.start[ld.wib_idx] + index];
            }
            for (; index < wd.total_vertices[ld.wib_idx] - 1; index += WARP_SIZE) {
                ld.vertices[index + 1] = ld.read_vertices[wd.start[ld.wib_idx] + index];
            }

            if ((ld.idx % WARP_SIZE) == 0) {
                ld.vertices[wd.number_of_members[ld.wib_idx]] = ld.read_vertices[wd.start[ld.wib_idx] + wd.total_vertices[ld.wib_idx] - 1];
            }
            __syncwarp();

            

            // ADD ONE VERTEX
            method_return = d_add_one_vertex(dd, wd, ld);



            // HANDLE CLIQUES
            if (wd.number_of_members[ld.wib_idx] >= (*dd.minimum_clique_size)) {
                d_check_for_clique(dd, wd, ld);
            }

            // if vertex in x found as not extendable continue to next iteration
            if (method_return == 1) {
                continue;
            }



            // WRITE TASKS TO BUFFERS
            // sort vertices in Quick efficient enumeration order before writing
            d_sort(ld.vertices, wd.total_vertices[ld.wib_idx], (ld.idx% WARP_SIZE), d_sort_vert_Q);

            // TODO - do we need this if?
            if (wd.number_of_candidates[ld.wib_idx] > 0) {
                d_write_to_tasks(dd, wd, ld);
            }
        }
    }



    if ((ld.idx % WARP_SIZE) == 0) {
        // sum to find tasks count
        atomicAdd(dd.total_tasks, dd.wtasks_count[(ld.idx / WARP_SIZE)]);
        atomicAdd(dd.total_cliques, dd.wcliques_count[(ld.idx / WARP_SIZE)]);
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
    int wib_idx = ((idx / WARP_SIZE) % (BLOCK_SIZE / WARP_SIZE));

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

    if ((*(dd.current_level)) % 2 == 0) {
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
        tasks_write[wib_idx] = 0;
        tasks_offset_write[wib_idx] = 1;
        cliques_write[wib_idx] = 0;
        cliques_offset_write[wib_idx] = 1;

        for (int i = 0; i < (idx / WARP_SIZE); i++) {
            tasks_offset_write[wib_idx] += dd.wtasks_count[i];
            tasks_write[wib_idx] += dd.wtasks_offset[(WTASKS_OFFSET_SIZE * i) + dd.wtasks_count[i]];

            cliques_offset_write[wib_idx] += dd.wcliques_count[i];
            cliques_write[wib_idx] += dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * i) + dd.wcliques_count[i]];
        }
    }
    __syncwarp();
    
    // move to tasks and buffer
    for (int i = (idx % WARP_SIZE) + 1; i <= dd.wtasks_count[(idx / WARP_SIZE)]; i += WARP_SIZE)
    {
        if (tasks_offset_write[wib_idx] + i - 1 <= EXPAND_THRESHOLD) {
            // to tasks
            write_offsets[tasks_offset_write[wib_idx] + i - 1] = dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + tasks_write[wib_idx];
        }
        else {
            // to buffer
            dd.buffer_offset[tasks_offset_write[wib_idx] + i - 2 - EXPAND_THRESHOLD + (*(dd.buffer_offset_start))] = dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + i] +
                tasks_write[wib_idx] - tasks_end + (*(dd.buffer_start));
        }
    }

    for (int i = (idx % WARP_SIZE); i < dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (idx / WARP_SIZE)) + dd.wtasks_count[(idx / WARP_SIZE)]]; i += WARP_SIZE) {
        if (tasks_write[wib_idx] + i < tasks_end) {
            // to tasks
            write_vertices[tasks_write[wib_idx] + i] = dd.wtasks_vertices[(WTASKS_SIZE * (idx / WARP_SIZE)) + i];
        }
        else {
            // to buffer
            dd.buffer_vertices[(*(dd.buffer_start)) + tasks_write[wib_idx] + i - tasks_end] = dd.wtasks_vertices[(WTASKS_SIZE * (idx / WARP_SIZE)) + i];
        }
    }

    //move to cliques
    for (int i = (idx % WARP_SIZE) + 1; i <= dd.wcliques_count[(idx / WARP_SIZE)]; i += WARP_SIZE) {
        dd.cliques_offset[(*(dd.cliques_offset_start)) + cliques_offset_write[wib_idx] + i - 2] = dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (idx / WARP_SIZE)) + i] + (*(dd.cliques_start)) + 
            cliques_write[wib_idx];
    }
    for (int i = (idx % WARP_SIZE); i < dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (idx / WARP_SIZE)) + dd.wcliques_count[(idx / WARP_SIZE)]]; i += WARP_SIZE) {
        dd.cliques_vertex[(*(dd.cliques_start)) + cliques_write[wib_idx] + i] = dd.wcliques_vertex[(WCLIQUES_SIZE * (idx / WARP_SIZE)) + i];
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
    if ((idx % WARP_SIZE) == 0 && cliques_write[wib_idx] > (CLIQUES_SIZE * (CLIQUES_PERCENT / 100.0))) {
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

    if ((*(dd.current_level)) % 2 == 0) {
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



// --- DEVICE EXPANSION KERNELS ---

// returns 1 if lookahead succesful, 0 otherwise 
__device__ int d_lookahead_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld) 
{
    bool lookahead_sucess;
    int pvertexid;
    int phelper1;
    int phelper2;

    lookahead_sucess = true;
    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning guarentees all members will be within 2hops of eveything
    for (int i = (ld.idx % WARP_SIZE); i < wd.num_mem[ld.wib_idx]; i += WARP_SIZE) {
        if (ld.read_vertices[wd.start[ld.wib_idx] + i].indeg + ld.read_vertices[wd.start[ld.wib_idx] + i].exdeg < dd.minimum_degrees[wd.tot_vert[ld.wib_idx]]) {
            lookahead_sucess = false;
            break;
        }
    }

    lookahead_sucess = !(__any_sync(0xFFFFFFFF, !lookahead_sucess));
    if (!lookahead_sucess) {
        return 0;
    }

    // update lvl2adj to candidates for all vertices
    for (int i = wd.num_mem[ld.wib_idx] + (ld.idx % WARP_SIZE); i < wd.tot_vert[ld.wib_idx]; i += WARP_SIZE) {
        pvertexid = ld.read_vertices[wd.start[ld.wib_idx] + i].vertexid;
        
        for (int j = wd.num_mem[ld.wib_idx]; j < wd.tot_vert[ld.wib_idx]; j++) {
            if (j == i) {
                continue;
            }

            phelper1 = ld.read_vertices[wd.start[ld.wib_idx] + j].vertexid;
            phelper2 = d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[phelper1], dd.twohop_offsets[phelper1 + 1] - dd.twohop_offsets[phelper1], pvertexid);
        
            if (phelper2 > -1) {
                ld.read_vertices[wd.start[ld.wib_idx] + i].lvl2adj++;
            }
        }
    }
    __syncwarp();

    // compares all vertices to the lemmas from Quick
    for (int j = wd.num_mem[ld.wib_idx] + (ld.idx % WARP_SIZE); j < wd.tot_vert[ld.wib_idx]; j += WARP_SIZE) {
        if (ld.read_vertices[wd.start[ld.wib_idx] + j].lvl2adj < wd.num_cand[ld.wib_idx] - 1 || ld.read_vertices[wd.start[ld.wib_idx] + j].indeg + ld.read_vertices[wd.start[ld.wib_idx] + j].exdeg < dd.minimum_degrees[wd.tot_vert[ld.wib_idx]]) {
            lookahead_sucess = false;
            break;
        }
    }
    lookahead_sucess = !(__any_sync(0xFFFFFFFF, !lookahead_sucess));

    if (lookahead_sucess) {
        // write to cliques
        uint64_t start_write = (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) + dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])];
        for (int j = (ld.idx % WARP_SIZE); j < wd.tot_vert[ld.wib_idx]; j += WARP_SIZE) {
            dd.wcliques_vertex[start_write + j] = ld.read_vertices[wd.start[ld.wib_idx] + j].vertexid;
        }
        if ((ld.idx % WARP_SIZE) == 0) {
            (dd.wcliques_count[(ld.idx / WARP_SIZE)])++;
            dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])] = start_write - (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) + wd.tot_vert[ld.wib_idx];
        }
        return 1;
    }

    return 0;
}

// returns 1 if failed found after removing, 0 otherwise
__device__ int d_remove_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld) 
{
    int pvertexid;
    int phelper1;
    int phelper2;

    int mindeg;
    bool failed_found;

    mindeg = d_get_mindeg(wd.num_mem[ld.wib_idx], dd);

    // remove the last candidate in vertices
    if ((ld.idx % WARP_SIZE) == 0) {
        wd.num_cand[ld.wib_idx]--;
        wd.tot_vert[ld.wib_idx]--;
    }
    __syncwarp();

    // update info of vertices connected to removed cand
    pvertexid = ld.read_vertices[wd.start[ld.wib_idx] + wd.tot_vert[ld.wib_idx]].vertexid;
    failed_found = false;

    for (int i = (ld.idx % WARP_SIZE); i < wd.tot_vert[ld.wib_idx]; i += WARP_SIZE) {
        phelper1 = ld.read_vertices[wd.start[ld.wib_idx] + i].vertexid;
        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            ld.read_vertices[wd.start[ld.wib_idx] + i].exdeg--;

            if (phelper1 < wd.num_mem[ld.wib_idx] && ld.read_vertices[wd.start[ld.wib_idx] + phelper1].indeg + ld.read_vertices[wd.start[ld.wib_idx] + phelper1].exdeg < mindeg) {
                failed_found = true;
                break;
            }
        }
    }

    failed_found = __any_sync(0xFFFFFFFF, failed_found);
    if (failed_found) {
        return 1;
    }

    return 0;
}

// returns 1 if failed found or invalid bound, 0 otherwise
__device__ int d_add_one_vertex(GPU_Data& dd, Warp_Data& wd, Local_Data& ld) 
{
    int pvertexid;
    int phelper1;
    int phelper2;
    bool failed_found;

    // ADD ONE VERTEX
    pvertexid = ld.vertices[wd.number_of_members[ld.wib_idx]].vertexid;

    if ((ld.idx % WARP_SIZE) == 0) {
        ld.vertices[wd.number_of_members[ld.wib_idx]].label = 1;
        wd.number_of_members[ld.wib_idx]++;
        wd.number_of_candidates[ld.wib_idx]--;
    }
    __syncwarp();

    for (int i = (ld.idx % WARP_SIZE); i < wd.tot_vert[ld.wib_idx]; i += WARP_SIZE) {
        phelper1 = ld.vertices[i].vertexid;
        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[pvertexid], dd.onehop_offsets[pvertexid + 1] - dd.onehop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            ld.vertices[i].exdeg--;
            ld.vertices[i].indeg++;
        }
    }
    __syncwarp();



    // DIAMETER PRUNING
    d_diameter_pruning(dd, wd, ld, pvertexid);



    // DEGREE BASED PRUNING
    failed_found = d_degree_pruning(dd, wd, ld);

    // if vertex in x found as not extendable continue to next iteration
    if (failed_found) {
        return 1;
    }
   
    return 0;
}

// TODO - check for extra syncs
// diameter pruning intitializes vertices labels and candidate indegs array for use in iterative degree pruning
__device__ void d_diameter_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int pvertexid)
{
    // vertices size * warp idx + (vertices size / warp size) * lane idx
    int lane_write = ((WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + ((WVERTICES_SIZE / WARP_SIZE) * (ld.idx % WARP_SIZE)));

    // intersection
    int phelper1;
    int phelper2;

    // vertex iteration
    int lane_remaining_count;

    lane_remaining_count = 0;

    for (int i = wd.number_of_members[ld.wib_idx] + (ld.idx % WARP_SIZE); i < wd.total_vertices[ld.wib_idx]; i += WARP_SIZE) {
        ld.vertices[i].label = -1;
    }
    __syncwarp();

    for (int i = wd.number_of_members[ld.wib_idx] + (ld.idx % WARP_SIZE); i < wd.total_vertices[ld.wib_idx]; i += WARP_SIZE) {
        phelper1 = ld.vertices[i].vertexid;
        phelper2 = d_bsearch_array(dd.twohop_neighbors + dd.twohop_offsets[pvertexid], dd.twohop_offsets[pvertexid + 1] - dd.twohop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            ld.vertices[i].label = 0;
            dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[i].indeg;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if ((ld.idx % WARP_SIZE) >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if ((ld.idx % WARP_SIZE) == WARP_SIZE - 1) {
        wd.remaining_count[ld.wib_idx] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    __syncwarp();

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd.candidate_indegs[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
    }
    __syncwarp();
}

// TODO - check for extra syncs
// returns true if invalid bounds or failed found
__device__ bool d_degree_pruning(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    // vertices size * warp idx + (vertices size / warp size) * lane idx
    int lane_write = ((WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + ((WVERTICES_SIZE / WARP_SIZE) * (ld.idx % WARP_SIZE)));

    // helper variables used throughout method to store various values, names have no meaning
    int pvertexid;
    int phelper1;
    int phelper2;

    // counter for lane intersection results
    int lane_remaining_count;
    int lane_removed_count;

    bool failed_found;



    d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * (ld.idx / WARP_SIZE)), wd.remaining_count[ld.wib_idx], (ld.idx % WARP_SIZE), d_sort_degs);

    // UNSURE - can we just set number of candidates and remaining count
    d_calculate_LU_bounds(dd, wd, ld, wd.remaining_count[ld.wib_idx]);
    if (wd.invalid_bounds[ld.wib_idx]) {
        return true;
    }

    // check for failed vertices
    failed_found = false;
    for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.wib_idx]; k += WARP_SIZE) {
        if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld)) {
            failed_found = true;
            break;
        }

    }
    failed_found = __any_sync(0xFFFFFFFF, failed_found);
    if (failed_found) {
        return true;
    }

    if ((ld.idx % WARP_SIZE) == 0) {
        wd.remaining_count[ld.wib_idx] = 0;
        wd.removed_count[ld.wib_idx] = 0;
        wd.rw_counter[ld.wib_idx] = 0;
    }



    lane_remaining_count = 0;
    lane_removed_count = 0;
    
    for (int i = wd.number_of_members[ld.wib_idx] + (ld.idx % WARP_SIZE); i < wd.total_vertices[ld.wib_idx]; i += WARP_SIZE) {
        if (ld.vertices[i].label == 0 && d_cand_isvalid_LU(ld.vertices[i], dd, wd, ld)) {
            dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
        }
        else {
            dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    pvertexid = lane_removed_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if ((ld.idx % WARP_SIZE) >= i) {
            lane_remaining_count += phelper1;
        }
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
        if ((ld.idx % WARP_SIZE) >= i) {
            lane_removed_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if ((ld.idx % WARP_SIZE) == WARP_SIZE - 1) {
        wd.remaining_count[ld.wib_idx] = lane_remaining_count;
        wd.removed_count[ld.wib_idx] = lane_removed_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    lane_removed_count -= pvertexid;

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_remaining_count + i] = ld.vertices[dd.lane_remaining_candidates[lane_write + i]];
    }
    // only need removed if going to be using removed to update degrees
    if (wd.remaining_count[ld.wib_idx] > wd.removed_count[ld.wib_idx]) {
        for (int i = 0; i < pvertexid; i++) {
            dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_removed_count + i] = ld.vertices[dd.lane_removed_candidates[lane_write + i]].vertexid;
        }
    }
    __syncwarp();


    
    while (wd.remaining_count[ld.wib_idx] > 0 && wd.removed_count[ld.wib_idx] > 0) {

        // different blocks for the read and write locations, vertices and remaining, this is done to avoid using extra variables and only one condition
        if (wd.rw_counter[ld.wib_idx] % 2 == 0) {
            // update degrees
            if (wd.remaining_count[ld.wib_idx] < wd.removed_count[ld.wib_idx]) {
                // via remaining
                // reset exdegs
                for (int i = (ld.idx % WARP_SIZE); i < wd.number_of_members[ld.wib_idx]; i += WARP_SIZE) {
                    ld.vertices[i].exdeg = 0;
                }
                for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                    dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i].exdeg = 0;
                }
                __syncwarp();

                // update exdeg based on remaining candidates
                for (int i = (ld.idx % WARP_SIZE); i < wd.number_of_members[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = ld.vertices[i].vertexid;

                    for (int j = 0; j < wd.remaining_count[ld.wib_idx]; j++) {
                        phelper1 = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + j].vertexid;
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            ld.vertices[i].exdeg++;
                        }
                    }
                }
                for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i].vertexid;

                    for (int j = 0; j < wd.remaining_count[ld.wib_idx]; j++) {
                        if (j == i) {
                            continue;
                        }

                        phelper1 = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + j].vertexid;
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i].exdeg++;
                        }
                    }
                }
            }
            else {
                // via removed
                // update exdeg based on remaining candidates
                for (int i = (ld.idx % WARP_SIZE); i < wd.number_of_members[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = ld.vertices[i].vertexid;

                    for (int j = 0; j < wd.removed_count[ld.wib_idx]; j++) {
                        phelper1 = dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + j];
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            ld.vertices[i].exdeg--;
                        }
                    }
                }
                for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i].vertexid;

                    for (int j = 0; j < wd.removed_count[ld.wib_idx]; j++) {
                        phelper1 = dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + j];
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i].exdeg--;
                        }
                    }
                }
            }
            __syncwarp();



            lane_remaining_count = 0;

            for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                if (d_cand_isvalid_LU(dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i], dd, wd, ld)) {
                    dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i].indeg;
                }
            }
            __syncwarp();

            // scan to calculate write postion in warp arrays
            phelper2 = lane_remaining_count;
            for (int i = 1; i < WARP_SIZE; i *= 2) {
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
                if ((ld.idx % WARP_SIZE) >= i) {
                    lane_remaining_count += phelper1;
                }
                __syncwarp();
            }
            // lane remaining count sum is scan for last lane and its value
            if ((ld.idx % WARP_SIZE) == WARP_SIZE - 1) {
                wd.num_val_cands[ld.wib_idx] = lane_remaining_count;
            }
            // make scan exclusive
            lane_remaining_count -= phelper2;

            // parallel write lane arrays to warp array
            for (int i = 0; i < phelper2; i++) {
                dd.candidate_indegs[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
            }
            __syncwarp();



            d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * (ld.idx / WARP_SIZE)), wd.num_val_cands[ld.wib_idx], (ld.idx % WARP_SIZE), d_sort_degs);

            // UNSURE - can we just set number of candidates and num val cands
            d_calculate_LU_bounds(dd, wd, ld, wd.num_val_cands[ld.wib_idx]);
            if (wd.invalid_bounds[ld.wib_idx]) {
                return true;
            }

            // check for failed vertices
            for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.wib_idx]; k += WARP_SIZE) {
                if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld)) {
                    failed_found = true;
                    break;
                }

            }
            failed_found = __any_sync(0xFFFFFFFF, failed_found);
            if (failed_found) {
                return true;
            }



            lane_remaining_count = 0;
            lane_removed_count = 0;

            // check for failed candidates
            for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                if (d_cand_isvalid_LU(dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i], dd, wd, ld)) {
                    dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
                }
                else {
                    dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
                }
            }
            __syncwarp();

            // scan to calculate write postion in warp arrays
            phelper2 = lane_remaining_count;
            pvertexid = lane_removed_count;
            for (int i = 1; i < WARP_SIZE; i *= 2) {
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
                if ((ld.idx % WARP_SIZE) >= i) {
                    lane_remaining_count += phelper1;
                }
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
                if ((ld.idx % WARP_SIZE) >= i) {
                    lane_removed_count += phelper1;
                }
                __syncwarp();
            }
            // lane remaining count sum is scan for last lane and its value
            if ((ld.idx % WARP_SIZE) == WARP_SIZE - 1) {
                wd.num_val_cands[ld.wib_idx] = lane_remaining_count;
                wd.removed_count[ld.wib_idx] = lane_removed_count;
            }
            // make scan exclusive
            lane_remaining_count -= phelper2;
            lane_removed_count -= pvertexid;

            // parallel write lane arrays to warp array
            for (int i = 0; i < phelper2; i++) {
                ld.vertices[wd.number_of_members[ld.wib_idx] + lane_remaining_count + i] = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + dd.lane_remaining_candidates[lane_write + i]];
            }
            // only need removed if going to be using removed to update degrees
            if (wd.num_val_cands[ld.wib_idx] > wd.removed_count[ld.wib_idx]) {
                for (int i = 0; i < pvertexid; i++) {
                    dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_removed_count + i] = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + dd.lane_removed_candidates[lane_write + i]].vertexid;
                }
            }
        }
        else {
            // update degrees
            if (wd.remaining_count[ld.wib_idx] < wd.removed_count[ld.wib_idx]) {
                // via remaining
                // reset exdegs
                for (int i = (ld.idx % WARP_SIZE); i < wd.number_of_members[ld.wib_idx]; i += WARP_SIZE) {
                    ld.vertices[i].exdeg = 0;
                }
                for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                    ld.vertices[wd.number_of_members[ld.wib_idx] + i].exdeg = 0;
                }
                __syncwarp();

                // update exdeg based on remaining candidates
                for (int i = (ld.idx % WARP_SIZE); i < wd.number_of_members[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = ld.vertices[i].vertexid;

                    for (int j = 0; j < wd.remaining_count[ld.wib_idx]; j++) {
                        phelper1 = ld.vertices[wd.number_of_members[ld.wib_idx] + j].vertexid;
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            ld.vertices[i].exdeg++;
                        }
                    }
                }
                for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = ld.vertices[wd.number_of_members[ld.wib_idx] + i].vertexid;

                    for (int j = 0; j < wd.remaining_count[ld.wib_idx]; j++) {
                        if (j == i) {
                            continue;
                        }

                        phelper1 = ld.vertices[wd.number_of_members[ld.wib_idx] + j].vertexid;
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            ld.vertices[wd.number_of_members[ld.wib_idx] + i].exdeg++;
                        }
                    }
                }
            }
            else {
                // via removed
                // update exdeg based on remaining candidates
                for (int i = (ld.idx % WARP_SIZE); i < wd.number_of_members[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = ld.vertices[i].vertexid;

                    for (int j = 0; j < wd.removed_count[ld.wib_idx]; j++) {
                        phelper1 = dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + j];
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            ld.vertices[i].exdeg--;
                        }
                    }
                }
                for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                    pvertexid = ld.vertices[wd.number_of_members[ld.wib_idx] + i].vertexid;

                    for (int j = 0; j < wd.removed_count[ld.wib_idx]; j++) {
                        phelper1 = dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + j];
                        phelper2 = d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[phelper1], dd.onehop_offsets[phelper1 + 1] - dd.onehop_offsets[phelper1], pvertexid);

                        if (phelper2 > -1) {
                            ld.vertices[wd.number_of_members[ld.wib_idx] + i].exdeg--;
                        }
                    }
                }
            }
            __syncwarp();



            lane_remaining_count = 0;

            for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                if (d_cand_isvalid_LU(ld.vertices[wd.number_of_members[ld.wib_idx] + i], dd, wd, ld)) {
                    dd.lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[wd.number_of_members[ld.wib_idx] + i].indeg;
                }
            }
            __syncwarp();

            // scan to calculate write postion in warp arrays
            phelper2 = lane_remaining_count;
            for (int i = 1; i < WARP_SIZE; i *= 2) {
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
                if ((ld.idx % WARP_SIZE) >= i) {
                    lane_remaining_count += phelper1;
                }
                __syncwarp();
            }
            // lane remaining count sum is scan for last lane and its value
            if ((ld.idx % WARP_SIZE) == WARP_SIZE - 1) {
                wd.num_val_cands[ld.wib_idx] = lane_remaining_count;
            }
            // make scan exclusive
            lane_remaining_count -= phelper2;

            // parallel write lane arrays to warp array
            for (int i = 0; i < phelper2; i++) {
                dd.candidate_indegs[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_remaining_count + i] = dd.lane_candidate_indegs[lane_write + i];
            }
            __syncwarp();



            d_sort_i(dd.candidate_indegs + (WVERTICES_SIZE * (ld.idx / WARP_SIZE)), wd.num_val_cands[ld.wib_idx], (ld.idx % WARP_SIZE), d_sort_degs);

            // UNSURE - can we just set number of candidates and num val cands
            d_calculate_LU_bounds(dd, wd, ld, wd.num_val_cands[ld.wib_idx]);
            if (wd.invalid_bounds[ld.wib_idx]) {
                return true;
            }

            // check for failed vertices
            for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.wib_idx]; k += WARP_SIZE) {
                if (!d_vert_isextendable_LU(ld.vertices[k], dd, wd, ld)) {
                    failed_found = true;
                    break;
                }

            }
            failed_found = __any_sync(0xFFFFFFFF, failed_found);
            if (failed_found) {
                return true;
            }



            lane_remaining_count = 0;
            lane_removed_count = 0;

            // check for failed candidates
            for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
                if (d_cand_isvalid_LU(ld.vertices[wd.number_of_members[ld.wib_idx] + i], dd, wd, ld)) {
                    dd.lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
                }
                else {
                    dd.lane_removed_candidates[lane_write + lane_removed_count++] = i;
                }
            }
            __syncwarp();

            // scan to calculate write postion in warp arrays
            phelper2 = lane_remaining_count;
            pvertexid = lane_removed_count;
            for (int i = 1; i < WARP_SIZE; i *= 2) {
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
                if ((ld.idx % WARP_SIZE) >= i) {
                    lane_remaining_count += phelper1;
                }
                phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
                if ((ld.idx % WARP_SIZE) >= i) {
                    lane_removed_count += phelper1;
                }
                __syncwarp();
            }
            // lane remaining count sum is scan for last lane and its value
            if ((ld.idx % WARP_SIZE) == WARP_SIZE - 1) {
                wd.num_val_cands[ld.wib_idx] = lane_remaining_count;
                wd.removed_count[ld.wib_idx] = lane_removed_count;
            }
            // make scan exclusive
            lane_remaining_count -= phelper2;
            lane_removed_count -= pvertexid;

            // parallel write lane arrays to warp array
            for (int i = 0; i < phelper2; i++) {
                dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_remaining_count + i] = ld.vertices[wd.number_of_members[ld.wib_idx] + dd.lane_remaining_candidates[lane_write + i]];
            }
            // only need removed if going to be using removed to update degrees
            if (wd.num_val_cands[ld.wib_idx] > wd.removed_count[ld.wib_idx]) {
                for (int i = 0; i < pvertexid; i++) {
                    dd.removed_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + lane_removed_count + i] = ld.vertices[wd.number_of_members[ld.wib_idx] + dd.lane_removed_candidates[lane_write + i]].vertexid;
                }
            }
        }




        if ((ld.idx % WARP_SIZE) == 0) {
            wd.remaining_count[ld.wib_idx] = wd.num_val_cands[ld.wib_idx];
            wd.rw_counter[ld.wib_idx]++;
        }
    }



    // condense vertices so remaining are after members, only needs to be done if they were not written into vertices last time
    if (wd.rw_counter[ld.wib_idx] % 2 == 0) {
        for (int i = (ld.idx % WARP_SIZE); i < wd.remaining_count[ld.wib_idx]; i += WARP_SIZE) {
            ld.vertices[wd.number_of_members[ld.wib_idx] + i] = dd.remaining_candidates[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + i];
        }
    }

    if ((ld.idx % WARP_SIZE) == 0) {
        wd.total_vertices[ld.wib_idx] = wd.total_vertices[ld.wib_idx] - wd.number_of_candidates[ld.wib_idx] + wd.remaining_count[ld.wib_idx];
        wd.number_of_candidates[ld.wib_idx] = wd.remaining_count[ld.wib_idx];
    }

    return false;
}

__device__ void d_calculate_LU_bounds(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates)
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
        wd.invalid_bounds[ld.wib_idx] = false;

        wd.sum_candidate_indeg[ld.wib_idx] = 0;
        wd.tightened_upper_bound[ld.wib_idx] = 0;

        wd.min_clq_indeg[ld.wib_idx] = ld.vertices[0].indeg;
        wd.min_indeg_exdeg[ld.wib_idx] = ld.vertices[0].exdeg;
        wd.min_clq_totaldeg[ld.wib_idx] = ld.vertices[0].indeg + ld.vertices[0].exdeg;
        wd.sum_clq_indeg[ld.wib_idx] = ld.vertices[0].indeg;

        wd.min_ext_deg[ld.wib_idx] = d_get_mindeg(wd.number_of_members[ld.wib_idx] + 1, dd);
    }
    __syncwarp();

    // each warp finds these values on their subsection of vertices
    for (index = 1 + (ld.idx % WARP_SIZE); index < wd.number_of_members[ld.wib_idx]; index += WARP_SIZE) {
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
        wd.sum_clq_indeg[ld.wib_idx] += sum_clq_indeg;
    }
    __syncwarp();

    // CRITICAL SECTION - each lane then compares their values to the next to get a warp level value
    for (int i = 0; i < WARP_SIZE; i++) {
        if ((ld.idx % WARP_SIZE) == i) {
            if (min_clq_indeg < wd.min_clq_indeg[ld.wib_idx]) {
                wd.min_clq_indeg[ld.wib_idx] = min_clq_indeg;
                wd.min_indeg_exdeg[ld.wib_idx] = min_indeg_exdeg;
            }
            else if (min_clq_indeg == wd.min_clq_indeg[ld.wib_idx]) {
                if (min_indeg_exdeg < wd.min_indeg_exdeg[ld.wib_idx]) {
                    wd.min_indeg_exdeg[ld.wib_idx] = min_indeg_exdeg;
                }
            }

            if (min_clq_totaldeg < wd.min_clq_totaldeg[ld.wib_idx]) {
                wd.min_clq_totaldeg[ld.wib_idx] = min_clq_totaldeg;
            }
        }
        __syncwarp();
    }

    // TODO - see if some of this can be parallelized
    if ((ld.idx % WARP_SIZE) == 0) {
        if (wd.min_clq_indeg[ld.wib_idx] < dd.minimum_degrees[wd.number_of_members[ld.wib_idx]])
        {
            // lower
            wd.lower_bound[ld.wib_idx] = d_get_mindeg(wd.number_of_members[ld.wib_idx], dd) - min_clq_indeg;

            while (wd.lower_bound[ld.wib_idx] <= wd.min_indeg_exdeg[ld.wib_idx] && wd.min_clq_indeg[ld.wib_idx] + wd.lower_bound[ld.wib_idx] <
                dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + wd.lower_bound[ld.wib_idx]]) {
                wd.lower_bound[ld.wib_idx]++;
            }

            if (wd.min_clq_indeg[ld.wib_idx] + wd.lower_bound[ld.wib_idx] < dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + wd.lower_bound[ld.wib_idx]]) {
                wd.invalid_bounds[ld.wib_idx] = true;
            }

            // upper
            wd.upper_bound[ld.wib_idx] = floor(wd.min_clq_totaldeg[ld.wib_idx] / (*(dd.minimum_degree_ratio))) + 1 - wd.number_of_members[ld.wib_idx];

            if (wd.upper_bound[ld.wib_idx] > number_of_candidates) {
                wd.upper_bound[ld.wib_idx] = number_of_candidates;
            }

            // tighten
            if (wd.lower_bound[ld.wib_idx] < wd.upper_bound[ld.wib_idx]) {
                // tighten lower
                for (index = 0; index < wd.lower_bound[ld.wib_idx]; index++) {
                    wd.sum_candidate_indeg[ld.wib_idx] += dd.candidate_indegs[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + index];
                }

                while (index < wd.upper_bound[ld.wib_idx] && wd.sum_clq_indeg[ld.wib_idx] + wd.sum_candidate_indeg[ld.wib_idx] < wd.number_of_members[ld.wib_idx] *
                    dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + index]) {
                    wd.sum_candidate_indeg[ld.wib_idx] += dd.candidate_indegs[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + index];
                    index++;
                }

                if (wd.sum_clq_indeg[ld.wib_idx] + wd.sum_candidate_indeg[ld.wib_idx] < wd.number_of_members[ld.wib_idx] * dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + index]) {
                    wd.invalid_bounds[ld.wib_idx] = true;
                }
                else {
                    wd.lower_bound[ld.wib_idx] = index;

                    wd.tightened_upper_bound[ld.wib_idx] = index;

                    while (index < wd.upper_bound[ld.wib_idx]) {
                        wd.sum_candidate_indeg[ld.wib_idx] += dd.candidate_indegs[(WVERTICES_SIZE * (ld.idx / WARP_SIZE)) + index];

                        index++;

                        if (wd.sum_clq_indeg[ld.wib_idx] + wd.sum_candidate_indeg[ld.wib_idx] >= wd.number_of_members[ld.wib_idx] *
                            dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + index]) {
                            wd.tightened_upper_bound[ld.wib_idx] = index;
                        }
                    }

                    if (wd.upper_bound[ld.wib_idx] > wd.tightened_upper_bound[ld.wib_idx]) {
                        wd.upper_bound[ld.wib_idx] = wd.tightened_upper_bound[ld.wib_idx];
                    }

                    if (wd.lower_bound[ld.wib_idx] > 1) {
                        wd.min_ext_deg[ld.wib_idx] = d_get_mindeg(wd.number_of_members[ld.wib_idx] + wd.lower_bound[ld.wib_idx], dd);
                    }
                }
            }
        }
        else {
            wd.min_ext_deg[ld.wib_idx] = d_get_mindeg(wd.number_of_members[ld.wib_idx] + 1,
                dd);

            wd.upper_bound[ld.wib_idx] = number_of_candidates;

            if (wd.number_of_members[ld.wib_idx] < (*(dd.minimum_clique_size))) {
                wd.lower_bound[ld.wib_idx] = (*(dd.minimum_clique_size)) - wd.number_of_members[ld.wib_idx];
            }
            else {
                wd.lower_bound[ld.wib_idx] = 0;
            }
        }

        if (wd.number_of_members[ld.wib_idx] + wd.upper_bound[ld.wib_idx] < (*(dd.minimum_clique_size))) {
            wd.invalid_bounds[ld.wib_idx] = true;
        }

        if (wd.upper_bound[ld.wib_idx] < 0 || wd.upper_bound[ld.wib_idx] < wd.lower_bound[ld.wib_idx]) {
            wd.invalid_bounds[ld.wib_idx] = true;
        }
    }
    __syncwarp();
}

// program updates degrees by : for each vertex, for each removed vertex, binary search neighbors of removed vertex for vertex
__device__ void d_update_degrees(GPU_Data& dd, Warp_Data& wd, Local_Data& ld, int number_of_removed)
{
    int pvertexid;

    for (int k = (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.wib_idx] - number_of_removed; k += WARP_SIZE) {
        pvertexid = ld.vertices[k].vertexid;
        for (int l = wd.total_vertices[ld.wib_idx] - number_of_removed; l < wd.total_vertices[ld.wib_idx]; l++) {
            if (d_bsearch_array(dd.onehop_neighbors + dd.onehop_offsets[ld.vertices[l].vertexid], dd.onehop_offsets[ld.vertices[l].vertexid + 1] - dd.onehop_offsets[ld.vertices[l].vertexid], pvertexid) != -1) {
                ld.vertices[k].exdeg--;
            }
        }
    }
    __syncwarp();
}

__device__ void d_check_for_clique(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    bool clique = true;

    for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.wib_idx]; k += WARP_SIZE) {
        if (ld.vertices[k].indeg < dd.minimum_degrees[wd.number_of_members[ld.wib_idx]]) {
            clique = false;
            break;
        }
    }
    // set to false if any threads in warp do not meet degree requirement
    clique = !(__any_sync(0xFFFFFFFF, !clique));

    // if clique write to warp buffer for cliques
    if (clique) {
        uint64_t start_write = (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) + dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])];
        for (int k = (ld.idx % WARP_SIZE); k < wd.number_of_members[ld.wib_idx]; k += WARP_SIZE) {
            dd.wcliques_vertex[start_write + k] = ld.vertices[k].vertexid;
        }
        if ((ld.idx % WARP_SIZE) == 0) {
            (dd.wcliques_count[(ld.idx / WARP_SIZE)])++;
            dd.wcliques_offset[(WCLIQUES_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wcliques_count[(ld.idx / WARP_SIZE)])] = start_write - (WCLIQUES_SIZE * (ld.idx / WARP_SIZE)) +
                wd.number_of_members[ld.wib_idx];
        }
    }
}

__device__ void d_write_to_tasks(GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    // CRITICAL
    atomicExch((int*)dd.maximal_expansion, false);

    uint64_t start_write = (WTASKS_SIZE * (ld.idx / WARP_SIZE)) + dd.wtasks_offset[WTASKS_OFFSET_SIZE * (ld.idx / WARP_SIZE) + (dd.wtasks_count[(ld.idx / WARP_SIZE)])];

    for (int k = (ld.idx % WARP_SIZE); k < wd.total_vertices[ld.wib_idx]; k += WARP_SIZE) {
        dd.wtasks_vertices[start_write + k].vertexid = ld.vertices[k].vertexid;
        dd.wtasks_vertices[start_write + k].label = ld.vertices[k].label;
        dd.wtasks_vertices[start_write + k].indeg = ld.vertices[k].indeg;
        dd.wtasks_vertices[start_write + k].exdeg = ld.vertices[k].exdeg;
        dd.wtasks_vertices[start_write + k].lvl2adj = 0;
    }
    if ((ld.idx % WARP_SIZE) == 0) {
        (dd.wtasks_count[(ld.idx / WARP_SIZE)])++;
        dd.wtasks_offset[(WTASKS_OFFSET_SIZE * (ld.idx / WARP_SIZE)) + (dd.wtasks_count[(ld.idx / WARP_SIZE)])] = start_write - (WTASKS_SIZE * (ld.idx / WARP_SIZE)) + wd.total_vertices[ld.wib_idx];
    }
}



// --- HELPER KERNELS ---

// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
__device__ int d_bsearch_array(int* search_array, int array_size, int search_number)
{
    // ALGO - binary
    // TYPE - serial
    // SPEED - 0(log(n))

    int low = 0;
    int high = array_size - 1;

    while (low <= high) {
        int mid = (low + high) / 2;

        if (search_array[mid] == search_number) {
            return mid;
        }
        else if (search_array[mid] > search_number) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }

    return -1;
}

// consider using merge
__device__ void d_sort(Vertex* target, int size, int lane_idx, int (*func)(Vertex&, Vertex&))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    Vertex vertex1;
    Vertex vertex2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (lane_idx * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            vertex1 = target[j];
            vertex2 = target[j + 1];

            if (func(vertex1, vertex2) == 1) {
                target[j] = vertex2;
                target[j + 1] = vertex1;
            }
        }
        __syncwarp();
    }
}

__device__ void d_sort_i(int* target, int size, int lane_idx, int (*func)(int, int))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    int num1;
    int num2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (lane_idx * 2); j < size - 1; j += (WARP_SIZE * 2)) {
            num1 = target[j];
            num2 = target[j + 1];

            if (func(num1, num2) == 1) {
                target[j] = num2;
                target[j + 1] = num1;
            }
        }
        __syncwarp();
    }
}

// Quick enumeration order sort keys
__device__ int d_sort_vert_Q(Vertex& v1, Vertex& v2)
{
    // order is: covered -> cands -> cover
    // keys are: indeg -> exdeg -> lvl2adj -> vertexid

    if (v1.label == 1 && v2.label != 1)
        return -1;
    else if (v1.label != 1 && v2.label == 1)
        return 1;
    else if (v1.label == 2 && v2.label != 2)
        return -1;
    else if (v1.label != 2 && v2.label == 2)
        return 1;
    else if (v1.label == 0 && v2.label != 0)
        return -1;
    else if (v1.label != 0 && v2.label == 0)
        return 1;
    else if (v1.label == 3 && v2.label != 3)
        return -1;
    else if (v1.label != 3 && v2.label == 3)
        return 1;
    else if (v1.label == 0 && v2.label != 0)
        return -1;
    else if (v1.indeg > v2.indeg)
        return -1;
    else if (v1.indeg < v2.indeg)
        return 1;
    else if (v1.exdeg > v2.exdeg)
        return -1;
    else if (v1.exdeg < v2.exdeg)
        return 1;
    else if (v1.lvl2adj > v2.lvl2adj)
        return -1;
    else if (v1.lvl2adj < v2.lvl2adj)
        return 1;
    else if (v1.vertexid > v2.vertexid)
        return -1;
    else if (v1.vertexid < v2.vertexid)
        return 1;
    else
        return 0;
}

__device__ int d_sort_degs(int n1, int n2)
{
    // descending order

    if (n1 > n2) {
        return -1;
    }
    else if (n1 < n2) {
        return 1;
    }
    else {
        return 0;
    }
}

// sort vetices only considering in clique, candidates, and pruned vertices
__device__ int d_sort_vert_cp(Vertex& vertex1, Vertex& vertex2)
{
    // order is: cands -> pruned

    if (vertex1.label == 0 && vertex2.label != 0) {
        return -1;
    }
    else if (vertex1.label != 0 && vertex2.label == 0) {
        return 1;
    }
    else {
        return 0;
    }
}

// sort vertices only considering in clique and candidates
__device__ int d_sort_vert_cc(Vertex& vertex1, Vertex& vertex2)
{
    // order is: in clique -> cands

    // in clique
    if (vertex1.label == 1 && vertex2.label != 1) {
        return -1;
    }
    else if (vertex1.label != 1 && vertex2.label == 1) {
        return 1;
    }
    else {
        return 0;
    }
}

// sort vertices with cands sorted from high indeg to low indeg
__device__ int d_sort_vert_lu(Vertex& vertex1, Vertex& vertex2)
{
    // order is: in clique -> covered -> crtical adjacent -> cands -> cover -> pruned

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

    // for ties: in clique low -> high, cand high indeg -> low indeg
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
        if (vertex1.indeg > vertex2.indeg) {
            return -1;
        }
        else if (vertex1.indeg < vertex2.indeg) {
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

// sort vertices with cands sorted from high idx to low idx
__device__ int d_sort_vert_ex(Vertex& vertex1, Vertex& vertex2)
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
            return 1;
        }
        else if (vertex1.vertexid < vertex2.vertexid) {
            return -1;
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

__device__ int d_get_mindeg(int number_of_members, GPU_Data& dd)
{
    if (number_of_members < (*(dd.minimum_clique_size))) {
        return dd.minimum_degrees[(*(dd.minimum_clique_size))];
    }
    else {
        return dd.minimum_degrees[number_of_members];
    }
}

__device__ bool d_cand_isvalid(Vertex& vertex, int number_of_members, GPU_Data& dd)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(number_of_members + vertex.exdeg + 1, dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool d_cand_isvalid_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[ld.wib_idx] + vertex.exdeg + 1, dd)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[ld.wib_idx]) {
        return false;
    }
    else if (vertex.indeg + wd.upper_bound[ld.wib_idx] - 1 < dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + wd.lower_bound[ld.wib_idx]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[ld.wib_idx] + wd.lower_bound[ld.wib_idx], dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool d_vert_isextendable(Vertex& vertex, int number_of_members, GPU_Data& dd)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(number_of_members + vertex.exdeg, dd)) {
        return false;
    }
    else {
        return true;
    }
}

__device__ bool d_vert_isextendable_LU(Vertex& vertex, GPU_Data& dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd.minimum_degrees[(*(dd.minimum_clique_size))]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[ld.wib_idx] + vertex.exdeg, dd)) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[ld.wib_idx]) {
        return false;
    }
    else if (vertex.exdeg == 0 && vertex.indeg < d_get_mindeg(wd.number_of_members[ld.wib_idx] + vertex.exdeg, dd)) {
        return false;
    }
    else if (vertex.indeg + wd.upper_bound[ld.wib_idx] < dd.minimum_degrees[wd.number_of_members[ld.wib_idx] + wd.upper_bound[ld.wib_idx]]) {
        return false;
    }
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[ld.wib_idx] + wd.lower_bound[ld.wib_idx], dd)) {
        return false;
    }
    else {
        return true;
    }
}



// --- DEBUG KERNELS ---

__device__ void d_print_vertices(Vertex* vertices, int size)
{
    printf("\nOffsets:\n0 %i\nVertex:\n", size);
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].vertexid);
    }
    printf("\nLabel:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].label);
    }
    printf("\nIndeg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].indeg);
    }
    printf("\nExdeg:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].exdeg);
    }
    printf("\nLvl2adj:\n");
    for (int i = 0; i < size; i++) {
        printf("%i ", vertices[i].lvl2adj);
    }
    printf("\n");
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