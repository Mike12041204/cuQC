#include "../inc/common.h"
#include "../inc/host_functions.h"
#include "../inc/host_debug.h"
#include "../inc/device_kernels.h"
#include "../inc/cuTS_MPI.h"

// --- PRIMARY FUNCTIONS ---
// initializes minimum degrees array 
void calculate_minimum_degrees(CPU_Graph& hg, int*& minimum_degrees, double minimum_degree_ratio)
{
    minimum_degrees = new int[hg.number_of_vertices + 1];
    minimum_degrees[0] = 0;
    for (int i = 1; i <= hg.number_of_vertices; i++) {
        minimum_degrees[i] = ceil(minimum_degree_ratio * (i - 1));
    }
}

void search(CPU_Graph& hg, ofstream& temp_results, DS_Sizes& dss, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size) 
{
    CPU_Data hd;                    // host vertex structure data
    CPU_Cliques hc;                 // host results data
    GPU_Data h_dd;                  // host pointers to device global memory
    GPU_Data* dd;                   // device pointers to device global memory
    uint64_t write_count;           // how many tasks in tasks
    uint64_t buffer_count;          // how many tasks in buffer
    uint64_t cliques_size;          // how many results stored
    uint64_t cliques_count;         // size of results stored
    uint64_t* mpiSizeBuffer;        // where data transfered intranode is stored
    Vertex* mpiVertexBuffer;
    bool help_others;
    int taker;
    bool divided_work;
    int from;

    mpiSizeBuffer = new uint64_t[MAX_MESSAGE];
    mpiVertexBuffer = new Vertex[MAX_MESSAGE];

    // open communication channels
    mpi_irecv_all(grank);

    for (int i = 0; i < wsize; ++i) {
        global_free_list[i] = false;
    }

    // HANDLE MEMORY
    allocate_memory(hd, h_dd, hc, hg, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size);
    cudaDeviceSynchronize();
    chkerr(cudaMalloc((void**)&dd, sizeof(GPU_Data)));
    chkerr(cudaMemcpy(dd, &h_dd, sizeof(GPU_Data), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    // TIME
    auto start = chrono::high_resolution_clock::now();

    // INITIALIZE TASKS
    if(grank == 0){
        cout << ">:INITIALIZING TASKS" << endl;
    }
    initialize_tasks(hg, hd, minimum_degrees, minimum_clique_size);

    // DEBUG
    if (DEBUG_TOGGLE) {
        mvs = (*(hd.tasks1_offset + (*hd.tasks1_count)));
        if ((*(hd.tasks1_offset + (*hd.tasks1_count))) > dss.wvertices_size) {
            cout << "!!! VERTICES SIZE ERROR !!!" << endl;
            return;
        }
        output_file << "CPU START" << endl;
        h_print_Data_Sizes(hd, hc);
    }

    // CPU EXPANSION
    // cpu expand must be called atleast one time to handle first round cover pruning as the gpu code cannot do this
    for (int i = 0; i < CPU_LEVELS + 1 && !(*hd.maximal_expansion); i++) {
        // will set maximal expansion false if no work generated
        h_expand_level(hg, hd, hc, dss, minimum_degrees, minimum_degree_ratio, minimum_clique_size);
    
        // if cliques is more than threshold dump
        if (hc.cliques_offset[(*hc.cliques_count)] > dss.cliques_dump) {
            flush_cliques(hc, temp_results);
        }

        // DEBUG
        if (DEBUG_TOGGLE) {
            h_print_Data_Sizes(hd, hc);
        }
    }

    flush_cliques(hc, temp_results);

    // TRANSFER TO GPU
    move_to_gpu(hd, h_dd, dss);
    cudaDeviceSynchronize();

    // EXPAND LEVEL
    if(grank == 0){
        cout << ">:BEGINNING EXPANSION" << endl;
    }

    help_others = false;

    // wait after all work in process has been completed, loop if work has been given from another process, break if all process complete work
    do{

        if(help_others){
            // decode buffer
            decode_com_buffer(h_dd, mpiSizeBuffer, mpiVertexBuffer);
            cudaDeviceSynchronize();
            // populate tasks from buffer
            fill_from_buffer<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
            cudaDeviceSynchronize();
            *hd.maximal_expansion = false;

            // DEBUG
            if (DEBUG_TOGGLE) {
                output_file << "RECIEVING WORK FROM PROCESS " << from << endl;
                print_Data_Sizes(h_dd, dss);
            }
        }

        // loop while not all work has been completed
        while (!(*hd.maximal_expansion)){

            *hd.maximal_expansion = true;
            chkerr(cudaMemset(h_dd.current_task, 0, sizeof(int)));
            cudaDeviceSynchronize();

            // expand all tasks in 'tasks' array, each warp will write to their respective warp tasks buffer in global memory
            d_expand_level<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
            cudaDeviceSynchronize();

            // DEBUG
            if (DEBUG_TOGGLE) {
                print_Warp_Data_Sizes_Every(h_dd, 1, dss);
            }

            // consolidate all the warp tasks/cliques buffers into the next global tasks array, buffer, and cliques
            transfer_buffers<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
            cudaDeviceSynchronize();

            // determine whether maximal expansion has been accomplished
            chkerr(cudaMemcpy(&buffer_count, h_dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            chkerr(cudaMemcpy(&write_count, h_dd.tasks_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));

            if (write_count > 0 || buffer_count > 0) {
                (*(hd.maximal_expansion)) = false;
            }

            chkerr(cudaMemset(h_dd.wtasks_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));
            chkerr(cudaMemset(h_dd.wcliques_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));
            if (write_count < dss.expand_threshold && buffer_count > 0) {
                // if not enough tasks were generated when expanding the previous level to fill the next tasks array the program will attempt to fill the tasks array by popping tasks from the buffer
                fill_from_buffer<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(dd);
                cudaDeviceSynchronize();
            }

            // determine whether cliques has exceeded defined threshold, if so dump them to a file
            chkerr(cudaMemcpy(&cliques_count, h_dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            chkerr(cudaMemcpy(&cliques_size, h_dd.cliques_offset + cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();

            // if cliques is more than threshold dump
            if (cliques_size > dss.cliques_dump) {
                dump_cliques(hc, h_dd, temp_results, dss);
            }

            // DEBUG
            if (DEBUG_TOGGLE) {
                print_Data_Sizes_Every(h_dd, 1, dss);
            }

            chkerr(cudaMemcpy(&buffer_count, h_dd.buffer_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            if(buffer_count > HELP_THRESHOLD){
                // return whether work was successfully given
                divided_work = give_work_wrapper(grank, taker, mpiSizeBuffer, mpiVertexBuffer, h_dd, buffer_count, dss);
                // update buffer count if work was given
                if(divided_work){
                    buffer_count -= (buffer_count > dss.expand_threshold) ? dss.expand_threshold + ((buffer_count - dss.expand_threshold) * ((100 - HELP_PERCENT) / 100.0)) : buffer_count;
                    chkerr(cudaMemcpy(h_dd.buffer_count, &buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice));

                    // DEBUG
                    if (DEBUG_TOGGLE) {
                        output_file << "SENDING WORK TO PROCESS " << taker << endl;
                        print_Data_Sizes(h_dd, dss);
                    }
                }
            }
        }

        // we have finished all our work, so if we get to the top of the loop again it is because we are helping someone else
        help_others = true;

    }while(wsize != take_work_wrap(grank, mpiSizeBuffer, mpiVertexBuffer, from));

    // TIME
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    MPI_Barrier(MPI_COMM_WORLD);
    if(grank == 0){
        cout << "--->:ENUMERATION TIME: " << duration.count() << " ms" << endl;
    }

    dump_cliques(hc, h_dd, temp_results, dss);

    free_memory(hd, h_dd, hc);
    chkerr(cudaFree(dd));
}

// allocates memory for the data structures on the host and device   
void allocate_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Cliques& hc, CPU_Graph& hg, DS_Sizes& dss, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size)
{
    // GPU GRAPH
    chkerr(cudaMalloc((void**)&h_dd.number_of_vertices, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.number_of_edges, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.onehop_neighbors, sizeof(int) * hg.number_of_edges));
    chkerr(cudaMalloc((void**)&h_dd.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&h_dd.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj));
    chkerr(cudaMalloc((void**)&h_dd.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1)));
    chkerr(cudaMemcpy(h_dd.number_of_vertices, &(hg.number_of_vertices), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.number_of_edges, &(hg.number_of_edges), sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.onehop_neighbors, hg.onehop_neighbors, sizeof(int) * hg.number_of_edges, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.onehop_offsets, hg.onehop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.twohop_neighbors, hg.twohop_neighbors, sizeof(int) * hg.number_of_lvl2adj, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.twohop_offsets, hg.twohop_offsets, sizeof(uint64_t) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    // CPU DATA
    hd.tasks1_count = new uint64_t;
    hd.tasks1_offset = new uint64_t[dss.expand_threshold + 1];
    hd.tasks1_vertices = new Vertex[dss.tasks_size];
    hd.tasks1_offset[0] = 0;
    (*(hd.tasks1_count)) = 0;
    hd.tasks2_count = new uint64_t;
    hd.tasks2_offset = new uint64_t[dss.expand_threshold + 1];
    hd.tasks2_vertices = new Vertex[dss.tasks_size];
    hd.tasks2_offset[0] = 0;
    (*(hd.tasks2_count)) = 0;
    hd.buffer_count = new uint64_t;
    hd.buffer_offset = new uint64_t[dss.buffer_offset_size];
    hd.buffer_vertices = new Vertex[dss.buffer_size];
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
    chkerr(cudaMalloc((void**)&h_dd.current_level, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_offset, sizeof(uint64_t) * (dss.expand_threshold + 1)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_vertices, sizeof(Vertex) * dss.tasks_size));
    chkerr(cudaMemset(h_dd.tasks_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(h_dd.tasks_count, 0, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_offset, sizeof(uint64_t) * dss.buffer_offset_size));
    chkerr(cudaMalloc((void**)&h_dd.buffer_vertices, sizeof(Vertex) * dss.buffer_size));
    chkerr(cudaMemset(h_dd.buffer_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(h_dd.buffer_count, 0, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_offset, (sizeof(uint64_t) * dss.wtasks_offset_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_vertices, (sizeof(Vertex) * dss.wtasks_size) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(h_dd.wtasks_offset, 0, (sizeof(uint64_t) * dss.wtasks_offset_size) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(h_dd.wtasks_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.global_vertices, (sizeof(Vertex) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.removed_candidates, (sizeof(int) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_removed_candidates, (sizeof(int) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.remaining_candidates, (sizeof(Vertex) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_remaining_candidates, (sizeof(int) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.candidate_indegs, (sizeof(int) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.lane_candidate_indegs, (sizeof(int) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.adjacencies, (sizeof(int) * dss.wvertices_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.minimum_degree_ratio, sizeof(double)));
    chkerr(cudaMalloc((void**)&h_dd.minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1)));
    chkerr(cudaMalloc((void**)&h_dd.minimum_clique_size, sizeof(int)));
    chkerr(cudaMemcpy(h_dd.minimum_degree_ratio, &minimum_degree_ratio, sizeof(double), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.minimum_degrees, minimum_degrees, sizeof(int) * (hg.number_of_vertices + 1), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.minimum_clique_size, &minimum_clique_size, sizeof(int), cudaMemcpyHostToDevice));
    chkerr(cudaMalloc((void**)&h_dd.total_tasks, sizeof(int)));
    chkerr(cudaMemset(h_dd.total_tasks, 0, sizeof(int)));
    // CPU CLIQUES
    hc.cliques_count = new uint64_t;
    hc.cliques_vertex = new int[dss.cliques_size];
    hc.cliques_offset = new uint64_t[dss.cliques_offset_size];
    hc.cliques_offset[0] = 0;
    (*(hc.cliques_count)) = 0;
    // GPU CLIQUES
    chkerr(cudaMalloc((void**)&h_dd.cliques_count, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_vertex, sizeof(int) * dss.cliques_size));
    chkerr(cudaMalloc((void**)&h_dd.cliques_offset, sizeof(uint64_t) * dss.cliques_offset_size));
    chkerr(cudaMemset(h_dd.cliques_offset, 0, sizeof(uint64_t)));
    chkerr(cudaMemset(h_dd.cliques_count, 0, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_count, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_offset, (sizeof(uint64_t) * dss.wcliques_offset_size) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_vertex, (sizeof(int) * dss.wcliques_size) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(h_dd.wcliques_offset, 0, (sizeof(uint64_t) * dss.wcliques_offset_size) * NUMBER_OF_WARPS));
    chkerr(cudaMemset(h_dd.wcliques_count, 0, sizeof(uint64_t) * NUMBER_OF_WARPS));
    chkerr(cudaMalloc((void**)&h_dd.total_cliques, sizeof(int)));
    chkerr(cudaMemset(h_dd.total_cliques, 0, sizeof(int)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_offset_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_start, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.current_task, sizeof(int)));
    // DATA STRUCTURE SIZES
    chkerr(cudaMalloc((void**)&h_dd.tasks_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.tasks_per_warp, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.buffer_offset_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_offset_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_percent, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wcliques_offset_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wtasks_offset_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.wvertices_size, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.expand_threshold, sizeof(uint64_t)));
    chkerr(cudaMalloc((void**)&h_dd.cliques_dump, sizeof(uint64_t)));
    chkerr(cudaMemcpy(h_dd.tasks_size, &dss.tasks_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.tasks_per_warp, &dss.tasks_per_warp, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_size, &dss.buffer_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_offset_size, &dss.buffer_offset_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.cliques_size, &dss.cliques_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.cliques_offset_size, &dss.cliques_offset_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.cliques_percent, &dss.cliques_percent, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.wcliques_size, &dss.wcliques_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.wcliques_offset_size, &dss.wcliques_offset_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.wtasks_size, &dss.wtasks_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.wtasks_offset_size, &dss.wtasks_offset_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.wvertices_size, &dss.wvertices_size, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.expand_threshold, &dss.expand_threshold, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.cliques_dump, &dss.cliques_dump, sizeof(uint64_t), cudaMemcpyHostToDevice));
}

// processes 0th level of expansion
void initialize_tasks(CPU_Graph& hg, CPU_Data& hd, int* minimum_degrees, int minimum_clique_size)
{
    // intersection
    int pvertexid;                      // vertex id for pruning
    uint64_t pneighbors_start;          // start of neighbors for pruning
    uint64_t pneighbors_end;            // end of neighbors for pruning
    int phelper1;                       // helper for pruning
    int maximum_degree;                 // maximum degree of any vertex
    int maximum_degree_index;           // index in vertices of max vertex
    int total_vertices;                 // total vertices
    int number_of_candidates;           // number of candidate vertices
    Vertex* vertices;                   // array of vertices
    int removed_start;                  // tracks how many updates done for pruning

    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // INITIALIZE VERTICES
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
    removed_start = 0;
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
    qsort(vertices, total_vertices, sizeof(Vertex), h_comp_vert_Q);
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

void h_expand_level(CPU_Graph& hg, CPU_Data& hd, CPU_Cliques& hc, DS_Sizes& dss, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size)
{
    uint64_t* read_count;              // read and write to tasks 1 and 2 in alternating manner
    uint64_t* read_offsets;
    Vertex* read_vertices;
    uint64_t* write_count;
    uint64_t* write_offsets;
    Vertex* write_vertices;
    uint64_t start;                    // old vertices information
    uint64_t end;
    int tot_vert;
    int num_mem;
    int num_cand;
    int expansions;
    int number_of_covered;
    Vertex* vertices;                   // new vertices information
    int number_of_members;
    int number_of_candidates;
    int total_vertices;
    int min_ext_deg;                    // calculate lower-upper bounds
    int lower_bound;
    int upper_bound;
    int method_return;                  // helper
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
        method_return = h_lookahead_pruning(hg, hc, hd, read_vertices, tot_vert, num_mem, num_cand, start, minimum_degrees);
        if (method_return) {
            continue;
        }

        // NEXT LEVEL
        for (int j = number_of_covered; j < expansions; j++) {

            // REMOVE ONE VERTEX
            if (j != number_of_covered) {
                method_return = h_remove_one_vertex(hg, hd, read_vertices, tot_vert, num_cand, num_mem, start, minimum_degrees, minimum_clique_size);
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
            method_return = h_add_one_vertex(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

            // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
            if (method_return == 1) {
                if (number_of_members >= minimum_clique_size) {
                    h_check_for_clique(hc, vertices, number_of_members, minimum_degrees);
                }

                delete vertices;
                continue;
            }

            // CRITICAL VERTEX PRUNING
            method_return = h_critical_vertex_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

            // if critical fail continue onto next iteration
            if (method_return == 2) {
                delete vertices;
                continue;
            }

            // CHECK FOR CLIQUE
            // all processes will do this, to prevent duplicates only process 0 will save cpu results
            if (grank == 0 && number_of_members >= minimum_clique_size) {
                h_check_for_clique(hc, vertices, number_of_members, minimum_degrees);
            }

            // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
            if (method_return == 1) {
                delete vertices;
                continue;
            }

            // WRITE TO TASKS
            //sort vertices so that lowest degree vertices are first in enumeration order before writing to tasks
            qsort(vertices, total_vertices, sizeof(Vertex), h_comp_vert_Q);

            if (number_of_candidates > 0) {
                h_write_to_tasks(hd, vertices, total_vertices, write_vertices, write_offsets, write_count);
            }

            delete vertices;
        }
    }

    // FILL TASKS FROM BUFFER
    // if last CPU round copy enough tasks for GPU expansion
    if ((*hd.current_level) == CPU_LEVELS && CPU_EXPAND_THRESHOLD < dss.expand_threshold && (*hd.buffer_count) > 0) {
        h_fill_from_buffer(hd, write_vertices, write_offsets, write_count, dss.expand_threshold);
    }
    // if not enough generated to fully populate fill from buffer
    if (*write_count < CPU_EXPAND_THRESHOLD && (*hd.buffer_count) > 0){
        h_fill_from_buffer(hd, write_vertices, write_offsets, write_count, CPU_EXPAND_THRESHOLD);
    }

    (*hd.current_level)++;
}

// distributes work amongst processes in strided manner
void move_to_gpu(CPU_Data& hd, GPU_Data& h_dd, DS_Sizes& dss)
{
    uint64_t* tasks_count;          // read vertices information
    uint64_t* tasks_offset;
    Vertex* tasks_vertices;         
    uint64_t offset_start;
    uint64_t count;

    // get proper read location for level
    if(CPU_LEVELS % 2 == 1){
        tasks_count = hd.tasks1_count;
        tasks_offset = hd.tasks1_offset;
        tasks_vertices = hd.tasks1_vertices;
    }
    else{
        tasks_count = hd.tasks2_count;
        tasks_offset = hd.tasks2_offset;
        tasks_vertices = hd.tasks2_vertices;
    }

    // each process is assigned tasks in a strided manner, this step condenses those tasks
    count = *tasks_count;
    *tasks_count = 0;
    offset_start = 0;
    for(int i = grank; i < count; i += wsize){
        // increment assigned tasks count
        (*tasks_count)++;

        // copy vertices before offets are changed
        for(int j = tasks_offset[i]; j < tasks_offset[i + 1]; j++){
            tasks_vertices[offset_start + j - tasks_offset[i]] = tasks_vertices[j];
        }

        // copy offset and adjust
        tasks_offset[*tasks_count] = tasks_offset[i + 1] - tasks_offset[i] + offset_start;
        offset_start = tasks_offset[*tasks_count];
    }

    // each process is assigned tasks in a strided manner, this step condenses those tasks
    count = *hd.buffer_count;
    *hd.buffer_count = 0;
    offset_start = 0;
    for(int i = grank; i < count; i += wsize){
        // increment assigned tasks count
        (*hd.buffer_count)++;

        // copy vertices before offets are changed
        for(int j = hd.buffer_offset[i]; j < hd.buffer_offset[i + 1]; j++){
            hd.buffer_vertices[offset_start + j - hd.buffer_offset[i]] = hd.buffer_vertices[j];
        }

        // copy offset and adjust
        hd.buffer_offset[*hd.buffer_count] = hd.buffer_offset[i + 1] - hd.buffer_offset[i] + offset_start;
        offset_start = hd.buffer_offset[*hd.buffer_count];
    }

    // condense tasks
    h_fill_from_buffer(hd, tasks_vertices, tasks_offset, tasks_count, dss.expand_threshold);

    // move to GPU
    chkerr(cudaMemcpy(h_dd.tasks_count, tasks_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.tasks_offset, tasks_offset, (*tasks_count + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.tasks_vertices, tasks_vertices, tasks_offset[*tasks_count] * sizeof(Vertex), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_count, hd.buffer_count, sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_offset, hd.buffer_offset, (*hd.buffer_count + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.buffer_vertices, hd.buffer_vertices, hd.buffer_offset[*hd.buffer_count] * sizeof(Vertex), cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(h_dd.current_level, hd.current_level, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // DEBUG
    if(DEBUG_TOGGLE){
        output_file << "GPU START" << endl;
        print_Data_Sizes(h_dd, dss);
    }
}

// move cliques from device to host
void dump_cliques(CPU_Cliques& hc, GPU_Data& h_dd, ofstream& temp_results, DS_Sizes& dss)
{
    // gpu cliques to cpu cliques
    chkerr(cudaMemcpy(hc.cliques_count, h_dd.cliques_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_offset, h_dd.cliques_offset, sizeof(uint64_t) * dss.cliques_offset_size, cudaMemcpyDeviceToHost));
    chkerr(cudaMemcpy(hc.cliques_vertex, h_dd.cliques_vertex, sizeof(int) * dss.cliques_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    flush_cliques(hc, temp_results);

    cudaMemset(h_dd.cliques_count, 0, sizeof(uint64_t));
}

// move cliques from host to file
void flush_cliques(CPU_Cliques& hc, ofstream& temp_results) 
{
    uint64_t start;         // start of current clique
    uint64_t end;           // end of current clique

    for (int i = 0; i < ((*hc.cliques_count)); i++) {
        start = hc.cliques_offset[i];
        end = hc.cliques_offset[i + 1];
        temp_results << end - start << " ";
        for (uint64_t j = start; j < end; j++) {
            temp_results << hc.cliques_vertex[j] << " ";
        }
        temp_results << "\n";
    }
    ((*hc.cliques_count)) = 0;
}

void free_memory(CPU_Data& hd, GPU_Data& h_dd, CPU_Cliques& hc)
{
    // GPU GRAPH
    chkerr(cudaFree(h_dd.number_of_vertices));
    chkerr(cudaFree(h_dd.number_of_edges));
    chkerr(cudaFree(h_dd.onehop_neighbors));
    chkerr(cudaFree(h_dd.onehop_offsets));
    chkerr(cudaFree(h_dd.twohop_neighbors));
    chkerr(cudaFree(h_dd.twohop_offsets));
    // CPU DATA
    delete hd.tasks1_count;
    delete[] hd.tasks1_offset;
    delete[] hd.tasks1_vertices;
    delete hd.tasks2_count;
    delete[] hd.tasks2_offset;
    delete[] hd.tasks2_vertices;
    delete hd.buffer_count;
    delete[] hd.buffer_offset;
    delete[] hd.buffer_vertices;
    delete hd.current_level;
    delete hd.maximal_expansion;
    delete hd.dumping_cliques;
    delete[] hd.vertex_order_map;
    delete[] hd.remaining_candidates;
    delete hd.remaining_count;
    delete[] hd.removed_candidates;
    delete hd.removed_count;
    delete[] hd.candidate_indegs;
    // GPU DATA
    chkerr(cudaFree(h_dd.current_level));
    chkerr(cudaFree(h_dd.tasks_count));
    chkerr(cudaFree(h_dd.tasks_offset));
    chkerr(cudaFree(h_dd.tasks_vertices));
    chkerr(cudaFree(h_dd.buffer_count));
    chkerr(cudaFree(h_dd.buffer_offset));
    chkerr(cudaFree(h_dd.buffer_vertices));
    chkerr(cudaFree(h_dd.wtasks_count));
    chkerr(cudaFree(h_dd.wtasks_offset));
    chkerr(cudaFree(h_dd.wtasks_vertices));
    chkerr(cudaFree(h_dd.global_vertices));
    chkerr(cudaFree(h_dd.remaining_candidates));
    chkerr(cudaFree(h_dd.lane_remaining_candidates));
    chkerr(cudaFree(h_dd.removed_candidates));
    chkerr(cudaFree(h_dd.lane_removed_candidates));
    chkerr(cudaFree(h_dd.candidate_indegs));
    chkerr(cudaFree(h_dd.lane_candidate_indegs));
    chkerr(cudaFree(h_dd.adjacencies));
    chkerr(cudaFree(h_dd.minimum_degree_ratio));
    chkerr(cudaFree(h_dd.minimum_degrees));
    chkerr(cudaFree(h_dd.minimum_clique_size));
    chkerr(cudaFree(h_dd.total_tasks));
    chkerr(cudaFree(h_dd.current_task));
    // CPU CLIQUES
    delete hc.cliques_count;
    delete[] hc.cliques_vertex;
    delete[] hc.cliques_offset;
    // GPU CLIQUES
    chkerr(cudaFree(h_dd.cliques_count));
    chkerr(cudaFree(h_dd.cliques_vertex));
    chkerr(cudaFree(h_dd.cliques_offset));
    chkerr(cudaFree(h_dd.wcliques_count));
    chkerr(cudaFree(h_dd.wcliques_vertex));
    chkerr(cudaFree(h_dd.wcliques_offset));
    chkerr(cudaFree(h_dd.buffer_offset_start));
    chkerr(cudaFree(h_dd.buffer_start));
    chkerr(cudaFree(h_dd.cliques_offset_start));
    chkerr(cudaFree(h_dd.cliques_start));
    // DATA STRUCTURE SIZES
    chkerr(cudaFree(h_dd.tasks_size));
    chkerr(cudaFree(h_dd.tasks_per_warp));
    chkerr(cudaFree(h_dd.buffer_size));
    chkerr(cudaFree(h_dd.buffer_offset_size));
    chkerr(cudaFree(h_dd.cliques_size));
    chkerr(cudaFree(h_dd.cliques_offset_size));
    chkerr(cudaFree(h_dd.cliques_percent));
    chkerr(cudaFree(h_dd.wcliques_size));
    chkerr(cudaFree(h_dd.wcliques_offset_size));
    chkerr(cudaFree(h_dd.wtasks_size));
    chkerr(cudaFree(h_dd.wtasks_offset_size));
    chkerr(cudaFree(h_dd.wvertices_size));
    chkerr(cudaFree(h_dd.expand_threshold));
    chkerr(cudaFree(h_dd.cliques_dump));
}

// --- SECONDARY EXPNASION FUNCTIONS ---
// returns 1 if lookahead was a success, else 0
int h_lookahead_pruning(CPU_Graph& hg, CPU_Cliques& hc, CPU_Data& hd, Vertex* read_vertices, int tot_vert, int num_mem, int num_cand, uint64_t start, int* minimum_degrees)
{
    int pvertexid;                      // used for intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    uint64_t start_write;               // starting write position for new cliques

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
    start_write = hc.cliques_offset[(*hc.cliques_count)];
    for (int j = 0; j < tot_vert; j++) {
        hc.cliques_vertex[start_write + j] = read_vertices[start + j].vertexid;
    }
    (*hc.cliques_count)++;
    hc.cliques_offset[(*hc.cliques_count)] = start_write + tot_vert;

    return 1;
}

// returns 1 is failed found or not enough vertices, else 0
int h_remove_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* read_vertices, int& tot_vert, int& num_cand, int& num_mem, uint64_t start, int* minimum_degrees, int minimum_clique_size)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;          
    uint64_t pneighbors_end;
    int phelper1;
    int mindeg;                         // helper variables
    bool failed_found;

    mindeg = h_get_mindeg(num_mem, minimum_degrees, minimum_clique_size);

    // remove one vertex
    num_cand--;
    tot_vert--;

    // initialize vertex order map
    for (int i = 0; i < tot_vert; i++) {
        hd.vertex_order_map[read_vertices[start + i].vertexid] = i;
    }

    failed_found = false;

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

// returns 1 if failed found or invalid bound, 0 otherwise
int h_add_one_vertex(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size)
{
    bool method_return;                 // helper variables
    int pvertexid;                      // intersection
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
    method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

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
int h_critical_vertex_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int& number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    bool critical_fail;                 // helper
    int number_of_crit_adj;
    int* adj_counters;
    bool method_return;                 // return

    // initialize vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = i;
    }

    // CRITICAL VERTEX PRUNING
    // adj_counter[0] = 10, means that the vertex at position 0 in new_vertices has 10 critical vertices neighbors within 2 hops
    adj_counters = new int[total_vertices];
    memset(adj_counters, 0, sizeof(int) * total_vertices);

    // iterate through all vertices in clique
    for (int k = 0; k < number_of_members; k++)
    {
        // if they are a critical vertex
        if (vertices[k].indeg + vertices[k].exdeg == minimum_degrees[number_of_members + lower_bound] && vertices[k].exdeg > 0) {
            pvertexid = vertices[k].vertexid;

            // iterate through all neighbors
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (uint64_t l = pneighbors_start; l < pneighbors_end; l++) {
                phelper1 = hd.vertex_order_map[hg.onehop_neighbors[l]];

                // if neighbor is cand
                if (phelper1 >= number_of_members) {
                    vertices[phelper1].label = 4;
                }
            }
        }
    }

    // reset vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = -1;
    }

    // sort vertices so that critical vertex adjacent candidates are immediately after vertices within the clique
    qsort(vertices + number_of_members, number_of_candidates, sizeof(Vertex), h_comp_vert_cv);

    // calculate number of critical adjacent vertices
    number_of_crit_adj = 0;
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 4) {
            number_of_crit_adj++;
        }
        else {
            break;
        }
    }

    // if there were any neighbors of critical vertices
    if (number_of_crit_adj > 0)
    {
        // initialize vertex order map
        for (int i = 0; i < total_vertices; i++) {
            hd.vertex_order_map[vertices[i].vertexid] = i;
        }

        // iterate through all neighbors
        for (int i = number_of_members; i < number_of_members + number_of_crit_adj; i++) {
            pvertexid = vertices[i].vertexid;

            // update 1hop adj
            pneighbors_start = hg.onehop_offsets[pvertexid];
            pneighbors_end = hg.onehop_offsets[pvertexid + 1];
            for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {
                phelper1 = hd.vertex_order_map[hg.onehop_neighbors[k]];

                if (phelper1 > -1) {
                    vertices[phelper1].indeg++;
                    vertices[phelper1].exdeg--;
                }
            }

            // track 2hop adj
            pneighbors_start = hg.twohop_offsets[pvertexid];
            pneighbors_end = hg.twohop_offsets[pvertexid + 1];
            for (uint64_t k = pneighbors_start; k < pneighbors_end; k++) {
                phelper1 = hd.vertex_order_map[hg.twohop_neighbors[k]];

                if (phelper1 > -1) {
                    adj_counters[phelper1]++;
                }
            }
        }

        critical_fail = false;

        // all vertices within the clique must be within 2hops of the newly ah_dded critical vertex adj vertices
        for (int k = 0; k < number_of_members; k++) {
            if (adj_counters[k] != number_of_crit_adj) {
                critical_fail = true;
            }
        }

        if (critical_fail) {
            // reset vertex order map
            for (int i = 0; i < total_vertices; i++) {
                hd.vertex_order_map[vertices[i].vertexid] = -1;
            }
            delete adj_counters;
            return 2;
        }

        // all critical adj vertices must all be within 2 hops of each other
        for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++) {
            if (adj_counters[k] < number_of_crit_adj - 1) {
                critical_fail = true;
            }
        }

        if (critical_fail) {
            // reset vertex order map
            for (int i = 0; i < total_vertices; i++) {
                hd.vertex_order_map[vertices[i].vertexid] = -1;
            }
            delete adj_counters;
            return 2;
        }

        // no failed vertices found so ah_dd all critical vertex adj candidates to clique
        for (int k = number_of_members; k < number_of_members + number_of_crit_adj; k++) {
            vertices[k].label = 1;
        }
        number_of_members += number_of_crit_adj;
        number_of_candidates -= number_of_crit_adj;
    }

    // DIAMTER PRUNING
    (*hd.remaining_count) = 0;

    // remove all cands who are not within 2hops of all newly ah_dded cands
    for (int k = number_of_members; k < total_vertices; k++) {
        if (adj_counters[k] == number_of_crit_adj) {
            hd.candidate_indegs[(*hd.remaining_count)++] = vertices[k].indeg;
        }
        else {
            vertices[k].label = -1;
        }
    }

    // DEGREE-BASED PRUNING
    method_return = h_degree_pruning(hg, hd, vertices, total_vertices, number_of_candidates, number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_degree_ratio, minimum_clique_size);

    // reset vertex order map
    for (int i = 0; i < total_vertices; i++) {
        hd.vertex_order_map[vertices[i].vertexid] = -1;
    }

    delete adj_counters;

    // if vertex in x found as not extendable, check if current set is clique and continue to next iteration
    if (method_return) {
        return 1;
    }

    return 0;
}

void h_diameter_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int pvertexid, int& total_vertices, int& number_of_candidates, int number_of_members)
{
    uint64_t pneighbors_start;          // intersection
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
bool h_degree_pruning(CPU_Graph& hg, CPU_Data& hd, Vertex* vertices, int& total_vertices, int& number_of_candidates, int number_of_members, int& upper_bound, int& lower_bound, int& min_ext_deg, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size)
{
    int pvertexid;                      // intersection
    uint64_t pneighbors_start;
    uint64_t pneighbors_end;
    int phelper1;
    int num_val_cands;                  // helper variables

    qsort(hd.candidate_indegs, (*hd.remaining_count), sizeof(int), h_comp_int_desc);

    // if invalid bounds found while calculating lower and upper bounds
    if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, (*hd.remaining_count), minimum_degrees, minimum_degree_ratio, minimum_clique_size)) {
        return true;
    }

    // check for failed vertices
    for (int k = 0; k < number_of_members; k++) {
        if (!h_vert_isextendable(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_clique_size)) {
            return true;
        }
    }

    (*hd.remaining_count) = 0;
    (*hd.removed_count) = 0;

    // check for invalid candidates
    for (int i = number_of_members; i < total_vertices; i++) {
        if (vertices[i].label == 0 && h_cand_isvalid(vertices[i], number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_clique_size)) {
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
            if (h_cand_isvalid(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_clique_size)) {
                hd.candidate_indegs[num_val_cands++] = vertices[hd.remaining_candidates[k]].indeg;
            }
        }

        qsort(hd.candidate_indegs, num_val_cands, sizeof(int), h_comp_int_desc);

        // if invalid bounds found while calculating lower and upper bounds
        if (h_calculate_LU_bounds(hd, upper_bound, lower_bound, min_ext_deg, vertices, number_of_members, num_val_cands, minimum_degrees, minimum_degree_ratio, minimum_clique_size)) {
            return true;
        }

        // check for failed vertices
        for (int k = 0; k < number_of_members; k++) {
            if (!h_vert_isextendable(vertices[k], number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_clique_size)) {
                return true;
            }
        }

        num_val_cands = 0;
        (*hd.removed_count) = 0;

        // check for invalid candidates
        for (int k = 0; k < (*hd.remaining_count); k++) {
            if (h_cand_isvalid(vertices[hd.remaining_candidates[k]], number_of_members, upper_bound, lower_bound, min_ext_deg, minimum_degrees, minimum_clique_size)) {
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

bool h_calculate_LU_bounds(CPU_Data& hd, int& upper_bound, int& lower_bound, int& min_ext_deg, Vertex* vertices, int number_of_members, int number_of_candidates, int* minimum_degrees, double minimum_degree_ratio, int minimum_clique_size)
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

    min_ext_deg = h_get_mindeg(number_of_members + 1, minimum_degrees, minimum_clique_size);

    if (min_clq_indeg < minimum_degrees[number_of_members])
    {
        // lower
        lower_bound = h_get_mindeg(number_of_members, minimum_degrees, minimum_clique_size) - min_clq_indeg;

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
                    min_ext_deg = h_get_mindeg(number_of_members + lower_bound, minimum_degrees, minimum_clique_size);
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

void h_check_for_clique(CPU_Cliques& hc, Vertex* vertices, int number_of_members, int* minimum_degrees)
{
    bool clique = true;
    int degree_requirement;

    degree_requirement = minimum_degrees[number_of_members];
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

    if ((*write_count) < CPU_EXPAND_THRESHOLD) {
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

void h_fill_from_buffer(CPU_Data& hd, Vertex* write_vertices, uint64_t* write_offsets, uint64_t* write_count, int threshold)
{
    int write_amount;
    uint64_t start_buffer;
    uint64_t end_buffer;
    uint64_t size_buffer;
    uint64_t start_write;

    // read from end of buffer, write to end of tasks, decrement buffer
    (*hd.maximal_expansion) = false;

    // get read and write locations
    write_amount = ((*hd.buffer_count) >= (threshold - *write_count)) ? threshold - *write_count : (*hd.buffer_count);
    start_buffer = hd.buffer_offset[(*hd.buffer_count) - write_amount];
    end_buffer = hd.buffer_offset[(*hd.buffer_count)];
    size_buffer = end_buffer - start_buffer;
    start_write = write_offsets[*write_count];

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