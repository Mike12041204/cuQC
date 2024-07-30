#include "../inc/common.h"
#include "../inc/device_kernels.h"

// --- PRIMARY KERNELS ---
__global__ void d_expand_level(GPU_Data* dd)
{
    __shared__ Warp_Data wd;        // data is stored in data structures to reduce the number of variables that need to be passed to methods
    Local_Data ld;
    int num_mem;                    // helper variables, not passed through to any methods
    int method_return;
    int index;

    // --- CURRENT LEVEL ---

    // initialize i for each warp
    int i = 0;
    if (LANE_IDX == 0) {
        i = atomicAdd(dd->current_task, 1);
    }
    i = __shfl_sync(0xFFFFFFFF, i, 0);

    while (i < *dd->tasks_count) {

        // get information on vertices being handled within tasks
        if (LANE_IDX == 0) {
            wd.start[WIB_IDX] = dd->tasks_offset[i];
            wd.end[WIB_IDX] = dd->tasks_offset[i + 1];
            wd.tot_vert[WIB_IDX] = wd.end[WIB_IDX] - wd.start[WIB_IDX];
        }
        __syncwarp();

        // each warp gets partial number of members
        num_mem = 0;
        for (uint64_t j = wd.start[WIB_IDX] + LANE_IDX; j < wd.end[WIB_IDX]; j += WARP_SIZE) {
            if (dd->tasks_vertices[j].label != 1) {
                break;
            }
            num_mem++;
        }
        // sum members across warp
        for (int k = 1; k < 32; k *= 2) {
            num_mem += __shfl_xor_sync(0xFFFFFFFF, num_mem, k);
        }

        if (LANE_IDX == 0) {
            wd.num_mem[WIB_IDX] = num_mem;
            wd.num_cand[WIB_IDX] = wd.tot_vert[WIB_IDX] - wd.num_mem[WIB_IDX];
            wd.expansions[WIB_IDX] = wd.num_cand[WIB_IDX];
        }
        __syncwarp();

        // LOOKAHEAD PRUNING
        method_return = d_lookahead_pruning(dd, wd, ld);
        if (method_return) {
            // schedule warps next task
            if (LANE_IDX == 0) {
                i = atomicAdd(dd->current_task, 1);
            }
            i = __shfl_sync(0xFFFFFFFF, i, 0);
            continue;
        }

        // --- NEXT LEVEL ---
        for (int j = 0; j < wd.expansions[WIB_IDX]; j++)
        {

            // REMOVE ONE VERTEX
            if (j > 0) {
                method_return = d_remove_one_vertex(dd, wd, ld);
                if (method_return) {
                    break;
                }
            }

            // INITIALIZE NEW VERTICES
            if (LANE_IDX == 0) {
                wd.number_of_members[WIB_IDX] = wd.num_mem[WIB_IDX];
                wd.number_of_candidates[WIB_IDX] = wd.num_cand[WIB_IDX];
                wd.total_vertices[WIB_IDX] = wd.tot_vert[WIB_IDX];
            }
            __syncwarp();

            // select whether to store vertices in global or shared memory based on size
            if (wd.total_vertices[WIB_IDX] <= VERTICES_SIZE) {
                ld.vertices = wd.shared_vertices + (VERTICES_SIZE * WIB_IDX);
            }
            else {
                ld.vertices = dd->global_vertices + (*dd->wvertices_size * WARP_IDX);
            }

            // copy vertices
            for (index = LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE) {
                ld.vertices[index] = dd->tasks_vertices[wd.start[WIB_IDX] + index];
            }
            for (; index < wd.total_vertices[WIB_IDX] - 1; index += WARP_SIZE) {
                ld.vertices[index + 1] = dd->tasks_vertices[wd.start[WIB_IDX] + index];
            }
            if (LANE_IDX == 0) {
                ld.vertices[wd.number_of_members[WIB_IDX]] = dd->tasks_vertices[wd.start[WIB_IDX] + wd.total_vertices[WIB_IDX] - 1];
            }
            __syncwarp();

            // ADD ONE VERTEX
            method_return = d_add_one_vertex(dd, wd, ld);

            // if failed found check for clique and continue on to the next iteration
            if (method_return == 1) {
                if (wd.number_of_members[WIB_IDX] >= (*dd->minimum_clique_size)) {
                    d_check_for_clique(dd, wd, ld);
                }
                continue;
            }

            // CRITICAL VERTEX PRUNING
            method_return = d_critical_vertex_pruning(dd, wd, ld);

            // critical fail, cannot be clique continue onto next iteration
            if (method_return == 2) {
                continue;
            }

            // HANDLE CLIQUES
            if (wd.number_of_members[WIB_IDX] >= (*dd->minimum_clique_size)) {
                d_check_for_clique(dd, wd, ld);
            }

            // if vertex in x found as not extendable continue to next iteration
            if (method_return == 1) {
                continue;
            }

            // WRITE TASKS TO BUFFERS
            // sort vertices in Quick efficient enumeration order before writing
            d_oe_sort_vert(ld.vertices, wd.total_vertices[WIB_IDX], d_comp_vert_Q);

            if (wd.number_of_candidates[WIB_IDX] > 0) {
                d_write_to_tasks(dd, wd, ld);
            }
        }

        // schedule warps next task
        if (LANE_IDX == 0) {
            i = atomicAdd(dd->current_task, 1);
        }
        i = __shfl_sync(0xFFFFFFFF, i, 0);
    }

    if (LANE_IDX == 0) {
        // sum to find tasks count
        atomicAdd(dd->total_tasks, dd->wtasks_count[WARP_IDX]);
        atomicAdd(dd->total_cliques, dd->wcliques_count[WARP_IDX]);
    }

    if (IDX == 0) {
        *dd->buffer_offset_start = *dd->buffer_count + 1;
        *dd->buffer_start = dd->buffer_offset[*dd->buffer_count];
        *dd->cliques_offset_start = *dd->cliques_count + 1;
        *dd->cliques_start = dd->cliques_offset[*dd->cliques_count];
    }
}

__global__ void transfer_buffers(GPU_Data* dd)
{
    __shared__ uint64_t tasks_write[WARPS_PER_BLOCK];
    __shared__ int tasks_offset_write[WARPS_PER_BLOCK];
    __shared__ uint64_t cliques_write[WARPS_PER_BLOCK];
    __shared__ int cliques_offset_write[WARPS_PER_BLOCK];
    __shared__ int twarp;
    __shared__ int toffsetwrite;
    __shared__ int twrite;
    __shared__ int tasks_end;

    // point of this is to find how many vertices will be transfered to tasks, it is easy to know how many tasks as it will just
    // be the expansion threshold, but to find how many vertices we must now the total size of all the tasks that will be copied.
    // each block does this but really could be done by one thread outside the GPU
    if (threadIdx.x == 0) {
        toffsetwrite = 0;
        twrite = 0;

        for (int i = 0; i < NUMBER_OF_WARPS; i++) {
            // if next warps count is more than expand threshold mark as such and break
            if (toffsetwrite + dd->wtasks_count[i] >= *dd->expand_threshold) {
                twarp = i;
                break;
            }
            // else adds its size and count
            twrite += dd->wtasks_offset[(*dd->wtasks_offset_size * i) + dd->wtasks_count[i]];
            toffsetwrite += dd->wtasks_count[i];
        }
        // final size is the size of all tasks up until last warp and the remaining tasks in the last warp until expand threshold is satisfied
        tasks_end = twrite + dd->wtasks_offset[(*dd->wtasks_offset_size * twarp) + (*dd->expand_threshold - toffsetwrite)];
    }
    __syncthreads();

    // warp level
    if (LANE_IDX == 0) {
        tasks_write[WIB_IDX] = 0;
        tasks_offset_write[WIB_IDX] = 1;
        cliques_write[WIB_IDX] = 0;
        cliques_offset_write[WIB_IDX] = 1;

        for (int i = 0; i < WARP_IDX; i++) {
            tasks_offset_write[WIB_IDX] += dd->wtasks_count[i];
            tasks_write[WIB_IDX] += dd->wtasks_offset[(*dd->wtasks_offset_size * i) + dd->wtasks_count[i]];

            cliques_offset_write[WIB_IDX] += dd->wcliques_count[i];
            cliques_write[WIB_IDX] += dd->wcliques_offset[(*dd->wcliques_offset_size * i) + dd->wcliques_count[i]];
        }
    }
    __syncwarp();
    
    // move to tasks and buffer
    for (int i = LANE_IDX + 1; i <= dd->wtasks_count[WARP_IDX]; i += WARP_SIZE) {
        if (tasks_offset_write[WIB_IDX] + i - 1 <= *dd->expand_threshold) {
            // to tasks
            dd->tasks_offset[tasks_offset_write[WIB_IDX] + i - 1] = dd->wtasks_offset[(*dd->wtasks_offset_size * WARP_IDX) + i] + tasks_write[WIB_IDX];
        }
        else {
            // to buffer
            dd->buffer_offset[tasks_offset_write[WIB_IDX] + i - 2 - *dd->expand_threshold + *dd->buffer_offset_start] = dd->wtasks_offset[(*dd->wtasks_offset_size * WARP_IDX) + i] +
                tasks_write[WIB_IDX] - tasks_end + *dd->buffer_start;
        }
    }

    for (int i = LANE_IDX; i < dd->wtasks_offset[(*dd->wtasks_offset_size * WARP_IDX) + dd->wtasks_count[WARP_IDX]]; i += WARP_SIZE) {
        if (tasks_write[WIB_IDX] + i < tasks_end) {
            // to tasks
            dd->tasks_vertices[tasks_write[WIB_IDX] + i] = dd->wtasks_vertices[(*dd->wtasks_size * WARP_IDX) + i];
        }
        else {
            // to buffer
            dd->buffer_vertices[*dd->buffer_start + tasks_write[WIB_IDX] + i - tasks_end] = dd->wtasks_vertices[(*dd->wtasks_size * WARP_IDX) + i];
        }
    }
    // NOTE - this sync is important for some reason, larger graphs/et dont work without it
    __syncthreads();

    //move to cliques
    for (int i = LANE_IDX + 1; i <= dd->wcliques_count[WARP_IDX]; i += WARP_SIZE) {
        dd->cliques_offset[*dd->cliques_offset_start + cliques_offset_write[WIB_IDX] + i - 2] = dd->wcliques_offset[(*dd->wcliques_offset_size * WARP_IDX) + i] + *dd->cliques_start + 
            cliques_write[WIB_IDX];
    }
    for (int i = LANE_IDX; i < dd->wcliques_offset[(*dd->wcliques_offset_size * WARP_IDX) + dd->wcliques_count[WARP_IDX]]; i += WARP_SIZE) {
        dd->cliques_vertex[*dd->cliques_start + cliques_write[WIB_IDX] + i] = dd->wcliques_vertex[(*dd->wcliques_size * WARP_IDX) + i];
    }

    if (IDX == 0) {
        // handle tasks and buffer counts
        if (*dd->total_tasks <= *dd->expand_threshold) {
            *dd->tasks_count = *dd->total_tasks;
        }
        else {
            *dd->tasks_count = *dd->expand_threshold;
            *dd->buffer_count += *dd->total_tasks - *dd->expand_threshold;
        }
        *dd->cliques_count += *dd->total_cliques;

        *dd->total_tasks = 0;
        *dd->total_cliques = 0;
        (*dd->current_level)++;
    }
}

__global__ void fill_from_buffer(GPU_Data* dd)
{
    // get read and write locations
    int write_amount = (*dd->buffer_count >= *dd->expand_threshold - *dd->tasks_count) ? *dd->expand_threshold - *dd->tasks_count : *dd->buffer_count;
    uint64_t start_buffer = dd->buffer_offset[*dd->buffer_count - write_amount];
    uint64_t end_buffer = dd->buffer_offset[*dd->buffer_count];
    uint64_t size_buffer = end_buffer - start_buffer;
    uint64_t start_write = dd->tasks_offset[*dd->tasks_count];

    // handle offsets
    for (int i = IDX + 1; i <= write_amount; i += NUMBER_OF_THREADS) {
        dd->tasks_offset[*dd->tasks_count + i] = start_write + dd->buffer_offset[*dd->buffer_count - write_amount + i] - start_buffer;
    }

    // handle data
    for (int i = IDX; i < size_buffer; i += NUMBER_OF_THREADS) {
        dd->tasks_vertices[start_write + i] = dd->buffer_vertices[start_buffer + i];
    }

    if (IDX == 0) {
        *dd->tasks_count += write_amount;
        *dd->buffer_count -= write_amount;
    }
}

// --- SECONDARY EXPANSION KERNELS ---
// returns 1 if lookahead succesful, 0 otherwise  
__device__ int d_lookahead_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    int phelper2;
    uint64_t start_write;

    if (LANE_IDX == 0) {
        wd.success[WIB_IDX] = true;
    }
    __syncwarp();

    // check if members meet degree requirement, dont need to check 2hop adj as diameter pruning guarentees all members will be within 2hops of eveything
    for (int i = LANE_IDX; i < wd.num_mem[WIB_IDX] && wd.success[WIB_IDX]; i += WARP_SIZE) {
        if (dd->tasks_vertices[wd.start[WIB_IDX] + i].indeg + dd->tasks_vertices[wd.start[WIB_IDX] + i].exdeg < dd->minimum_degrees[wd.tot_vert[WIB_IDX]]) {
            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();

    if (!wd.success[WIB_IDX]) {
        return 0;
    }

    // update lvl2adj to candidates for all vertices
    for (int i = wd.num_mem[WIB_IDX] + LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE) {
        pvertexid = dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid;
        
        for (int j = wd.num_mem[WIB_IDX]; j < wd.tot_vert[WIB_IDX]; j++) {
            if (j == i) {
                continue;
            }

            phelper1 = dd->tasks_vertices[wd.start[WIB_IDX] + j].vertexid;
            phelper2 = d_b_search_int(dd->twohop_neighbors + dd->twohop_offsets[phelper1], dd->twohop_offsets[phelper1 + 1] - dd->twohop_offsets[phelper1], pvertexid);
        
            if (phelper2 > -1) {
                dd->tasks_vertices[wd.start[WIB_IDX] + i].lvl2adj++;
            }
        }
    }
    __syncwarp();

    // compares all vertices to the lemmas from Quick
    for (int j = wd.num_mem[WIB_IDX] + LANE_IDX; j < wd.tot_vert[WIB_IDX] && wd.success[WIB_IDX]; j += WARP_SIZE) {
        if (dd->tasks_vertices[wd.start[WIB_IDX] + j].lvl2adj < wd.num_cand[WIB_IDX] - 1 || dd->tasks_vertices[wd.start[WIB_IDX] + j].indeg + dd->tasks_vertices[wd.start[WIB_IDX] + j].exdeg < dd->minimum_degrees[wd.tot_vert[WIB_IDX]]) {
            wd.success[WIB_IDX] = false;
            break;
        }
    }
    __syncwarp();

    if (wd.success[WIB_IDX]) {
        // write to cliques
        start_write = (*dd->wcliques_size * WARP_IDX) + dd->wcliques_offset[(*dd->wcliques_offset_size * WARP_IDX) + dd->wcliques_count[WARP_IDX]];
        for (int j = LANE_IDX; j < wd.tot_vert[WIB_IDX]; j += WARP_SIZE) {
            dd->wcliques_vertex[start_write + j] = dd->tasks_vertices[wd.start[WIB_IDX] + j].vertexid;
        }
        if (LANE_IDX == 0) {
            (dd->wcliques_count[WARP_IDX])++;
            dd->wcliques_offset[(*dd->wcliques_offset_size * WARP_IDX) + dd->wcliques_count[WARP_IDX]] = start_write - (*dd->wcliques_size * WARP_IDX) + wd.tot_vert[WIB_IDX];
        }
        return 1;
    }

    return 0;
}

// returns 1 if failed found after removing, 0 otherwise
__device__ int d_remove_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    int phelper2;
    int mindeg;

    mindeg = d_get_mindeg(wd.num_mem[WIB_IDX], dd);

    // remove the last candidate in vertices
    if (LANE_IDX == 0) {
        wd.num_cand[WIB_IDX]--;
        wd.tot_vert[WIB_IDX]--;
        wd.success[WIB_IDX] = false;
    }
    __syncwarp();

    // update info of vertices connected to removed cand
    pvertexid = dd->tasks_vertices[wd.start[WIB_IDX] + wd.tot_vert[WIB_IDX]].vertexid;

    for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX] && !wd.success[WIB_IDX]; i += WARP_SIZE) {
        phelper1 = dd->tasks_vertices[wd.start[WIB_IDX] + i].vertexid;
        phelper2 = d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[pvertexid], dd->onehop_offsets[pvertexid + 1] - dd->onehop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            dd->tasks_vertices[wd.start[WIB_IDX] + i].exdeg--;

            if (phelper1 < wd.num_mem[WIB_IDX] && dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].indeg + dd->tasks_vertices[wd.start[WIB_IDX] + phelper1].exdeg < mindeg) {
                wd.success[WIB_IDX] = true;
                break;
            }
        }
    }
    __syncwarp();

    if (wd.success[WIB_IDX]) {
        return 1;
    }

    return 0;
}

// returns 1 if failed found or invalid bound, 0 otherwise 
__device__ int d_add_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int pvertexid;
    int phelper1;
    int phelper2;
    bool failed_found;

    // ADD ONE VERTEX
    pvertexid = ld.vertices[wd.number_of_members[WIB_IDX]].vertexid;

    if (LANE_IDX == 0) {
        ld.vertices[wd.number_of_members[WIB_IDX]].label = 1;
        wd.number_of_members[WIB_IDX]++;
        wd.number_of_candidates[WIB_IDX]--;
    }
    __syncwarp();

    for (int i = LANE_IDX; i < wd.tot_vert[WIB_IDX]; i += WARP_SIZE) {
        phelper1 = ld.vertices[i].vertexid;
        phelper2 = d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[pvertexid], dd->onehop_offsets[pvertexid + 1] - dd->onehop_offsets[pvertexid], phelper1);

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

// returns 2, if critical fail, 1 if failed found or invalid bound, 0 otherwise
__device__ int d_critical_vertex_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    int phelper1;                   // intersection
    int number_of_crit_adj;         // pruning
    bool failed_found;

    // CRITICAL VERTEX PRUNING 
    // iterate through all vertices in clique
    for (int k = 0; k < wd.number_of_members[WIB_IDX]; k++) {

        // if they are a critical vertex
        if (ld.vertices[k].indeg + ld.vertices[k].exdeg == dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]] && ld.vertices[k].exdeg > 0) {
            phelper1 = ld.vertices[k].vertexid;

            // iterate through all candidates
            for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
                if (ld.vertices[i].label != 4) {
                    // if candidate is neighbor of critical vertex mark as such
                    if (d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[phelper1], dd->onehop_offsets[phelper1 + 1] - dd->onehop_offsets[phelper1], ld.vertices[i].vertexid) > -1) {
                        ld.vertices[i].label = 4;
                    }
                }
            }
        }
        __syncwarp();
    }

    // sort vertices so that critical vertex adjacent candidates are immediately after vertices within the clique
    d_oe_sort_vert(ld.vertices + wd.number_of_members[WIB_IDX], wd.number_of_candidates[WIB_IDX], d_comp_vert_cv);

    // count number of critical adjacent vertices
    number_of_crit_adj = 0;
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        if (ld.vertices[i].label == 4) {
            number_of_crit_adj++;
        }
        else {
            break;
        }
    }
    // get sum
    for (int i = 1; i < 32; i *= 2) {
        number_of_crit_adj += __shfl_xor_sync(0xFFFFFFFF, number_of_crit_adj, i);
    }

    failed_found = false;

    // reset adjacencies
    for (int i = LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        dd->adjacencies[(*dd->wvertices_size * WARP_IDX) + i] = 0;
    }

    // if there were any neighbors of critical vertices
    if (number_of_crit_adj > 0)
    {
        // iterate through all vertices and update their degrees as if critical adjacencies were added and keep track of how many critical adjacencies they are adjacent to
        for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
            phelper1 = ld.vertices[k].vertexid;

            for (int i = wd.number_of_members[WIB_IDX]; i < wd.number_of_members[WIB_IDX] + number_of_crit_adj; i++) {
                if (d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[phelper1], dd->onehop_offsets[phelper1 + 1] - dd->onehop_offsets[phelper1], ld.vertices[i].vertexid) > -1) {
                    ld.vertices[k].indeg++;
                    ld.vertices[k].exdeg--;
                }

                if (d_b_search_int(dd->twohop_neighbors + dd->twohop_offsets[phelper1], dd->twohop_offsets[phelper1 + 1] - dd->twohop_offsets[phelper1], ld.vertices[i].vertexid) > -1) {
                    dd->adjacencies[(*dd->wvertices_size * WARP_IDX) + k]++;
                }
            }
        }
        __syncwarp();

        // all vertices within the clique must be within 2hops of the newly added critical vertex adj vertices
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
            if (dd->adjacencies[(*dd->wvertices_size * WARP_IDX) + k] != number_of_crit_adj) {
                failed_found = true;
                break;
            }
        }
        failed_found = __any_sync(0xFFFFFFFF, failed_found);
        if (failed_found) {
            return 2;
        }

        // all critical adj vertices must all be within 2 hops of each other
        for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.number_of_members[WIB_IDX] + number_of_crit_adj; k += WARP_SIZE) {
            if (dd->adjacencies[(*dd->wvertices_size * WARP_IDX) + k] < number_of_crit_adj - 1) {
                failed_found = true;
                break;
            }
        }
        failed_found = __any_sync(0xFFFFFFFF, failed_found);
        if (failed_found) {
            return 2;
        }

        // no failed vertices found so add all critical vertex adj candidates to clique
        for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.number_of_members[WIB_IDX] + number_of_crit_adj; k += WARP_SIZE) {
            ld.vertices[k].label = 1;
        }

        if (LANE_IDX == 0) {
            wd.number_of_members[WIB_IDX] += number_of_crit_adj;
            wd.number_of_candidates[WIB_IDX] -= number_of_crit_adj;
        }
        __syncwarp();
    }

    // DIAMTER PRUNING
    d_diameter_pruning_cv(dd, wd, ld, number_of_crit_adj);

    // DEGREE BASED PRUNING
    failed_found = d_degree_pruning(dd, wd, ld);

    // if vertex in x found as not extendable continue to next iteration
    if (failed_found) {
        return 1;
    }

    return 0;
}

// diameter pruning intitializes vertices labels and candidate indegs array for use in iterative degree pruning
__device__ void d_diameter_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int pvertexid)
{
    int lane_write;
    int phelper1;                       // intersection
    int phelper2;
    int lane_remaining_count;           // vertex iteration

    lane_write = (*dd->wvertices_size * WARP_IDX) + ((*dd->wvertices_size / WARP_SIZE) * LANE_IDX);
    lane_remaining_count = 0;

    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        ld.vertices[i].label = -1;
    }
    __syncwarp();

    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        phelper1 = ld.vertices[i].vertexid;
        phelper2 = d_b_search_int(dd->twohop_neighbors + dd->twohop_offsets[pvertexid], dd->twohop_offsets[pvertexid + 1] - dd->twohop_offsets[pvertexid], phelper1);

        if (phelper2 > -1) {
            ld.vertices[i].label = 0;
            dd->lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[i].indeg;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    __syncwarp();

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd->candidate_indegs[(*dd->wvertices_size * WARP_IDX) + lane_remaining_count + i] = dd->lane_candidate_indegs[lane_write + i];
    }
    __syncwarp();
}

__device__ void d_diameter_pruning_cv(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int number_of_crit_adj)
{
    int lane_write;
    int lane_remaining_count;           // vertex iteration
    int phelper1;                       // intersection
    int phelper2;

    lane_write = (*dd->wvertices_size * WARP_IDX) + ((*dd->wvertices_size / WARP_SIZE) * LANE_IDX);
    lane_remaining_count = 0;

    // remove all cands who are not within 2hops of all newly added cands
    for (int k = wd.number_of_members[WIB_IDX] + LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
        if (dd->adjacencies[(*dd->wvertices_size * WARP_IDX) + k] == number_of_crit_adj) {
            dd->lane_candidate_indegs[lane_write + lane_remaining_count++] = ld.vertices[k].indeg;
        }
        else {
            ld.vertices[k].label = -1;
        }
    }

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    __syncwarp();

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd->candidate_indegs[(*dd->wvertices_size * WARP_IDX) + lane_remaining_count + i] = dd->lane_candidate_indegs[lane_write + i];
    }
    __syncwarp();
}

// returns true if invalid bounds or failed found
__device__ bool d_degree_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    // vertices size * warp idx + (vertices size / warp size) * lane idx
    int lane_write = ((*dd->wvertices_size * WARP_IDX) + ((*dd->wvertices_size / WARP_SIZE) * LANE_IDX));

    // helper variables used throughout method to store various values, names have no meaning
    int pvertexid;
    int phelper1;
    int phelper2;
    Vertex* read;
    Vertex* write;
    // counter for lane intersection results
    int lane_remaining_count;
    int lane_removed_count;

    d_oe_sort_int(dd->candidate_indegs + (*dd->wvertices_size * WARP_IDX), wd.remaining_count[WIB_IDX], d_comp_int_desc);

    d_calculate_LU_bounds(dd, wd, ld, wd.remaining_count[WIB_IDX]);
    if (wd.success[WIB_IDX]) {
        return true;
    }

    // check for failed vertices
    __syncwarp();
    for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX] && !wd.success[WIB_IDX]; k += WARP_SIZE) {
        if (!d_vert_isextendable(ld.vertices[k], dd, wd, ld)) {
            wd.success[WIB_IDX] = true;
            break;
        }

    }
    __syncwarp();
    if (wd.success[WIB_IDX]) {
        return true;
    }

    if (LANE_IDX == 0) {
        wd.remaining_count[WIB_IDX] = 0;
        wd.removed_count[WIB_IDX] = 0;
        wd.rw_counter[WIB_IDX] = 0;
    }

    lane_remaining_count = 0;
    lane_removed_count = 0;
    
    for (int i = wd.number_of_members[WIB_IDX] + LANE_IDX; i < wd.total_vertices[WIB_IDX]; i += WARP_SIZE) {
        if (ld.vertices[i].label == 0 && d_cand_isvalid(ld.vertices[i], dd, wd, ld)) {
            dd->lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
        }
        else {
            dd->lane_removed_candidates[lane_write + lane_removed_count++] = i;
        }
    }
    __syncwarp();

    // scan to calculate write postion in warp arrays
    phelper2 = lane_remaining_count;
    pvertexid = lane_removed_count;
    for (int i = 1; i < WARP_SIZE; i *= 2) {
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_remaining_count += phelper1;
        }
        phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
        if (LANE_IDX >= i) {
            lane_removed_count += phelper1;
        }
        __syncwarp();
    }
    // lane remaining count sum is scan for last lane and its value
    if (LANE_IDX == WARP_SIZE - 1) {
        wd.remaining_count[WIB_IDX] = lane_remaining_count;
        wd.removed_count[WIB_IDX] = lane_removed_count;
    }
    // make scan exclusive
    lane_remaining_count -= phelper2;
    lane_removed_count -= pvertexid;

    // parallel write lane arrays to warp array
    for (int i = 0; i < phelper2; i++) {
        dd->remaining_candidates[(*dd->wvertices_size * WARP_IDX) + lane_remaining_count + i] = ld.vertices[dd->lane_remaining_candidates[lane_write + i]];
    }
    // only need removed if going to be using removed to update degrees
    if (!(wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX])) {
        for (int i = 0; i < pvertexid; i++) {
            dd->removed_candidates[(*dd->wvertices_size * WARP_IDX) + lane_removed_count + i] = ld.vertices[dd->lane_removed_candidates[lane_write + i]].vertexid;
        }
    }
    __syncwarp();
    
    while (wd.remaining_count[WIB_IDX] > 0 && wd.removed_count[WIB_IDX] > 0) {
        // we alternate reading and writing remaining variables from two arrays
        if (wd.rw_counter[WIB_IDX] % 2 == 0) {
            read = dd->remaining_candidates + (*dd->wvertices_size * WARP_IDX);
            write = ld.vertices + wd.number_of_members[WIB_IDX];
        }
        else {
            read = ld.vertices + wd.number_of_members[WIB_IDX];
            write = dd->remaining_candidates + (*dd->wvertices_size * WARP_IDX);
        }

        // update degrees
        if (wd.remaining_count[WIB_IDX] < wd.removed_count[WIB_IDX]) {
            // via remaining, reset exdegs
            for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE) {
                ld.vertices[i].exdeg = 0;
            }
            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
                read[i].exdeg = 0;
            }
            __syncwarp();

            // update exdeg based on remaining candidates, every lane should get the next vertex to intersect dynamically
            for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = ld.vertices[i].vertexid;

                for (int j = 0; j < wd.remaining_count[WIB_IDX]; j++) {
                    phelper1 = read[j].vertexid;
                    phelper2 = d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[phelper1], dd->onehop_offsets[phelper1 + 1] - dd->onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        ld.vertices[i].exdeg++;
                    }
                }
            }

            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = read[i].vertexid;

                for (int j = 0; j < wd.remaining_count[WIB_IDX]; j++) {
                    if (j == i) {
                        continue;
                    }

                    phelper1 = read[j].vertexid;
                    phelper2 = d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[phelper1], dd->onehop_offsets[phelper1 + 1] - dd->onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        read[i].exdeg++;
                    }
                }
            }
        }
        else {
            // via removed, update exdeg based on remaining candidates, again lane scheduling should be dynamic
            for (int i = LANE_IDX; i < wd.number_of_members[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = ld.vertices[i].vertexid;

                for (int j = 0; j < wd.removed_count[WIB_IDX]; j++) {
                    phelper1 = dd->removed_candidates[(*dd->wvertices_size * WARP_IDX) + j];
                    phelper2 = d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[phelper1], dd->onehop_offsets[phelper1 + 1] - dd->onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        ld.vertices[i].exdeg--;
                    }
                }
            }

            for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
                pvertexid = read[i].vertexid;

                for (int j = 0; j < wd.removed_count[WIB_IDX]; j++) {
                    phelper1 = dd->removed_candidates[(*dd->wvertices_size * WARP_IDX) + j];
                    phelper2 = d_b_search_int(dd->onehop_neighbors + dd->onehop_offsets[phelper1], dd->onehop_offsets[phelper1 + 1] - dd->onehop_offsets[phelper1], pvertexid);

                    if (phelper2 > -1) {
                        read[i].exdeg--;
                    }
                }
            }
        }
        __syncwarp();

        lane_remaining_count = 0;

        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            if (d_cand_isvalid(read[i], dd, wd, ld)) {
                dd->lane_candidate_indegs[lane_write + lane_remaining_count++] = read[i].indeg;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_remaining_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1) {
            wd.num_val_cands[WIB_IDX] = lane_remaining_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++) {
            dd->candidate_indegs[(*dd->wvertices_size * WARP_IDX) + lane_remaining_count + i] = dd->lane_candidate_indegs[lane_write + i];
        }
        __syncwarp();

        d_oe_sort_int(dd->candidate_indegs + (*dd->wvertices_size * WARP_IDX), wd.num_val_cands[WIB_IDX], d_comp_int_desc);

        d_calculate_LU_bounds(dd, wd, ld, wd.num_val_cands[WIB_IDX]);
        if (wd.success[WIB_IDX]) {
            return true;
        }

        // check for failed vertices
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX] && !wd.success[WIB_IDX]; k += WARP_SIZE) {
            if (!d_vert_isextendable(ld.vertices[k], dd, wd, ld)) {
                wd.success[WIB_IDX] = true;
                break;
            }

        }
        __syncwarp();
        if (wd.success[WIB_IDX]) {
            return true;
        }

        lane_remaining_count = 0;
        lane_removed_count = 0;

        // check for failed candidates
        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            if (d_cand_isvalid(read[i], dd, wd, ld)) {
                dd->lane_remaining_candidates[lane_write + lane_remaining_count++] = i;
            }
            else {
                dd->lane_removed_candidates[lane_write + lane_removed_count++] = i;
            }
        }
        __syncwarp();

        // scan to calculate write postion in warp arrays
        phelper2 = lane_remaining_count;
        pvertexid = lane_removed_count;
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_remaining_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_remaining_count += phelper1;
            }
            phelper1 = __shfl_up_sync(0xFFFFFFFF, lane_removed_count, i, WARP_SIZE);
            if (LANE_IDX >= i) {
                lane_removed_count += phelper1;
            }
            __syncwarp();
        }
        // lane remaining count sum is scan for last lane and its value
        if (LANE_IDX == WARP_SIZE - 1) {
            wd.num_val_cands[WIB_IDX] = lane_remaining_count;
            wd.removed_count[WIB_IDX] = lane_removed_count;
        }
        // make scan exclusive
        lane_remaining_count -= phelper2;
        lane_removed_count -= pvertexid;

        // parallel write lane arrays to warp array
        for (int i = 0; i < phelper2; i++) {
            write[lane_remaining_count + i] = read[dd->lane_remaining_candidates[lane_write + i]];
        }
        // only need removed if going to be using removed to update degrees
        if (!(wd.num_val_cands[WIB_IDX] < wd.removed_count[WIB_IDX])) {
            for (int i = 0; i < pvertexid; i++) {
                dd->removed_candidates[(*dd->wvertices_size * WARP_IDX) + lane_removed_count + i] = read[dd->lane_removed_candidates[lane_write + i]].vertexid;
            }
        }

        if (LANE_IDX == 0) {
            wd.remaining_count[WIB_IDX] = wd.num_val_cands[WIB_IDX];
            wd.rw_counter[WIB_IDX]++;
        }
    }

    // condense vertices so remaining are after members, only needs to be done if they were not written into vertices last time
    if (wd.rw_counter[WIB_IDX] % 2 == 0) {
        for (int i = LANE_IDX; i < wd.remaining_count[WIB_IDX]; i += WARP_SIZE) {
            ld.vertices[wd.number_of_members[WIB_IDX] + i] = dd->remaining_candidates[(*dd->wvertices_size * WARP_IDX) + i];
        }
    }

    if (LANE_IDX == 0) {
        wd.total_vertices[WIB_IDX] = wd.total_vertices[WIB_IDX] - wd.number_of_candidates[WIB_IDX] + wd.remaining_count[WIB_IDX];
        wd.number_of_candidates[WIB_IDX] = wd.remaining_count[WIB_IDX];
    }

    return false;
}

__device__ void d_calculate_LU_bounds(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates)
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
    if (LANE_IDX == 0) {
        wd.success[WIB_IDX] = false;

        wd.sum_candidate_indeg[WIB_IDX] = 0;
        wd.tightened_upper_bound[WIB_IDX] = 0;

        wd.min_clq_indeg[WIB_IDX] = ld.vertices[0].indeg;
        wd.min_indeg_exdeg[WIB_IDX] = ld.vertices[0].exdeg;
        wd.min_clq_totaldeg[WIB_IDX] = ld.vertices[0].indeg + ld.vertices[0].exdeg;
        wd.sum_clq_indeg[WIB_IDX] = ld.vertices[0].indeg;

        wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + 1, dd);
    }
    __syncwarp();

    // each warp finds these values on their subsection of vertices
    for (index = 1 + LANE_IDX; index < wd.number_of_members[WIB_IDX]; index += WARP_SIZE) {
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
    if (LANE_IDX == 0) {
        // add to shared memory sum
        wd.sum_clq_indeg[WIB_IDX] += sum_clq_indeg;
    }
    __syncwarp();

    // CRITICAL SECTION - each lane then compares their values to the next to get a warp level value
    for (int i = 0; i < WARP_SIZE; i++) {
        if (LANE_IDX == i) {
            if (min_clq_indeg < wd.min_clq_indeg[WIB_IDX]) {
                wd.min_clq_indeg[WIB_IDX] = min_clq_indeg;
                wd.min_indeg_exdeg[WIB_IDX] = min_indeg_exdeg;
            }
            else if (min_clq_indeg == wd.min_clq_indeg[WIB_IDX]) {
                if (min_indeg_exdeg < wd.min_indeg_exdeg[WIB_IDX]) {
                    wd.min_indeg_exdeg[WIB_IDX] = min_indeg_exdeg;
                }
            }

            if (min_clq_totaldeg < wd.min_clq_totaldeg[WIB_IDX]) {
                wd.min_clq_totaldeg[WIB_IDX] = min_clq_totaldeg;
            }
        }
        __syncwarp();
    }

    // CRITICAL SECTION - only first lane does this as there are little calculations
    if (LANE_IDX == 0) {
        if (wd.min_clq_indeg[WIB_IDX] < dd->minimum_degrees[wd.number_of_members[WIB_IDX]])
        {
            // lower
            wd.lower_bound[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX], dd) - min_clq_indeg;

            while (wd.lower_bound[WIB_IDX] <= wd.min_indeg_exdeg[WIB_IDX] && wd.min_clq_indeg[WIB_IDX] + wd.lower_bound[WIB_IDX] <
                dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]]) {
                wd.lower_bound[WIB_IDX]++;
            }

            if (wd.min_clq_indeg[WIB_IDX] + wd.lower_bound[WIB_IDX] < dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]]) {
                wd.success[WIB_IDX] = true;
            }

            // upper
            wd.upper_bound[WIB_IDX] = floor(wd.min_clq_totaldeg[WIB_IDX] / (*(dd->minimum_degree_ratio))) + 1 - wd.number_of_members[WIB_IDX];

            if (wd.upper_bound[WIB_IDX] > number_of_candidates) {
                wd.upper_bound[WIB_IDX] = number_of_candidates;
            }

            // tighten
            if (wd.lower_bound[WIB_IDX] < wd.upper_bound[WIB_IDX]) {
                // tighten lower
                for (index = 0; index < wd.lower_bound[WIB_IDX]; index++) {
                    wd.sum_candidate_indeg[WIB_IDX] += dd->candidate_indegs[(*dd->wvertices_size * WARP_IDX) + index];
                }

                while (index < wd.upper_bound[WIB_IDX] && wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] < wd.number_of_members[WIB_IDX] *
                    dd->minimum_degrees[wd.number_of_members[WIB_IDX] + index]) {
                    wd.sum_candidate_indeg[WIB_IDX] += dd->candidate_indegs[(*dd->wvertices_size * WARP_IDX) + index];
                    index++;
                }

                if (wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] < wd.number_of_members[WIB_IDX] * dd->minimum_degrees[wd.number_of_members[WIB_IDX] + index]) {
                    wd.success[WIB_IDX] = true;
                }
                else {
                    wd.lower_bound[WIB_IDX] = index;

                    wd.tightened_upper_bound[WIB_IDX] = index;

                    while (index < wd.upper_bound[WIB_IDX]) {
                        wd.sum_candidate_indeg[WIB_IDX] += dd->candidate_indegs[(*dd->wvertices_size * WARP_IDX) + index];

                        index++;

                        if (wd.sum_clq_indeg[WIB_IDX] + wd.sum_candidate_indeg[WIB_IDX] >= wd.number_of_members[WIB_IDX] *
                            dd->minimum_degrees[wd.number_of_members[WIB_IDX] + index]) {
                            wd.tightened_upper_bound[WIB_IDX] = index;
                        }
                    }

                    if (wd.upper_bound[WIB_IDX] > wd.tightened_upper_bound[WIB_IDX]) {
                        wd.upper_bound[WIB_IDX] = wd.tightened_upper_bound[WIB_IDX];
                    }

                    if (wd.lower_bound[WIB_IDX] > 1) {
                        wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd);
                    }
                }
            }
        }
        else {
            wd.min_ext_deg[WIB_IDX] = d_get_mindeg(wd.number_of_members[WIB_IDX] + 1,
                dd);

            wd.upper_bound[WIB_IDX] = number_of_candidates;

            if (wd.number_of_members[WIB_IDX] < (*(dd->minimum_clique_size))) {
                wd.lower_bound[WIB_IDX] = (*(dd->minimum_clique_size)) - wd.number_of_members[WIB_IDX];
            }
            else {
                wd.lower_bound[WIB_IDX] = 0;
            }
        }

        if (wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX] < (*(dd->minimum_clique_size))) {
            wd.success[WIB_IDX] = true;
        }

        if (wd.upper_bound[WIB_IDX] < 0 || wd.upper_bound[WIB_IDX] < wd.lower_bound[WIB_IDX]) {
            wd.success[WIB_IDX] = true;
        }
    }
    __syncwarp();
}

__device__ void d_check_for_clique(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    bool clique;

    clique = true;

    for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
        if (ld.vertices[k].indeg < dd->minimum_degrees[wd.number_of_members[WIB_IDX]]) {
            clique = false;
            break;
        }
    }
    // set to false if any threads in warp do not meet degree requirement
    clique = !(__any_sync(0xFFFFFFFF, !clique));

    // if clique write to warp buffer for cliques
    if (clique) {
        uint64_t start_write = (*dd->wcliques_size * WARP_IDX) + dd->wcliques_offset[(*dd->wcliques_offset_size * WARP_IDX) + dd->wcliques_count[WARP_IDX]];
        for (int k = LANE_IDX; k < wd.number_of_members[WIB_IDX]; k += WARP_SIZE) {
            dd->wcliques_vertex[start_write + k] = ld.vertices[k].vertexid;
        }
        if (LANE_IDX == 0) {
            (dd->wcliques_count[WARP_IDX])++;
            dd->wcliques_offset[*dd->wcliques_offset_size * WARP_IDX + dd->wcliques_count[WARP_IDX]] = start_write - (*dd->wcliques_size * WARP_IDX) + wd.number_of_members[WIB_IDX];
        }
    }
}

__device__ void d_write_to_tasks(GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    uint64_t start_write;

    start_write = (*dd->wtasks_size * WARP_IDX) + dd->wtasks_offset[*dd->wtasks_offset_size * WARP_IDX + dd->wtasks_count[WARP_IDX]];

    for (int k = LANE_IDX; k < wd.total_vertices[WIB_IDX]; k += WARP_SIZE) {
        dd->wtasks_vertices[start_write + k].vertexid = ld.vertices[k].vertexid;
        dd->wtasks_vertices[start_write + k].label = ld.vertices[k].label;
        dd->wtasks_vertices[start_write + k].indeg = ld.vertices[k].indeg;
        dd->wtasks_vertices[start_write + k].exdeg = ld.vertices[k].exdeg;
        dd->wtasks_vertices[start_write + k].lvl2adj = 0;
    }
    if (LANE_IDX == 0) {
        dd->wtasks_count[WARP_IDX]++;
        dd->wtasks_offset[(*dd->wtasks_offset_size * WARP_IDX) + dd->wtasks_count[WARP_IDX]] = start_write - (*dd->wtasks_size * WARP_IDX) + wd.total_vertices[WIB_IDX];
    }
}

// --- TERTIARY KENERLS ---
// searches an int array for a certain int, returns the position in the array that item was found, or -1 if not found
__device__ int d_b_search_int(int* search_array, int array_size, int search_number)
{
    // ALGO - BINARY
    // TYPE - SERIAL
    // SPEED - O(log(n))
    
    int low;
    int high;
    int mid;
    int mid_value;
    int comp;

    low = 0;
    high = array_size - 1;

    while (low < high) {
        mid = (low + high) / 2;
        mid_value = search_array[mid];
        comp = (mid_value < search_number);

        low = low + comp * (mid + 1 - low);
        high = high - !comp * (high - mid);
    }

    // Now low == high, check if it's the search_number
    return (search_array[low] == search_number) ? low : -1;
}

// consider using merge
__device__ void d_oe_sort_vert(Vertex* target, int size, int (*func)(Vertex&, Vertex&))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    Vertex vertex1;
    Vertex vertex2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2)) {
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

__device__ void d_oe_sort_int(int* target, int size, int (*func)(int, int))
{
    // ALGO - ODD/EVEN
    // TYPE - PARALLEL
    // SPEED - O(n^2)

    int num1;
    int num2;

    for (int i = 0; i < size; i++) {
        for (int j = (i % 2) + (LANE_IDX * 2); j < size - 1; j += (WARP_SIZE * 2)) {
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