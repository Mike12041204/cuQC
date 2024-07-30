#ifndef DCUQC_DEVICE_KERNELS_H
#define DCUQC_DEVICE_KERNELS_H

#include "./common.h"

// --- PRIMARY KERNELS ---
__global__ void d_expand_level(GPU_Data* dd);
__global__ void transfer_buffers(GPU_Data* dd);
__global__ void fill_from_buffer(GPU_Data* dd);

// --- SECONDARY EXPANSION KERNELS ---
__device__ int d_lookahead_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_remove_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_add_one_vertex(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ int d_critical_vertex_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_check_for_clique(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_write_to_tasks(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);
__device__ void d_diameter_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int pvertexid);
__device__ void d_diameter_pruning_cv(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int number_of_crit_adj);
__device__ void d_calculate_LU_bounds(GPU_Data* dd, Warp_Data& wd, Local_Data& ld, int number_of_candidates);
__device__ bool d_degree_pruning(GPU_Data* dd, Warp_Data& wd, Local_Data& ld);

// --- TERTIARY KERNELS ---
__device__ void d_oe_sort_vert(Vertex* target, int size, int (*func)(Vertex&, Vertex&));
__device__ void d_oe_sort_int(int* target, int size, int (*func)(int, int));
__device__ int d_b_search_int(int* search_array, int array_size, int search_number);
__device__ __forceinline__ int d_comp_vert_Q(Vertex& v1, Vertex& v2)
{
    // order is: member -> covered -> cands -> cover
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
__device__ __forceinline__ int d_comp_vert_cv(Vertex& v1, Vertex& v2)
{
    // put crit adj vertices before candidates

    if (v1.label == 4 && v2.label != 4)
        return -1;
    else if (v1.label != 4 && v2.label == 4)
        return 1;
    else
        return 0;
}
__device__ __forceinline__ int d_comp_int_desc(int n1, int n2)
{
    // descending order

    if (n1 > n2)
        return -1;
    else if (n1 < n2)
        return 1;
    else
        return 0;
}
__device__ __forceinline__ int d_get_mindeg(int number_of_members, GPU_Data* dd)
{
    if (number_of_members < (*(dd->minimum_clique_size)))
        return dd->minimum_degrees[(*(dd->minimum_clique_size))];
    else
        return dd->minimum_degrees[number_of_members];
}
__device__ __forceinline__ bool d_cand_isvalid(Vertex& vertex, GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd->minimum_degrees[(*(dd->minimum_clique_size))])
        return false;
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg + 1, dd))
        return false;
    else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX])
        return false;
    else if (vertex.indeg + wd.upper_bound[WIB_IDX] - 1 < dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX]])
        return false;
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd))
        return false;
    else
        return true;
}
__device__ __forceinline__ bool d_vert_isextendable(Vertex& vertex, GPU_Data* dd, Warp_Data& wd, Local_Data& ld)
{
    if (vertex.indeg + vertex.exdeg < dd->minimum_degrees[(*(dd->minimum_clique_size))])
        return false;
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg, dd))
        return false;
    else if (vertex.indeg + vertex.exdeg < wd.min_ext_deg[WIB_IDX])
        return false;
    else if (vertex.exdeg == 0 && vertex.indeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + vertex.exdeg, dd))
        return false;
    else if (vertex.indeg + wd.upper_bound[WIB_IDX] < dd->minimum_degrees[wd.number_of_members[WIB_IDX] + wd.upper_bound[WIB_IDX]])
        return false;
    else if (vertex.indeg + vertex.exdeg < d_get_mindeg(wd.number_of_members[WIB_IDX] + wd.lower_bound[WIB_IDX], dd))
        return false;
    else
        return true;
}

// --- DEBUG KERNELS ---
__device__ void d_print_vertices(Vertex* vertices, int size);

#endif // DCUQC_DEVICE_KERNELS_H