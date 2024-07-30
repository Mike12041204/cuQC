#ifndef DCUQC_HOST_DEBUG_H
#define DCUQC_HOST_DEBUG_H

#include "./common.h"

void print_CPU_Data(CPU_Data& hd);
void print_GPU_Data(GPU_Data& h_dd, DS_Sizes& dss);
void print_CPU_Graph(CPU_Graph& hg);
void print_GPU_Graph(GPU_Data& h_dd, CPU_Graph& hg);
void print_WTask_Buffers(GPU_Data& h_dd, DS_Sizes& dss);
void print_WClique_Buffers(GPU_Data& h_dd, DS_Sizes& dss);
void print_GPU_Cliques(GPU_Data& h_dd, DS_Sizes& dss); 
void print_CPU_Cliques(CPU_Cliques& hc);
bool print_Data_Sizes(GPU_Data& h_dd, DS_Sizes& dss);
void h_print_Data_Sizes(CPU_Data& hd, CPU_Cliques& hc);
void print_vertices(Vertex* vertices, int size);
bool print_Data_Sizes_Every(GPU_Data& h_dd, int every, DS_Sizes& dss);
bool print_Warp_Data_Sizes(GPU_Data& h_dd, DS_Sizes& dss);
void print_All_Warp_Data_Sizes(GPU_Data& h_dd, DS_Sizes& dss);
bool print_Warp_Data_Sizes_Every(GPU_Data& h_dd, int every, DS_Sizes& dss);
void print_All_Warp_Data_Sizes_Every(GPU_Data& h_dd, int every, DS_Sizes& dss);
void initialize_maxes();
void print_maxes();

#endif // DCUQC_HOST_DEBUG_H