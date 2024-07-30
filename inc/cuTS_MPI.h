#ifndef DCUQC_CUTS_MPI_H
#define DCUQC_CUTS_MPI_H

#include "./common.h"

void mpi_irecv_all(int rank);
void decode_com_buffer(GPU_Data& h_dd, uint64_t* mpiSizeBuffer, Vertex* mpiVertexBuffer);
void encode_com_buffer(GPU_Data& h_dd, uint64_t* mpiSizeBuffer, Vertex* mpiVertexBuffer, uint64_t buffer_count, DS_Sizes& dss);
bool give_work_wrapper(int rank, int &taker, uint64_t* mpiSizeBuffer, Vertex* mpiVertexBuffer, GPU_Data& h_dd, uint64_t buffer_count, DS_Sizes& dss);
int take_work_wrap(int rank, uint64_t* mpiSizeBuffer, Vertex* mpiVertexBuffer, int& from);

#endif // DCUQC_CUTS_MPI_H