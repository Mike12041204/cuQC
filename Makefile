# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
# NOTE - -O optimization flags cause bugs, don't use them
NVCCFLAGS = -gencode arch=compute_80,code=sm_80 -std=c++11
CXXFLAGS = -std=c++11
LDFLAGS := -lmpi
INCLUDES = -Iinc

OBJECTS = src/common.o src/host_debug.o src/Quick_rmnonmax.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o main.o
TARGET = DcuQC

$(TARGET): $(OBJECTS)
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/common.o: src/common.cpp inc/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/host_functions.o: src/host_functions.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/device_kernels.h inc/cuTS_MPI.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/host_debug.o: src/host_debug.cpp inc/common.h inc/host_debug.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/device_kernels.o: src/device_kernels.cu inc/common.h inc/device_kernels.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/Quick_rmnonmax.o: src/Quick_rmnonmax.cpp inc/common.h inc/Quick_rmnonmax.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/cuTS_MPI.o: src/cuTS_MPI.cpp inc/common.h inc/cuTS_MPI.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(TARGET) e_* t_* r_*

.PHONY: wipe
wipe:
	rm -f o_*