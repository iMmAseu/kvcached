NVCC ?= nvcc

all: bench_vmm.bin bench_vmm_batch_unmap.bin bench_vmm_unmap_compare.bin

bench_vmm.bin: bench_vmm.cpp
	$(NVCC) $^ -o $@ -O2 -g -lcuda -std=c++17

bench_vmm_batch_unmap.bin: bench_vmm_batch_unmap.cpp
	$(NVCC) $^ -o $@ -O2 -g -lcuda -std=c++17

bench_vmm_unmap_compare.bin: bench_vmm_unmap_compare.cpp
	$(NVCC) $^ -o $@ -O2 -g -lcuda -std=c++17

clean:
	$(RM) bench_vmm.bin bench_vmm_batch_unmap.bin bench_vmm_unmap_compare.bin
