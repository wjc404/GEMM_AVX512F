CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -mavx512f -O2

CONFIG_FILES = dgemm_tune.h sgemm_tune.h src/gemm_set_parameters.h
KERNEL_PREFIX = src/kernel/gemm_kernel
SRCFILE = $(KERNEL_PREFIX).S src/gemm_driver.c
INCFILE = src/gemm_copy.c $(KERNEL_PREFIX)_wrapper.c

default: DGEMM.so SGEMM.so AVX512_CPUTEST

DGEMM.so: $(SRCFILE) $(INCFILE) $(CONFIG_FILES)
	$(CC) -DDOUBLE $(CCFLAGS) $(SRCFILE) -o $@
  
SGEMM.so: $(SRCFILE) $(INCFILE) $(CONFIG_FILES)
	$(CC) $(CCFLAGS) $(SRCFILE) -o $@

AVX512_CPUTEST: AVX512_CPUTEST.c
	$(CC) -mavx512f $^ -o $@

clean:
	rm -f *GEMM.so
  
