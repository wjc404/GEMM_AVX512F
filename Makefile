CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -mavx512f -O2

CONFIG_FILES = dgemm_tune.h sgemm_tune.h src/gemm_set_parameters.h
KERNEL_PREFIX = src/kernel/gemm_kernel
KERNEL_INCS = $(KERNEL_PREFIX)_wrapper.c $(KERNEL_PREFIX)_unroll2.S $(KERNEL_PREFIX)_unroll3.S $(KERNEL_PREFIX)_unroll4.S $(KERNEL_PREFIX)_unroll6.S
SRCFILE = $(KERNEL_PREFIX).S src/gemm_driver.c
INCFILE = src/gemm_copy.c $(KERNEL_INCS)

default: DGEMM.so SGEMM.so

DGEMM.so: $(SRCFILE) $(INCFILE) $(CONFIG_FILES)
	$(CC) -DDOUBLE $(CCFLAGS) $(SRCFILE) -o $@
  
SGEMM.so: $(SRCFILE) $(INCFILE) $(CONFIG_FILES)
	$(CC) $(CCFLAGS) $(SRCFILE) -o $@

clean:
	rm -f *GEMM.so
  
