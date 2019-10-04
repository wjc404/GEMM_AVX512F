//this file should be included by gemm_driver.c and gemm_kernel.S

//read tuning parameters
# ifdef DOUBLE
 # include "../dgemm_tune.h"
# else
 # include "../sgemm_tune.h"
# endif

//GEMM_UNROLL_M_VEC: 2,3,4 or 6
# if GEMM_UNROLL_M_VEC < 2
 # undef GEMM_UNROLL_M_VEC
 # define GEMM_UNROLL_M_VEC 2
# endif
# if GEMM_UNROLL_M_VEC == 5
 # undef GEMM_UNROLL_M_VEC
 # define GEMM_UNROLL_M_VEC 4
# endif
# if GEMM_UNROLL_M_VEC > 5
 # undef GEMM_UNROLL_M_VEC
 # define GEMM_UNROLL_M_VEC 6
# endif

//set GEMM_UNROLL_N
# if GEMM_UNROLL_N < 2
 # undef GEMM_UNROLL_N
 # define GEMM_UNROLL_N 2
# endif
# if GEMM_UNROLL_N > (24/GEMM_UNROLL_M_VEC) //maximum 24 zmm for accumulators
 # undef GEMM_UNROLL_N
 # define GEMM_UNROLL_N (24/GEMM_UNROLL_M_VEC)
# endif

//restrict other parameters
# if GEMM_LOOP_TIMES_N < 2
 # undef GEMM_LOOP_TIMES_N
 # define GEMM_LOOP_TIMES_N 2
# endif
# if GEMM_LOOP_TIMES_N > 200
 # undef GEMM_LOOP_TIMES_N
 # define GEMM_LOOP_TIMES_N 200
# endif
# if GEMM_LOOP_TIMES_K < 2
 # undef GEMM_LOOP_TIMES_K
 # define GEMM_LOOP_TIMES_K 2
# endif
# if PREF_CYCLES_PACKED_A < 1
 # undef PREF_CYCLES_PACKED_A
 # define PREF_CYCLES_PACKED_A 12
# endif
# if PREF_CYCLES_PACKED_B < 1
 # undef PREF_CYCLES_PACKED_B
 # define PREF_CYCLES_PACKED_B 64
# endif
# if CPU_NUM_512FMA_UNITS < 1
 # undef CPU_NUM_512FMA_UNITS
 # define CPU_NUM_512FMA_UNITS 1
# endif

//setting prefetch parameters
# define A_PR_BYTE (PREF_CYCLES_PACKED_A*64*CPU_NUM_512FMA_UNITS/GEMM_UNROLL_N)
# define B_PR_ELEM (PREF_CYCLES_PACKED_B*CPU_NUM_512FMA_UNITS/GEMM_UNROLL_M_VEC)

//setting common block dimensions
# define GEMM_BLOCK_DIM_N (GEMM_LOOP_TIMES_N*GEMM_UNROLL_N)
# define GEMM_BLOCK_L1DIM_K (GEMM_LOOP_TIMES_K*4*GEMM_UNROLL_N) //GEMM_UNROLL_LOOP_K = 4 in current implementation
# define GEMM_BLOCK_DIM_K (GEMM_BLOCK_L1DIM_K*GEMM_UNROLL_M_VEC)
# ifdef DOUBLE
 # define GEMM_BLOCK_DIM_M (8*GEMM_UNROLL_M_VEC) //8 double elements per avx512 vector
# else
 # define GEMM_BLOCK_DIM_M (16*GEMM_UNROLL_M_VEC) //16 float elements per avx512 vector
# endif
