//this file should be included by gemm_driver.c and gemm_kernel.S

//read tuning parameters
# ifdef DOUBLE
 # include "../dgemm_tune.h"
# else
 # include "../sgemm_tune.h"
# endif

//GEMM_UNROLL_M_VEC: 2,3,4 or 6
# if GEMM_UNROLL_M_VEC < 2
 # define GEMM_UNROLL_M_VEC 2
# endif
# if GEMM_UNROLL_M_VEC == 5
 # define GEMM_UNROLL_M_VEC 4
# endif
# if GEMM_UNROLL_M_VEC > 5
 # define GEMM_UNROLL_M_VEC 6
# endif

//GEMM_NUM_ACCUMULATORS: 12 or 24
# if GEMM_NUM_ACCUMULATORS < 18
 # define GEMM_NUM_ACCUMULATORS 12
# endif
# if GEMM_NUM_ACCUMULATORS > 17
 # define GEMM_NUM_ACCUMULATORS 24
# endif

# define GEMM_UNROLL_N (GEMM_NUM_ACCUMULATORS/GEMM_UNROLL_M_VEC)

//restrict other parameters
# if GEMM_LOOP_TIMES_N < 2
 # define GEMM_LOOP_TIMES_N 2
# endif
# if GEMM_LOOP_TIMES_N > 200
 # define GEMM_LOOP_TIMES_N 200
# endif
# if GEMM_LOOP_TIMES_K < 2
 # define GEMM_LOOP_TIMES_K 2
# endif
# if PREF_CYCLES_PACKED_A < 1
 # define PREF_CYCLES_PACKED_A 12
# endif
# if PREF_CYCLES_PACKED_B < 1
 # define PREF_CYCLES_PACKED_B 64
# endif

//setting prefetch parameters, assuming 2x512bit FMA units per core
# define A_PR_BYTE (PREF_CYCLES_PACKED_A*GEMM_UNROLL_M_VEC*64/(GEMM_NUM_ACCUMULATORS/2))
# define B_PR_ELEM (PREF_CYCLES_PACKED_B*GEMM_UNROLL_N/(GEMM_NUM_ACCUMULATORS/2))

//setting common block dimensions
# define GEMM_BLOCK_DIM_N (GEMM_LOOP_TIMES_N*GEMM_UNROLL_N)
# define GEMM_BLOCK_L1DIM_K (GEMM_LOOP_TIMES_K*4*GEMM_UNROLL_N) //GEMM_UNROLL_LOOP_K = 4 in current implementation
# define GEMM_BLOCK_DIM_K (GEMM_BLOCK_L1DIM_K*GEMM_UNROLL_M_VEC)
# ifdef DOUBLE
 # define GEMM_BLOCK_DIM_M (8*GEMM_UNROLL_M_VEC)
# else
 # define GEMM_BLOCK_DIM_M (16*GEMM_UNROLL_M_VEC)
# endif
