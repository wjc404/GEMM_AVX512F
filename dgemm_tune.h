//The size of packed A matrix is 64B*GEMM_LOOP_TIMES_K*GEMM_UNROLL_K*GEMM_UNROLL_N*GEMM_UNROL_M_VEC, which should not exceed the capacity of L1 cache.
//The size of packed B matrix is 8B*(GEMM_UNROLL_N^2)*GEMM_LOOP_TIMES_N*GEMM_UNROLL_K*GEMM_LOOP_TIMES_K(ifndef NO_REPEAT_C_BLOCK:*GEMM_UNROLL_M_VEC), which should not exceed the capacity of L3 cache per core.
//GEMM_UNROLL_M_VEC*GEMM_UNROLL_N should be no less than CPU_NUM_512FMA_UNITS*"latency of fma instructions in cycles".

/* parameters tuned on Xeon Platinum 8269CY */
# define CPU_NUM_512FMA_UNITS 2
//# define NO_REPEAT_C_BLOCK
//# define HIGH_MEM_LATENCY
# define GEMM_UNROLL_M_VEC 3
# define GEMM_UNROLL_N 8
# define GEMM_UNROLL_K 4 //4-8
# define GEMM_LOOP_TIMES_N 20
# define GEMM_LOOP_TIMES_K 4
# define PREF_CYCLES_PACKED_A 36 //should be greater than the latency of L2 cache (in cycles).
# define PREF_CYCLES_PACKED_B 128 //should be greater than the latency of L3 cache (in cycles).
