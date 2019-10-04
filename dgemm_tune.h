// The size of packed A matrix is 256B * GEMM_LOOP_TIMES_K * GEMM_UNROLL_N * GEMM_UNROL_M_VEC, which should not succeeds the capacity of L1 cache.
// The size of packed B matrix is 32B * (GEMM_UNROLL_N^2) * GEMM_LOOP_TIMES_N * GEMM_LOOP_TIMES_K * GEMM_UNROLL_M_VEC, which should not exceed the capacity of L3 cache per core.
// GEMM_UNROLL_M_VEC * GEMM_UNROLL_N should be no less than CPU_NUM_512FMA_UNITS * "latency of fma instructions in cycles".
// The value of PREF_CYCLES_PACKED_A should be greater than the latency of L2 cache (in cycles).
// The value of PREF_CYCLES_PACKED_B should be greater than the latency of L3 cache (in cycles).


/* here is an example */
# define CPU_NUM_512FMA_UNITS 2 //how many 512-bit fma execution units per CPU core ?
# define GEMM_UNROLL_M_VEC 2
# define GEMM_UNROLL_N 6
# define GEMM_LOOP_TIMES_N 32
# define GEMM_LOOP_TIMES_K 8
# define PREF_CYCLES_PACKED_A 16
# define PREF_CYCLES_PACKED_B 64

