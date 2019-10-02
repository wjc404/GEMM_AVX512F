// The size of packed A matrix is 6kB * GEMM_LOOP_TIMES_K, which should not succeeds the capacity of L1 cache.
// The value of PREF_CYCLES_PACKED_A should be greater than the latency of L2 cache (in cycles).
// The value of PREF_CYCLES_PACKED_B should be greater than the latency of L3 cache (in cycles).

# define GEMM_UNROLL_M_VEC 6
# define GEMM_LOOP_TIMES_N 32
# define GEMM_LOOP_TIMES_K 6
# define PREF_CYCLES_PACKED_A 12
# define PREF_CYCLES_PACKED_B 36

