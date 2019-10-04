//GEMM_UNROLL_M_VEC = 2, GEMM_UNROLL_N=2,3,4,5,6,7,8,9,10,11,12
.macro KERNEL_0
    vmovaps (A0),%zmm0
    vmovaps 64(A0),%zmm1
    addq $128,A0
    VEC_BROAD (B0),%zmm2
    addq $SIZE,B0
    VEC_FMA231 %zmm0,%zmm2,%zmm8
    VEC_FMA231 %zmm1,%zmm2,%zmm9
.endm

.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%zmm0
    vmovaps \Aoff+64(A0),%zmm1
    VEC_BROAD \Boff(B0),%zmm2
    VEC_FMA231 %zmm0,%zmm2,%zmm8
    VEC_FMA231 %zmm1,%zmm2,%zmm9
    VEC_BROAD \Boff+SIZE(B0),%zmm3
    VEC_FMA231 %zmm0,%zmm3,%zmm10
    VEC_FMA231 %zmm1,%zmm3,%zmm11
# if GEMM_UNROLL_N > 2
    VEC_BROAD \Boff+SIZE*2(B0),%zmm4
    VEC_FMA231 %zmm0,%zmm4,%zmm12
    VEC_FMA231 %zmm1,%zmm4,%zmm13
 # if GEMM_UNROLL_N > 3
    VEC_BROAD \Boff+SIZE*3(B0),%zmm5
    VEC_FMA231 %zmm0,%zmm5,%zmm14
    VEC_FMA231 %zmm1,%zmm5,%zmm15
  # if GEMM_UNROLL_N > 4
    VEC_BROAD \Boff+SIZE*4(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm16
    VEC_FMA231 %zmm1,%zmm6,%zmm17
   # if GEMM_UNROLL_N > 5
    VEC_BROAD \Boff+SIZE*5(B0),%zmm7
    VEC_FMA231 %zmm0,%zmm7,%zmm18
    VEC_FMA231 %zmm1,%zmm7,%zmm19
    # if GEMM_UNROLL_N > 6
    VEC_BROAD \Boff+SIZE*6(B0),%zmm2
    VEC_FMA231 %zmm0,%zmm2,%zmm20
    VEC_FMA231 %zmm1,%zmm2,%zmm21
     # if GEMM_UNROLL_N > 7
    VEC_BROAD \Boff+SIZE*7(B0),%zmm3
    VEC_FMA231 %zmm0,%zmm3,%zmm22
    VEC_FMA231 %zmm1,%zmm3,%zmm23
      # if GEMM_UNROLL_N > 8
    VEC_BROAD \Boff+SIZE*8(B0),%zmm4
    VEC_FMA231 %zmm0,%zmm4,%zmm24
    VEC_FMA231 %zmm1,%zmm4,%zmm25
       # if GEMM_UNROLL_N > 9
    VEC_BROAD \Boff+SIZE*9(B0),%zmm5
    VEC_FMA231 %zmm0,%zmm5,%zmm26
    VEC_FMA231 %zmm1,%zmm5,%zmm27
        # if GEMM_UNROLL_N > 10
    VEC_BROAD \Boff+SIZE*10(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm28
    VEC_FMA231 %zmm1,%zmm6,%zmm29
         # if GEMM_UNROLL_N > 11
    VEC_BROAD \Boff+SIZE*11(B0),%zmm7
    VEC_FMA231 %zmm0,%zmm7,%zmm30
    VEC_FMA231 %zmm1,%zmm7,%zmm31
         # endif
        # endif
       # endif
      # endif
     # endif
    # endif
   # endif
  # endif
 # endif
# endif
.endm

.macro KERNEL_f Aoff,Boff,delta,deltb
    KERNEL_1 \Aoff,\Boff
    addq $\delta,A0
    addq $\deltb,B0
.endm

.macro KERNEL_2 nextablk
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
# if GEMM_UNROLL_N > 8
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
# endif
    prefetcht1 (\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N)*SIZE(B0)
# if GEMM_UNROLL_N > 8
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N+8)*SIZE(B0)
# endif
    prefetcht1 64(\nextablk)
    addq $128,\nextablk
    incq %r11
    KERNEL_f 128,GEMM_UNROLL_N*SIZE,256,GEMM_UNROLL_N*2*SIZE
.endm

.macro KERNEL_4_edgen
    KERNEL_1 0,0
    KERNEL_1 128,GEMM_UNROLL_N*SIZE
    KERNEL_1 256,GEMM_UNROLL_N*SIZE*2
    incq %r11
    KERNEL_f 384,GEMM_UNROLL_N*SIZE*3,512,GEMM_UNROLL_N*SIZE*4
.endm

.macro KERNEL_4 Arefpos,Areset
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
# if GEMM_UNROLL_N > 8
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
# endif
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N)*SIZE(B0)
# if GEMM_UNROLL_N > 8
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N+8)*SIZE(B0)
# endif
    KERNEL_1 128,GEMM_UNROLL_N*SIZE
# if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+256(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+256(A0)
# endif
# if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+320(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+320(A0)
# endif
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N*2)*SIZE(B0)
# if GEMM_UNROLL_N > 8
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N*2+8)*SIZE(B0)
# endif
    KERNEL_1 256,GEMM_UNROLL_N*SIZE*2
# if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+384(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+384(A0)
# endif
    prefetcht0 A_PR_BYTE+448(A0,AD,1)
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N*3)*SIZE(B0)
# if GEMM_UNROLL_N > 8
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N*3+8)*SIZE(B0)
# endif
    incq %r11
    KERNEL_f 384,GEMM_UNROLL_N*SIZE*3,512,GEMM_UNROLL_N*SIZE*4
.endm

.macro SHIFTYMM
    vmovaps %zmm10,%zmm8
    vmovaps %zmm11,%zmm9
# if GEMM_UNROLL_N > 2
    vmovaps %zmm12,%zmm10
    vmovaps %zmm13,%zmm11
 # if GEMM_UNROLL_N > 3
    vmovaps %zmm14,%zmm12
    vmovaps %zmm15,%zmm13
  # if GEMM_UNROLL_N > 4
    vmovaps %zmm16,%zmm14
    vmovaps %zmm17,%zmm15
   # if GEMM_UNROLL_N > 5
    vmovaps %zmm18,%zmm16
    vmovaps %zmm19,%zmm17
    # if GEMM_UNROLL_N > 6
    vmovaps %zmm20,%zmm18
    vmovaps %zmm21,%zmm19
     # if GEMM_UNROLL_N > 7
    vmovaps %zmm22,%zmm20
    vmovaps %zmm23,%zmm21
      # if GEMM_UNROLL_N > 8
    vmovaps %zmm24,%zmm22
    vmovaps %zmm25,%zmm23
       # if GEMM_UNROLL_N > 9
    vmovaps %zmm26,%zmm24
    vmovaps %zmm27,%zmm25
        # if GEMM_UNROLL_N > 10
    vmovaps %zmm28,%zmm26
    vmovaps %zmm29,%zmm27
         # if GEMM_UNROLL_N > 11
    vmovaps %zmm30,%zmm28
    vmovaps %zmm31,%zmm29
         # endif
        # endif
       # endif
      # endif
     # endif
    # endif
   # endif
  # endif
 # endif
# endif
.endm

.macro CLEAR zmmno1,zmmno2
    vpxorq %zmm\zmmno1,%zmm\zmmno1,%zmm\zmmno1
    vpxorq %zmm\zmmno2,%zmm\zmmno2,%zmm\zmmno2
.endm

# if GEMM_UNROLL_N == 12
 # define TAIL_ZMMLIST 30,31
# endif
# if GEMM_UNROLL_N == 11
 # define TAIL_ZMMLIST 28,29
# endif
# if GEMM_UNROLL_N == 10
 # define TAIL_ZMMLIST 26,27
# endif
# if GEMM_UNROLL_N == 9
 # define TAIL_ZMMLIST 24,25
# endif
# if GEMM_UNROLL_N == 8
 # define TAIL_ZMMLIST 22,23
# endif
# if GEMM_UNROLL_N == 7
 # define TAIL_ZMMLIST 20,21
# endif
# if GEMM_UNROLL_N == 6
 # define TAIL_ZMMLIST 18,19
# endif
# if GEMM_UNROLL_N == 5
 # define TAIL_ZMMLIST 16,17
# endif
# if GEMM_UNROLL_N == 4
 # define TAIL_ZMMLIST 14,15
# endif
# if GEMM_UNROLL_N == 3
 # define TAIL_ZMMLIST 12,13
# endif
# if GEMM_UNROLL_N == 2
 # define TAIL_ZMMLIST 10,11
# endif

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR TAIL_ZMMLIST
.endm

.macro SUB_STORE_1col zmmno1,zmmno2
    VEC_ADD (CS),%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD 64(CS),%zmm\zmmno2,%zmm\zmmno2
    vmovups %zmm\zmmno1,(CS)
    vmovups %zmm\zmmno2,64(CS)
.endm

.macro SUB_STORE_1col_edgem zmmno1,zmmno2
    CLEAR 0,1
    VEC_MASKMOV (CS),%zmm0{%k1}
    VEC_MASKMOV 64(CS),%zmm1{%k2}
    VEC_ADD %zmm0,%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD %zmm1,%zmm\zmmno2,%zmm\zmmno2
    VEC_MASKMOV %zmm\zmmno1,(CS){%k1}
    VEC_MASKMOV %zmm\zmmno2,64(CS){%k2}
.endm

.macro STORECBLK_1col
    SUB_STORE_1col 8,9
    addq LDC,CS
.endm

.macro STORECBLK_1col_edgem
    SUB_STORE_1col_edgem 8,9
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR 10,11
# if GEMM_UNROLL_N > 2
    CLEAR 12,13
 # if GEMM_UNROLL_N > 3
    CLEAR 14,15
  # if GEMM_UNROLL_N > 4
    CLEAR 16,17
   # if GEMM_UNROLL_N > 5
    CLEAR 18,19
    # if GEMM_UNROLL_N > 6
    CLEAR 20,21
     # if GEMM_UNROLL_N > 7
    CLEAR 22,23
      # if GEMM_UNROLL_N > 8
    CLEAR 24,25
       # if GEMM_UNROLL_N > 9
    CLEAR 26,27
        # if GEMM_UNROLL_N > 10
    CLEAR 28,29
         # if GEMM_UNROLL_N > 11
    CLEAR 30,31
         # endif
        # endif
       # endif
      # endif
     # endif
    # endif
   # endif
  # endif
 # endif
# endif
.endm

.macro FIN_C
    SUB_STORE_1col 8,9
# if GEMM_UNROLL_N > 2
    addq LDC,CS
    SUB_STORE_1col 10,11
 # if GEMM_UNROLL_N > 3
    addq LDC,CS
    SUB_STORE_1col 12,13
  # if GEMM_UNROLL_N > 4
    addq LDC,CS
    SUB_STORE_1col 14,15
   # if GEMM_UNROLL_N > 5
    addq LDC,CS
    SUB_STORE_1col 16,17
    # if GEMM_UNROLL_N > 6
    addq LDC,CS
    SUB_STORE_1col 18,19
     # if GEMM_UNROLL_N > 7
    addq LDC,CS
    SUB_STORE_1col 20,21
      # if GEMM_UNROLL_N > 8
    addq LDC,CS
    SUB_STORE_1col 22,23
       # if GEMM_UNROLL_N > 9
    addq LDC,CS
    SUB_STORE_1col 24,25
        # if GEMM_UNROLL_N > 10
    addq LDC,CS
    SUB_STORE_1col 26,27
         # if GEMM_UNROLL_N > 11
    addq LDC,CS
    SUB_STORE_1col 28,29
         # endif
        # endif
       # endif
      # endif
     # endif
    # endif
   # endif
  # endif
 # endif
# endif
.endm

.macro FIN_C_edgem
    SUB_STORE_1col_edgem 10,11
# if GEMM_UNROLL_N > 2
    addq LDC,CS
    SUB_STORE_1col_edgem 12,13
 # if GEMM_UNROLL_N > 3
    addq LDC,CS
    SUB_STORE_1col_edgem 14,15
  # if GEMM_UNROLL_N > 4
    addq LDC,CS
    SUB_STORE_1col_edgem 16,17
   # if GEMM_UNROLL_N > 5
    addq LDC,CS
    SUB_STORE_1col_edgem 18,19
    # if GEMM_UNROLL_N > 6
    addq LDC,CS
    SUB_STORE_1col_edgem 20,21
     # if GEMM_UNROLL_N > 7
    addq LDC,CS
    SUB_STORE_1col_edgem 22,23
      # if GEMM_UNROLL_N > 8
    addq LDC,CS
    SUB_STORE_1col_edgem 24,25
       # if GEMM_UNROLL_N > 9
    addq LDC,CS
    SUB_STORE_1col_edgem 26,27
        # if GEMM_UNROLL_N > 10
    addq LDC,CS
    SUB_STORE_1col_edgem 28,29
         # if GEMM_UNROLL_N > 11
    addq LDC,CS
    SUB_STORE_1col_edgem 30,31
         # endif
        # endif
       # endif
      # endif
     # endif
    # endif
   # endif
  # endif
 # endif
# endif
.endm

.macro ZERO_C_1col //used in edge part
    CLEAR 8,9
.endm

.macro ZERO_C_ncol //used in edge part
    CLEAR 8,9
    INIT_C
.endm

.macro STORECBLK_ncol //used in edge part
    FIN_C
    addq LDC,CS
    SUB_STORE_1col TAIL_ZMMLIST
    addq LDC,CS
.endm

.macro STORECBLK_ncol_edgem //used in edge part
    SUB_STORE_1col_edgem 8,9
    addq LDC,CS
    FIN_C_edgem
    addq LDC,CS
.endm
