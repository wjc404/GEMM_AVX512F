//GEMM_UNROLL_M_VEC = 4, GEMM_UNROLL_N=2,3,4,5,6
.macro KERNEL_0
    vmovaps (A0),%zmm0
    vmovaps 64(A0),%zmm1
    vmovaps 128(A0),%zmm2
    vmovaps 192(A0),%zmm3
    addq $256,A0
    VEC_BROAD (B0),%zmm4
    addq $SIZE,B0
    VEC_FMA231 %zmm0,%zmm4,%zmm8
    VEC_FMA231 %zmm1,%zmm4,%zmm9
    VEC_FMA231 %zmm2,%zmm4,%zmm10
    VEC_FMA231 %zmm3,%zmm4,%zmm11
.endm

.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%zmm0
    vmovaps \Aoff+64(A0),%zmm1
    vmovaps \Aoff+128(A0),%zmm2
    vmovaps \Aoff+192(A0),%zmm3
    VEC_BROAD \Boff(B0),%zmm4
    VEC_FMA231 %zmm0,%zmm4,%zmm8
    VEC_FMA231 %zmm1,%zmm4,%zmm9
    VEC_FMA231 %zmm2,%zmm4,%zmm10
    VEC_FMA231 %zmm3,%zmm4,%zmm11
    VEC_BROAD \Boff+SIZE(B0),%zmm5
    VEC_FMA231 %zmm0,%zmm5,%zmm12
    VEC_FMA231 %zmm1,%zmm5,%zmm13
    VEC_FMA231 %zmm2,%zmm5,%zmm14
    VEC_FMA231 %zmm3,%zmm5,%zmm15
# if GEMM_UNROLL_N > 2
    VEC_BROAD \Boff+SIZE*2(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm16
    VEC_FMA231 %zmm1,%zmm6,%zmm17
    VEC_FMA231 %zmm2,%zmm6,%zmm18
    VEC_FMA231 %zmm3,%zmm6,%zmm19
 # if GEMM_UNROLL_N > 3
    VEC_BROAD \Boff+SIZE*3(B0),%zmm4
    VEC_FMA231 %zmm0,%zmm4,%zmm20
    VEC_FMA231 %zmm1,%zmm4,%zmm21
    VEC_FMA231 %zmm2,%zmm4,%zmm22
    VEC_FMA231 %zmm3,%zmm4,%zmm23
  # if GEMM_UNROLL_N > 4
    VEC_BROAD \Boff+SIZE*4(B0),%zmm5
    VEC_FMA231 %zmm0,%zmm5,%zmm24
    VEC_FMA231 %zmm1,%zmm5,%zmm25
    VEC_FMA231 %zmm2,%zmm5,%zmm26
    VEC_FMA231 %zmm3,%zmm5,%zmm27
   # if GEMM_UNROLL_N > 5
    VEC_BROAD \Boff+SIZE*5(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm28
    VEC_FMA231 %zmm1,%zmm6,%zmm29
    VEC_FMA231 %zmm2,%zmm6,%zmm30
    VEC_FMA231 %zmm3,%zmm6,%zmm31
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
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 (\nextablk)
    prefetcht1 64(\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N)*SIZE(B0)
    prefetcht1 128(\nextablk)
    prefetcht1 192(\nextablk)
    addq $256,\nextablk
    incq %r11
    KERNEL_f 256,GEMM_UNROLL_N*SIZE,512,GEMM_UNROLL_N*SIZE*2
.endm

.macro KERNEL_4_edgen
    KERNEL_1 0,0
    KERNEL_1 256,GEMM_UNROLL_N*SIZE
    KERNEL_1 512,GEMM_UNROLL_N*SIZE*2
    incq %r11
    KERNEL_f 768,GEMM_UNROLL_N*SIZE*3,1024,GEMM_UNROLL_N*SIZE*4
.endm

.macro KERNEL_4 Arefpos,Areset
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N)*SIZE(B0)
    KERNEL_1 256,GEMM_UNROLL_N*SIZE
# if A_PR_BYTE > 511
    prefetcht0 A_PR_BYTE+512(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+512(A0)
# endif
# if A_PR_BYTE > 447
    prefetcht0 A_PR_BYTE+576(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+576(A0)
# endif
# if A_PR_BYTE > 383
    prefetcht0 A_PR_BYTE+640(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+640(A0)
# endif
# if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+704(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+704(A0)
# endif
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N*2)*SIZE(B0)
    KERNEL_1 512,GEMM_UNROLL_N*SIZE*2
# if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+768(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+768(A0)
# endif
# if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+832(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+832(A0)
# endif
# if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+896(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+896(A0)
# endif
    prefetcht0 A_PR_BYTE+960(A0,AD,1)
    prefetcht0 (B_PR_ELEM+GEMM_UNROLL_N*3)*SIZE(B0)
    incq %r11
    KERNEL_f 768,GEMM_UNROLL_N*SIZE*3,1024,GEMM_UNROLL_N*SIZE*4
.endm

.macro SHIFTYMM
    vmovaps %zmm12,%zmm8
    vmovaps %zmm13,%zmm9
    vmovaps %zmm14,%zmm10
    vmovaps %zmm15,%zmm11
# if GEMM_UNROLL_N > 2
    vmovaps %zmm16,%zmm12
    vmovaps %zmm17,%zmm13
    vmovaps %zmm18,%zmm14
    vmovaps %zmm19,%zmm15
 # if GEMM_UNROLL_N > 3
    vmovaps %zmm20,%zmm16
    vmovaps %zmm21,%zmm17
    vmovaps %zmm22,%zmm18
    vmovaps %zmm23,%zmm19
  # if GEMM_UNROLL_N > 4
    vmovaps %zmm24,%zmm20
    vmovaps %zmm25,%zmm21
    vmovaps %zmm26,%zmm22
    vmovaps %zmm27,%zmm23
   # if GEMM_UNROLL_N > 5
    vmovaps %zmm28,%zmm24
    vmovaps %zmm29,%zmm25
    vmovaps %zmm30,%zmm26
    vmovaps %zmm31,%zmm27
   # endif
  # endif
 # endif
# endif
.endm

.macro CLEAR zmmno1,zmmno2,zmmno3,zmmno4
    vpxorq %zmm\zmmno1,%zmm\zmmno1,%zmm\zmmno1
    vpxorq %zmm\zmmno2,%zmm\zmmno2,%zmm\zmmno2
    vpxorq %zmm\zmmno3,%zmm\zmmno3,%zmm\zmmno3
    vpxorq %zmm\zmmno4,%zmm\zmmno4,%zmm\zmmno4
.endm

# if GEMM_UNROLL_N == 6
 # define TAIL_ZMMLIST 28,29,30,31
# endif
# if GEMM_UNROLL_N == 5
 # define TAIL_ZMMLIST 24,25,26,27
# endif
# if GEMM_UNROLL_N == 4
 # define TAIL_ZMMLIST 20,21,22,23
# endif
# if GEMM_UNROLL_N == 3
 # define TAIL_ZMMLIST 16,17,18,19
# endif
# if GEMM_UNROLL_N == 2
 # define TAIL_ZMMLIST 12,13,14,15
# endif

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR TAIL_ZMMLIST
.endm

.macro SUB_STORE_1col zmmno1,zmmno2,zmmno3,zmmno4
    VEC_ADD (CS),%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD 64(CS),%zmm\zmmno2,%zmm\zmmno2
    VEC_ADD 128(CS),%zmm\zmmno3,%zmm\zmmno3
    VEC_ADD 192(CS),%zmm\zmmno4,%zmm\zmmno4
    vmovups %zmm\zmmno1,(CS)
    vmovups %zmm\zmmno2,64(CS)
    vmovups %zmm\zmmno3,128(CS)
    vmovups %zmm\zmmno4,192(CS)
.endm

.macro SUB_STORE_1col_edgem zmmno1,zmmno2,zmmno3,zmmno4
    CLEAR 0,1,2,3
    VEC_MASKMOV (CS),%zmm0{%k1}
    VEC_MASKMOV 64(CS),%zmm1{%k2}
    VEC_MASKMOV 128(CS),%zmm2{%k3}
    VEC_MASKMOV 192(CS),%zmm3{%k4}
    VEC_ADD %zmm0,%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD %zmm1,%zmm\zmmno2,%zmm\zmmno2
    VEC_ADD %zmm2,%zmm\zmmno3,%zmm\zmmno3
    VEC_ADD %zmm3,%zmm\zmmno4,%zmm\zmmno4
    VEC_MASKMOV %zmm\zmmno1,(CS){%k1}
    VEC_MASKMOV %zmm\zmmno2,64(CS){%k2}
    VEC_MASKMOV %zmm\zmmno3,128(CS){%k3}
    VEC_MASKMOV %zmm\zmmno4,192(CS){%k4}
.endm

.macro STORECBLK_1col
    SUB_STORE_1col 8,9,10,11
    addq LDC,CS
.endm

.macro STORECBLK_1col_edgem
    SUB_STORE_1col_edgem 8,9,10,11
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR 12,13,14,15
# if GEMM_UNROLL_N > 2
    CLEAR 16,17,18,19
 # if GEMM_UNROLL_N > 3
    CLEAR 20,21,22,23
  # if GEMM_UNROLL_N > 4
    CLEAR 24,25,26,27
   # if GEMM_UNROLL_N > 5
    CLEAR 28,29,30,31
   # endif
  # endif
 # endif
# endif
.endm

.macro FIN_C
    SUB_STORE_1col 8,9,10,11
# if GEMM_UNROLL_N > 2
    addq LDC,CS
    SUB_STORE_1col 12,13,14,15
 # if GEMM_UNROLL_N > 3
    addq LDC,CS
    SUB_STORE_1col 16,17,18,19
  # if GEMM_UNROLL_N > 4
    addq LDC,CS
    SUB_STORE_1col 20,21,22,23
   # if GEMM_UNROLL_N > 5
    addq LDC,CS
    SUB_STORE_1col 24,25,26,27
   # endif
  # endif
 # endif
# endif
.endm

.macro FIN_C_edgem
    SUB_STORE_1col_edgem 12,13,14,15
# if GEMM_UNROLL_N > 2
    addq LDC,CS
    SUB_STORE_1col_edgem 16,17,18,19
 # if GEMM_UNROLL_N > 3
    addq LDC,CS
    SUB_STORE_1col_edgem 20,21,22,23
  # if GEMM_UNROLL_N > 4
    addq LDC,CS
    SUB_STORE_1col_edgem 24,25,26,27
   # if GEMM_UNROLL_N > 5
    addq LDC,CS
    SUB_STORE_1col_edgem 28,29,30,31
   # endif
  # endif
 # endif
# endif
.endm

.macro ZERO_C_1col //used in edge part
    CLEAR 8,9,10,11
.endm

.macro ZERO_C_ncol //used in edge part
    CLEAR 8,9,10,11
    INIT_C
.endm

.macro STORECBLK_ncol //used in edge part
    FIN_C
    addq LDC,CS
    SUB_STORE_1col TAIL_ZMMLIST
    addq LDC,CS
.endm

.macro STORECBLK_ncol_edgem //used in edge part
    SUB_STORE_1col_edgem 8,9,10,11
    addq LDC,CS
    FIN_C_edgem
    addq LDC,CS
.endm
