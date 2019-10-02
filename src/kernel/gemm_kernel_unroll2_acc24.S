//GEMM_UNROLL_M_VEC = 2, GEMM_UNROLL_N=12
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
    VEC_BROAD \Boff+SIZE*2(B0),%zmm4
    VEC_FMA231 %zmm0,%zmm4,%zmm12
    VEC_FMA231 %zmm1,%zmm4,%zmm13
    VEC_BROAD \Boff+SIZE*3(B0),%zmm5
    VEC_FMA231 %zmm0,%zmm5,%zmm14
    VEC_FMA231 %zmm1,%zmm5,%zmm15
    VEC_BROAD \Boff+SIZE*4(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm16
    VEC_FMA231 %zmm1,%zmm6,%zmm17
    VEC_BROAD \Boff+SIZE*5(B0),%zmm7
    VEC_FMA231 %zmm0,%zmm7,%zmm18
    VEC_FMA231 %zmm1,%zmm7,%zmm19
    VEC_BROAD \Boff+SIZE*6(B0),%zmm2
    VEC_FMA231 %zmm0,%zmm2,%zmm20
    VEC_FMA231 %zmm1,%zmm2,%zmm21
    VEC_BROAD \Boff+SIZE*7(B0),%zmm3
    VEC_FMA231 %zmm0,%zmm3,%zmm22
    VEC_FMA231 %zmm1,%zmm3,%zmm23
    VEC_BROAD \Boff+SIZE*8(B0),%zmm4
    VEC_FMA231 %zmm0,%zmm4,%zmm24
    VEC_FMA231 %zmm1,%zmm4,%zmm25
    VEC_BROAD \Boff+SIZE*9(B0),%zmm5
    VEC_FMA231 %zmm0,%zmm5,%zmm26
    VEC_FMA231 %zmm1,%zmm5,%zmm27
    VEC_BROAD \Boff+SIZE*10(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm28
    VEC_FMA231 %zmm1,%zmm6,%zmm29
    VEC_BROAD \Boff+SIZE*11(B0),%zmm7
    VEC_FMA231 %zmm0,%zmm7,%zmm30
    VEC_FMA231 %zmm1,%zmm7,%zmm31
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
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    prefetcht1 (\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    prefetcht1 64(\nextablk)
    addq $128,\nextablk
    incq %r11
    KERNEL_f 128,12*SIZE,256,24*SIZE
.endm

.macro KERNEL_4_edgen
    KERNEL_1 0,0
    KERNEL_1 128,12*SIZE
    KERNEL_1 256,24*SIZE
    incq %r11
    KERNEL_f 384,36*SIZE,512,48*SIZE
.endm

.macro KERNEL_4 Arefpos,Areset
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    KERNEL_1 128,12*SIZE
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
    prefetcht0 (B_PR_ELEM+24)*SIZE(B0)
    prefetcht0 (B_PR_ELEM+32)*SIZE(B0)
    KERNEL_1 256,24*SIZE
# if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+384(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+384(A0)
# endif
    prefetcht0 A_PR_BYTE+448(A0,AD,1)
    prefetcht0 (B_PR_ELEM+40)*SIZE(B0)
    incq %r11
    KERNEL_f 384,36*SIZE,512,48*SIZE
.endm

.macro SHIFTYMM
    vmovaps %zmm10,%zmm8
    vmovaps %zmm11,%zmm9
    vmovaps %zmm12,%zmm10
    vmovaps %zmm13,%zmm11
    vmovaps %zmm14,%zmm12
    vmovaps %zmm15,%zmm13
    vmovaps %zmm16,%zmm14
    vmovaps %zmm17,%zmm15
    vmovaps %zmm18,%zmm16
    vmovaps %zmm19,%zmm17
    vmovaps %zmm20,%zmm18
    vmovaps %zmm21,%zmm19
    vmovaps %zmm22,%zmm20
    vmovaps %zmm23,%zmm21
    vmovaps %zmm24,%zmm22
    vmovaps %zmm25,%zmm23
    vmovaps %zmm26,%zmm24
    vmovaps %zmm27,%zmm25
    vmovaps %zmm28,%zmm26
    vmovaps %zmm29,%zmm27
    vmovaps %zmm30,%zmm28
    vmovaps %zmm31,%zmm29
.endm

.macro CLEAR zmmno1,zmmno2
    vpxorq %zmm\zmmno1,%zmm\zmmno1,%zmm\zmmno1
    vpxorq %zmm\zmmno2,%zmm\zmmno2,%zmm\zmmno2
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR 30,31
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
    CLEAR 12,13
    CLEAR 14,15
    CLEAR 16,17
    CLEAR 18,19
    CLEAR 20,21
    CLEAR 22,23
    CLEAR 24,25
    CLEAR 26,27
    CLEAR 28,29
    CLEAR 30,31
.endm

.macro FIN_C
    SUB_STORE_1col 8,9
    addq LDC,CS
    SUB_STORE_1col 10,11
    addq LDC,CS
    SUB_STORE_1col 12,13
    addq LDC,CS
    SUB_STORE_1col 14,15
    addq LDC,CS
    SUB_STORE_1col 16,17
    addq LDC,CS
    SUB_STORE_1col 18,19
    addq LDC,CS
    SUB_STORE_1col 20,21
    addq LDC,CS
    SUB_STORE_1col 22,23
    addq LDC,CS
    SUB_STORE_1col 24,25
    addq LDC,CS
    SUB_STORE_1col 26,27
    addq LDC,CS
    SUB_STORE_1col 28,29
.endm

.macro FIN_C_edgem
    SUB_STORE_1col_edgem 10,11
    addq LDC,CS
    SUB_STORE_1col_edgem 12,13
    addq LDC,CS
    SUB_STORE_1col_edgem 14,15
    addq LDC,CS
    SUB_STORE_1col_edgem 16,17
    addq LDC,CS
    SUB_STORE_1col_edgem 18,19
    addq LDC,CS
    SUB_STORE_1col_edgem 20,21
    addq LDC,CS
    SUB_STORE_1col_edgem 22,23
    addq LDC,CS
    SUB_STORE_1col_edgem 24,25
    addq LDC,CS
    SUB_STORE_1col_edgem 26,27
    addq LDC,CS
    SUB_STORE_1col_edgem 28,29
    addq LDC,CS
    SUB_STORE_1col_edgem 30,31
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
    SUB_STORE_1col 30,31
    addq LDC,CS
.endm

.macro STORECBLK_ncol_edgem //used in edge part
    SUB_STORE_1col_edgem 8,9
    addq LDC,CS
    FIN_C_edgem
    addq LDC,CS
.endm
