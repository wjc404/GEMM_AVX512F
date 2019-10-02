//GEMM_UNROLL_M_VEC = 6, GEMM_UNROLL_N=4
.macro KERNEL_0
    vmovaps (A0),%zmm0
    vmovaps 64(A0),%zmm1
    vmovaps 128(A0),%zmm2
    vmovaps 192(A0),%zmm3
    vmovaps 256(A0),%zmm4
    vmovaps 320(A0),%zmm5
    addq $384,A0
    VEC_BROAD (B0),%zmm6
    addq $SIZE,B0
    VEC_FMA231 %zmm0,%zmm6,%zmm8
    VEC_FMA231 %zmm1,%zmm6,%zmm9
    VEC_FMA231 %zmm2,%zmm6,%zmm10
    VEC_FMA231 %zmm3,%zmm6,%zmm11
    VEC_FMA231 %zmm4,%zmm6,%zmm12
    VEC_FMA231 %zmm5,%zmm6,%zmm13
.endm

.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%zmm0
    vmovaps \Aoff+64(A0),%zmm1
    vmovaps \Aoff+128(A0),%zmm2
    vmovaps \Aoff+192(A0),%zmm3
    vmovaps \Aoff+256(A0),%zmm4
    vmovaps \Aoff+320(A0),%zmm5
    VEC_BROAD \Boff(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm8
    VEC_FMA231 %zmm1,%zmm6,%zmm9
    VEC_FMA231 %zmm2,%zmm6,%zmm10
    VEC_FMA231 %zmm3,%zmm6,%zmm11
    VEC_FMA231 %zmm4,%zmm6,%zmm12
    VEC_FMA231 %zmm5,%zmm6,%zmm13
    VEC_BROAD \Boff+SIZE(B0),%zmm7
    VEC_FMA231 %zmm0,%zmm7,%zmm14
    VEC_FMA231 %zmm1,%zmm7,%zmm15
    VEC_FMA231 %zmm2,%zmm7,%zmm16
    VEC_FMA231 %zmm3,%zmm7,%zmm17
    VEC_FMA231 %zmm4,%zmm7,%zmm18
    VEC_FMA231 %zmm5,%zmm7,%zmm19
    VEC_BROAD \Boff+SIZE*2(B0),%zmm6
    VEC_FMA231 %zmm0,%zmm6,%zmm20
    VEC_FMA231 %zmm1,%zmm6,%zmm21
    VEC_FMA231 %zmm2,%zmm6,%zmm22
    VEC_FMA231 %zmm3,%zmm6,%zmm23
    VEC_FMA231 %zmm4,%zmm6,%zmm24
    VEC_FMA231 %zmm5,%zmm6,%zmm25
    VEC_BROAD \Boff+SIZE*3(B0),%zmm7
    VEC_FMA231 %zmm0,%zmm7,%zmm26
    VEC_FMA231 %zmm1,%zmm7,%zmm27
    VEC_FMA231 %zmm2,%zmm7,%zmm28
    VEC_FMA231 %zmm3,%zmm7,%zmm29
    VEC_FMA231 %zmm4,%zmm7,%zmm30
    VEC_FMA231 %zmm5,%zmm7,%zmm31
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
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 (\nextablk)
    prefetcht1 64(\nextablk)
    prefetcht1 128(\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 A_PR_BYTE+512(A0)
    prefetcht0 A_PR_BYTE+576(A0)
    prefetcht0 A_PR_BYTE+640(A0)
    prefetcht0 A_PR_BYTE+704(A0)
    prefetcht1 192(\nextablk)
    prefetcht1 256(\nextablk)
    prefetcht1 320(\nextablk)
    addq $384,\nextablk
    incq %r11
    KERNEL_f 384,4*SIZE,768,8*SIZE
.endm

.macro KERNEL_4_edgen
    KERNEL_1 0,0
    KERNEL_1 384,4*SIZE
    KERNEL_1 768,8*SIZE
    incq %r11
    KERNEL_f 1152,12*SIZE,1536,16*SIZE
.endm

.macro KERNEL_4 Arefpos,Areset
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 A_PR_BYTE+512(A0)
    prefetcht0 A_PR_BYTE+576(A0)
    prefetcht0 A_PR_BYTE+640(A0)
    prefetcht0 A_PR_BYTE+704(A0)
    KERNEL_1 384,4*SIZE
# if A_PR_BYTE > 767
    prefetcht0 A_PR_BYTE+768(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+768(A0)
# endif
# if A_PR_BYTE > 703
    prefetcht0 A_PR_BYTE+832(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+832(A0)
# endif
# if A_PR_BYTE > 639
    prefetcht0 A_PR_BYTE+896(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+896(A0)
# endif
# if A_PR_BYTE > 575
    prefetcht0 A_PR_BYTE+960(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+960(A0)
# endif
# if A_PR_BYTE > 511
    prefetcht0 A_PR_BYTE+1024(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1024(A0)
# endif
# if A_PR_BYTE > 447
    prefetcht0 A_PR_BYTE+1088(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1088(A0)
# endif
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    KERNEL_1 768,8*SIZE
# if A_PR_BYTE > 383
    prefetcht0 A_PR_BYTE+1152(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1152(A0)
# endif
# if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+1216(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1216(A0)
# endif
# if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+1280(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1280(A0)
# endif
# if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+1344(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1344(A0)
# endif
# if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+1408(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+1408(A0)
# endif
    prefetcht0 A_PR_BYTE+1472(A0,AD,1)
    incq %r11
    KERNEL_f 1152,12*SIZE,1536,16*SIZE
.endm

.macro SHIFTYMM
    vmovaps %zmm14,%zmm8
    vmovaps %zmm15,%zmm9
    vmovaps %zmm16,%zmm10
    vmovaps %zmm17,%zmm11
    vmovaps %zmm18,%zmm12
    vmovaps %zmm19,%zmm13
    vmovaps %zmm20,%zmm14
    vmovaps %zmm21,%zmm15
    vmovaps %zmm22,%zmm16
    vmovaps %zmm23,%zmm17
    vmovaps %zmm24,%zmm18
    vmovaps %zmm25,%zmm19
    vmovaps %zmm26,%zmm20
    vmovaps %zmm27,%zmm21
    vmovaps %zmm28,%zmm22
    vmovaps %zmm29,%zmm23
    vmovaps %zmm30,%zmm24
    vmovaps %zmm31,%zmm25
.endm

.macro CLEAR zmmno1,zmmno2,zmmno3,zmmno4,zmmno5,zmmno6
    vpxorq %zmm\zmmno1,%zmm\zmmno1,%zmm\zmmno1
    vpxorq %zmm\zmmno2,%zmm\zmmno2,%zmm\zmmno2
    vpxorq %zmm\zmmno3,%zmm\zmmno3,%zmm\zmmno3
    vpxorq %zmm\zmmno4,%zmm\zmmno4,%zmm\zmmno4
    vpxorq %zmm\zmmno5,%zmm\zmmno5,%zmm\zmmno5
    vpxorq %zmm\zmmno6,%zmm\zmmno6,%zmm\zmmno6
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR 26,27,28,29,30,31
.endm

.macro SUB_STORE_1col zmmno1,zmmno2,zmmno3,zmmno4,zmmno5,zmmno6
    VEC_ADD (CS),%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD 64(CS),%zmm\zmmno2,%zmm\zmmno2
    VEC_ADD 128(CS),%zmm\zmmno3,%zmm\zmmno3
    VEC_ADD 192(CS),%zmm\zmmno4,%zmm\zmmno4
    VEC_ADD 256(CS),%zmm\zmmno5,%zmm\zmmno5
    VEC_ADD 320(CS),%zmm\zmmno6,%zmm\zmmno6
    vmovups %zmm\zmmno1,(CS)
    vmovups %zmm\zmmno2,64(CS)
    vmovups %zmm\zmmno3,128(CS)
    vmovups %zmm\zmmno4,192(CS)
    vmovups %zmm\zmmno5,256(CS)
    vmovups %zmm\zmmno6,320(CS)
.endm

.macro SUB_STORE_1col_edgem zmmno1,zmmno2,zmmno3,zmmno4,zmmno5,zmmno6
    CLEAR 0,1,2,3,4,5
    VEC_MASKMOV (CS),%zmm0{%k1}
    VEC_MASKMOV 64(CS),%zmm1{%k2}
    VEC_MASKMOV 128(CS),%zmm2{%k3}
    VEC_MASKMOV 192(CS),%zmm3{%k4}
    VEC_MASKMOV 256(CS),%zmm4{%k5}
    VEC_MASKMOV 320(CS),%zmm5{%k6}
    VEC_ADD %zmm0,%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD %zmm1,%zmm\zmmno2,%zmm\zmmno2
    VEC_ADD %zmm2,%zmm\zmmno3,%zmm\zmmno3
    VEC_ADD %zmm3,%zmm\zmmno4,%zmm\zmmno4
    VEC_ADD %zmm4,%zmm\zmmno5,%zmm\zmmno5
    VEC_ADD %zmm5,%zmm\zmmno6,%zmm\zmmno6
    VEC_MASKMOV %zmm\zmmno1,(CS){%k1}
    VEC_MASKMOV %zmm\zmmno2,64(CS){%k2}
    VEC_MASKMOV %zmm\zmmno3,128(CS){%k3}
    VEC_MASKMOV %zmm\zmmno4,192(CS){%k4}
    VEC_MASKMOV %zmm\zmmno5,256(CS){%k5}
    VEC_MASKMOV %zmm\zmmno6,320(CS){%k6}
.endm

.macro STORECBLK_1col
    SUB_STORE_1col 8,9,10,11,12,13
    addq LDC,CS
.endm

.macro STORECBLK_1col_edgem
    SUB_STORE_1col_edgem 8,9,10,11,12,13
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR 14,15,16,17,18,19
    CLEAR 20,21,22,23,24,25
    CLEAR 26,27,28,29,30,31
.endm

.macro FIN_C
    SUB_STORE_1col 8,9,10,11,12,13
    addq LDC,CS
    SUB_STORE_1col 14,15,16,17,18,19
    addq LDC,CS
    SUB_STORE_1col 20,21,22,23,24,25
.endm

.macro FIN_C_edgem
    SUB_STORE_1col_edgem 14,15,16,17,18,19
    addq LDC,CS
    SUB_STORE_1col_edgem 20,21,22,23,24,25
    addq LDC,CS
    SUB_STORE_1col_edgem 26,27,28,29,30,31
.endm

.macro ZERO_C_1col //used in edge part
    CLEAR 8,9,10,11,12,13
.endm

.macro ZERO_C_ncol //used in edge part
    CLEAR 8,9,10,11,12,13
    INIT_C
.endm

.macro STORECBLK_ncol //used in edge part
    FIN_C
    addq LDC,CS
    SUB_STORE_1col 26,27,28,29,30,31
    addq LDC,CS
.endm

.macro STORECBLK_ncol_edgem //used in edge part
    SUB_STORE_1col_edgem 8,9,10,11,12,13
    addq LDC,CS
    FIN_C_edgem
    addq LDC,CS
.endm
