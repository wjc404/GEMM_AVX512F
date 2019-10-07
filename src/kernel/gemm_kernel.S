# include "../gemm_set_parameters.h"
# define A0      %rdi //ablk pointer
# define B0      %rsi //bblk pointer
# define CL      %r14 //cload pointer
# define CS      %r15 //cstore pointer
# define LDC     %rcx //ldc * sizeof(float)
# define AL      %rax //aload pointer
# define CIP  -8(%rsp)//cstartpos
# define AD      %r10 //A offset

# ifdef DOUBLE
 # define VEC_BROAD vbroadcastsd
 # define VEC_FMA231 vfmadd231pd
 # define VEC_ADD vaddpd
 # define VEC_MASKMOV vmovupd
 # define SIZE 8
# else
 # define VEC_BROAD vbroadcastss
 # define VEC_FMA231 vfmadd231ps
 # define VEC_ADD vaddps
 # define VEC_MASKMOV vmovups
 # define SIZE 4
# endif

# define BYTES_PER_COL (GEMM_UNROLL_M_VEC*64)
# define NEXT_A_PREF_STEP (BYTES_PER_COL*GEMM_BLOCK_L1DIM_K/GEMM_BLOCK_DIM_N) //in bytes

# if GEMM_UNROLL_M_VEC == 2
 # define LAST_PREFNEXTA_UNIT 64 // in bytes
 # define ZMMPARLIST zmmno1,zmmno2
 # define ZMMDSTLIST zmmds1,zmmds2
 # define ZMMCOLT 0,1
 # define ZMMCOL1 8,9
 # define ZMMCOL2 10,11
 # define ZMMCOL3 12,13
 # define ZMMCOL4 14,15
 # define ZMMCOL5 16,17
 # define ZMMCOL6 18,19
 # define ZMMCOL7 20,21
 # define ZMMCOL8 22,23
 # define ZMMCOL9 24,25
 # define ZMMCOL10 26,27
 # define ZMMCOL11 28,29
 # define ZMMCOL12 30,31
# endif
# if GEMM_UNROLL_M_VEC == 3
 # define LAST_PREFNEXTA_UNIT 128
 # define ZMMPARLIST zmmno1,zmmno2,zmmno3
 # define ZMMDSTLIST zmmds1,zmmds2,zmmds3
 # define ZMMCOLT 0,1,2
 # define ZMMCOL1 8,9,10
 # define ZMMCOL2 11,12,13
 # define ZMMCOL3 14,15,16
 # define ZMMCOL4 17,18,19
 # define ZMMCOL5 20,21,22
 # define ZMMCOL6 23,24,25
 # define ZMMCOL7 26,27,28
 # define ZMMCOL8 29,30,31
# endif
# if GEMM_UNROLL_M_VEC == 4
 # define LAST_PREFNEXTA_UNIT 128
 # define ZMMPARLIST zmmno1,zmmno2,zmmno3,zmmno4
 # define ZMMDSTLIST zmmds1,zmmds2,zmmds3,zmmds4
 # define ZMMCOLT 0,1,2,3
 # define ZMMCOL1 8,9,10,11
 # define ZMMCOL2 12,13,14,15
 # define ZMMCOL3 16,17,18,19
 # define ZMMCOL4 20,21,22,23
 # define ZMMCOL5 24,25,26,27
 # define ZMMCOL6 28,29,30,31
# endif
# if GEMM_UNROLL_M_VEC == 5
 # define LAST_PREFNEXTA_UNIT 192
 # define ZMMPARLIST zmmno1,zmmno2,zmmno3,zmmno4,zmmno5
 # define ZMMDSTLIST zmmds1,zmmds2,zmmds3,zmmds4,zmmds5
 # define ZMMCOLT 0,1,2,3,4
 # define ZMMCOL1 8,9,10,11,12
 # define ZMMCOL2 13,14,15,16,17
 # define ZMMCOL3 18,19,20,21,22
 # define ZMMCOL4 23,24,25,26,27
# endif
# if GEMM_UNROLL_M_VEC == 6
 # define LAST_PREFNEXTA_UNIT 192
 # define ZMMPARLIST zmmno1,zmmno2,zmmno3,zmmno4,zmmno5,zmmno6
 # define ZMMDSTLIST zmmds1,zmmds2,zmmds3,zmmds4,zmmds5,zmmds6
 # define ZMMCOLT 0,1,2,3,4,5
 # define ZMMCOL1 8,9,10,11,12,13
 # define ZMMCOL2 14,15,16,17,18,19
 # define ZMMCOL3 20,21,22,23,24,25
 # define ZMMCOL4 26,27,28,29,30,31
# endif

# define ZMMCOL_FIRST ZMMCOL1
# define ZMMCOL_last(imm) ZMMCOL##imm
# define zmmcol_LAST(imm) ZMMCOL_last(imm)
# define ZMMCOL_LAST zmmcol_LAST(GEMM_UNROLL_N)

.macro LOAD_A_1COL Aoffset
    vmovaps \Aoffset(A0),%zmm0
    vmovaps \Aoffset+64(A0),%zmm1
# if GEMM_UNROLL_M_VEC > 2
    vmovaps \Aoffset+128(A0),%zmm2
# endif
# if GEMM_UNROLL_M_VEC > 3
    vmovaps \Aoffset+192(A0),%zmm3
# endif
# if GEMM_UNROLL_M_VEC > 4
    vmovaps \Aoffset+256(A0),%zmm4
# endif
# if GEMM_UNROLL_M_VEC > 5
    vmovaps \Aoffset+320(A0),%zmm5
# endif
.endm

.macro FMA_1COL ZMMPARLIST
    VEC_FMA231 %zmm0,%zmm7,%zmm\zmmno1
    VEC_FMA231 %zmm1,%zmm7,%zmm\zmmno2
# if GEMM_UNROLL_M_VEC > 2
    VEC_FMA231 %zmm2,%zmm7,%zmm\zmmno3
# endif
# if GEMM_UNROLL_M_VEC > 3
    VEC_FMA231 %zmm3,%zmm7,%zmm\zmmno4
# endif
# if GEMM_UNROLL_M_VEC > 4
    VEC_FMA231 %zmm4,%zmm7,%zmm\zmmno5
# endif
# if GEMM_UNROLL_M_VEC > 5
    VEC_FMA231 %zmm5,%zmm7,%zmm\zmmno6
# endif
.endm

.macro KERNEL_0
    LOAD_A_1COL 0
    addq $BYTES_PER_COL,A0
    VEC_BROAD (B0),%zmm7
    addq $SIZE,B0
    FMA_1COL ZMMCOL1
.endm

.macro KERNEL_1 Aoff,Boff
    LOAD_A_1COL \Aoff
    VEC_BROAD \Boff(B0),%zmm7
    FMA_1COL ZMMCOL1
    VEC_BROAD \Boff+SIZE(B0),%zmm7
    FMA_1COL ZMMCOL2
# if GEMM_UNROLL_N > 2
    VEC_BROAD \Boff+SIZE*2(B0),%zmm7
    FMA_1COL ZMMCOL3
 # if GEMM_UNROLL_N > 3
    VEC_BROAD \Boff+SIZE*3(B0),%zmm7
    FMA_1COL ZMMCOL4
  # if GEMM_UNROLL_N > 4
    VEC_BROAD \Boff+SIZE*4(B0),%zmm7
    FMA_1COL ZMMCOL5
   # if GEMM_UNROLL_N > 5
    VEC_BROAD \Boff+SIZE*5(B0),%zmm7
    FMA_1COL ZMMCOL6
    # if GEMM_UNROLL_N > 6
    VEC_BROAD \Boff+SIZE*6(B0),%zmm7
    FMA_1COL ZMMCOL7
     # if GEMM_UNROLL_N > 7
    VEC_BROAD \Boff+SIZE*7(B0),%zmm7
    FMA_1COL ZMMCOL8
      # if GEMM_UNROLL_N > 8
    VEC_BROAD \Boff+SIZE*8(B0),%zmm7
    FMA_1COL ZMMCOL9
       # if GEMM_UNROLL_N > 9
    VEC_BROAD \Boff+SIZE*9(B0),%zmm7
    FMA_1COL ZMMCOL10
        # if GEMM_UNROLL_N > 10
    VEC_BROAD \Boff+SIZE*10(B0),%zmm7
    FMA_1COL ZMMCOL11
         # if GEMM_UNROLL_N > 11
    VEC_BROAD \Boff+SIZE*11(B0),%zmm7
    FMA_1COL ZMMCOL12
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

.macro PREF_BULK_UNIT_B notation
    prefetcht0 (B_PR_ELEM+\notation*GEMM_UNROLL_N)*SIZE(B0)
# if GEMM_UNROLL_N > 64/SIZE
    prefetcht0 (B_PR_ELEM+\notation*GEMM_UNROLL_N)*SIZE+64(B0)
# endif
.endm

.macro PREF_LAST_UNIT notation,nextablk
    prefetcht0 A_PR_BYTE+\notation*BYTES_PER_COL(A0)
    prefetcht0 A_PR_BYTE+\notation*BYTES_PER_COL+64(A0)
# if GEMM_UNROLL_M_VEC > 2
    prefetcht0 A_PR_BYTE+\notation*BYTES_PER_COL+128(A0)
# endif
# if GEMM_UNROLL_M_VEC > 3
    prefetcht0 A_PR_BYTE+\notation*BYTES_PER_COL+192(A0)
# endif
# if GEMM_UNROLL_M_VEC > 4
    prefetcht0 A_PR_BYTE+\notation*BYTES_PER_COL+256(A0)
# endif
# if GEMM_UNROLL_M_VEC > 5
    prefetcht0 A_PR_BYTE+\notation*BYTES_PER_COL+320(A0)
# endif
    PREF_BULK_UNIT_B \notation
    prefetcht1 \notation*LAST_PREFNEXTA_UNIT(\nextablk)
# if LAST_PREFNEXTA_UNIT > 64
    prefetcht1 \notation*LAST_PREFNEXTA_UNIT+64(\nextablk)
# endif
# if LAST_PREFNEXTA_UNIT > 128
    prefetcht1 \notation*LAST_PREFNEXTA_UNIT+128(\nextablk)
# endif
.endm

.macro KERNEL_UNROLL_K_edgen
    KERNEL_1 0,0
    KERNEL_1 BYTES_PER_COL,GEMM_UNROLL_N*SIZE
    KERNEL_1 BYTES_PER_COL*2,GEMM_UNROLL_N*SIZE*2
# if GEMM_UNROLL_K > 4
    KERNEL_1 BYTES_PER_COL*3,GEMM_UNROLL_N*SIZE*3
# endif
# if GEMM_UNROLL_K > 5
    KERNEL_1 BYTES_PER_COL*4,GEMM_UNROLL_N*SIZE*4
# endif
# if GEMM_UNROLL_K > 6
    KERNEL_1 BYTES_PER_COL*5,GEMM_UNROLL_N*SIZE*5
# endif
# if GEMM_UNROLL_K > 7
    KERNEL_1 BYTES_PER_COL*6,GEMM_UNROLL_N*SIZE*6
# endif
    KERNEL_f BYTES_PER_COL*(GEMM_UNROLL_K-1),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-1),BYTES_PER_COL*GEMM_UNROLL_K,GEMM_UNROLL_N*SIZE*GEMM_UNROLL_K
.endm

.macro KERNEL_UNROLL_K_last nextablk
    PREF_LAST_UNIT 0,\nextablk
    KERNEL_1 0,0
    PREF_LAST_UNIT 1,\nextablk
    KERNEL_1 BYTES_PER_COL,GEMM_UNROLL_N*SIZE
    PREF_LAST_UNIT 2,\nextablk
    KERNEL_1 BYTES_PER_COL*2,GEMM_UNROLL_N*SIZE*2
    PREF_LAST_UNIT 3,\nextablk
# if GEMM_UNROLL_K > 4
    KERNEL_1 BYTES_PER_COL*3,GEMM_UNROLL_N*SIZE*3
    PREF_LAST_UNIT 4,\nextablk
# endif
# if GEMM_UNROLL_K > 5
    KERNEL_1 BYTES_PER_COL*4,GEMM_UNROLL_N*SIZE*4
    PREF_LAST_UNIT 5,\nextablk
# endif
# if GEMM_UNROLL_K > 6
    KERNEL_1 BYTES_PER_COL*5,GEMM_UNROLL_N*SIZE*5
    PREF_LAST_UNIT 6,\nextablk
# endif
# if GEMM_UNROLL_K > 7
    KERNEL_1 BYTES_PER_COL*6,GEMM_UNROLL_N*SIZE*6
    PREF_LAST_UNIT 7,\nextablk
# endif
    addq $LAST_PREFNEXTA_UNIT*GEMM_UNROLL_K,\nextablk
    KERNEL_f BYTES_PER_COL*(GEMM_UNROLL_K-1),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-1),BYTES_PER_COL*GEMM_UNROLL_K,GEMM_UNROLL_N*SIZE*GEMM_UNROLL_K
.endm

.macro KERNEL_BULK_UNIT_FRONT notation
# if GEMM_UNROLL_M_VEC > 5
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-\notation)-384(A0)
# endif
# if GEMM_UNROLL_M_VEC > 4
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-\notation)-320(A0)
# endif
# if GEMM_UNROLL_M_VEC > 3
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-\notation)-256(A0)
# endif
# if GEMM_UNROLL_M_VEC > 2
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-\notation)-192(A0)
# endif
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-\notation)-128(A0)
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-\notation)-64(A0)
    PREF_BULK_UNIT_B (GEMM_UNROLL_K-\notation-1)
    KERNEL_1 BYTES_PER_COL*(GEMM_UNROLL_K-\notation-1),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-\notation-1)
.endm

.macro KERNEL_UNROLL_K Arefpos,Areset
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
# if GEMM_UNROLL_K > 7
    KERNEL_BULK_UNIT_FRONT 7
# endif
# if GEMM_UNROLL_K > 6
    KERNEL_BULK_UNIT_FRONT 6
# endif
# if GEMM_UNROLL_K > 5
    KERNEL_BULK_UNIT_FRONT 5
# endif
# if GEMM_UNROLL_K > 4
    KERNEL_BULK_UNIT_FRONT 4
# endif
# if GEMM_UNROLL_M_VEC > 5
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-384(A0)
# endif
# if GEMM_UNROLL_M_VEC > 4
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-320(A0)
# endif
# if GEMM_UNROLL_M_VEC > 3
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-256(A0)
# endif
# if GEMM_UNROLL_M_VEC > 2
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-192(A0)
# endif
# if A_PR_BYTE > BYTES_PER_COL*3+127
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-128(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-128(A0)
# endif
# if A_PR_BYTE > BYTES_PER_COL*3+63
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-64(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-3)-64(A0)
# endif
    PREF_BULK_UNIT_B (GEMM_UNROLL_K-4)
    KERNEL_1 BYTES_PER_COL*(GEMM_UNROLL_K-4),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-4)
# if GEMM_UNROLL_M_VEC > 5
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-384(A0)
# endif
# if GEMM_UNROLL_M_VEC > 4
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-320(A0)
# endif
# if GEMM_UNROLL_M_VEC > 3
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-256(A0)
# endif
# if GEMM_UNROLL_M_VEC > 2
 # if A_PR_BYTE > BYTES_PER_COL*2+191
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-192(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-192(A0)
 # endif
# endif
# if A_PR_BYTE > BYTES_PER_COL*2+127
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-128(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-128(A0)
# endif
# if A_PR_BYTE > BYTES_PER_COL*2+63
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-64(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-2)-64(A0)
# endif
    PREF_BULK_UNIT_B (GEMM_UNROLL_K-3)
    KERNEL_1 BYTES_PER_COL*(GEMM_UNROLL_K-3),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-3)
# if GEMM_UNROLL_M_VEC > 5
 # if A_PR_BYTE > BYTES_PER_COL+383
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-384(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-384(A0)
 # endif
# endif
# if GEMM_UNROLL_M_VEC > 4
 # if A_PR_BYTE > BYTES_PER_COL+319
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-320(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-320(A0)
 # endif
# endif
# if GEMM_UNROLL_M_VEC > 3
 # if A_PR_BYTE > BYTES_PER_COL+255
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-256(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-256(A0)
 # endif
# endif
# if GEMM_UNROLL_M_VEC > 2
 # if A_PR_BYTE > BYTES_PER_COL+191
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-192(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-192(A0)
 # endif
# endif
# if A_PR_BYTE > BYTES_PER_COL+127
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-128(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-128(A0)
# endif
# if A_PR_BYTE > BYTES_PER_COL+63
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-64(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*(GEMM_UNROLL_K-1)-64(A0)
# endif
    PREF_BULK_UNIT_B (GEMM_UNROLL_K-2)
    KERNEL_1 BYTES_PER_COL*(GEMM_UNROLL_K-2),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-2)
# if GEMM_UNROLL_M_VEC > 5
 # if A_PR_BYTE > 383
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-384(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-384(A0)
 # endif
# endif
# if GEMM_UNROLL_M_VEC > 4
 # if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-320(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-320(A0)
 # endif
# endif
# if GEMM_UNROLL_M_VEC > 3
 # if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-256(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-256(A0)
 # endif
# endif
# if GEMM_UNROLL_M_VEC > 2
 # if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-192(A0,AD,1)
 # else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-192(A0)
 # endif
# endif
# if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-128(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-128(A0)
# endif
    prefetcht0 A_PR_BYTE+BYTES_PER_COL*GEMM_UNROLL_K-64(A0,AD,1)
    PREF_BULK_UNIT_B (GEMM_UNROLL_K-1)
    KERNEL_f BYTES_PER_COL*(GEMM_UNROLL_K-1),GEMM_UNROLL_N*SIZE*(GEMM_UNROLL_K-1),BYTES_PER_COL*GEMM_UNROLL_K,GEMM_UNROLL_N*SIZE*GEMM_UNROLL_K
.endm

.macro CLEAR ZMMPARLIST
    vpxorq %zmm\zmmno1,%zmm\zmmno1,%zmm\zmmno1
    vpxorq %zmm\zmmno2,%zmm\zmmno2,%zmm\zmmno2
# if GEMM_UNROLL_M_VEC > 2
    vpxorq %zmm\zmmno3,%zmm\zmmno3,%zmm\zmmno3
# endif
# if GEMM_UNROLL_M_VEC > 3
    vpxorq %zmm\zmmno4,%zmm\zmmno4,%zmm\zmmno4
# endif
# if GEMM_UNROLL_M_VEC > 4
    vpxorq %zmm\zmmno5,%zmm\zmmno5,%zmm\zmmno5
# endif
# if GEMM_UNROLL_M_VEC > 5
    vpxorq %zmm\zmmno6,%zmm\zmmno6,%zmm\zmmno6
# endif
.endm

.macro UNIT_SHIFTZMM ZMMPARLIST,ZMMDSTLIST
    vmovaps %zmm\zmmno1,%zmm\zmmds1
    vmovaps %zmm\zmmno2,%zmm\zmmds2
# if GEMM_UNROLL_M_VEC > 2
    vmovaps %zmm\zmmno3,%zmm\zmmds3
# endif
# if GEMM_UNROLL_M_VEC > 3
    vmovaps %zmm\zmmno4,%zmm\zmmds4
# endif
# if GEMM_UNROLL_M_VEC > 4
    vmovaps %zmm\zmmno5,%zmm\zmmds5
# endif
# if GEMM_UNROLL_M_VEC > 5
    vmovaps %zmm\zmmno6,%zmm\zmmds6
# endif
.endm

.macro SHIFTZMM
    UNIT_SHIFTZMM ZMMCOL2,ZMMCOL1
# if GEMM_UNROLL_N > 2
    UNIT_SHIFTZMM ZMMCOL3,ZMMCOL2
# endif
# if GEMM_UNROLL_N > 3
    UNIT_SHIFTZMM ZMMCOL4,ZMMCOL3
# endif
# if GEMM_UNROLL_N > 4
    UNIT_SHIFTZMM ZMMCOL5,ZMMCOL4
# endif
# if GEMM_UNROLL_N > 5
    UNIT_SHIFTZMM ZMMCOL6,ZMMCOL5
# endif
# if GEMM_UNROLL_N > 6
    UNIT_SHIFTZMM ZMMCOL7,ZMMCOL6
# endif
# if GEMM_UNROLL_N > 7
    UNIT_SHIFTZMM ZMMCOL8,ZMMCOL7
# endif
# if GEMM_UNROLL_N > 8
    UNIT_SHIFTZMM ZMMCOL9,ZMMCOL8
# endif
# if GEMM_UNROLL_N > 9
    UNIT_SHIFTZMM ZMMCOL10,ZMMCOL9
# endif
# if GEMM_UNROLL_N > 10
    UNIT_SHIFTZMM ZMMCOL11,ZMMCOL10
# endif
# if GEMM_UNROLL_N > 11
    UNIT_SHIFTZMM ZMMCOL12,ZMMCOL11
# endif
.endm

.macro UPDATECBLK_1col
    SHIFTZMM
    CLEAR ZMMCOL_LAST
.endm

.macro SUB_STORE_1col ZMMPARLIST
    VEC_ADD (CS),%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD 64(CS),%zmm\zmmno2,%zmm\zmmno2
    vmovups %zmm\zmmno1,(CS)
    vmovups %zmm\zmmno2,64(CS)
# if GEMM_UNROLL_M_VEC > 2
    VEC_ADD 128(CS),%zmm\zmmno3,%zmm\zmmno3
    vmovups %zmm\zmmno3,128(CS)
# endif
# if GEMM_UNROLL_M_VEC > 3
    VEC_ADD 192(CS),%zmm\zmmno4,%zmm\zmmno4
    vmovups %zmm\zmmno4,192(CS)
# endif
# if GEMM_UNROLL_M_VEC > 4
    VEC_ADD 256(CS),%zmm\zmmno5,%zmm\zmmno5
    vmovups %zmm\zmmno5,256(CS)
# endif
# if GEMM_UNROLL_M_VEC > 5
    VEC_ADD 320(CS),%zmm\zmmno6,%zmm\zmmno6
    vmovups %zmm\zmmno6,320(CS)
# endif
.endm

.macro SUB_STORE_1col_edgem ZMMPARLIST
    CLEAR ZMMCOLT
    VEC_MASKMOV (CS),%zmm0{%k1}
    VEC_MASKMOV 64(CS),%zmm1{%k2}
    VEC_ADD %zmm0,%zmm\zmmno1,%zmm\zmmno1
    VEC_ADD %zmm1,%zmm\zmmno2,%zmm\zmmno2
    VEC_MASKMOV %zmm\zmmno1,(CS){%k1}
    VEC_MASKMOV %zmm\zmmno2,64(CS){%k2}
# if GEMM_UNROLL_M_VEC > 2
    VEC_MASKMOV 128(CS),%zmm2{%k3}
    VEC_ADD %zmm2,%zmm\zmmno3,%zmm\zmmno3
    VEC_MASKMOV %zmm\zmmno3,128(CS){%k3}
# endif
# if GEMM_UNROLL_M_VEC > 3
    VEC_MASKMOV 192(CS),%zmm3{%k4}
    VEC_ADD %zmm3,%zmm\zmmno4,%zmm\zmmno4
    VEC_MASKMOV %zmm\zmmno4,192(CS){%k4}
# endif
# if GEMM_UNROLL_M_VEC > 4
    VEC_MASKMOV 256(CS),%zmm4{%k5}
    VEC_ADD %zmm4,%zmm\zmmno5,%zmm\zmmno5
    VEC_MASKMOV %zmm\zmmno5,256(CS){%k5}
# endif
# if GEMM_UNROLL_M_VEC > 5
    VEC_MASKMOV 320(CS),%zmm5{%k6}
    VEC_ADD %zmm5,%zmm\zmmno6,%zmm\zmmno6
    VEC_MASKMOV %zmm\zmmno6,320(CS){%k6}
# endif
.endm

.macro STORECBLK_1col
    SUB_STORE_1col ZMMCOL1
    addq LDC,CS
.endm

.macro STORECBLK_1col_edgem
    SUB_STORE_1col_edgem ZMMCOL1
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR ZMMCOL2
# if GEMM_UNROLL_N > 2
    CLEAR ZMMCOL3
 # if GEMM_UNROLL_N > 3
    CLEAR ZMMCOL4
  # if GEMM_UNROLL_N > 4
    CLEAR ZMMCOL5
   # if GEMM_UNROLL_N > 5
    CLEAR ZMMCOL6
    # if GEMM_UNROLL_N > 6
    CLEAR ZMMCOL7
     # if GEMM_UNROLL_N > 7
    CLEAR ZMMCOL8
      # if GEMM_UNROLL_N > 8
    CLEAR ZMMCOL9
       # if GEMM_UNROLL_N > 9
    CLEAR ZMMCOL10
        # if GEMM_UNROLL_N > 10
    CLEAR ZMMCOL11
         # if GEMM_UNROLL_N > 11
    CLEAR ZMMCOL12
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
    SUB_STORE_1col ZMMCOL1
# if GEMM_UNROLL_N > 2
    addq LDC,CS
    SUB_STORE_1col ZMMCOL2
 # if GEMM_UNROLL_N > 3
    addq LDC,CS
    SUB_STORE_1col ZMMCOL3
  # if GEMM_UNROLL_N > 4
    addq LDC,CS
    SUB_STORE_1col ZMMCOL4
   # if GEMM_UNROLL_N > 5
    addq LDC,CS
    SUB_STORE_1col ZMMCOL5
    # if GEMM_UNROLL_N > 6
    addq LDC,CS
    SUB_STORE_1col ZMMCOL6
     # if GEMM_UNROLL_N > 7
    addq LDC,CS
    SUB_STORE_1col ZMMCOL7
      # if GEMM_UNROLL_N > 8
    addq LDC,CS
    SUB_STORE_1col ZMMCOL8
       # if GEMM_UNROLL_N > 9
    addq LDC,CS
    SUB_STORE_1col ZMMCOL9
        # if GEMM_UNROLL_N > 10
    addq LDC,CS
    SUB_STORE_1col ZMMCOL10
         # if GEMM_UNROLL_N > 11
    addq LDC,CS
    SUB_STORE_1col ZMMCOL11
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
    SUB_STORE_1col_edgem ZMMCOL2
# if GEMM_UNROLL_N > 2
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL3
 # if GEMM_UNROLL_N > 3
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL4
  # if GEMM_UNROLL_N > 4
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL5
   # if GEMM_UNROLL_N > 5
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL6
    # if GEMM_UNROLL_N > 6
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL7
     # if GEMM_UNROLL_N > 7
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL8
      # if GEMM_UNROLL_N > 8
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL9
       # if GEMM_UNROLL_N > 9
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL10
        # if GEMM_UNROLL_N > 10
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL11
         # if GEMM_UNROLL_N > 11
    addq LDC,CS
    SUB_STORE_1col_edgem ZMMCOL12
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
    CLEAR ZMMCOL1
.endm

.macro ZERO_C_ncol //used in edge part
    CLEAR ZMMCOL1
    INIT_C
.endm

.macro STORECBLK_ncol //used in edge part
    FIN_C
    addq LDC,CS
    SUB_STORE_1col ZMMCOL_LAST
    addq LDC,CS
.endm

.macro STORECBLK_ncol_edgem //used in edge part
    SUB_STORE_1col_edgem ZMMCOL1
    addq LDC,CS
    FIN_C_edgem
    addq LDC,CS
.endm

.macro PREF_1_col insn,src //insn=prefetcht0,prefetcht1 or prefetcht2; src=c_address_register
    \insn (\src)
# if BYTES_PER_COL > 65
    \insn 64(\src)
# endif
# if BYTES_PER_COL > 129
    \insn 128(\src)
# endif
# if BYTES_PER_COL > 193
    \insn 192(\src)
# endif
# if BYTES_PER_COL > 257
    \insn 256(\src)
# endif
# if BYTES_PER_COL > 321
    \insn 320(\src)
# endif
# if BYTES_PER_COL > 385
    \insn 384(\src)
# endif
# if BYTES_PER_COL > 449
    \insn 448(\src)
# endif
# if BYTES_PER_COL > 513
    \insn 512(\src)
# endif
    \insn BYTES_PER_COL-1(\src)
.endm

.macro SET_LDC
# ifdef DOUBLE
    salq $3,LDC
# else
    salq $2,LDC
# endif
.endm

.macro SETMASKm_r32 r32_mdim,r32_temp,r32_zero,max_digit //please note temporary use of %rcx here.
    movl $0xffffffff,\r32_temp
    movl $\max_digit,%ecx
    subl \r32_mdim,%ecx
    testl $0x80000000,%ecx
    cmovnz \r32_zero,%ecx //if(ecx<0) ecx=0;
    shrl %cl,\r32_temp
    cmpl $32,%ecx
    cmovnb \r32_zero,\r32_temp //shrl only reads 5 lowest bits of %cl, so cases when %ecx >= 32 must be considered!
.endm

.macro SETMASKm r32_mdim,r32_temp,r32_zero //please note temporary use of %rcx here.
    xorl \r32_zero,\r32_zero
# ifdef DOUBLE
    SETMASKm_r32 \r32_mdim,\r32_temp,\r32_zero,32
/*now divide r32_temp to 4 parts and move them to k1-k4 */
    kmovw \r32_temp,%k1 //lowest bits of edge-c column
    shrl $8,\r32_temp
    kmovw \r32_temp,%k2
  # if GEMM_UNROLL_M_VEC > 2
    shrl $8,\r32_temp
    kmovw \r32_temp,%k3
  # endif
  # if GEMM_UNROLL_M_VEC > 3
    shrl $8,\r32_temp
    kmovw \r32_temp,%k4
  # endif
  # if GEMM_UNROLL_M_VEC > 4
    SETMASKm_r32 \r32_mdim,\r32_temp,\r32_zero,64
/*now divide r32_temp to 4 parts and move them to k5-k6 */
    kmovw \r32_temp,%k5
    # if GEMM_UNROLL_M_VEC > 5
    shrl $8,\r32_temp
    kmovw \r32_temp,%k6
    # endif
  # endif
# else
    SETMASKm_r32 \r32_mdim,\r32_temp,\r32_zero,32
/*now divide r32_temp to 2 parts and move them to k1-k2 */
    kmovw \r32_temp,%k1
    shrl $16,\r32_temp
    kmovw \r32_temp,%k2
  # if GEMM_UNROLL_M_VEC > 2
    SETMASKm_r32 \r32_mdim,\r32_temp,\r32_zero,64
/*now divide r32_temp to 2 parts and move them to k3-k4 */
    kmovw \r32_temp,%k3
    # if GEMM_UNROLL_M_VEC > 3
    shrl $16,\r32_temp
    kmovw \r32_temp,%k4
    # endif
  # endif
  # if GEMM_UNROLL_M_VEC > 4
    SETMASKm_r32 \r32_mdim,\r32_temp,\r32_zero,96
/*now divide r32_temp to 2 parts and move them to k5-k6 */
    kmovw \r32_temp,%k5
    # if GEMM_UNROLL_M_VEC > 5
    shrl $16,\r32_temp
    kmovw \r32_temp,%k6
    # endif
  # endif
# endif
.endm

.section .text
//enter the function unit_gemmblkregccc, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=prefcoffset(in bytes)
.globl unit_gemmblkregccc
.type unit_gemmblkregccc,@function
unit_gemmblkregccc:

    push %r15
    push %r14
    push %r13
    push %r12
    movq %rdx,CIP
    movq %rdi,AL
    movslq %r8d,%r13 //pref_c_offset in bytes
    addq $BYTES_PER_COL*2*GEMM_BLOCK_L1DIM_K,AL //point to (prefetch) next ablk zone of abuffer, start from the tail part
    movslq %ecx,LDC
    SET_LDC
    movq CIP,CS

    INIT_C
    xorq %r12,%r12
    movq A0,%r9
    addq $(GEMM_BLOCK_L1DIM_K-GEMM_UNROLL_K)*BYTES_PER_COL,%r9 //Arefpos
    movq $(-GEMM_BLOCK_L1DIM_K)*BYTES_PER_COL,%r8 //Areset
.Louter_gemmblkregccc:
    xorq AD,AD
    UPDATECBLK_1col
    PREF_1_col prefetcht0,CS
    subq $NEXT_A_PREF_STEP,AL
    prefetcht1 (AL)
# if NEXT_A_PREF_STEP > 64
    prefetcht1 64(AL)
# endif
# if NEXT_A_PREF_STEP > 128
    prefetcht1 128(AL)
# endif
# if NEXT_A_PREF_STEP > 192
    prefetcht1 192(AL)
# endif
# if NEXT_A_PREF_STEP > 256
    prefetcht1 256(AL)
# endif
# if NEXT_A_PREF_STEP > 320
    prefetcht1 320(AL)
# endif
# if NEXT_A_PREF_STEP > 384
    prefetcht1 384(AL)
# endif
# if NEXT_A_PREF_STEP > 448
    prefetcht1 448(AL)
# endif
# if NEXT_A_PREF_STEP > 512
    prefetcht1 512(AL)
# endif
# ifndef NO_REPEAT_C_BLOCK
    prefetcht2 (CS,%r13,1)
# endif
    xorq %r11,%r11
.Linner_gemmblkregccc:
    incq %r11
    KERNEL_UNROLL_K %r9,%r8
    cmpq $GEMM_LOOP_TIMES_K,%r11
    jb .Linner_gemmblkregccc

    addq AD,A0
    incq %r12
    STORECBLK_1col
    cmpq $GEMM_BLOCK_DIM_N-GEMM_UNROLL_N,%r12
    jb .Louter_gemmblkregccc

    movq A0,%r9
    addq $BYTES_PER_COL*GEMM_BLOCK_L1DIM_K,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_gemmblkregccc_last:
    PREF_1_col prefetcht0,CS
    PREF_1_col prefetcht1,CL
    addq LDC,CL
    prefetcht2 (CS,%r13,1)
    xorq %r11,%r11
.Linner_gemmblkregccc_last:
    incq %r11
    KERNEL_UNROLL_K_last %r9
    cmpq $GEMM_LOOP_TIMES_K,%r11
    jb .Linner_gemmblkregccc_last

    incq %r12
    STORECBLK_1col
    UPDATECBLK_1col
    cmpq $GEMM_BLOCK_DIM_N,%r12
    jb .Louter_gemmblkregccc_last

    movq CIP,CS
    FIN_C

    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function unit_gemmblktailccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim
.globl unit_gemmblktailccc
.type unit_gemmblktailccc,@function
unit_gemmblktailccc:

    push %r15
    push %r14
    push %r13
    push %r12
    movq %rdx,CIP
    movslq %ecx,%r13 //temporarily store LDC
    SETMASKm %r8d,%r14d,%r15d //fill k0-k7
    movq %r13,LDC
    SET_LDC

    movq CIP,CS
    INIT_C
    xorq %r12,%r12
    movq A0,%r9
    addq $(GEMM_BLOCK_L1DIM_K-GEMM_UNROLL_K)*BYTES_PER_COL,%r9 //Arefpos
    movq $(-GEMM_BLOCK_L1DIM_K)*BYTES_PER_COL,%r8 //Areset
.Louter_tail:
    xorq AD,AD
    UPDATECBLK_1col
    PREF_1_col prefetcht0,CS
    xorq %r11,%r11
.Linner_tail:
    incq %r11
    KERNEL_UNROLL_K %r9,%r8
    cmpq $GEMM_LOOP_TIMES_K,%r11
    jb .Linner_tail

    addq AD,A0
    STORECBLK_1col_edgem //use k1-k7 instead of stack
    incq %r12
    cmpq $GEMM_BLOCK_DIM_N,%r12
    jb .Louter_tail

    movq CIP,CS
    FIN_C_edgem //use k1-k7 instead of stack
    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

.macro PREF_C_1col pref_c_offset,c_addr_reg
    PREF_1_col prefetcht0,\c_addr_reg
# ifndef NO_REPEAT_C_BLOCK
    prefetcht2 (\c_addr_reg,\pref_c_offset,1)
# endif
.endm

.macro PREF_C_ncol pref_c_offset
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# if GEMM_UNROLL_N > 2
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 3
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 4
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 5
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 6
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 7
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 8
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 9
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 10
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
# if GEMM_UNROLL_N > 11
    PREF_C_1col \pref_c_offset,CL
    addq LDC,CL
# endif
.endm

//enter the function unit_gemmblkirregkccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=kdim, r9d=pref_c_offset(in bytes)
.globl unit_gemmblkirregkccc
.type unit_gemmblkirregkccc,@function
unit_gemmblkirregkccc: //CS for store, CL for prefetcht0

    push %r15
    push %r14
    push %r13
    push %r12
    movq %rdx,CS
    movq CS,CL
    movslq %ecx,LDC
    movq %rdi,AL //save ablk_head address
    movslq %r9d,%r9 //pref_c_offset
    movslq %r8d,%r8 //kdim
    testq %r8,%r8
    jz .Ledgek_end
    SET_LDC

    xorq %r12,%r12
.Ledgek_outer:
    ZERO_C_ncol
    PREF_C_ncol %r9
    xorq %r11,%r11
    movq AL,A0
.Ledgek_inner:
    incq %r11
    KERNEL_f 0,0,BYTES_PER_COL,GEMM_UNROLL_N*SIZE
    cmpq %r8,%r11
    jb .Ledgek_inner

    STORECBLK_ncol
    incq %r12
    cmpq $GEMM_LOOP_TIMES_N,%r12
    jb .Ledgek_outer

.Ledgek_end:
    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function unit_gemmblkirregnccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=ndim, r9d=pref_c_offset(in bytes)
.globl unit_gemmblkirregnccc
.type unit_gemmblkirregnccc,@function
unit_gemmblkirregnccc:

    push %r15
    push %r14
    push %r13
    push %r12
    movq %rdx,CS
    movq CS,CL
    movslq %ecx,LDC
    movq %rdi,AL //save ablk_head address
    movslq %r9d,%r9 //pref_c_offset
    movslq %r8d,%r8 //ndim
    SET_LDC

.Ledgen_ncol_outer:
    cmpq $GEMM_UNROLL_N,%r8
    jb .Ledgen_1col_part
    ZERO_C_ncol
    PREF_C_ncol %r9
    xorq %r11,%r11
    movq AL,A0
.Ledgen_ncol_inner:
    incq %r11
    KERNEL_UNROLL_K_edgen
    cmpq $GEMM_LOOP_TIMES_K*GEMM_UNROLL_N,%r11
    jb .Ledgen_ncol_inner

    STORECBLK_ncol
    subq $GEMM_UNROLL_N,%r8
    jmp .Ledgen_ncol_outer

.Ledgen_1col_part:
    testq %r8,%r8
    jz .Ledgen_end

.Ledgen_1col_outer:
    ZERO_C_1col
    PREF_C_1col %r9,CS
    xorq %r11,%r11
    movq AL,A0
.Ledgen_1col_inner:
    incq %r11
    KERNEL_0
    cmpq $GEMM_BLOCK_L1DIM_K,%r11
    jb .Ledgen_1col_inner

    STORECBLK_1col
    decq %r8
    jnz .Ledgen_1col_outer

.Ledgen_end:
    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function unit_gemmblkirregccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8=&{mdim,ndim,kdim}(int64_t par[3]), r9d=pref_c_offset(in bytes)
.globl unit_gemmblkirregccc
.type unit_gemmblkirregccc,@function
unit_gemmblkirregccc:

    push %r15
    push %r14
    push %r13
    push %r12
    movq (%r8),%r13 //mdim(temp)
    movslq %ecx,%r14
    SETMASKm %r13d,%r11d,%r12d //only modifies r11,r12,rcx,k1-k7
    movq %r14,LDC
    SET_LDC //%rcx=LDC_in_bytes(constant)
    movq 16(%r8),%r13 //kdim(constant)
    movq 8(%r8),%r8 //ndim(counter)
    movq %rdi,AL //ablk_head(constant)
    movq %rdx,CS
    movq CS,CL
    movslq %r9d,%r9 //pref_c_offset(constant)
    testq %r13,%r13
    jz .Ledge_end

.Ledge_ncol_outer:
    cmpq $GEMM_UNROLL_N,%r8
    jb .Ledge_1col_part
    ZERO_C_ncol
    PREF_C_ncol %r9
    xorq %r11,%r11
    movq AL,A0
.Ledge_ncol_inner:
    incq %r11
    KERNEL_f 0,0,BYTES_PER_COL,GEMM_UNROLL_N*SIZE
    cmpq %r13,%r11
    jb .Ledge_ncol_inner

    STORECBLK_ncol_edgem
    subq $GEMM_UNROLL_N,%r8
    jmp .Ledge_ncol_outer

.Ledge_1col_part:
    testq %r8,%r8
    jz .Ledge_end

.Ledge_1col_outer:
    ZERO_C_1col
    PREF_C_1col %r9,CS
    xorq %r11,%r11
    movq AL,A0
.Ledge_1col_inner:
    incq %r11
    KERNEL_0
    cmpq %r13,%r11
    jb .Ledge_1col_inner

    STORECBLK_1col_edgem
    decq %r8
    jnz .Ledge_1col_outer

.Ledge_end:
    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function timedelay
.globl timedelay
.type timedelay,@function
timedelay:
    xorq %r11,%r11
.Ltimedelay:
    incq %r11
    vhaddpd %ymm0,%ymm0,%ymm0
    vhaddpd %ymm0,%ymm0,%ymm0
    vhaddpd %ymm0,%ymm0,%ymm0
    vhaddpd %ymm0,%ymm0,%ymm0
    cmpq $500,%r11
    jb .Ltimedelay

    vzeroupper
    retq
    
