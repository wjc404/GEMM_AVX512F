/***************************************************************************
 * Here for utilities common in all dtrsm operations.
 **************************************************************************/

/** do transposition on 4x4 matrix of float64.
 * the 4x4 float64 is held on 4 ymm registers %ymm{yc1-yc4}
 * additional 2 ymm registers %ymm{yt1} and %ymm{yt2} are for temporary use */
.macro TRANSPOSE_4X4_FLOAT64 yc1,yc2,yc3,yc4,yt1,yt2
    vperm2f128 $0x20,%ymm\yc3,%ymm\yc1,%ymm\yt1
    vperm2f128 $0x20,%ymm\yc4,%ymm\yc2,%ymm\yt2
    vperm2f128 $0x31,%ymm\yc4,%ymm\yc2,%ymm\yc4
    vperm2f128 $0x31,%ymm\yc3,%ymm\yc1,%ymm\yc2
    vunpcklpd %ymm\yt2,%ymm\yt1,%ymm\yc1
    vunpcklpd %ymm\yc4,%ymm\yc2,%ymm\yc3
    vunpckhpd %ymm\yc4,%ymm\yc2,%ymm\yc4
    vunpckhpd %ymm\yt2,%ymm\yt1,%ymm\yc2
.endm

/** do transposition on 4x4 matrix of float128.
 * the 4x4 float128 is held on 4 zmm registers %zmm{zc1-zc4}
 * additional 2 zmm registers %zmm{zt1} and %zmm{zt2} are for temporary use */
.macro TRANSPOSE_4X4_FLOAT128 zc1,zc2,zc3,zc4,zt1,zt2
    vshuff64x2 $0x44,%zmm\zc2,%zmm\zc1,%zmm\zt1
    vshuff64x2 $0x44,%zmm\zc4,%zmm\zc3,%zmm\zt2
    vshuff64x2 $0xee,%zmm\zc2,%zmm\zc1,%zmm\zc2
    vshuff64x2 $0xee,%zmm\zc4,%zmm\zc3,%zmm\zc3
    vshuff64x2 $0x88,%zmm\zt2,%zmm\zt1,%zmm\zc1
    vshuff64x2 $0xdd,%zmm\zc3,%zmm\zc2,%zmm\zc4
    vshuff64x2 $0x88,%zmm\zc3,%zmm\zc2,%zmm\zc3
    vshuff64x2 $0xdd,%zmm\zt2,%zmm\zt1,%zmm\zc2
.endm

/** set the value of kmask register, using register "r32" temporarily */
.macro INIT_K8 ctrl,r32,kmask_no
    movl $\ctrl,%\r32
    kmovw %\r32,%k\kmask_no
.endm

/** setting uint64x8 in %%zmm${zmm_no} according to the given
 *  uint8x8 immediate "ctrl", using register "r64" temporarily */
.macro INIT_U64X8 ctrl,r64,zmm_no
    movq $\ctrl,%\r64
    vmovq %\r64,%xmm\zmm_no
    vpmovzxbq %xmm\zmm_no,%zmm\zmm_no
.endm

/**************************************************************************
 * register usage for DTRSM_SOLVE_MxNy_L[N/T] macros:
 * input registers: pointer_a, pointer_b, pointer_c,
      (ldb_bytes), (ldc_bytes)
 * ISA: avx512f for x >= 8 && y >= 2 and also M4N2, avx2 for the rest.
 * temporary registers: r10 for all, ymm0-3 for avx2 routines,
 *    zmm0-7 and k2-k3 for avx512f routines.
 * accumulation registers: ymm4-15 for avx2 routines,
 *    zmm8-31 for avx512f routines.
 *    (must be identical to R solvers and kernels)
 *************************************************************************/
/**************************************************************************
 * Here enter the dtrsm_l_m8n4 zone.
 * Macros here are specially for dtrsm_l[n/t]_m[8/16]n[4/8/12] solvers
 *************************************************************************/

/** the dtrsm_l_storeb_m8n4 operation reorder elements of C,
 *  then store them to row-major matrix block of B.
 *  assignment before this operation for per m2n4 block:
 *    v[0] v[2] v[4] v[6]
 *    v[1] v[3] v[5] v[7]
 *  assignment after this operation for per m2n4 block:
 *    v[0] v[1] v[2] v[3]
 *    v[4] v[5] v[6] v[7] */

.macro DTRSM_L_STOREB_M8N4_INIT r64tmp,zperm_no
    INIT_U64X8 0x0705030106040200,\r64tmp,\zperm_no
.endm

.macro DTRSM_L_STOREB_M8N4_RUN zc1,zc2,zc3,zc4,zperm,zt,off,mem:vararg
    vpermpd %zmm\zc1,%zmm\zperm,%zmm\zt
    vmovupd %zmm\zt,\off(\mem)
    vpermpd %zmm\zc2,%zmm\zperm,%zmm\zt
    vmovupd %zmm\zt,\off+64(\mem)
    vpermpd %zmm\zc3,%zmm\zperm,%zmm\zt
    vmovupd %zmm\zt,\off+128(\mem)
    vpermpd %zmm\zc4,%zmm\zperm,%zmm\zt
    vmovupd %zmm\zt,\off+192(\mem)
.endm

/** the dtrsm_l_storec_m8n4 operation reorder output elements of C,
 *  then store them to column-major matrix block of C.
 *  partition (assignment) before this operation:
 *    v0[0] v0[2] v0[4] v0[6]
 *    v0[1] v0[3] v0[5] v0[7]
 *    v1[0] v1[2] v1[4] v1[6]
 *    v1[1] v1[3] v1[5] v1[7]
 *    v2[0] v2[2] v2[4] v2[6]
 *    v2[1] v2[3] v2[5] v2[7]
 *    v3[0] v3[2] v3[4] v3[6]
 *    v3[1] v3[3] v3[5] v3[7]
 *  partition after this operation:
 *    v0[0] v1[0] v2[0] v3[0]
 *    v0[1] v1[1] v2[1] v3[1]
 *    v0[2] v1[2] v2[2] v3[2]
 *    v0[3] v1[3] v2[3] v3[3]
 *    v0[4] v1[4] v2[4] v3[4]
 *    v0[5] v1[5] v2[5] v3[5]
 *    v0[6] v1[6] v2[6] v3[6]
 *    v0[7] v1[7] v2[7] v3[7] */

.macro DTRSM_L_STOREC_M8N4_RUN zc1,zc2,zc3,zc4,zt1,zt2,ptc,ldc_bytes
    TRANSPOSE_4X4_FLOAT128 \zc1,\zc2,\zc3,\zc4,\zt1,\zt2
    vmovupd %zmm\zc1,(%\ptc)
    vmovupd %zmm\zc2,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vmovupd %zmm\zc3,(%\ptc)
    vmovupd %zmm\zc4,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

.macro DTRSM_L_STOREC_M16N4_RUN zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8,zt1,zt2,ptc,ldc_bytes
    TRANSPOSE_4X4_FLOAT128 \zc1,\zc2,\zc3,\zc4,\zt1,\zt2
    TRANSPOSE_4X4_FLOAT128 \zc5,\zc6,\zc7,\zc8,\zt1,\zt2
    vmovupd %zmm\zc1,(%\ptc)
    vmovupd %zmm\zc5,64(%\ptc)
    vmovupd %zmm\zc2,(%\ptc,%\ldc_bytes,1)
    vmovupd %zmm\zc6,64(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vmovupd %zmm\zc3,(%\ptc)
    vmovupd %zmm\zc7,64(%\ptc)
    vmovupd %zmm\zc4,(%\ptc,%\ldc_bytes,1)
    vmovupd %zmm\zc8,64(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

/** the dtrsm_l_preprocess_m8n4 operation reorder input elements and
 *  add them with a M8N4 matrix block of C
 *  partition (assignment) before this operation:
 *    v0[0] v0[1] v2[0] v2[1]
 *    v1[0] v1[1] v3[0] v3[1]
 *    v0[2] v0[3] v2[2] v2[3]
 *    v1[2] v1[3] v3[2] v3[3]
 *    v0[4] v0[5] v2[4] v2[5]
 *    v1[4] v1[5] v3[4] v3[5]
 *    v0[6] v0[7] v2[6] v2[7]
 *    v1[6] v1[7] v3[7] v3[7]
 *  partition after this operation:
 *    v0[0] v0[2] v0[4] v0[6]
 *    v0[1] v0[3] v0[5] v0[7]
 *    v1[0] v1[2] v1[4] v1[6]
 *    v1[1] v1[3] v1[5] v1[7]
 *    v2[0] v2[2] v2[4] v2[6]
 *    v2[1] v2[3] v2[5] v2[7]
 *    v3[0] v3[2] v3[4] v3[6]
 *    v3[1] v3[3] v3[5] v3[7] */

/** warning: this will change the address in ptc */
.macro DTRSM_L_PREPROCESS_M8N4_RUN zc1,zc2,zc3,zc4,zt1,zt2,ptc,ldc_bytes
    vunpcklpd %zmm\zc2,%zmm\zc1,%zmm\zt1
    vunpckhpd %zmm\zc2,%zmm\zc1,%zmm\zc2
    vunpcklpd %zmm\zc4,%zmm\zc3,%zmm\zt2
    vunpckhpd %zmm\zc4,%zmm\zc3,%zmm\zc4
    vaddpd (%\ptc),%zmm\zt1,%zmm\zc1
    vaddpd (%\ptc,%\ldc_bytes,1),%zmm\zc2,%zmm\zc2
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vaddpd (%\ptc),%zmm\zt2,%zmm\zc3
    vaddpd (%\ptc,%\ldc_bytes,1),%zmm\zc4,%zmm\zc4
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    TRANSPOSE_4X4_FLOAT128 \zc1,\zc2,\zc3,\zc4,\zt1,\zt2
.endm

.macro DTRSM_L_PREPROCESS_M16N4_RUN zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8,zt1,zt2,ptc,ldc_bytes
    vunpcklpd %zmm\zc2,%zmm\zc1,%zmm\zt1
    vunpcklpd %zmm\zc6,%zmm\zc5,%zmm\zt2
    vunpckhpd %zmm\zc2,%zmm\zc1,%zmm\zc2
    vunpckhpd %zmm\zc6,%zmm\zc5,%zmm\zc6
    vaddpd (%\ptc),%zmm\zt1,%zmm\zc1
    vaddpd 64(%\ptc),%zmm\zt2,%zmm\zc5
    vaddpd (%\ptc,%\ldc_bytes,1),%zmm\zc2,%zmm\zc2
    vaddpd 64(%\ptc,%\ldc_bytes,1),%zmm\zc6,%zmm\zc6
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vunpcklpd %zmm\zc4,%zmm\zc3,%zmm\zt1
    vunpcklpd %zmm\zc8,%zmm\zc7,%zmm\zt2
    vunpckhpd %zmm\zc4,%zmm\zc3,%zmm\zc4
    vunpckhpd %zmm\zc8,%zmm\zc7,%zmm\zc8
    vaddpd (%\ptc),%zmm\zt1,%zmm\zc3
    vaddpd 64(%\ptc),%zmm\zt2,%zmm\zc7
    vaddpd (%\ptc,%\ldc_bytes,1),%zmm\zc4,%zmm\zc4
    vaddpd 64(%\ptc,%\ldc_bytes,1),%zmm\zc8,%zmm\zc8
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    TRANSPOSE_4X4_FLOAT128 \zc1,\zc2,\zc3,\zc4,\zt1,\zt2
    TRANSPOSE_4X4_FLOAT128 \zc5,\zc6,\zc7,\zc8,\zt1,\zt2
.endm

/** zc1: c[0][0] c[1][0] c[0][1] c[1][1] c[0][2] c[1][2] c[0][3] c[1][3]
 *  zc2: c[2][0] c[3][0] c[2][1] c[3][1] c[2][2] c[3][2] c[2][3] c[3][3]
 *  zc3: c[4][0] c[5][0] c[4][1] c[5][1] c[4][2] c[5][2] c[4][3] c[5][3]
 *  zc4: c[6][0] c[7][0] c[6][1] c[7][1] c[6][2] c[7][2] c[6][3] c[7][3]
 *  za1: a[0][k] a[1][k] a[0][k] a[1][k] a[0][k] a[1][k] a[0][k] a[1][k]
 *  za2: a[2][k] a[3][k] a[2][k] a[3][k] a[2][k] a[3][k] a[2][k] a[3][k]
 *  za3: a[4][k] a[5][k] a[4][k] a[5][k] a[4][k] a[5][k] a[4][k] a[5][k]
 *  za4: a[6][k] a[7][k] a[6][k] a[7][k] a[6][k] a[7][k] a[6][k] a[7][k]
 *  zcoef: the output updated values of solved C for updating outer elements
 *  off: the starting offset of the current row-major m8n4 block of matrix B
 *  kup: the kmask activating upper c elements(id = 0,2,4,6), with ctrl = 0x55
 *  kdown: the kmask activating lower c elements(id = 1,3,5,7), with ctrl = 0xaa */

.macro SOLVE_LN_M2N4_DOWN_IN_M8N4 zc1,za1,zcoef,kup,kdown
    vmulpd %zmm\zc1,%zmm\za1,%zmm\zc1{%k\kdown}
    vunpckhpd %zmm\zc1,%zmm\zc1,%zmm\zcoef
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1{%k\kup}
.endm

.macro SOLVE_LN_M2N4_UP_IN_M8N4 zc1,za1,zcoef,kup
    vmulpd %zmm\zc1,%zmm\za1,%zmm\zc1{%k\kup}
    vunpcklpd %zmm\zc1,%zmm\zc1,%zmm\zcoef
.endm

.macro SOLVE_LT_M2N4_UP_IN_M8N4 zc1,za1,zcoef,kup,kdown
    vmulpd %zmm\zc1,%zmm\za1,%zmm\zc1{%k\kup}
    vunpcklpd %zmm\zc1,%zmm\zc1,%zmm\zcoef
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1{%k\kdown}
.endm

.macro SOLVE_LT_M2N4_DOWN_IN_M8N4 zc1,za1,zcoef,kdown
    vmulpd %zmm\zc1,%zmm\za1,%zmm\zc1{%k\kdown}
    vunpckhpd %zmm\zc1,%zmm\zc1,%zmm\zcoef
.endm

.macro SOLVE_LN_M8N4_mpos0 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M2N4_UP_IN_M8N4 \zc1,\za1,\zcoef,\kup
.endm

.macro SOLVE_LN_M8N4_mpos1 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M2N4_DOWN_IN_M8N4 \zc1,\za1,\zcoef,\kup,\kdown
.endm

.macro SOLVE_LN_M8N4_mpos2 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M8N4_mpos0 \zc2,\zc2,\zc3,\zc4,\za2,\za2,\za3,\za4,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
.endm

.macro SOLVE_LN_M8N4_mpos3 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M8N4_mpos1 \zc2,\zc2,\zc3,\zc4,\za2,\za2,\za3,\za4,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
.endm

.macro SOLVE_LN_M8N4_mpos4 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M8N4_mpos2 \zc2,\zc3,\zc3,\zc4,\za2,\za3,\za3,\za4,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
.endm

.macro SOLVE_LN_M8N4_mpos5 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M8N4_mpos3 \zc2,\zc3,\zc3,\zc4,\za2,\za3,\za3,\za4,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
.endm

.macro SOLVE_LN_M8N4_mpos6 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M8N4_mpos4 \zc2,\zc3,\zc4,\zc4,\za2,\za3,\za4,\za4,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
.endm

.macro SOLVE_LN_M8N4_mpos7 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LN_M8N4_mpos5 \zc2,\zc3,\zc4,\zc4,\za2,\za3,\za4,\za4,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
.endm

.macro SOLVE_LT_M8N4_mpos7 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M2N4_DOWN_IN_M8N4 \zc4,\za4,\zcoef,\kdown
.endm

.macro SOLVE_LT_M8N4_mpos6 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M2N4_UP_IN_M8N4 \zc4,\za4,\zcoef,\kup,\kdown
.endm

.macro SOLVE_LT_M8N4_mpos5 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M8N4_mpos7 \zc1,\zc2,\zc3,\zc3,\za1,\za2,\za3,\za3,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro SOLVE_LT_M8N4_mpos4 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M8N4_mpos6 \zc1,\zc2,\zc3,\zc3,\za1,\za2,\za3,\za3,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro SOLVE_LT_M8N4_mpos3 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M8N4_mpos5 \zc1,\zc2,\zc2,\zc3,\za1,\za2,\za2,\za3,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro SOLVE_LT_M8N4_mpos2 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M8N4_mpos4 \zc1,\zc2,\zc2,\zc3,\za1,\za2,\za2,\za3,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro SOLVE_LT_M8N4_mpos1 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M8N4_mpos3 \zc1,\zc1,\zc2,\zc3,\za1,\za1,\za2,\za3,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro SOLVE_LT_M8N4_mpos0 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef,kup,kdown
    SOLVE_LT_M8N4_mpos2 \zc1,\zc1,\zc2,\zc3,\za1,\za1,\za2,\za3,\zcoef,\kup,\kdown
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro UPDATE_L_M8N4 zc1,zc2,zc3,zc4,za1,za2,za3,za4,zcoef
    vfnmadd231pd %zmm\za1,%zmm\zcoef,%zmm\zc1
    vfnmadd231pd %zmm\za2,%zmm\zcoef,%zmm\zc2
    vfnmadd231pd %zmm\za3,%zmm\zcoef,%zmm\zc3
    vfnmadd231pd %zmm\za4,%zmm\zcoef,%zmm\zc4
.endm

.macro LOAD_L_A_M8_IN_M8N4 za1,za2,za3,za4,pta,base_off
    vbroadcastf32x4 \base_off(%\pta),%zmm\za1
    vbroadcastf32x4 \base_off+16(%\pta),%zmm\za2
    vbroadcastf32x4 \base_off+32(%\pta),%zmm\za3
    vbroadcastf32x4 \base_off+48(%\pta),%zmm\za4
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos7 za1,za2,za3,za4,pta,base_off
    vbroadcastsd \base_off+56(%\pta),%zmm\za4
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos6 za1,za2,za3,za4,pta,base_off
    vbroadcastf32x4 \base_off+48(%\pta),%zmm\za4
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos5 za1,za2,za3,za4,pta,base_off
    vbroadcastsd \base_off+40(%\pta),%zmm\za3
    LOAD_LT_A_M8_IN_M8N4_mpos6 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos4 za1,za2,za3,za4,pta,base_off
    vbroadcastf32x4 \base_off+32(%\pta),%zmm\za3
    LOAD_LT_A_M8_IN_M8N4_mpos6 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos3 za1,za2,za3,za4,pta,base_off
    vbroadcastsd \base_off+24(%\pta),%zmm\za2
    LOAD_LT_A_M8_IN_M8N4_mpos4 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos2 za1,za2,za3,za4,pta,base_off
    vbroadcastf32x4 \base_off+16(%\pta),%zmm\za2
    LOAD_LT_A_M8_IN_M8N4_mpos4 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos1 za1,za2,za3,za4,pta,base_off
    vbroadcastsd \base_off+8(%\pta),%zmm\za1
    LOAD_LT_A_M8_IN_M8N4_mpos2 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

.macro LOAD_LT_A_M8_IN_M8N4_mpos0 za1,za2,za3,za4,pta,base_off
    LOAD_L_A_M8_IN_M8N4 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos0 za1,za2,za3,za4,pta,base_off
    vbroadcastsd \base_off(%\pta),%zmm\za1
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos1 za1,za2,za3,za4,pta,base_off
    vbroadcastf32x4 \base_off(%\pta),%zmm\za1
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos2 za1,za2,za3,za4,pta,base_off
    LOAD_LN_A_M8_IN_M8N4_mpos1 \za1,\za2,\za3,\za4,\pta,\base_off
    vbroadcastsd \base_off+16(%\pta),%zmm\za2
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos3 za1,za2,za3,za4,pta,base_off
    LOAD_LN_A_M8_IN_M8N4_mpos1 \za1,\za2,\za3,\za4,\pta,\base_off
    vbroadcastf32x4 \base_off+16(%\pta),%zmm\za2
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos4 za1,za2,za3,za4,pta,base_off
    LOAD_LN_A_M8_IN_M8N4_mpos3 \za1,\za2,\za3,\za4,\pta,\base_off
    vbroadcastsd \base_off+32(%\pta),%zmm\za3
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos5 za1,za2,za3,za4,pta,base_off
    LOAD_LN_A_M8_IN_M8N4_mpos3 \za1,\za2,\za3,\za4,\pta,\base_off
    vbroadcastf32x4 \base_off+32(%\pta),%zmm\za3
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos6 za1,za2,za3,za4,pta,base_off
    LOAD_LN_A_M8_IN_M8N4_mpos5 \za1,\za2,\za3,\za4,\pta,\base_off
    vbroadcastsd \base_off+48(%\pta),%zmm\za4
.endm

.macro LOAD_LN_A_M8_IN_M8N4_mpos7 za1,za2,za3,za4,pta,base_off
    LOAD_L_A_M8_IN_M8N4 \za1,\za2,\za3,\za4,\pta,\base_off
.endm

/*************************************************************************
 * warning (in dtrsm_l_m8n4 zone):
 * The macros below have predefined rules of register usage where care
 * should be taken (or data corruption would occur)
*************************************************************************/

/** zmm8-zmm31 for accumulators, should be preserved */
/** r10 is used for temporary variable */

.macro SOLVE_L_M8N4_RUN1 mode,mpos,pta,offa,zc1,zc2,zc3,zc4
    LOAD_L\mode\()_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa
    SOLVE_L\mode\()_M8N4_mpos\mpos \zc1,\zc2,\zc3,\zc4,4,5,6,7,3,2,3
.endm

.macro SOLVE_L_M8N8_RUN1 mode,mpos,pta,offa,zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8
    SOLVE_L_M8N4_RUN1 \mode,\mpos,\pta,\offa,\zc1,\zc2,\zc3,\zc4
    SOLVE_L\mode\()_M8N4_mpos\mpos \zc5,\zc6,\zc7,\zc8,4,5,6,7,3,2,3
.endm

.macro SOLVE_L_M8N12_RUN1 mode,mpos,pta,offa,zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8,zc9,zca,zcb,zcc
    SOLVE_L_M8N8_RUN1 \mode,\mpos,\pta,\offa,\zc1,\zc2,\zc3,\zc4,\zc5,\zc6,\zc7,\zc8
    SOLVE_L\mode\()_M8N4_mpos\mpos \zc9,\zca,\zcb,\zcc,4,5,6,7,3,2,3
.endm

.macro SOLVE_L_M8_N4X_RUN1 mode,ndim,mpos,pta,offa,zc:vararg
    SOLVE_L_M8N\ndim\()_RUN1 \mode,\mpos,\pta,\offa,\zc
.endm

.macro SOLVE_LN_M16N4_RUN1 mpos,pta,offa
    LOAD_LN_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa+64
    SOLVE_LN_M8N4_mpos\mpos 12,13,14,15,4,5,6,7,3,2,3
    LOAD_L_A_M8_IN_M8N4 4,5,6,7,\pta,\offa
    UPDATE_L_M8N4 8,9,10,11,4,5,6,7,3
.endm

.macro SOLVE_LN_M16N8_RUN1 mpos,pta,offa
    LOAD_LN_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa+64
    SOLVE_LN_M8N4_mpos\mpos 12,13,14,15,4,5,6,7,3,2,3
    SOLVE_LN_M8N4_mpos\mpos 20,21,22,23,4,5,6,7,2,2,3
    LOAD_L_A_M8_IN_M8N4 4,5,6,7,\pta,\offa
    UPDATE_L_M8N4 8,9,10,11,4,5,6,7,3
    UPDATE_L_M8N4 16,17,18,19,4,5,6,7,2
.endm

.macro SOLVE_LN_M16N12_RUN1 mpos,pta,offa
    LOAD_LN_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa+64
    SOLVE_LN_M8N4_mpos\mpos 12,13,14,15,4,5,6,7,3,2,3
    SOLVE_LN_M8N4_mpos\mpos 20,21,22,23,4,5,6,7,2,2,3
    SOLVE_LN_M8N4_mpos\mpos 28,29,30,31,4,5,6,7,1,2,3
    LOAD_L_A_M8_IN_M8N4 4,5,6,7,\pta,\offa
    UPDATE_L_M8N4 8,9,10,11,4,5,6,7,3
    UPDATE_L_M8N4 16,17,18,19,4,5,6,7,2
    UPDATE_L_M8N4 24,25,26,27,4,5,6,7,1
.endm

.macro SOLVE_LT_M16N4_RUN1 mpos,pta,offa
    LOAD_LT_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa
    SOLVE_LT_M8N4_mpos\mpos 8,9,10,11,4,5,6,7,3,2,3
    LOAD_L_A_M8_IN_M8N4 4,5,6,7,\pta,\offa+64
    UPDATE_L_M8N4 12,13,14,15,4,5,6,7,3
.endm

.macro SOLVE_LT_M16N8_RUN1 mpos,pta,offa
    LOAD_LT_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa
    SOLVE_LT_M8N4_mpos\mpos 8,9,10,11,4,5,6,7,3,2,3
    SOLVE_LT_M8N4_mpos\mpos 16,17,18,19,4,5,6,7,2,2,3
    LOAD_L_A_M8_IN_M8N4 4,5,6,7,\pta,\offa+64
    UPDATE_L_M8N4 12,13,14,15,4,5,6,7,3
    UPDATE_L_M8N4 20,21,22,23,4,5,6,7,2
.endm

.macro SOLVE_LT_M16N12_RUN1 mpos,pta,offa
    LOAD_LT_A_M8_IN_M8N4_mpos\mpos 4,5,6,7,\pta,\offa
    SOLVE_LT_M8N4_mpos\mpos 8,9,10,11,4,5,6,7,3,2,3
    SOLVE_LT_M8N4_mpos\mpos 16,17,18,19,4,5,6,7,2,2,3
    SOLVE_LT_M8N4_mpos\mpos 24,25,26,27,4,5,6,7,1,2,3
    LOAD_L_A_M8_IN_M8N4 4,5,6,7,\pta,\offa+64
    UPDATE_L_M8N4 12,13,14,15,4,5,6,7,3
    UPDATE_L_M8N4 20,21,22,23,4,5,6,7,2
    UPDATE_L_M8N4 28,29,30,31,4,5,6,7,1
.endm

.macro SOLVE_L_M16_N4X_RUN1 mode,ndim,mpos,pta,offa,dummy:vararg
    SOLVE_L\mode\()_M16N\ndim\()_RUN1 \mpos,\pta,\offa
.endm

.macro SOLVE_LN_M8X_N4X_WRAPPER mdim,ndim,mld,pta,offa,zc:vararg
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,7,\pta,\offa-\mld*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,6,\pta,\offa-\mld*16,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,5,\pta,\offa-\mld*24,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,4,\pta,\offa-\mld*32,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,3,\pta,\offa-\mld*40,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,2,\pta,\offa-\mld*48,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,1,\pta,\offa-\mld*56,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 N,\ndim,0,\pta,\offa-\mld*64,\zc
.endm

.macro SOLVE_LT_M8X_N4X_WRAPPER mdim,ndim,mld,pta,offa,zc:vararg
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,0,\pta,\offa+\mld*8-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,1,\pta,\offa+\mld*16-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,2,\pta,\offa+\mld*24-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,3,\pta,\offa+\mld*32-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,4,\pta,\offa+\mld*40-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,5,\pta,\offa+\mld*48-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,6,\pta,\offa+\mld*56-\mdim*8,\zc
    SOLVE_L_M\mdim\()_N4X_RUN1 T,\ndim,7,\pta,\offa+\mld*64-\mdim*8,\zc
.endm

.macro DTRSM_L_STORE_M8N4BLOCK_INIT
    DTRSM_L_STOREB_M8N4_INIT r10,7
.endm

/** warning: this operation will update the pointer ptc */
.macro DTRSM_L_STORE_M8N4BLOCK_RUN zc1,zc2,zc3,zc4,ptc,ldc_bytes,offb,memb:vararg
    DTRSM_L_STOREB_M8N4_RUN \zc1,\zc2,\zc3,\zc4,7,6,\offb,\memb
    DTRSM_L_STOREC_M8N4_RUN \zc1,\zc2,\zc3,\zc4,6,5,\ptc,\ldc_bytes
.endm

/** warning: this operation will update the pointer ptc */
.macro DTRSM_L_STORE_M16N4BLOCK_RUN zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8,ptc,ldc_bytes,offb,memb:vararg
    DTRSM_L_STOREB_M8N4_RUN \zc1,\zc2,\zc3,\zc4,7,6,\offb,\memb
    DTRSM_L_STOREB_M8N4_RUN \zc5,\zc6,\zc7,\zc8,7,6,\offb+256,\memb
    DTRSM_L_STOREC_M16N4_RUN \zc1,\zc2,\zc3,\zc4,\zc5,\zc6,\zc7,\zc8,6,5,\ptc,\ldc_bytes
.endm

.macro DTRSM_L_SOLVE_M8N4BLOCK_INIT kup,kdown
    INIT_K8 0x55,r10d,\kup
    INIT_K8 0xaa,r10d,\kdown
.endm

/*   Here are the characteristics for the solver macros DTRSM_SOLVE_MmNm_L[N/T]
 * with m = 8,16 and n = 4,8,12.
 *   All parameters to these macros are registers holding pointer/stride values,
 * which will not be modified in the execution. "pt[a/b/c]" is for pointer to
 * array [A/B/C].
 *   These macros will use registers zmm1-zmm7, k2-k3 and r10 temporarily.
 *   Additional input elements (negative accumulated results from GEMM KERNEL)
 * are provided through registers zmm8-zmmX (X is the minimum number for the
 * register group to hold all elements).
 *   DTRSM_SOLVE_MmNn_LN assumes that all input pointers POINT TO THE END of
 * the array or the lowest matrix in a combined input, while
 * DTRSM_SOLVE_MmNn_LT assumes those pointers POINT TO THE BEGINNING.
 *   For these solver macros, matrix A is a constant column-major RmCm array,
 * array B is a combination of n/4 row-major RmC4 matrix blocks with a stride
 * of 4*LDB between nearby blocks, matrix C is a column-major RmCn matrix with
 * leading dimention = LDC. */

.macro DTRSM_SOLVE_M8N4_LN pta,ptb,ptc,ldc_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    leaq -64(%\ptc),%r10
    DTRSM_L_PREPROCESS_M8N4_RUN 8,9,10,11,6,7,r10,\ldc_bytes
    SOLVE_LN_M8X_N4X_WRAPPER 8,4,8,\pta,0,8,9,10,11
    DTRSM_L_STORE_M8N4BLOCK_INIT
    leaq -64(%\ptc),%r10
    DTRSM_L_STORE_M8N4BLOCK_RUN 8,9,10,11,r10,\ldc_bytes,-256,%\ptb
.endm

.macro DTRSM_SOLVE_M8N4_LT pta,ptb,ptc,ldc_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M8N4_RUN 8,9,10,11,6,7,r10,\ldc_bytes
    SOLVE_LT_M8X_N4X_WRAPPER 8,4,8,\pta,0,8,9,10,11
    DTRSM_L_STORE_M8N4BLOCK_INIT
    movq %\ptc,%r10
    DTRSM_L_STORE_M8N4BLOCK_RUN 8,9,10,11,r10,\ldc_bytes,0,%\ptb
.endm

.macro DTRSM_SOLVE_M8N8_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    leaq -64(%\ptc),%r10
    DTRSM_L_PREPROCESS_M8N4_RUN 8,9,10,11,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N4_RUN 12,13,14,15,6,7,r10,\ldc_bytes
    SOLVE_LN_M8X_N4X_WRAPPER 8,8,8,\pta,0,8,9,10,11,12,13,14,15
    DTRSM_L_STORE_M8N4BLOCK_INIT
    leaq -64(%\ptc),%r10
    DTRSM_L_STORE_M8N4BLOCK_RUN 8,9,10,11,r10,\ldc_bytes,-256,%\ptb
    DTRSM_L_STORE_M8N4BLOCK_RUN 12,13,14,15,r10,\ldc_bytes,-256,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M8N8_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M8N4_RUN 8,9,10,11,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N4_RUN 12,13,14,15,6,7,r10,\ldc_bytes
    SOLVE_LT_M8X_N4X_WRAPPER 8,8,8,\pta,0,8,9,10,11,12,13,14,15
    DTRSM_L_STORE_M8N4BLOCK_INIT
    movq %\ptc,%r10
    DTRSM_L_STORE_M8N4BLOCK_RUN 8,9,10,11,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M8N4BLOCK_RUN 12,13,14,15,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M8N12_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    leaq -64(%\ptc),%r10
    DTRSM_L_PREPROCESS_M8N4_RUN 8,9,10,11,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N4_RUN 12,13,14,15,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N4_RUN 16,17,18,19,6,7,r10,\ldc_bytes
    SOLVE_LN_M8X_N4X_WRAPPER 8,12,8,\pta,0,8,9,10,11,12,13,14,15,16,17,18,19
    DTRSM_L_STORE_M8N4BLOCK_INIT
    leaq -64(%\ptc),%r10
    DTRSM_L_STORE_M8N4BLOCK_RUN 8,9,10,11,r10,\ldc_bytes,-256,%\ptb
    DTRSM_L_STORE_M8N4BLOCK_RUN 12,13,14,15,r10,\ldc_bytes,-256,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M8N4BLOCK_RUN 16,17,18,19,r10,\ldc_bytes,-256,%\ptb,%\ldb_bytes,8
.endm

.macro DTRSM_SOLVE_M8N12_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M8N4_RUN 8,9,10,11,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N4_RUN 12,13,14,15,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N4_RUN 16,17,18,19,6,7,r10,\ldc_bytes
    SOLVE_LT_M8X_N4X_WRAPPER 8,12,8,\pta,0,8,9,10,11,12,13,14,15,16,17,18,19
    DTRSM_L_STORE_M8N4BLOCK_INIT
    movq %\ptc,%r10
    DTRSM_L_STORE_M8N4BLOCK_RUN 8,9,10,11,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M8N4BLOCK_RUN 12,13,14,15,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M8N4BLOCK_RUN 16,17,18,19,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,8
.endm

.macro DTRSM_SOLVE_M16N4_LN pta,ptb,ptc,ldc_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    leaq -128(%\ptc),%r10
    DTRSM_L_PREPROCESS_M16N4_RUN 8,9,10,11,12,13,14,15,6,7,r10,\ldc_bytes
    SOLVE_LN_M8X_N4X_WRAPPER 16,4,16,\pta,0,dummy
    SOLVE_LN_M8X_N4X_WRAPPER 8,4,16,\pta,-1024,8,9,10,11
    DTRSM_L_STORE_M8N4BLOCK_INIT
    leaq -128(%\ptc),%r10
    DTRSM_L_STORE_M16N4BLOCK_RUN 8,9,10,11,12,13,14,15,r10,\ldc_bytes,-512,%\ptb
.endm

.macro DTRSM_SOLVE_M16N4_LT pta,ptb,ptc,ldc_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M16N4_RUN 8,9,10,11,12,13,14,15,6,7,r10,\ldc_bytes
    SOLVE_LT_M8X_N4X_WRAPPER 16,4,16,\pta,0,dummy
    SOLVE_LT_M8X_N4X_WRAPPER 8,4,16,\pta,1024,12,13,14,15
    DTRSM_L_STORE_M8N4BLOCK_INIT
    movq %\ptc,%r10
    DTRSM_L_STORE_M16N4BLOCK_RUN 8,9,10,11,12,13,14,15,r10,\ldc_bytes,0,%\ptb
.endm

.macro DTRSM_SOLVE_M16N8_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    leaq -128(%\ptc),%r10
    DTRSM_L_PREPROCESS_M16N4_RUN 8,9,10,11,12,13,14,15,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M16N4_RUN 16,17,18,19,20,21,22,23,6,7,r10,\ldc_bytes
    SOLVE_LN_M8X_N4X_WRAPPER 16,8,16,\pta,0,dummy
    SOLVE_LN_M8X_N4X_WRAPPER 8,8,16,\pta,-1024,8,9,10,11,16,17,18,19
    DTRSM_L_STORE_M8N4BLOCK_INIT
    leaq -128(%\ptc),%r10
    DTRSM_L_STORE_M16N4BLOCK_RUN 8,9,10,11,12,13,14,15,r10,\ldc_bytes,-512,%\ptb
    DTRSM_L_STORE_M16N4BLOCK_RUN 16,17,18,19,20,21,22,23,r10,\ldc_bytes,-512,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M16N8_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M16N4_RUN 8,9,10,11,12,13,14,15,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M16N4_RUN 16,17,18,19,20,21,22,23,6,7,r10,\ldc_bytes
    SOLVE_LT_M8X_N4X_WRAPPER 16,8,16,\pta,0,dummy
    SOLVE_LT_M8X_N4X_WRAPPER 8,8,16,\pta,1024,12,13,14,15,20,21,22,23
    DTRSM_L_STORE_M8N4BLOCK_INIT
    movq %\ptc,%r10
    DTRSM_L_STORE_M16N4BLOCK_RUN 8,9,10,11,12,13,14,15,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M16N4BLOCK_RUN 16,17,18,19,20,21,22,23,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M16N12_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    leaq -128(%\ptc),%r10
    DTRSM_L_PREPROCESS_M16N4_RUN 8,9,10,11,12,13,14,15,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M16N4_RUN 16,17,18,19,20,21,22,23,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M16N4_RUN 24,25,26,27,28,29,30,31,6,7,r10,\ldc_bytes
    SOLVE_LN_M8X_N4X_WRAPPER 16,12,16,\pta,0,dummy
    SOLVE_LN_M8X_N4X_WRAPPER 8,12,16,\pta,-1024,8,9,10,11,16,17,18,19,24,25,26,27
    DTRSM_L_STORE_M8N4BLOCK_INIT
    leaq -128(%\ptc),%r10
    DTRSM_L_STORE_M16N4BLOCK_RUN 8,9,10,11,12,13,14,15,r10,\ldc_bytes,-512,%\ptb
    DTRSM_L_STORE_M16N4BLOCK_RUN 16,17,18,19,20,21,22,23,r10,\ldc_bytes,-512,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M16N4BLOCK_RUN 24,25,26,27,28,29,30,31,r10,\ldc_bytes,-512,%\ptb,%\ldb_bytes,8
.endm

.macro DTRSM_SOLVE_M16N12_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_L_SOLVE_M8N4BLOCK_INIT 2,3
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M16N4_RUN 8,9,10,11,12,13,14,15,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M16N4_RUN 16,17,18,19,20,21,22,23,6,7,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M16N4_RUN 24,25,26,27,28,29,30,31,6,7,r10,\ldc_bytes
    SOLVE_LT_M8X_N4X_WRAPPER 16,12,16,\pta,0,dummy
    SOLVE_LT_M8X_N4X_WRAPPER 8,12,16,\pta,1024,12,13,14,15,20,21,22,23,28,29,30,31
    DTRSM_L_STORE_M8N4BLOCK_INIT
    movq %\ptc,%r10
    DTRSM_L_STORE_M16N4BLOCK_RUN 8,9,10,11,12,13,14,15,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M16N4BLOCK_RUN 16,17,18,19,20,21,22,23,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M16N4BLOCK_RUN 24,25,26,27,28,29,30,31,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,8
.endm

/**************************************************************************
 * End of dtrsm_l_m8n4 zone
 * Here enter the dtrsm_l_m4n4 zone.
 * Macros here are specially for dtrsm_l[n/t]_m[1/2/4]n[4/8/12] solvers
 *************************************************************************/

/** warning: the register ptc will be updated */
.macro DTRSM_L_PREPROCESS_M4N4_RUN yc1,yc2,yc3,yc4,yt1,yt2,ptc,ldc_bytes
    vunpcklpd %ymm\yc2,%ymm\yc1,%ymm\yt1
    vunpckhpd %ymm\yc2,%ymm\yc1,%ymm\yc2
    vmovupd (%\ptc),%xmm\yt2
    vinsertf128 $1,(%\ptc,%\ldc_bytes,2),%ymm\yt2,%ymm\yt2
    vaddpd %ymm\yt2,%ymm\yt1,%ymm\yc1
    vunpcklpd %ymm\yc4,%ymm\yc3,%ymm\yt1
    vunpckhpd %ymm\yc4,%ymm\yc3,%ymm\yc4
    vmovupd 16(%\ptc),%xmm\yt2
    vinsertf128 $1,16(%\ptc,%\ldc_bytes,2),%ymm\yt2,%ymm\yt2
    vaddpd %ymm\yt2,%ymm\yt1,%ymm\yc3
    addq %\ldc_bytes,%\ptc
    vmovupd (%\ptc),%xmm\yt2
    vinsertf128 $1,(%\ptc,%\ldc_bytes,2),%ymm\yt2,%ymm\yt2
    vaddpd %ymm\yt2,%ymm\yc2,%ymm\yt1
    vunpckhpd %ymm\yt1,%ymm\yc1,%ymm\yc2
    vunpcklpd %ymm\yt1,%ymm\yc1,%ymm\yc1
    vmovupd 16(%\ptc),%xmm\yt2
    vinsertf128 $1,16(%\ptc,%\ldc_bytes,2),%ymm\yt2,%ymm\yt2
    vaddpd %ymm\yt2,%ymm\yc4,%ymm\yt1
    vunpckhpd %ymm\yt1,%ymm\yc3,%ymm\yc4
    vunpcklpd %ymm\yt1,%ymm\yc3,%ymm\yc3
    addq %\ldc_bytes,%\ptc
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

/** warning: the register ptc will be updated */
.macro DTRSM_L_PREPROCESS_M2N4_RUN yc1,yc2,yt1,yt2,ptc,ldc_bytes
    vunpcklpd %ymm\yc2,%ymm\yc1,%ymm\yt1
    vunpckhpd %ymm\yc2,%ymm\yc1,%ymm\yc2
    vmovupd (%\ptc),%xmm\yt2
    vinsertf128 $1,(%\ptc,%\ldc_bytes,2),%ymm\yt2,%ymm\yt2
    addq %\ldc_bytes,%\ptc
    vaddpd %ymm\yt2,%ymm\yt1,%ymm\yt1
    vmovupd (%\ptc),%xmm\yt2
    vinsertf128 $1,(%\ptc,%\ldc_bytes,2),%ymm\yt2,%ymm\yt2
    addq %\ldc_bytes,%\ptc
    vaddpd %ymm\yt2,%ymm\yc2,%ymm\yt2
    vunpcklpd %ymm\yt2,%ymm\yt1,%ymm\yc1
    vunpckhpd %ymm\yt2,%ymm\yt1,%ymm\yc2
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

/** warning: the register ptc will be updated */
.macro DTRSM_L_PREPROCESS_M1N4_RUN yc1,yt1,yt2,ptc,ldc_bytes
    vmovsd (%\ptc),%xmm\yt1
    vmovhpd (%\ptc,%\ldc_bytes,1),%xmm\yt1,%xmm\yt1
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vmovsd (%\ptc),%xmm\yt2
    vmovhpd (%\ptc,%\ldc_bytes,1),%xmm\yt2,%xmm\yt2
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vinsertf128 $1,%xmm\yt2,%ymm\yt1,%ymm\yt1
    vaddpd %ymm\yt1,%ymm\yc1,%ymm\yc1
.endm

/** warning: the register ptc will be updated */
.macro DTRSM_L_STORE_M4N4_RUN yc1,yc2,yc3,yc4,yt1,yt2,ptc,ldc_bytes,offb,memb:vararg
    vmovupd %ymm\yc1,\offb(\memb)
    vmovupd %ymm\yc2,\offb+32(\memb)
    vmovupd %ymm\yc3,\offb+64(\memb)
    vmovupd %ymm\yc4,\offb+96(\memb)
    TRANSPOSE_4X4_FLOAT64 \yc1,\yc2,\yc3,\yc4,\yt1,\yt2
    vmovupd %ymm\yc1,(%\ptc)
    vmovupd %ymm\yc2,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vmovupd %ymm\yc3,(%\ptc)
    vmovupd %ymm\yc4,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

/** warning: the register ptc will be updated */
.macro DTRSM_L_STORE_M2N4_RUN yc1,yc2,yt1,ptc,ldc_bytes,offb,memb:vararg
    vmovupd %ymm\yc1,\offb(\memb)
    vmovupd %ymm\yc2,\offb+32(\memb)
    vunpcklpd %ymm\yc2,%ymm\yc1,%ymm\yt1
    vunpckhpd %ymm\yc2,%ymm\yc1,%ymm\yc2
    vmovupd %xmm\yt1,(%\ptc)
    vmovupd %xmm\yc2,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vextractf128 $1,%ymm\yt1,(%\ptc)
    vextractf128 $1,%ymm\yc2,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

/** warning: the register ptc will be updated */
.macro DTRSM_L_STORE_M1N4_RUN yc1,ptc,ldc_bytes,offb,memb:vararg
    vmovupd %ymm\yc1,\offb(\memb)
    vmovsd %xmm\yc1,(%\ptc)
    vmovhpd %xmm\yc1,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
    vextractf128 $1,%ymm\yc1,%xmm\yc1
    vmovsd %xmm\yc1,(%\ptc)
    vmovhpd %xmm\yc1,(%\ptc,%\ldc_bytes,1)
    leaq (%\ptc,%\ldc_bytes,2),%\ptc
.endm

.macro SOLVE_L_M1N4UNIT yc1,yt1,pta,a_base_off
    vbroadcastsd \a_base_off(%\pta),%ymm\yt1
    vmulpd %ymm\yt1,%ymm\yc1,%ymm\yc1
.endm

.macro SOLVE_L_M1N8UNIT yc1,yc2,yt1,pta,a_base_off
    SOLVE_L_M1N4UNIT \yc1,\yt1,\pta,\a_base_off
    vmulpd %ymm\yt1,%ymm\yc2,%ymm\yc2
.endm

.macro SOLVE_L_M1N12UNIT yc1,yc2,yc3,yt1,pta,a_base_off
    SOLVE_L_M1N8UNIT \yc1,\yc2,\yt1,\pta,\a_base_off
    vmulpd %ymm\yt1,%ymm\yc3,%ymm\yc3
.endm

.macro SOLVE_L_M2N4UNIT yc1,yc2,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M1N4UNIT \yc1,\yt1,\pta,\a_base_off
    vbroadcastsd \a_base_off+(\a_inc_off)(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc2
.endm

.macro SOLVE_L_M2N8UNIT yc1,yc2,yc3,yc4,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M1N8UNIT \yc1,\yc3,\yt1,\pta,\a_base_off
    vbroadcastsd \a_base_off+(\a_inc_off)(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc2
    vfnmadd231pd %ymm\yt1,%ymm\yc3,%ymm\yc4
.endm

.macro SOLVE_L_M2N12UNIT yc1,yc2,yc3,yc4,yc5,yc6,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M1N12UNIT \yc1,\yc3,\yc5,\yt1,\pta,\a_base_off
    vbroadcastsd \a_base_off+(\a_inc_off)(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc2
    vfnmadd231pd %ymm\yt1,%ymm\yc3,%ymm\yc4
    vfnmadd231pd %ymm\yt1,%ymm\yc5,%ymm\yc6
.endm

.macro SOLVE_L_M3N4UNIT yc1,yc2,yc3,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M2N4UNIT \yc1,\yc2,\yt1,\pta,\a_base_off,\a_inc_off
    vbroadcastsd \a_base_off+(\a_inc_off)*2(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc3
.endm

.macro SOLVE_L_M3N8UNIT yc1,yc2,yc3,yc4,yc5,yc6,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M2N8UNIT \yc1,\yc2,\yc4,\yc5,\yt1,\pta,\a_base_off,\a_inc_off
    vbroadcastsd \a_base_off+(\a_inc_off)*2(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc3
    vfnmadd231pd %ymm\yt1,%ymm\yc4,%ymm\yc6
.endm

.macro SOLVE_L_M3N12UNIT yc1,yc2,yc3,yc4,yc5,yc6,yc7,yc8,yc9,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M2N12UNIT \yc1,\yc2,\yc4,\yc5,\yc7,\yc8,\yt1,\pta,\a_base_off,\a_inc_off
    vbroadcastsd \a_base_off+(\a_inc_off)*2(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc3
    vfnmadd231pd %ymm\yt1,%ymm\yc4,%ymm\yc6
    vfnmadd231pd %ymm\yt1,%ymm\yc7,%ymm\yc9
.endm

.macro SOLVE_L_M4N4UNIT yc1,yc2,yc3,yc4,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M3N4UNIT \yc1,\yc2,\yc3,\yt1,\pta,\a_base_off,\a_inc_off
    vbroadcastsd \a_base_off+(\a_inc_off)*3(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc4
.endm

.macro SOLVE_L_M4N8UNIT yc1,yc2,yc3,yc4,yc5,yc6,yc7,yc8,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M3N8UNIT \yc1,\yc2,\yc3,\yc5,\yc6,\yc7,\yt1,\pta,\a_base_off,\a_inc_off
    vbroadcastsd \a_base_off+(\a_inc_off)*3(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc4
    vfnmadd231pd %ymm\yt1,%ymm\yc5,%ymm\yc8
.endm

.macro SOLVE_L_M4N12UNIT yc1,yc2,yc3,yc4,yc5,yc6,yc7,yc8,yc9,yca,ycb,ycc,yt1,pta,a_base_off,a_inc_off
    SOLVE_L_M3N12UNIT \yc1,\yc2,\yc3,\yc5,\yc6,\yc7,\yc9,\yca,\ycb,\yt1,\pta,\a_base_off,\a_inc_off
    vbroadcastsd \a_base_off+(\a_inc_off)*3(%\pta),%ymm\yt1
    vfnmadd231pd %ymm\yt1,%ymm\yc1,%ymm\yc4
    vfnmadd231pd %ymm\yt1,%ymm\yc5,%ymm\yc8
    vfnmadd231pd %ymm\yt1,%ymm\yc9,%ymm\ycc
.endm

/*************************************************************************
 * warning (in dtrsm_l_m4n4 zone):
 * The macros below have predefined rules of register usage where care
 * should be taken (or data corruption would occur)
*************************************************************************/

.macro DTRSM_SOLVE_M1N4_L_RUN pta,ptb,ptc,ldc_bytes,a_off
    leaq \a_off(%\ptc),%r10
    DTRSM_L_PREPROCESS_M1N4_RUN 4,2,3,r10,\ldc_bytes
    SOLVE_L_M1N4UNIT 4,3,\pta,\a_off
    leaq \a_off(%\ptc),%r10
    DTRSM_L_STORE_M1N4_RUN 4,r10,\ldc_bytes,\a_off*4,%\ptb
.endm

.macro DTRSM_SOLVE_M1N8_L_RUN pta,ptb,ptc,ldc_bytes,a_off,ldb_bytes
    leaq \a_off(%\ptc),%r10
    DTRSM_L_PREPROCESS_M1N4_RUN 4,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M1N4_RUN 5,2,3,r10,\ldc_bytes
    SOLVE_L_M1N8UNIT 4,5,3,\pta,\a_off
    leaq \a_off(%\ptc),%r10
    DTRSM_L_STORE_M1N4_RUN 4,r10,\ldc_bytes,\a_off*4,%\ptb
    DTRSM_L_STORE_M1N4_RUN 5,r10,\ldc_bytes,\a_off*4,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M1N12_L_RUN pta,ptb,ptc,ldc_bytes,a_off,ldb_bytes
    leaq \a_off(%\ptc),%r10
    DTRSM_L_PREPROCESS_M1N4_RUN 4,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M1N4_RUN 5,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M1N4_RUN 6,2,3,r10,\ldc_bytes
    SOLVE_L_M1N12UNIT 4,5,6,3,\pta,\a_off
    leaq \a_off(%\ptc),%r10
    DTRSM_L_STORE_M1N4_RUN 4,r10,\ldc_bytes,\a_off*4,%\ptb
    DTRSM_L_STORE_M1N4_RUN 5,r10,\ldc_bytes,\a_off*4,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M1N4_RUN 6,r10,\ldc_bytes,\a_off*4,%\ptb,%\ldb_bytes,8
.endm

/*   Here are the characteristics for the solver macros DTRSM_SOLVE_MmNm_L[N/T]
 * with m = 1,2,4 and n = 4,8,12.
 *   All parameters to these macros are registers holding pointer/stride values,
 * which will not be modified in the execution. "pt[a/b/c]" is for pointer to
 * array [A/B/C].
 *   These macros will use registers ymm2-ymm3 and r10 temporarily.
 *   Additional input elements (negative accumulated results from GEMM KERNEL)
 * are provided through registers ymm4-ymmX (X is the minimum number for the
 * register group to hold all elements). Each ymm register holds m1n4 elements.
 * They are assigned from left to right(downwards in the same column block).
 *   These macros are also suitable for haswell DTRSM_L kernels.
 *   DTRSM_SOLVE_MmNn_LN assumes that all input pointers POINT TO THE END of
 * the array or the lowest matrix in a combined input, while
 * DTRSM_SOLVE_MmNn_LT assumes those pointers POINT TO THE BEGINNING.
 *   For these solver macros, matrix A is a constant column-major RmCm array,
 * array B is a combination of n/4 row-major RmC4 matrix blocks with a stride
 * of 4*LDB between nearby blocks, matrix C is a column-major RmCn matrix with
 * leading dimention = LDC. */

.macro DTRSM_SOLVE_M1N4_LN pta,ptb,ptc,ldc_bytes
    DTRSM_SOLVE_M1N4_L_RUN \pta,\ptb,\ptc,\ldc_bytes,-8
.endm

.macro DTRSM_SOLVE_M1N8_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_SOLVE_M1N8_L_RUN \pta,\ptb,\ptc,\ldc_bytes,-8,\ldb_bytes
.endm

.macro DTRSM_SOLVE_M1N12_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_SOLVE_M1N12_L_RUN \pta,\ptb,\ptc,\ldc_bytes,-8,\ldb_bytes
.endm

.macro DTRSM_SOLVE_M1N4_LT pta,ptb,ptc,ldc_bytes
    DTRSM_SOLVE_M1N4_L_RUN \pta,\ptb,\ptc,\ldc_bytes,0
.endm

.macro DTRSM_SOLVE_M1N8_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_SOLVE_M1N8_L_RUN \pta,\ptb,\ptc,\ldc_bytes,0,\ldb_bytes
.endm

.macro DTRSM_SOLVE_M1N12_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    DTRSM_SOLVE_M1N12_L_RUN \pta,\ptb,\ptc,\ldc_bytes,0,\ldb_bytes
.endm

.macro DTRSM_SOLVE_M2N4_LN pta,ptb,ptc,ldc_bytes
    leaq -16(%\ptc),%r10
    DTRSM_L_PREPROCESS_M2N4_RUN 4,5,2,3,r10,\ldc_bytes
    SOLVE_L_M2N4UNIT 5,4,3,\pta,-8,-8
    SOLVE_L_M1N4UNIT 4,3,\pta,-32
    leaq -16(%\ptc),%r10
    DTRSM_L_STORE_M2N4_RUN 4,5,3,r10,\ldc_bytes,-64,%\ptb
.endm

.macro DTRSM_SOLVE_M2N8_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    leaq -16(%\ptc),%r10
    DTRSM_L_PREPROCESS_M2N4_RUN 4,5,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M2N4_RUN 6,7,2,3,r10,\ldc_bytes
    SOLVE_L_M2N8UNIT 5,4,7,6,3,\pta,-8,-8
    SOLVE_L_M1N8UNIT 4,6,3,\pta,-32
    leaq -16(%\ptc),%r10
    DTRSM_L_STORE_M2N4_RUN 4,5,3,r10,\ldc_bytes,-64,%\ptb
    DTRSM_L_STORE_M2N4_RUN 6,7,3,r10,\ldc_bytes,-64,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M2N12_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    leaq -16(%\ptc),%r10
    DTRSM_L_PREPROCESS_M2N4_RUN 4,5,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M2N4_RUN 6,7,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M2N4_RUN 8,9,2,3,r10,\ldc_bytes
    SOLVE_L_M2N12UNIT 5,4,7,6,9,8,3,\pta,-8,-8
    SOLVE_L_M1N12UNIT 4,6,8,3,\pta,-32
    leaq -16(%\ptc),%r10
    DTRSM_L_STORE_M2N4_RUN 4,5,3,r10,\ldc_bytes,-64,%\ptb
    DTRSM_L_STORE_M2N4_RUN 6,7,3,r10,\ldc_bytes,-64,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M2N4_RUN 8,9,3,r10,\ldc_bytes,-64,%\ptb,%\ldb_bytes,8
.endm

.macro DTRSM_SOLVE_M2N4_LT pta,ptb,ptc,ldc_bytes
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M2N4_RUN 4,5,2,3,r10,\ldc_bytes
    SOLVE_L_M2N4UNIT 4,5,3,\pta,0,8
    SOLVE_L_M1N4UNIT 5,3,\pta,24
    movq %\ptc,%r10
    DTRSM_L_STORE_M2N4_RUN 4,5,3,r10,\ldc_bytes,0,%\ptb
.endm

.macro DTRSM_SOLVE_M2N8_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M2N4_RUN 4,5,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M2N4_RUN 6,7,2,3,r10,\ldc_bytes
    SOLVE_L_M2N8UNIT 4,5,6,7,3,\pta,0,8
    SOLVE_L_M1N8UNIT 5,7,3,\pta,24
    movq %\ptc,%r10
    DTRSM_L_STORE_M2N4_RUN 4,5,3,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M2N4_RUN 6,7,3,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M2N12_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M2N4_RUN 4,5,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M2N4_RUN 6,7,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M2N4_RUN 8,9,2,3,r10,\ldc_bytes
    SOLVE_L_M2N12UNIT 4,5,6,7,8,9,3,\pta,0,8
    SOLVE_L_M1N12UNIT 5,7,9,3,\pta,24
    movq %\ptc,%r10
    DTRSM_L_STORE_M2N4_RUN 4,5,3,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M2N4_RUN 6,7,3,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M2N4_RUN 8,9,3,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,8
.endm

.macro DTRSM_SOLVE_M4N4_LN pta,ptb,ptc,ldc_bytes
    leaq -32(%\ptc),%r10
    DTRSM_L_PREPROCESS_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes
    SOLVE_L_M4N4UNIT 7,6,5,4,3,\pta,-8,-8
    SOLVE_L_M3N4UNIT 6,5,4,3,\pta,-48,-8
    SOLVE_L_M2N4UNIT 5,4,3,\pta,-88,-8
    SOLVE_L_M1N4UNIT 4,3,\pta,-128
    leaq -32(%\ptc),%r10
    DTRSM_L_STORE_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes,-128,%\ptb
.endm

.macro DTRSM_SOLVE_M4N8_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    leaq -32(%\ptc),%r10
    DTRSM_L_PREPROCESS_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes
    SOLVE_L_M4N8UNIT 7,6,5,4,11,10,9,8,3,\pta,-8,-8
    SOLVE_L_M3N8UNIT 6,5,4,10,9,8,3,\pta,-48,-8
    SOLVE_L_M2N8UNIT 5,4,9,8,3,\pta,-88,-8
    SOLVE_L_M1N8UNIT 4,8,3,\pta,-128
    leaq -32(%\ptc),%r10
    DTRSM_L_STORE_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes,-128,%\ptb
    DTRSM_L_STORE_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes,-128,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M4N12_LN pta,ptb,ptc,ldc_bytes,ldb_bytes
    leaq -32(%\ptc),%r10
    DTRSM_L_PREPROCESS_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M4N4_RUN 12,13,14,15,2,3,r10,\ldc_bytes
    SOLVE_L_M4N12UNIT 7,6,5,4,11,10,9,8,15,14,13,12,3,\pta,-8,-8
    SOLVE_L_M3N12UNIT 6,5,4,10,9,8,14,13,12,3,\pta,-48,-8
    SOLVE_L_M2N12UNIT 5,4,9,8,13,12,3,\pta,-88,-8
    SOLVE_L_M1N12UNIT 4,8,12,3,\pta,-128
    leaq -32(%\ptc),%r10
    DTRSM_L_STORE_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes,-128,%\ptb
    DTRSM_L_STORE_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes,-128,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M4N4_RUN 12,13,14,15,2,3,r10,\ldc_bytes,-128,%\ptb,%\ldb_bytes,8
.endm

.macro DTRSM_SOLVE_M4N4_LT pta,ptb,ptc,ldc_bytes
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes
    SOLVE_L_M4N4UNIT 4,5,6,7,3,\pta,0,8
    SOLVE_L_M3N4UNIT 5,6,7,3,\pta,40,8
    SOLVE_L_M2N4UNIT 6,7,3,\pta,80,8
    SOLVE_L_M1N4UNIT 7,3,\pta,120
    movq %\ptc,%r10
    DTRSM_L_STORE_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes,0,%\ptb
.endm

.macro DTRSM_SOLVE_M4N8_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes
    SOLVE_L_M4N8UNIT 4,5,6,7,8,9,10,11,3,\pta,0,8
    SOLVE_L_M3N8UNIT 5,6,7,9,10,11,3,\pta,40,8
    SOLVE_L_M2N8UNIT 6,7,10,11,3,\pta,80,8
    SOLVE_L_M1N8UNIT 7,11,3,\pta,120
    movq %\ptc,%r10
    DTRSM_L_STORE_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
.endm

.macro DTRSM_SOLVE_M4N12_LT pta,ptb,ptc,ldc_bytes,ldb_bytes
    movq %\ptc,%r10
    DTRSM_L_PREPROCESS_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes
    DTRSM_L_PREPROCESS_M4N4_RUN 12,13,14,15,2,3,r10,\ldc_bytes
    SOLVE_L_M4N12UNIT 4,5,6,7,8,9,10,11,12,13,14,15,3,\pta,0,8
    SOLVE_L_M3N12UNIT 5,6,7,9,10,11,13,14,15,3,\pta,40,8
    SOLVE_L_M2N12UNIT 6,7,10,11,14,15,3,\pta,80,8
    SOLVE_L_M1N12UNIT 7,11,15,3,\pta,120
    movq %\ptc,%r10
    DTRSM_L_STORE_M4N4_RUN 4,5,6,7,2,3,r10,\ldc_bytes,0,%\ptb
    DTRSM_L_STORE_M4N4_RUN 8,9,10,11,2,3,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,4
    DTRSM_L_STORE_M4N4_RUN 12,13,14,15,2,3,r10,\ldc_bytes,0,%\ptb,%\ldb_bytes,8
.endm

/**************************************************************************
 * End of dtrsm_l_m4n4 zone
 * Here enter the dtrsm_l_m4n2 zone.
 * Macros here are specially for dtrsm_l[n/t]_m[4/8/16]n2 solvers
 *************************************************************************/

.macro SOLVE_L_M4N2_TAIL zc,zt,kmul,pta,offa
    vbroadcastf64x4 \offa(%\pta),%zmm\zt
    vmulpd %zmm\zt,%zmm\zc,%zmm\zc{%k\kmul}
.endm

.macro SOLVE_L_M4N2_NOTAIL mpos,zc,zt,zcoef,kmul,kfma,pta,offa,kmul_shiftdir
    SOLVE_L_M4N2_TAIL \zc,\zt,\kmul,\pta,\offa
    kandnw %k\kfma,%k\kmul,%k\kfma
    vpermpd $0x55*\mpos,%zmm\zc,%zmm\zcoef
    vfnmadd231pd %zmm\zt,%zmm\zcoef,%zmm\zc{%k\kfma}
    kshift\kmul_shiftdir\()w $1,%k\kmul,%k\kmul
.endm

.macro UPDATE_L_M4N2 zc,zt,zcoef,pta,offa
    vbroadcastf64x4 \offa(%\pta),%zmm\zt
    vfnmadd231pd %zmm\zt,%zmm\zcoef,%zmm\zc
.endm

.macro UPDATE_L_M8N2 zc1,zc2,zt,zcoef,pta,offa
    UPDATE_L_M4N2 \zc1,\zt,\zcoef,\pta,\offa
    UPDATE_L_M4N2 \zc2,\zt,\zcoef,\pta,\offa+32
.endm

.macro UPDATE_L_M12N2 zc1,zc2,zc3,zt,zcoef,pta,offa
    UPDATE_L_M8N2 \zc1,\zc2,\zt,\zcoef,\pta,\offa
    UPDATE_L_M4N2 \zc3,\zt,\zcoef,\pta,\offa+64
.endm

.macro DTRSM_L_STORE_M4N2_INIT r64tmp,zperm_no
    INIT_U64X8 0x0703060205010400,\r64tmp,\zperm_no
.endm

.macro DTRSM_L_STORE_M4N2_RUN zc,zperm,zt,offb,ptb,offc,ptc,ldc_bytes
    vpermpd %zmm\zc,%zmm\zperm,%zmm\zt
    vmovupd %zmm\zt,\offb(%\ptb)
    vmovupd %ymm\zc,\offc(%\ptc)
    vextractf64x4 $1,%zmm\zc,\offc(%\ptc,%\ldc_bytes,1)
.endm

.macro DTRSM_L_PREPROCESS_M8N2_RUN zc1,zc2,zt,offc,ptc,ldc_bytes
    vaddpd \offc(%\ptc),%zmm\zc1,%zmm\zt
    vaddpd \offc(%\ptc,%\ldc_bytes,1),%zmm\zc2,%zmm\zc2
    vshuff64x2 $0x44,%zmm\zc2,%zmm\zt,%zmm\zc1
    vshuff64x2 $0xee,%zmm\zc2,%zmm\zt,%zmm\zc2
.endm

/*************************************************************************
 * warning (in dtrsm_l_m4n2 zone):
 * The macros below have predefined rules of register usage where care
 * should be taken (or data corruption would occur)
*************************************************************************/

/** k2 for vmulpd merge mask, k3 for vfnmadd231pd merge mask */
/** zmm6 for updated coefficients, zmm7 for temporary register */
/** zmm8-zmm11 for input data */

.macro DTRSM_SOLVE_M4N2BLOCK_LN_TAIL_RUN zc,pta,base,ld
    SOLVE_L_M4N2_NOTAIL 3,\zc,7,6,2,3,\pta,\base-32,r
    SOLVE_L_M4N2_NOTAIL 2,\zc,7,6,2,3,\pta,\base-32-\ld,r
    SOLVE_L_M4N2_NOTAIL 1,\zc,7,6,2,3,\pta,\base-32-\ld*2,r
    SOLVE_L_M4N2_TAIL \zc,7,2,\pta,\base-32-\ld*3
.endm

.macro DTRSM_SOLVE_M4N2BLOCK_LT_TAIL_RUN zc,pta,base,ld
    SOLVE_L_M4N2_NOTAIL 0,\zc,7,6,2,3,\pta,\base,l
    SOLVE_L_M4N2_NOTAIL 1,\zc,7,6,2,3,\pta,\base+\ld,l
    SOLVE_L_M4N2_NOTAIL 2,\zc,7,6,2,3,\pta,\base+\ld*2,l
    SOLVE_L_M4N2_TAIL \zc,7,2,\pta,\base+\ld*3
.endm

.macro DTRSM_SOLVE_M4N2BLOCK_LN_NOTAIL_RUN zcsolve,pta,base,update_m,ld,update_list:vararg
    SOLVE_L_M4N2_NOTAIL 3,\zcsolve,7,6,2,3,\pta,\base-32,r
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base-32-\update_m*8
    SOLVE_L_M4N2_NOTAIL 2,\zcsolve,7,6,2,3,\pta,\base-32-\ld,r
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base-32-\update_m*8-\ld
    SOLVE_L_M4N2_NOTAIL 1,\zcsolve,7,6,2,3,\pta,\base-32-\ld*2,r
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base-32-\update_m*8-\ld*2
    SOLVE_L_M4N2_TAIL \zcsolve,7,2,\pta,\base-32-\ld*3
    kshiftlw $3,%k2,%k2
    knotw %k2,%k3
    vpermpd $0,%zmm\zcsolve,%zmm6
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base-32-\update_m*8-\ld*3
.endm

.macro DTRSM_SOLVE_M4N2BLOCK_LT_NOTAIL_RUN zcsolve,pta,base,update_m,ld,update_list:vararg
    SOLVE_L_M4N2_NOTAIL 0,\zcsolve,7,6,2,3,\pta,\base,l
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base+32
    SOLVE_L_M4N2_NOTAIL 1,\zcsolve,7,6,2,3,\pta,\base+\ld,l
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base+\ld+32
    SOLVE_L_M4N2_NOTAIL 2,\zcsolve,7,6,2,3,\pta,\base+\ld*2,l
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base+\ld*2+32
    SOLVE_L_M4N2_TAIL \zcsolve,7,2,\pta,\base+\ld*3
    kshiftrw $3,%k2,%k2
    knotw %k2,%k3
    vpermpd $0xff,%zmm\zcsolve,%zmm6
    UPDATE_L_M\update_m\()N2 \update_list,7,6,\pta,\base+\ld*3+32
.endm

.macro DTRSM_SOLVE_M4N2_LN pta,ptb,ptc,ldc_bytes
    vmovupd -32(%\ptc),%ymm7
    vinsertf64x4 $1,-32(%\ptc,%\ldc_bytes,1),%zmm7,%zmm7
    vaddpd %zmm7,%zmm8,%zmm8
    INIT_K8 0xff,r10d,3
    INIT_K8 0x88,r10d,2
    DTRSM_SOLVE_M4N2BLOCK_LN_TAIL_RUN 8,\pta,0,32
    DTRSM_L_STORE_M4N2_INIT r10,6
    DTRSM_L_STORE_M4N2_RUN 8,6,7,-64,\ptb,-32,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M4N2_LT pta,ptb,ptc,ldc_bytes
    vmovupd (%\ptc),%ymm7
    vinsertf64x4 $1,(%\ptc,%\ldc_bytes,1),%zmm7,%zmm7
    vaddpd %zmm7,%zmm8,%zmm8
    INIT_K8 0xff,r10d,3
    INIT_K8 0x11,r10d,2
    DTRSM_SOLVE_M4N2BLOCK_LT_TAIL_RUN 8,\pta,0,32
    DTRSM_L_STORE_M4N2_INIT r10,6
    DTRSM_L_STORE_M4N2_RUN 8,6,7,0,\ptb,0,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M8N2_LN pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M8N2_RUN 8,9,7,-64,\ptc,\ldc_bytes
    INIT_K8 0xff,r10d,3
    INIT_K8 0x88,r10d,2
    DTRSM_SOLVE_M4N2BLOCK_LN_NOTAIL_RUN 9,\pta,0,4,64,8
    DTRSM_SOLVE_M4N2BLOCK_LN_TAIL_RUN 8,\pta,-288,64
    DTRSM_L_STORE_M4N2_INIT r10,6
    DTRSM_L_STORE_M4N2_RUN 8,6,7,-128,\ptb,-64,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 9,6,7,-64,\ptb,-32,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M8N2_LT pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M8N2_RUN 8,9,7,0,\ptc,\ldc_bytes
    INIT_K8 0xff,r10d,3
    INIT_K8 0x11,r10d,2
    DTRSM_SOLVE_M4N2BLOCK_LT_NOTAIL_RUN 8,\pta,0,4,64,9
    DTRSM_SOLVE_M4N2BLOCK_LT_TAIL_RUN 9,\pta,288,64
    DTRSM_L_STORE_M4N2_INIT r10,6
    DTRSM_L_STORE_M4N2_RUN 8,6,7,0,\ptb,0,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 9,6,7,64,\ptb,32,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M16N2_LN pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M8N2_RUN 8,9,7,-128,\ptc,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N2_RUN 10,11,7,-64,\ptc,\ldc_bytes
    INIT_K8 0xff,r10d,3
    INIT_K8 0x88,r10d,2
    DTRSM_SOLVE_M4N2BLOCK_LN_NOTAIL_RUN 11,\pta,0,12,128,8,9,10
    DTRSM_SOLVE_M4N2BLOCK_LN_NOTAIL_RUN 10,\pta,-544,8,128,8,9
    DTRSM_SOLVE_M4N2BLOCK_LN_NOTAIL_RUN 9,\pta,-1088,4,128,8
    DTRSM_SOLVE_M4N2BLOCK_LN_TAIL_RUN 8,\pta,-1632,128
    DTRSM_L_STORE_M4N2_INIT r10,6
    DTRSM_L_STORE_M4N2_RUN 8,6,7,-256,\ptb,-128,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 9,6,7,-192,\ptb,-96,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 10,6,7,-128,\ptb,-64,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 11,6,7,-64,\ptb,-32,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M16N2_LT pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M8N2_RUN 8,9,7,0,\ptc,\ldc_bytes
    DTRSM_L_PREPROCESS_M8N2_RUN 10,11,7,64,\ptc,\ldc_bytes
    INIT_K8 0xff,r10d,3
    INIT_K8 0x11,r10d,2
    DTRSM_SOLVE_M4N2BLOCK_LT_NOTAIL_RUN 8,\pta,0,12,128,9,10,11
    DTRSM_SOLVE_M4N2BLOCK_LT_NOTAIL_RUN 9,\pta,544,8,128,10,11
    DTRSM_SOLVE_M4N2BLOCK_LT_NOTAIL_RUN 10,\pta,1088,4,128,11
    DTRSM_SOLVE_M4N2BLOCK_LT_TAIL_RUN 11,\pta,1632,128
    DTRSM_L_STORE_M4N2_INIT r10,6
    DTRSM_L_STORE_M4N2_RUN 8,6,7,0,\ptb,0,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 9,6,7,64,\ptb,32,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 10,6,7,128,\ptb,64,\ptc,\ldc_bytes
    DTRSM_L_STORE_M4N2_RUN 11,6,7,192,\ptb,96,\ptc,\ldc_bytes
.endm

/**************************************************************************
 * End of dtrsm_l_m4n2 zone
 * Here enter the dtrsm_l_m4n1 zone.
 * Macros here are specially for dtrsm_l[n/t]_m[4/8/16]n1 solvers
 *************************************************************************/

.macro UPDATE_L_M4N1 yc1,ycoef,pta,offa
    vfnmadd231pd \offa(%\pta),%ymm\ycoef,%ymm\yc1
.endm

.macro UPDATE_L_M8N1 yc1,yc2,ycoef,pta,offa
    vfnmadd231pd \offa(%\pta),%ymm\ycoef,%ymm\yc1
    vfnmadd231pd \offa+32(%\pta),%ymm\ycoef,%ymm\yc2
.endm

.macro UPDATE_L_M12N1 yc1,yc2,yc3,ycoef,pta,offa
    vfnmadd231pd \offa(%\pta),%ymm\ycoef,%ymm\yc1
    vfnmadd231pd \offa+32(%\pta),%ymm\ycoef,%ymm\yc2
    vfnmadd231pd \offa+64(%\pta),%ymm\ycoef,%ymm\yc3
.endm

.macro SOLVE_L_M4N1BLOCK mpos,mnextinc,yc,yt,ycoef,pta,offa,update_m,update_list:vararg
    vmovupd \offa(%\pta),%ymm\yt
    vmulpd %ymm\yt,%ymm\yc,%ymm\ycoef
    vblendpd $1<<\mpos,%ymm\ycoef,%ymm\yc,%ymm\yc
    vpermpd $0x55*\mpos,%ymm\ycoef,%ymm\ycoef
.if \mpos != (3 + 3 * \mnextinc) / 2
    vfnmadd132pd %ymm\ycoef,%ymm\yc,%ymm\yt
.if \mnextinc > 0
    vblendpd $0xF<<(\mpos+1),%ymm\yt,%ymm\yc,%ymm\yc
.else
    vblendpd $(1<<\mpos)-1,%ymm\yt,%ymm\yc,%ymm\yc
.endif
.endif
.if \update_m > 0
    UPDATE_L_M\update_m\()N1 \update_list,\ycoef,\pta,\offa+32-(1-(\mnextinc))*(\update_m+4)*4
.endif
.endm

.macro SOLVE_LN_M4N1BLOCK_RUN update_m,mld,yc,yt,ycoef,pta,offa,update_list:vararg
    SOLVE_L_M4N1BLOCK 3,-1,\yc,\yt,\ycoef,\pta,\offa-(\mld-\update_m)*8,\update_m,\update_list
    SOLVE_L_M4N1BLOCK 2,-1,\yc,\yt,\ycoef,\pta,\offa-(\mld*2-\update_m)*8,\update_m,\update_list
    SOLVE_L_M4N1BLOCK 1,-1,\yc,\yt,\ycoef,\pta,\offa-(\mld*3-\update_m)*8,\update_m,\update_list
    SOLVE_L_M4N1BLOCK 0,-1,\yc,\yt,\ycoef,\pta,\offa-(\mld*4-\update_m)*8,\update_m,\update_list
.endm

.macro SOLVE_LT_M4N1BLOCK_RUN update_m,mld,yc,yt,ycoef,pta,offa,update_list:vararg
    SOLVE_L_M4N1BLOCK 0,1,\yc,\yt,\ycoef,\pta,\offa+(\mld-\update_m-4)*8,\update_m,\update_list
    SOLVE_L_M4N1BLOCK 1,1,\yc,\yt,\ycoef,\pta,\offa+(\mld*2-\update_m-4)*8,\update_m,\update_list
    SOLVE_L_M4N1BLOCK 2,1,\yc,\yt,\ycoef,\pta,\offa+(\mld*3-\update_m-4)*8,\update_m,\update_list
    SOLVE_L_M4N1BLOCK 3,1,\yc,\yt,\ycoef,\pta,\offa+(\mld*4-\update_m-4)*8,\update_m,\update_list
.endm

/*************************************************************************
 * warning (in dtrsm_l_m4n1 zone):
 * The macros below have predefined rules of register usage where care
 * should be taken (or data corruption would occur)
*************************************************************************/

.macro DTRSM_SOLVE_M4N1_LN pta,ptb,ptc
    vaddpd -32(%\ptc),%ymm4,%ymm4
    SOLVE_LN_M4N1BLOCK_RUN 0,4,4,2,3,\pta,0,dummy
    vmovupd %ymm4,-32(%\ptb)
    vmovupd %ymm4,-32(%\ptc)
.endm

.macro DTRSM_SOLVE_M4N1_LT pta,ptb,ptc
    vaddpd (%\ptc),%ymm4,%ymm4
    SOLVE_LT_M4N1BLOCK_RUN 0,4,4,2,3,\pta,0,dummy
    vmovupd %ymm4,(%\ptb)
    vmovupd %ymm4,(%\ptc)
.endm

.macro DTRSM_SOLVE_M8N1_LN pta,ptb,ptc
    vaddpd -64(%\ptc),%ymm4,%ymm4
    vaddpd -32(%\ptc),%ymm5,%ymm5
    SOLVE_LN_M4N1BLOCK_RUN 4,8,5,2,3,\pta,0,4
    SOLVE_LN_M4N1BLOCK_RUN 0,8,4,2,3,\pta,-256,dummy
    vmovupd %ymm4,-64(%\ptb)
    vmovupd %ymm5,-32(%\ptb)
    vmovupd %ymm4,-64(%\ptc)
    vmovupd %ymm5,-32(%\ptc)        
.endm

.macro DTRSM_SOLVE_M8N1_LT pta,ptb,ptc
    vaddpd (%\ptc),%ymm4,%ymm4
    vaddpd 32(%\ptc),%ymm5,%ymm5
    SOLVE_LT_M4N1BLOCK_RUN 4,8,4,2,3,\pta,0,5
    SOLVE_LT_M4N1BLOCK_RUN 0,8,5,2,3,\pta,256,dummy
    vmovupd %ymm4,(%\ptb)
    vmovupd %ymm5,32(%\ptb)
    vmovupd %ymm4,(%\ptc)
    vmovupd %ymm5,32(%\ptc)
.endm

.macro DTRSM_SOLVE_M16N1_LN pta,ptb,ptc
    vaddpd -128(%\ptc),%ymm4,%ymm4
    vaddpd -96(%\ptc),%ymm5,%ymm5
    vaddpd -64(%\ptc),%ymm6,%ymm6
    vaddpd -32(%\ptc),%ymm7,%ymm7
    SOLVE_LN_M4N1BLOCK_RUN 12,16,7,2,3,\pta,0,4,5,6
    SOLVE_LN_M4N1BLOCK_RUN 8,16,6,2,3,\pta,-512,4,5
    SOLVE_LN_M4N1BLOCK_RUN 4,16,5,2,3,\pta,-1024,4
    SOLVE_LN_M4N1BLOCK_RUN 0,16,4,2,3,\pta,-1536,dummy
    vmovupd %ymm4,-128(%\ptb)
    vmovupd %ymm5,-96(%\ptb)
    vmovupd %ymm6,-64(%\ptb)
    vmovupd %ymm7,-32(%\ptb)
    vmovupd %ymm4,-128(%\ptc)
    vmovupd %ymm5,-96(%\ptc)
    vmovupd %ymm6,-64(%\ptc)
    vmovupd %ymm7,-32(%\ptc)
.endm

.macro DTRSM_SOLVE_M16N1_LT pta,ptb,ptc
    vaddpd (%\ptc),%ymm4,%ymm4
    vaddpd 32(%\ptc),%ymm5,%ymm5
    vaddpd 64(%\ptc),%ymm6,%ymm6
    vaddpd 96(%\ptc),%ymm7,%ymm7
    SOLVE_LT_M4N1BLOCK_RUN 12,16,4,2,3,\pta,0,5,6,7
    SOLVE_LT_M4N1BLOCK_RUN 8,16,5,2,3,\pta,512,6,7
    SOLVE_LT_M4N1BLOCK_RUN 4,16,6,2,3,\pta,1024,7
    SOLVE_LT_M4N1BLOCK_RUN 0,16,7,2,3,\pta,1536,dummy
    vmovupd %ymm4,(%\ptb)
    vmovupd %ymm5,32(%\ptb)
    vmovupd %ymm6,64(%\ptb)
    vmovupd %ymm7,96(%\ptb)
    vmovupd %ymm4,(%\ptc)
    vmovupd %ymm5,32(%\ptc)
    vmovupd %ymm6,64(%\ptc)
    vmovupd %ymm7,96(%\ptc)
.endm

/**************************************************************************
 * End of dtrsm_l_m4n1 zone
 * Here enter the dtrsm_l_m2n2 zone.
 * Macros here are specially for dtrsm_l[n/t]_m[1/2]n2 solvers
 *************************************************************************/

.macro DTRSM_L_PREPROCESS_M2N2 xc1,xc2,xt,offc,ptc,ldc_bytes
    vaddpd \offc(%\ptc),%xmm\xc1,%xmm\xc1
    vaddpd \offc(%\ptc,%\ldc_bytes,1),%xmm\xc2,%xmm\xt
    vunpckhpd %xmm\xt,%xmm\xc1,%xmm\xc2
    vunpcklpd %xmm\xt,%xmm\xc1,%xmm\xc1
.endm

.macro DTRSM_L_STORE_M2N2 xc1,xc2,xt,offb,ptb,offc,ptc,ldc_bytes
    vmovupd %xmm\xc1,\offb(%\ptb)
    vmovupd %xmm\xc2,\offb+16(%\ptb)
    vunpcklpd %xmm\xc2,%xmm\xc1,%xmm\xt
    vunpckhpd %xmm\xc2,%xmm\xc1,%xmm\xc2
    vmovupd %xmm\xt,\offc(%\ptc)
    vmovupd %xmm\xc2,\offc(%\ptc,%\ldc_bytes,1)
.endm

.macro DTRSM_L_PREPROCESS_M1N2 xc,xt,offc,ptc,ldc_bytes
    vmovsd \offc(%\ptc),%xmm\xt
    vmovhpd \offc(%\ptc,%\ldc_bytes,1),%xmm\xt,%xmm\xt
    vaddpd %xmm\xt,%xmm\xc,%xmm\xc
.endm

.macro DTRSM_L_STORE_M1N2 xc,offb,ptb,offc,ptc,ldc_bytes
    vmovupd %xmm\xc,\offb(%\ptb)
    vmovsd %xmm\xc,\offc(%\ptc)
    vmovhpd %xmm\xc,\offc(%\ptc,%\ldc_bytes,1)
.endm

.macro DTRSM_L_SOLVE_M1N2UNIT xc,xt,offa,pta
    vmovddup \offa(%\pta),%xmm\xt
    vmulpd %xmm\xt,%xmm\xc,%xmm\xc
.endm

.macro DTRSM_L_UPDATE_M1N2UNIT xc,xt,xcoef,offa,pta
    vmovddup \offa(%\pta),%xmm\xt
    vfnmadd231pd %xmm\xt,%xmm\xcoef,%xmm\xc
.endm

/*************************************************************************
 * warning (in dtrsm_l_m2n2 zone):
 * The macros below have predefined rules of register usage where care
 * should be taken (or data corruption would occur)
*************************************************************************/

.macro DTRSM_SOLVE_M2N2_LN pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M2N2 4,5,3,-16,\ptc,\ldc_bytes
    DTRSM_L_SOLVE_M1N2UNIT 5,3,-8,\pta
    DTRSM_L_UPDATE_M1N2UNIT 4,3,5,-16,\pta
    DTRSM_L_SOLVE_M1N2UNIT 4,3,-32,\pta
    DTRSM_L_STORE_M2N2 4,5,3,-32,\ptb,-16,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M2N2_LT pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M2N2 4,5,3,0,\ptc,\ldc_bytes
    DTRSM_L_SOLVE_M1N2UNIT 4,3,0,\pta
    DTRSM_L_UPDATE_M1N2UNIT 5,3,4,8,\pta
    DTRSM_L_SOLVE_M1N2UNIT 5,3,24,\pta
    DTRSM_L_STORE_M2N2 4,5,3,0,\ptb,0,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M1N2_LN pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M1N2 4,3,-8,\ptc,\ldc_bytes
    DTRSM_L_SOLVE_M1N2UNIT 4,3,-8,\pta
    DTRSM_L_STORE_M1N2 4,-16,\ptb,-8,\ptc,\ldc_bytes
.endm

.macro DTRSM_SOLVE_M1N2_LT pta,ptb,ptc,ldc_bytes
    DTRSM_L_PREPROCESS_M1N2 4,3,0,\ptc,\ldc_bytes
    DTRSM_L_SOLVE_M1N2UNIT 4,3,0,\pta
    DTRSM_L_STORE_M1N2 4,0,\ptb,0,\ptc,\ldc_bytes
.endm

/**************************************************************************
 * End of dtrsm_l_m2n2 zone
 * Here enter the dtrsm_l_m2n1 zone.
 * Macros here are specially for dtrsm_l[n/t]_m[1/2]n1 solvers
 * warning: The macros below have predefined rules of register usage
 * where care should be taken (or data corruption would occur)
 *************************************************************************/

.macro DTRSM_SOLVE_M1N1_L pta,ptb,ptc,off
    vaddsd \off(%\ptc),%xmm4,%xmm4
    vmulsd \off(%\pta),%xmm4,%xmm4
    vmovsd %xmm4,\off(%\ptb)
    vmovsd %xmm4,\off(%\ptc)
.endm

.macro DTRSM_SOLVE_M1N1_LN pta,ptb,ptc
    DTRSM_SOLVE_M1N1_L \pta,\ptb,\ptc,-8
.endm

.macro DTRSM_SOLVE_M1N1_LT pta,ptb,ptc
    DTRSM_SOLVE_M1N1_L \pta,\ptb,\ptc,0
.endm

.macro DTRSM_SOLVE_M2N1_LN pta,ptb,ptc
    vaddpd -16(%\ptc),%xmm4,%xmm4
    vunpckhpd %xmm4,%xmm4,%xmm5
    vmulsd -8(%\pta),%xmm5,%xmm5
    vfnmadd231sd -16(%\pta),%xmm5,%xmm4
    vmulsd -32(%\pta),%xmm4,%xmm4
    vunpcklpd %xmm5,%xmm4,%xmm4
    vmovupd %xmm4,-16(%\ptb)
    vmovupd %xmm4,-16(%\ptc)
.endm

.macro DTRSM_SOLVE_M2N1_LT pta,ptb,ptc
    vaddpd (%\ptc),%xmm4,%xmm4
    vunpckhpd %xmm4,%xmm4,%xmm5
    vmulsd (%\pta),%xmm4,%xmm4
    vfnmadd231sd 8(%\pta),%xmm4,%xmm5
    vmulsd 24(%\pta),%xmm5,%xmm5
    vunpcklpd %xmm5,%xmm4,%xmm4
    vmovupd %xmm4,(%\ptb)
    vmovupd %xmm4,(%\ptc)
.endm

/**************************************************************************
 * End of dtrsm_l_m2n1 zone
**************************************************************************/


.macro funct name
.globl \name
.type \name,@function
\name :
.endm

.macro TEST_LOAD_M2N2 xc1,xc2,offset,ptb
    vmovsd \offset(%\ptb),%xmm\xc1
    vmovsd \offset+8(%\ptb),%xmm\xc2
    vmovhpd \offset+16(%\ptb),%xmm\xc1,%xmm\xc1
    vmovhpd \offset+24(%\ptb),%xmm\xc2,%xmm\xc2
.endm

.macro TEST_LOAD_M4N2 zc1,offset,ptb
    vmovupd \offset(%\ptb),%zmm\zc1
    vpermpd $0xd8,%zmm\zc1,%zmm\zc1
    vshuff64x2 $0xd8,%zmm\zc1,%zmm\zc1,%zmm\zc1
.endm

/** warning: this macro temporarily uses zmm7 */
.macro TEST_LOAD_M8N2 zc1,zc2,offset,ptb
    vmovupd \offset(%\ptb),%zmm\zc1
    vpermpd $0xd8,%zmm\zc1,%zmm7
    vmovupd \offset+64(%\ptb),%zmm\zc2
    vpermpd $0xd8,%zmm\zc2,%zmm\zc2
    vshuff64x2 $0x88,%zmm\zc2,%zmm7,%zmm\zc1
    vshuff64x2 $0xdd,%zmm\zc2,%zmm7,%zmm\zc2
.endm

.macro TEST_LOAD_M2N4 yc1,yc2,offset,mem:vararg
    vmovupd \offset(\mem),%ymm\yc1
    vmovupd \offset+32(\mem),%ymm\yc2
.endm

.macro TEST_LOAD_M4N4 yc1,yc2,yc3,yc4,offset,mem:vararg
    TEST_LOAD_M2N4 \yc1,\yc2,\offset,\mem
    TEST_LOAD_M2N4 \yc3,\yc4,\offset+64,\mem
.endm

/** warning: this macro temporarily uses zmm6 and zmm7 */
.macro TEST_LOAD_M8N4 zc1,zc2,zc3,zc4,offset,mem:vararg
    vmovupd \offset(\mem),%zmm\zc1
    vmovupd \offset+64(\mem),%zmm\zc3
    vmovupd \offset+128(\mem),%zmm\zc2
    vmovupd \offset+192(\mem),%zmm\zc4
    TRANSPOSE_4X4_FLOAT128 \zc1,\zc3,\zc2,\zc4,6,7
.endm

/** warning: this macro temporarily uses zmm6 and zmm7 */
.macro TEST_LOAD_M16N4 zc1,zc2,zc3,zc4,zc5,zc6,zc7,zc8,offset,mem:vararg
    TEST_LOAD_M8N4 \zc1,\zc2,\zc3,\zc4,\offset,\mem
    TEST_LOAD_M8N4 \zc5,\zc6,\zc7,\zc8,\offset+256,\mem
.endm

.section .text

funct dtrsm_solve_ln_m1n1
    vmovsd (%rsi),%xmm4
    addq $8,%rdi
    addq $8,%rsi
    addq $8,%rdx
    DTRSM_SOLVE_M1N1_LN rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_lt_m1n1
    vmovsd (%rsi),%xmm4
    DTRSM_SOLVE_M1N1_LT rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_ln_m2n1
    vmovupd (%rsi),%xmm4
    addq $32,%rdi
    addq $16,%rsi
    addq $16,%rdx
    DTRSM_SOLVE_M2N1_LN rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_lt_m2n1
    vmovupd (%rsi),%xmm4
    DTRSM_SOLVE_M2N1_LT rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_ln_m2n2
    TEST_LOAD_M2N2 4,5,0,rsi
    addq $32,%rdi
    addq $32,%rsi
    addq $16,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M2N2_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m2n2
    TEST_LOAD_M2N2 4,5,0,rsi
    salq $3,%r8
    DTRSM_SOLVE_M2N2_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m1n2
    vmovupd (%rsi),%xmm4
    addq $8,%rdi
    addq $16,%rsi
    addq $8,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M1N2_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m1n2
    vmovupd (%rsi),%xmm4
    salq $3,%r8
    DTRSM_SOLVE_M1N2_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m4n1
    vmovupd (%rsi),%ymm4
    addq $128,%rdi
    addq $32,%rsi
    addq $32,%rdx
    DTRSM_SOLVE_M4N1_LN rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_lt_m4n1
    vmovupd (%rsi),%ymm4
    DTRSM_SOLVE_M4N1_LT rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_ln_m8n1
    TEST_LOAD_M2N4 4,5,0,%rsi
    addq $512,%rdi
    addq $64,%rsi
    addq $64,%rdx
    DTRSM_SOLVE_M8N1_LN rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_lt_m8n1
    TEST_LOAD_M2N4 4,5,0,%rsi
    DTRSM_SOLVE_M8N1_LT rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_ln_m16n1
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    addq $2048,%rdi
    addq $128,%rsi
    addq $128,%rdx
    DTRSM_SOLVE_M16N1_LN rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_lt_m16n1
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    DTRSM_SOLVE_M16N1_LT rdi,rsi,rdx
    vzeroupper
    retq

funct dtrsm_solve_ln_m4n2
    TEST_LOAD_M4N2 8,0,rsi
    addq $128,%rdi
    addq $64,%rsi
    addq $32,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M4N2_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m4n2
    TEST_LOAD_M4N2 8,0,rsi
    salq $3,%r8
    DTRSM_SOLVE_M4N2_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m8n2
    TEST_LOAD_M8N2 8,9,0,rsi
    addq $512,%rdi
    addq $128,%rsi
    addq $64,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M8N2_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m8n2
    TEST_LOAD_M8N2 8,9,0,rsi
    salq $3,%r8
    DTRSM_SOLVE_M8N2_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m16n2
    TEST_LOAD_M8N2 8,9,0,rsi
    TEST_LOAD_M8N2 10,11,128,rsi
    addq $2048,%rdi
    addq $256,%rsi
    addq $128,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M16N2_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m16n2
    TEST_LOAD_M8N2 8,9,0,rsi
    TEST_LOAD_M8N2 10,11,128,rsi
    salq $3,%r8
    DTRSM_SOLVE_M16N2_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m1n4
    vmovupd (%rsi),%ymm4
    addq $8,%rdi
    addq $32,%rsi
    addq $8,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M1N4_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m1n4
    vmovupd (%rsi),%ymm4
    salq $3,%r8
    DTRSM_SOLVE_M1N4_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m1n8
    salq $3,%rcx
    salq $3,%r8
    vmovupd (%rsi),%ymm4
    vmovupd (%rsi,%rcx,4),%ymm5
    addq $8,%rdi
    addq $32,%rsi
    addq $8,%rdx
    DTRSM_SOLVE_M1N8_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m1n8
    salq $3,%rcx
    salq $3,%r8
    vmovupd (%rsi),%ymm4
    vmovupd (%rsi,%rcx,4),%ymm5
    DTRSM_SOLVE_M1N8_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m1n12
    salq $3,%rcx
    salq $3,%r8
    vmovupd (%rsi),%ymm4
    vmovupd (%rsi,%rcx,4),%ymm5
    vmovupd (%rsi,%rcx,8),%ymm6
    addq $8,%rdi
    addq $32,%rsi
    addq $8,%rdx
    DTRSM_SOLVE_M1N12_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m1n12
    salq $3,%rcx
    salq $3,%r8
    vmovupd (%rsi),%ymm4
    vmovupd (%rsi,%rcx,4),%ymm5
    vmovupd (%rsi,%rcx,8),%ymm6
    DTRSM_SOLVE_M1N12_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m2n4
    TEST_LOAD_M2N4 4,5,0,%rsi
    addq $32,%rdi
    addq $64,%rsi
    addq $16,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M2N4_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m2n4
    TEST_LOAD_M2N4 4,5,0,%rsi
    salq $3,%r8
    DTRSM_SOLVE_M2N4_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m2n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M2N4 4,5,0,%rsi
    TEST_LOAD_M2N4 6,7,0,%rsi,%rcx,4
    addq $32,%rdi
    addq $64,%rsi
    addq $16,%rdx
    DTRSM_SOLVE_M2N8_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m2n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M2N4 4,5,0,%rsi
    TEST_LOAD_M2N4 6,7,0,%rsi,%rcx,4
    DTRSM_SOLVE_M2N8_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m2n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M2N4 4,5,0,%rsi
    TEST_LOAD_M2N4 6,7,0,%rsi,%rcx,4
    TEST_LOAD_M2N4 8,9,0,%rsi,%rcx,8
    addq $32,%rdi
    addq $64,%rsi
    addq $16,%rdx
    DTRSM_SOLVE_M2N12_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m2n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M2N4 4,5,0,%rsi
    TEST_LOAD_M2N4 6,7,0,%rsi,%rcx,4
    TEST_LOAD_M2N4 8,9,0,%rsi,%rcx,8
    DTRSM_SOLVE_M2N12_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m4n4
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    addq $128,%rdi
    addq $128,%rsi
    addq $32,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M4N4_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m4n4
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    salq $3,%r8
    DTRSM_SOLVE_M4N4_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m4n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    TEST_LOAD_M4N4 8,9,10,11,0,%rsi,%rcx,4
    addq $128,%rdi
    addq $128,%rsi
    addq $32,%rdx
    DTRSM_SOLVE_M4N8_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m4n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    TEST_LOAD_M4N4 8,9,10,11,0,%rsi,%rcx,4
    DTRSM_SOLVE_M4N8_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m4n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    TEST_LOAD_M4N4 8,9,10,11,0,%rsi,%rcx,4
    TEST_LOAD_M4N4 12,13,14,15,0,%rsi,%rcx,8
    addq $128,%rdi
    addq $128,%rsi
    addq $32,%rdx
    DTRSM_SOLVE_M4N12_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m4n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M4N4 4,5,6,7,0,%rsi
    TEST_LOAD_M4N4 8,9,10,11,0,%rsi,%rcx,4
    TEST_LOAD_M4N4 12,13,14,15,0,%rsi,%rcx,8
    DTRSM_SOLVE_M4N12_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

/** rdi = pta, rsi = ptb, rdx = ptc, rcx = *, r8 = ldc */
funct dtrsm_solve_ln_m8n4
    TEST_LOAD_M8N4 8,9,10,11,0,%rsi
    addq $512,%rdi
    addq $256,%rsi
    addq $64,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M8N4_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m8n4
    TEST_LOAD_M8N4 8,9,10,11,0,%rsi
    salq $3,%r8
    DTRSM_SOLVE_M8N4_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

/** rdi = pta, rsi = ptb, rdx = ptc, rcx = ldb, r8 = ldc */
funct dtrsm_solve_ln_m8n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M8N4 8,9,10,11,0,%rsi
    TEST_LOAD_M8N4 12,13,14,15,0,%rsi,%rcx,4
    addq $512,%rdi
    addq $256,%rsi
    addq $64,%rdx
    DTRSM_SOLVE_M8N8_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m8n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M8N4 8,9,10,11,0,%rsi
    TEST_LOAD_M8N4 12,13,14,15,0,%rsi,%rcx,4
    DTRSM_SOLVE_M8N8_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m8n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M8N4 8,9,10,11,0,%rsi
    TEST_LOAD_M8N4 12,13,14,15,0,%rsi,%rcx,4
    TEST_LOAD_M8N4 16,17,18,19,0,%rsi,%rcx,8
    addq $512,%rdi
    addq $256,%rsi
    addq $64,%rdx
    DTRSM_SOLVE_M8N12_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m8n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M8N4 8,9,10,11,0,%rsi
    TEST_LOAD_M8N4 12,13,14,15,0,%rsi,%rcx,4
    TEST_LOAD_M8N4 16,17,18,19,0,%rsi,%rcx,8
    DTRSM_SOLVE_M8N12_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m16n4
    TEST_LOAD_M16N4 8,9,10,11,12,13,14,15,0,%rsi
    addq $2048,%rdi
    addq $512,%rsi
    addq $128,%rdx
    salq $3,%r8
    DTRSM_SOLVE_M16N4_LN rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_lt_m16n4
    TEST_LOAD_M16N4 8,9,10,11,12,13,14,15,0,%rsi
    salq $3,%r8
    DTRSM_SOLVE_M16N4_LT rdi,rsi,rdx,r8
    vzeroupper
    retq

funct dtrsm_solve_ln_m16n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M16N4 8,9,10,11,12,13,14,15,0,%rsi
    TEST_LOAD_M16N4 16,17,18,19,20,21,22,23,0,%rsi,%rcx,4
    addq $2048,%rdi
    addq $512,%rsi
    addq $128,%rdx
    DTRSM_SOLVE_M16N8_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m16n8
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M16N4 8,9,10,11,12,13,14,15,0,%rsi
    TEST_LOAD_M16N4 16,17,18,19,20,21,22,23,0,%rsi,%rcx,4
    DTRSM_SOLVE_M16N8_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_ln_m16n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M16N4 8,9,10,11,12,13,14,15,0,%rsi
    TEST_LOAD_M16N4 16,17,18,19,20,21,22,23,0,%rsi,%rcx,4
    TEST_LOAD_M16N4 24,25,26,27,28,29,30,31,0,%rsi,%rcx,8
    addq $2048,%rdi
    addq $512,%rsi
    addq $128,%rdx
    DTRSM_SOLVE_M16N12_LN rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

funct dtrsm_solve_lt_m16n12
    salq $3,%rcx
    salq $3,%r8
    TEST_LOAD_M16N4 8,9,10,11,12,13,14,15,0,%rsi
    TEST_LOAD_M16N4 16,17,18,19,20,21,22,23,0,%rsi,%rcx,4
    TEST_LOAD_M16N4 24,25,26,27,28,29,30,31,0,%rsi,%rcx,8
    DTRSM_SOLVE_M16N12_LT rdi,rsi,rdx,r8,rcx
    vzeroupper
    retq

