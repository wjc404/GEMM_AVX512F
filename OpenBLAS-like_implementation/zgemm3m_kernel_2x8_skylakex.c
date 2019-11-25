#include "common.h"
#include <stdint.h>
#include <immintrin.h>

// recommended settings: GEMM_DEFAULT_Q = 144, GEMM_DEFAULT_P = 384
// %0=a_pointer, %1=b_pointer, %2=c_pointer, %3=c_store, %4=ldc(bytes), %5=&constval, %6 = k_counter, %7 = m_counter, %8 = b_pref, %9 = k2
// const double constval[2] = {alpha_r, alpha_i};
// r11 = m; r12=16*k; r13 = k; r14 = b_head; r15=%0+3*r12.

#define GENERAL_INIT "movq %7,%%r11; movq %1,%%r14; movq %6,%%r13; movq %6,%%r12; salq $4,%%r12;"
#define GENERAL_RECOVER "movq %%r11,%7; movq %%r13,%6; movq %%r14,%1;"
#define CONSTZMM_INIT "vbroadcastf32x4 (%5),%%zmm0;"
#define COMPUTE_INIT "movq %%r13,%6; movq %%r14,%1; leaq (%0,%%r12,2),%%r15; addq %%r12,%%r15;"

/* n=8,16 && m=2,4,8,12; zmm0={alpha_r,alpha_i,...,alpha_r,alpha_i}, zmm1-zmm7 for temporary use, zmm8-zmm31 for accumulators; k2=(11001100)b=0xcc. */
/* c_block row_major; accumulators assigning: column-major */
#define unit_init_m2n8(c1,c2) "vpxorq "#c1","#c1","#c1"; vpxorq "#c2","#c2","#c2";"
#define INIT_m2n8 unit_init_m2n8(%%zmm8,%%zmm9)
#define INIT_m4n8 INIT_m2n8 unit_init_m2n8(%%zmm10,%%zmm11)
#define INIT_m8n8 INIT_m4n8 unit_init_m2n8(%%zmm12,%%zmm13) unit_init_m2n8(%%zmm14,%%zmm15)
#define INIT_m12n8 INIT_m8n8 unit_init_m2n8(%%zmm16,%%zmm17) unit_init_m2n8(%%zmm18,%%zmm19)
#define INIT_m2n16 INIT_m4n8
#define INIT_m4n16 INIT_m8n8
#define INIT_m8n16 INIT_m12n8 unit_init_m2n8(%%zmm20,%%zmm21) unit_init_m2n8(%%zmm22,%%zmm23)
#define INIT_m12n16 INIT_m8n16 \
    unit_init_m2n8(%%zmm24,%%zmm25) unit_init_m2n8(%%zmm26,%%zmm27) unit_init_m2n8(%%zmm28,%%zmm29) unit_init_m2n8(%%zmm30,%%zmm31)
#define unit_kernel_n8_loadb "vmovupd (%1),%%zmm1; addq $64,%1;"
#define unit_kernel_n16_loadb "vmovupd (%1),%%zmm1; vmovupd (%1,%%r12,4),%%zmm2; addq $64,%1;"
#define unit_kernel_m1n8_acc(c1,a_off,...) "vbroadcastsd "#a_off"("#__VA_ARGS__"),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,"#c1";"
#define unit_kernel_m1n16_acc(c1,c2,a_off,...) unit_kernel_m1n8_acc(c1,a_off,__VA_ARGS__) "vfmadd231pd %%zmm2,%%zmm3,"#c2";"
#define unit_kernel_m2n8_acc(c1,c2,...) unit_kernel_m1n8_acc(c1,0,__VA_ARGS__) unit_kernel_m1n8_acc(c2,8,__VA_ARGS__)
#define unit_kernel_m2n16_acc(c1,c2,c3,c4,...) unit_kernel_m1n16_acc(c1,c3,0,__VA_ARGS__) unit_kernel_m1n16_acc(c2,c4,8,__VA_ARGS__)
#define KERNEL_h_k1m2n8 unit_kernel_n8_loadb unit_kernel_m2n8_acc(%%zmm8,%%zmm9,%0)
#define KERNEL_t_k1m2n8 KERNEL_h_k1m2n8 "addq $16,%0;"
#define KERNEL_h_k1m4n8 KERNEL_h_k1m2n8 unit_kernel_m2n8_acc(%%zmm10,%%zmm11,%0,%%r12,1)
#define KERNEL_t_k1m4n8 KERNEL_h_k1m4n8 "addq $16,%0;"
#define KERNEL_h_k1m8n8 KERNEL_h_k1m4n8 unit_kernel_m2n8_acc(%%zmm12,%%zmm13,%0,%%r12,2) "addq $16,%0;" unit_kernel_m2n8_acc(%%zmm14,%%zmm15,%%r15)
#define KERNEL_t_k1m8n8 KERNEL_h_k1m8n8 "addq $16,%%r15;"
#define KERNEL_t_k1m12n8 \
    KERNEL_h_k1m8n8 unit_kernel_m2n8_acc(%%zmm16,%%zmm17,%%r15,%%r12,1) unit_kernel_m2n8_acc(%%zmm18,%%zmm19,%%r15,%%r12,2) "addq $16,%%r15;"
#define KERNEL_t_k1m2n16 unit_kernel_n16_loadb unit_kernel_m2n16_acc(%%zmm8,%%zmm9,%%zmm10,%%zmm11,%0) "addq $16,%0;"
#define KERNEL_t_k1m4n16 unit_kernel_n16_loadb \
    unit_kernel_m2n16_acc(%%zmm8,%%zmm9,%%zmm12,%%zmm13,%0)\
    unit_kernel_m2n16_acc(%%zmm10,%%zmm11,%%zmm14,%%zmm15,%0,%%r12,1)  "addq $16,%0;"
#define KERNEL_t_k1m8n16 unit_kernel_n16_loadb \
    unit_kernel_m2n16_acc(%%zmm8,%%zmm9,%%zmm16,%%zmm17,%0)\
    unit_kernel_m2n16_acc(%%zmm10,%%zmm11,%%zmm18,%%zmm19,%0,%%r12,1)\
    unit_kernel_m2n16_acc(%%zmm12,%%zmm13,%%zmm20,%%zmm21,%0,%%r12,2)  "addq $16,%0;"\
    unit_kernel_m2n16_acc(%%zmm14,%%zmm15,%%zmm22,%%zmm23,%%r15)  "addq $16,%%r15;"
#define KERNEL_t_k1m12n16 unit_kernel_n16_loadb \
    unit_kernel_m2n16_acc(%%zmm8,%%zmm9,%%zmm20,%%zmm21,%0)\
    unit_kernel_m2n16_acc(%%zmm10,%%zmm11,%%zmm22,%%zmm23,%0,%%r12,1)\
    unit_kernel_m2n16_acc(%%zmm12,%%zmm13,%%zmm24,%%zmm25,%0,%%r12,2)  "addq $16,%0;"\
    unit_kernel_m2n16_acc(%%zmm14,%%zmm15,%%zmm26,%%zmm27,%%r15)\
    unit_kernel_m2n16_acc(%%zmm16,%%zmm17,%%zmm28,%%zmm29,%%r15,%%r12,1)\
    unit_kernel_m2n16_acc(%%zmm18,%%zmm19,%%zmm30,%%zmm31,%%r15,%%r12,2)  "addq $16,%%r15;"
#define unit_trans3m_m2n8(c1,c2) \
    "vblendmpd "#c1","#c2",%%zmm1%{%9%}; vshuff64x2 $177,%%zmm1,%%zmm1,%%zmm1;"\
    "vblendmpd %%zmm1,"#c1","#c1"%{%9%}; vblendmpd "#c2",%%zmm1,"#c2"%{%9%};"
#define unit_trans3m_m4n8(c1,c2,c3,c4) \
    unit_trans3m_m2n8(c1,c2)  unit_trans3m_m2n8(c3,c4)\
    "vshuff64x2 $68,"#c3","#c1",%%zmm1; vshuff64x2 $238,"#c3","#c1","#c3"; vmovapd %%zmm1,"#c1";"\
    "vshuff64x2 $68,"#c4","#c2",%%zmm1; vshuff64x2 $238,"#c4","#c2","#c4"; vmovapd %%zmm1,"#c2";"
#define unit_store_m4n2(c1,off) \
    "vunpcklpd "#c1","#c1",%%zmm1; vunpckhpd "#c1","#c1",%%zmm2;"\
    "vfmadd213pd "#off"(%3),%%zmm0,%%zmm1; vmovupd %%zmm1,"#off"(%3);"\
    "vfmadd213pd "#off"(%3,%4,1),%%zmm0,%%zmm2; vmovupd %%zmm2,"#off"(%3,%4,1);"
#define unit_store_m2_n2n2(c1) \
    "vunpcklpd "#c1","#c1",%%zmm1; vunpckhpd "#c1","#c1",%%zmm2;"\
    "vmovupd (%3),%%ymm3; vinsertf64x4 $1,(%3,%4,4),%%zmm3,%%zmm3; vfmadd213pd %%zmm3,%%zmm0,%%zmm1;"\
    "vmovupd %%ymm1,(%3); vextractf64x4 $1,%%zmm1,(%3,%4,4); addq %4,%3;"\
    "vmovupd (%3),%%ymm3; vinsertf64x4 $1,(%3,%4,4),%%zmm3,%%zmm3; vfmadd213pd %%zmm3,%%zmm0,%%zmm2;"\
    "vmovupd %%ymm2,(%3); vextractf64x4 $1,%%zmm2,(%3,%4,4); addq %4,%3;"
#define save_start_m2 "movq %2,%3; addq $32,%2;"
#define SAVE_m2n8 save_start_m2 \
    unit_trans3m_m2n8(%%zmm8,%%zmm9)  unit_store_m2_n2n2(%%zmm8) unit_store_m2_n2n2(%%zmm9)
#define SAVE_m2n16 SAVE_m2n8 "leaq (%3,%4,4),%3;" \
    unit_trans3m_m2n8(%%zmm10,%%zmm11)  unit_store_m2_n2n2(%%zmm10) unit_store_m2_n2n2(%%zmm11)
#define save_start_m4 "movq %2,%3; addq $64,%2;"
#define unit_save_m4n8(c1,c2,c3,c4) \
    unit_trans3m_m4n8(c1,c2,c3,c4)\
    unit_store_m4n2(c1,0) "leaq (%3,%4,2),%3;" unit_store_m4n2(c2,0) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c3,0) "leaq (%3,%4,2),%3;" unit_store_m4n2(c4,0) "leaq (%3,%4,2),%3;"
#define SAVE_m4n8 save_start_m4 unit_save_m4n8(%%zmm8,%%zmm9,%%zmm10,%%zmm11)
#define SAVE_m4n16 SAVE_m4n8 unit_save_m4n8(%%zmm12,%%zmm13,%%zmm14,%%zmm15)
#define save_start_m8 "movq %2,%3; addq $128,%2;"
#define unit_save_m8n8(c1,c2,c3,c4,c5,c6,c7,c8) \
    unit_trans3m_m4n8(c1,c2,c3,c4)\
    unit_trans3m_m4n8(c5,c6,c7,c8)\
    unit_store_m4n2(c1,0) unit_store_m4n2(c5,64) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c2,0) unit_store_m4n2(c6,64) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c3,0) unit_store_m4n2(c7,64) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c4,0) unit_store_m4n2(c8,64) "leaq (%3,%4,2),%3;"
#define SAVE_m8n8 save_start_m8 unit_save_m8n8(%%zmm8,%%zmm9,%%zmm10,%%zmm11,%%zmm12,%%zmm13,%%zmm14,%%zmm15)
#define SAVE_m8n16 SAVE_m8n8 unit_save_m8n8(%%zmm16,%%zmm17,%%zmm18,%%zmm19,%%zmm20,%%zmm21,%%zmm22,%%zmm23)
#define save_start_m12 "movq %2,%3; addq $192,%2;"
#define unit_save_m12n8(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12) \
    unit_trans3m_m4n8(c1,c2,c3,c4)\
    unit_trans3m_m4n8(c5,c6,c7,c8)\
    unit_trans3m_m4n8(c9,c10,c11,c12)\
    unit_store_m4n2(c1,0) unit_store_m4n2(c5,64) unit_store_m4n2(c9,128) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c2,0) unit_store_m4n2(c6,64) unit_store_m4n2(c10,128) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c3,0) unit_store_m4n2(c7,64) unit_store_m4n2(c11,128) "leaq (%3,%4,2),%3;"\
    unit_store_m4n2(c4,0) unit_store_m4n2(c8,64) unit_store_m4n2(c12,128) "leaq (%3,%4,2),%3;"
#define SAVE_m12n8 save_start_m12 \
    unit_save_m12n8(%%zmm8,%%zmm9,%%zmm10,%%zmm11,%%zmm12,%%zmm13,%%zmm14,%%zmm15,%%zmm16,%%zmm17,%%zmm18,%%zmm19)
#define SAVE_m12n16 SAVE_m12n8 \
    unit_save_m12n8(%%zmm20,%%zmm21,%%zmm22,%%zmm23,%%zmm24,%%zmm25,%%zmm26,%%zmm27,%%zmm28,%%zmm29,%%zmm30,%%zmm31)

/* n=4 && m=2,4,8,12; ymm0={alpha_r,alpha_i,alpha_r,alpha_i}, ymm1-ymm3 for temporary use, ymm4-ymm15 for accumulators; */
/* c_block row_major; accumulators assigning: column-major */
#define unit_init_m2n4(c1,c2) "vpxor "#c1","#c1","#c1"; vpxor "#c2","#c2","#c2";"
#define INIT_m2n4 unit_init_m2n4(%%ymm4,%%ymm5)
#define INIT_m4n4 INIT_m2n4 unit_init_m2n4(%%ymm6,%%ymm7)
#define INIT_m8n4 INIT_m4n4 unit_init_m2n4(%%ymm8,%%ymm9) unit_init_m2n4(%%ymm10,%%ymm11)
#define INIT_m12n4 INIT_m8n4 unit_init_m2n4(%%ymm12,%%ymm13) unit_init_m2n4(%%ymm14,%%ymm15)
#define unit_kernel_n4_loadb "vmovupd (%1),%%ymm1; addq $32,%1;"
#define unit_kernel_m1n4_acc(c1,a_off,...) "vbroadcastsd "#a_off"("#__VA_ARGS__"),%%ymm2; vfmadd231pd %%ymm1,%%ymm2,"#c1";"
#define unit_kernel_m2n4_acc(c1,c2,...) unit_kernel_m1n4_acc(c1,0,__VA_ARGS__) unit_kernel_m1n4_acc(c2,8,__VA_ARGS__)
#define KERNEL_h_k1m2n4 unit_kernel_n4_loadb unit_kernel_m2n4_acc(%%ymm4,%%ymm5,%0)
#define KERNEL_t_k1m2n4 KERNEL_h_k1m2n4 "addq $16,%0;"
#define KERNEL_h_k1m4n4 KERNEL_h_k1m2n4 unit_kernel_m2n4_acc(%%ymm6,%%ymm7,%0,%%r12,1)
#define KERNEL_t_k1m4n4 KERNEL_h_k1m4n4 "addq $16,%0;"
#define KERNEL_h_k1m8n4 KERNEL_h_k1m4n4 unit_kernel_m2n4_acc(%%ymm8,%%ymm9,%0,%%r12,2) "addq $16,%0;" unit_kernel_m2n4_acc(%%ymm10,%%ymm11,%%r15)
#define KERNEL_t_k1m8n4 KERNEL_h_k1m8n4 "addq $16,%%r15;"
#define KERNEL_t_k1m12n4 KERNEL_h_k1m8n4 \
    unit_kernel_m2n4_acc(%%ymm12,%%ymm13,%%r15,%%r12,1) unit_kernel_m2n4_acc(%%ymm14,%%ymm15,%%r15,%%r12,2) "addq $16,%%r15;"
#define unit_trans3m_m2n4(c1,c2) "vperm2f128 $2,"#c1","#c2",%%ymm1; vperm2f128 $19,"#c1","#c2","#c2"; vmovapd %%ymm1,"#c1";"
#define unit_store_m2n2(c1,off) \
    "vunpcklpd "#c1","#c1",%%ymm1; vunpckhpd "#c1","#c1",%%ymm2;"\
    "vfmadd213pd "#off"(%3),%%ymm0,%%ymm1; vmovupd %%ymm1,"#off"(%3);"\
    "vfmadd213pd "#off"(%3,%4,1),%%ymm0,%%ymm2; vmovupd %%ymm2,"#off"(%3,%4,1);"
#define SAVE_m2n4 save_start_m2 \
    unit_trans3m_m2n4(%%ymm4,%%ymm5)\
    unit_store_m2n2(%%ymm4,0) "leaq (%3,%4,2),%3;" unit_store_m2n2(%%ymm5,0)
#define SAVE_m4n4 save_start_m4 \
    unit_trans3m_m2n4(%%ymm4,%%ymm5)\
    unit_trans3m_m2n4(%%ymm6,%%ymm7)\
    unit_store_m2n2(%%ymm4,0) unit_store_m2n2(%%ymm6,32) "leaq (%3,%4,2),%3;"\
    unit_store_m2n2(%%ymm5,0) unit_store_m2n2(%%ymm7,32)
#define SAVE_m8n4 save_start_m8 \
    unit_trans3m_m2n4(%%ymm4,%%ymm5)\
    unit_trans3m_m2n4(%%ymm6,%%ymm7)\
    unit_trans3m_m2n4(%%ymm8,%%ymm9)\
    unit_trans3m_m2n4(%%ymm10,%%ymm11)\
    unit_store_m2n2(%%ymm4,0) unit_store_m2n2(%%ymm6,32) unit_store_m2n2(%%ymm8,64) unit_store_m2n2(%%ymm10,96) "leaq (%3,%4,2),%3;"\
    unit_store_m2n2(%%ymm5,0) unit_store_m2n2(%%ymm7,32) unit_store_m2n2(%%ymm9,64) unit_store_m2n2(%%ymm11,96)
#define SAVE_m12n4 save_start_m12 \
    unit_trans3m_m2n4(%%ymm4,%%ymm5)\
    unit_trans3m_m2n4(%%ymm6,%%ymm7)\
    unit_trans3m_m2n4(%%ymm8,%%ymm9)\
    unit_trans3m_m2n4(%%ymm10,%%ymm11)\
    unit_trans3m_m2n4(%%ymm12,%%ymm13)\
    unit_trans3m_m2n4(%%ymm14,%%ymm15)\
    unit_store_m2n2(%%ymm4,0) unit_store_m2n2(%%ymm6,32) unit_store_m2n2(%%ymm8,64)\
    unit_store_m2n2(%%ymm10,96) unit_store_m2n2(%%ymm12,128) unit_store_m2n2(%%ymm14,160) "leaq (%3,%4,2),%3;"\
    unit_store_m2n2(%%ymm5,0) unit_store_m2n2(%%ymm7,32) unit_store_m2n2(%%ymm9,64)\
    unit_store_m2n2(%%ymm11,96) unit_store_m2n2(%%ymm13,128) unit_store_m2n2(%%ymm15,160)

/* n=2 && m=2,4,8,12; xmm0={alpha_r,alpha_i}, xmm1-xmm3 for temporary use, xmm4-xmm15 for accumulators; */
/* c_block row-major; accumulators assigning: column-major */
#define unit_init_m2n2(c1,c2) unit_init_m2n4(c1,c2)
#define INIT_m2n2 unit_init_m2n2(%%xmm4,%%xmm5)
#define INIT_m4n2 INIT_m2n2 unit_init_m2n2(%%xmm6,%%xmm7)
#define INIT_m8n2 INIT_m4n2 unit_init_m2n2(%%xmm8,%%xmm9) unit_init_m2n2(%%xmm10,%%xmm11)
#define INIT_m12n2 INIT_m8n2 unit_init_m2n2(%%xmm12,%%xmm13) unit_init_m2n2(%%xmm14,%%xmm15)
#define unit_kernel_n2_loadb "vmovupd (%1),%%xmm1; addq $16,%1;"
#define unit_kernel_m1n2_acc(c1,a_off,...) "vmovddup "#a_off"("#__VA_ARGS__"),%%xmm2; vfmadd231pd %%xmm1,%%xmm2,"#c1";"
#define unit_kernel_m2n2_acc(c1,c2,...) unit_kernel_m1n2_acc(c1,0,__VA_ARGS__) unit_kernel_m1n2_acc(c2,8,__VA_ARGS__)
#define KERNEL_h_k1m2n2 unit_kernel_n2_loadb unit_kernel_m2n2_acc(%%xmm4,%%xmm5,%0)
#define KERNEL_t_k1m2n2 KERNEL_h_k1m2n2 "addq $16,%0;"
#define KERNEL_h_k1m4n2 KERNEL_h_k1m2n2 unit_kernel_m2n2_acc(%%xmm6,%%xmm7,%0,%%r12,1)
#define KERNEL_t_k1m4n2 KERNEL_h_k1m4n2 "addq $16,%0;"
#define KERNEL_h_k1m8n2 KERNEL_h_k1m4n2 unit_kernel_m2n2_acc(%%xmm8,%%xmm9,%0,%%r12,2) "addq $16,%0;" unit_kernel_m2n2_acc(%%xmm10,%%xmm11,%%r15)
#define KERNEL_t_k1m8n2 KERNEL_h_k1m8n2 "addq $16,%%r15;"
#define KERNEL_t_k1m12n2 KERNEL_h_k1m8n2 \
    unit_kernel_m2n2_acc(%%xmm12,%%xmm13,%%r15,%%r12,1) unit_kernel_m2n2_acc(%%xmm14,%%xmm15,%%r15,%%r12,2) "addq $16,%%r15;"
#define unit_store_xmm_m2n2(c1,c2,off) \
    "vmovapd "#c1",%%xmm3; vinsertf128 $1,"#c2",%%ymm3,%%ymm3;" unit_store_m2n2(%%ymm3,off)
#define SAVE_m2n2 save_start_m2 unit_store_xmm_m2n2(%%xmm4,%%xmm5,0)
#define SAVE_m4n2 save_start_m4 unit_store_xmm_m2n2(%%xmm4,%%xmm5,0) unit_store_xmm_m2n2(%%xmm6,%%xmm7,32)
#define SAVE_m8n2 save_start_m8 \
    unit_store_xmm_m2n2(%%xmm4,%%xmm5,0) unit_store_xmm_m2n2(%%xmm6,%%xmm7,32)\
    unit_store_xmm_m2n2(%%xmm8,%%xmm9,64) unit_store_xmm_m2n2(%%xmm10,%%xmm11,96)
#define SAVE_m12n2 save_start_m12 \
    unit_store_xmm_m2n2(%%xmm4,%%xmm5,0) unit_store_xmm_m2n2(%%xmm6,%%xmm7,32)\
    unit_store_xmm_m2n2(%%xmm8,%%xmm9,64) unit_store_xmm_m2n2(%%xmm10,%%xmm11,96)\
    unit_store_xmm_m2n2(%%xmm12,%%xmm13,128) unit_store_xmm_m2n2(%%xmm14,%%xmm15,160)

/* m=2,4,8,12 && n=1; ymm0={alpha_r,alpha_i,alpha_r,alpha_i}, xmm1-xmm3 for temporary use, xmm4-xmm9 for accumulators; */
/* c_block column_major; accumulators assigning: column-major */
#define unit_init_m2n1(c1) "vpxor "#c1","#c1","#c1";"
#define INIT_m2n1 unit_init_m2n1(%%xmm4)
#define INIT_m4n1 INIT_m2n1 unit_init_m2n1(%%xmm5)
#define INIT_m8n1 INIT_m4n1 unit_init_m2n1(%%xmm6) unit_init_m2n1(%%xmm7)
#define INIT_m12n1 INIT_m8n1 unit_init_m2n1(%%xmm8) unit_init_m2n1(%%xmm9)
#define unit_kernel_n1_loadb "vmovddup (%1),%%xmm1; addq $8,%1;"
#define unit_kernel_m1n1_acc(c1,...) "vfmadd231sd ("#__VA_ARGS__"),%%xmm1,"#c1";"
#define unit_kernel_m2n1_acc(c1,...) "vfmadd231pd ("#__VA_ARGS__"),%%xmm1,"#c1";"
#define KERNEL_h_k1m2n1 unit_kernel_n1_loadb unit_kernel_m2n1_acc(%%xmm4,%0)
#define KERNEL_t_k1m2n1 KERNEL_h_k1m2n1 "addq $16,%0;"
#define KERNEL_h_k1m4n1 KERNEL_h_k1m2n1 unit_kernel_m2n1_acc(%%xmm5,%0,%%r12,1)
#define KERNEL_t_k1m4n1 KERNEL_h_k1m4n1 "addq $16,%0;"
#define KERNEL_h_k1m8n1 KERNEL_h_k1m4n1 unit_kernel_m2n1_acc(%%xmm6,%0,%%r12,2) "addq $16,%0;" unit_kernel_m2n1_acc(%%xmm7,%%r15)
#define KERNEL_t_k1m8n1 KERNEL_h_k1m8n1 "addq $16,%%r15;"
#define KERNEL_t_k1m12n1 KERNEL_h_k1m8n1 \
    unit_kernel_m2n1_acc(%%xmm8,%%r15,%%r12,1) unit_kernel_m2n1_acc(%%xmm9,%%r15,%%r12,2) "addq $16,%%r15;"
#define unit_store_m2n1(c1,off) \
    "vmovapd "#c1",%%xmm1; vinsertf128 $1,"#c1",%%ymm1,%%ymm1; vpermilpd $12,%%ymm1,%%ymm1;"\
    "vfmadd213pd "#off"(%3),%%ymm0,%%ymm1; vmovupd %%ymm1,"#off"(%3);"
#define SAVE_m2n1 save_start_m2 unit_store_m2n1(%%xmm4,0)
#define SAVE_m4n1 save_start_m4 unit_store_m2n1(%%xmm4,0) unit_store_m2n1(%%xmm5,32)
#define SAVE_m8n1 save_start_m8 \
    unit_store_m2n1(%%xmm4,0) unit_store_m2n1(%%xmm5,32) unit_store_m2n1(%%xmm6,64) unit_store_m2n1(%%xmm7,96)
#define SAVE_m12n1 save_start_m12 \
    unit_store_m2n1(%%xmm4,0) unit_store_m2n1(%%xmm5,32) unit_store_m2n1(%%xmm6,64)\
    unit_store_m2n1(%%xmm7,96) unit_store_m2n1(%%xmm8,128) unit_store_m2n1(%%xmm9,160)

/* m=1 && n=1,2,4,8,16; zmm0={alpha_r,alpha_i,...,alpha_r,alpha_i}, vmm1-vmm3 and vmm6-vmm15 for temporary use, vmm4-vmm5 for accumulators; */
/* c_block row_major; accumulators assigning: row-major */
#define INIT_m1n1 INIT_m1n2
#define INIT_m1n2 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m1n4 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define INIT_m1n8 "vpxorq %%zmm4,%%zmm4,%%zmm4;"
#define INIT_m1n16 unit_init_m2n8(%%zmm4,%%zmm5)
#define KERNEL_t_k1m1n1 unit_kernel_n1_loadb unit_kernel_m1n1_acc(%%xmm4,%0) "addq $8,%0;"
#define KERNEL_t_k1m1n2 unit_kernel_n2_loadb unit_kernel_m1n2_acc(%%xmm4,0,%0) "addq $8,%0;"
#define KERNEL_t_k1m1n4 unit_kernel_n4_loadb unit_kernel_m1n4_acc(%%ymm4,0,%0) "addq $8,%0;"
#define KERNEL_t_k1m1n8 unit_kernel_n8_loadb unit_kernel_m1n8_acc(%%zmm4,0,%0) "addq $8,%0;"
#define KERNEL_t_k1m1n16 unit_kernel_n16_loadb unit_kernel_m1n16_acc(%%zmm4,%%zmm5,0,%0) "addq $8,%0;"
#define save_start_m1 "movq %2,%3; addq $16,%2;"
#define SAVE_m1n1 save_start_m1 \
    "vmovddup %%xmm4,%%xmm4; vfmadd213pd (%3),%%xmm0,%%xmm4; vmovupd %%xmm4,(%3);"
#define SAVE_m1n2 save_start_m1 \
    "vunpcklpd %%xmm4,%%xmm4,%%xmm1; vunpckhpd %%xmm4,%%xmm4,%%xmm2;"\
    "vfmadd213pd (%3),%%xmm0,%%xmm1; vmovupd %%xmm1,(%3);"\
    "vfmadd213pd (%3,%4,1),%%xmm0,%%xmm2; vmovupd %%xmm2,(%3,%4,1);"
#define unit_store_m1n4(c1) \
    "vunpcklpd "#c1","#c1",%%ymm1; vunpckhpd "#c1","#c1",%%ymm2;"\
    "vmovupd (%3),%%xmm3; vinsertf128 $1,(%3,%4,2),%%ymm3,%%ymm3; vfmadd213pd %%ymm3,%%ymm0,%%ymm1;"\
    "vmovupd %%xmm1,(%3); vextractf128 $1,%%ymm1,(%3,%4,2); addq %4,%3;"\
    "vmovupd (%3),%%xmm3; vinsertf128 $1,(%3,%4,2),%%ymm3,%%ymm3; vfmadd213pd %%ymm3,%%ymm0,%%ymm2;"\
    "vmovupd %%xmm2,(%3); vextractf128 $1,%%ymm2,(%3,%4,2); addq %4,%3; leaq (%3,%4,2),%3;"
#define SAVE_m1n4 save_start_m1 unit_store_m1n4(%%ymm4)
#define unit_store_m1n8(c1) \
    "vextractf64x4 $0,"#c1",%%ymm6; vextractf64x4 $1,"#c1",%%ymm7;" unit_store_m1n4(%%ymm6) unit_store_m1n4(%%ymm7)
#define SAVE_m1n8 save_start_m1 unit_store_m1n8(%%zmm4)
#define SAVE_m1n16 SAVE_m1n8 unit_store_m1n8(%%zmm5)

/* modules for computing a block of C */
#define COMPUTE_m1(ndim) \
    INIT_m1n##ndim COMPUTE_INIT\
    #ndim"88110:\n\t"\
    "testq %6,%6; jz "#ndim"88111f;"\
    KERNEL_t_k1m1n##ndim\
    "decq %6; jmp "#ndim"88110b;"\
    #ndim"88111:\n\t"\
    SAVE_m1n##ndim
#define COMPUTE_m2(ndim) \
    INIT_m2n##ndim COMPUTE_INIT\
    #ndim"88220:\n\t"\
    "testq %6,%6; jz "#ndim"88221f;"\
    KERNEL_t_k1m2n##ndim\
    "decq %6; jmp "#ndim"88220b;"\
    #ndim"88221:\n\t"\
    SAVE_m2n##ndim
#define COMPUTE_m4(ndim) \
    INIT_m4n##ndim COMPUTE_INIT\
    #ndim"88440:\n\t"\
    "testq %6,%6; jz "#ndim"88441f;"\
    KERNEL_t_k1m4n##ndim\
    "decq %6; jmp "#ndim"88440b;"\
    #ndim"88441:\n\t"\
    "addq %%r12,%0;"\
    SAVE_m4n##ndim
#define COMPUTE_m8(ndim) \
    INIT_m8n##ndim COMPUTE_INIT\
    #ndim"88880:\n\t"\
    "testq %6,%6; jz "#ndim"88881f;"\
    KERNEL_t_k1m8n##ndim\
    "decq %6; jmp "#ndim"88880b;"\
    #ndim"88881:\n\t"\
    "addq %%r12,%0;leaq (%0,%%r12,2),%0;"\
    SAVE_m8n##ndim
#define COMPUTE_m12(ndim) \
    INIT_m12n##ndim\
    COMPUTE_INIT "movq %2,%3;"\
    "cmpq $20,%6; jb "#ndim"88990f;"\
    #ndim"88999:\n\t"\
    KERNEL_t_k1m12n##ndim\
    KERNEL_t_k1m12n##ndim\
    "prefetcht1 (%3); prefetcht1 64(%3); prefetcht1 128(%3); prefetcht1 191(%3); addq %4,%3;"\
    KERNEL_t_k1m12n##ndim\
    KERNEL_t_k1m12n##ndim\
    "prefetcht1 (%8); addq $24,%8;"\
    "subq $4,%6; cmpq $20,%6; jnb "#ndim"88999b;"\
    "movq %2,%3;"\
    #ndim"88990:\n\t"\
    "testq %6,%6; jz "#ndim"88991f;"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3); prefetcht0 191(%3); addq %4,%3;"\
    KERNEL_t_k1m12n##ndim\
    "decq %6; jmp "#ndim"88990b;"\
    #ndim"88991:\n\t"\
    "addq %%r12,%0;leaq (%0,%%r12,4),%0;"\
    SAVE_m12n##ndim
#define COMPUTE(ndim) {\
    b_pref = b_pointer + ndim * K;\
    __asm__ __volatile__(\
    GENERAL_INIT\
    CONSTZMM_INIT\
    "cmpq $12,%7;jb 33101"#ndim"f;"\
    "33109"#ndim":\n\t"\
    COMPUTE_m12(ndim)\
    "subq $12,%7;cmpq $12,%7;jnb 33109"#ndim"b;"\
    "33101"#ndim":\n\t"\
    "cmpq $8,%7;jb 33102"#ndim"f;"\
    COMPUTE_m8(ndim) "subq $8,%7;"\
    "33102"#ndim":\n\t"\
    "cmpq $4,%7;jb 33103"#ndim"f;"\
    COMPUTE_m4(ndim) "subq $4,%7;"\
    "33103"#ndim":\n\t"\
    "cmpq $2,%7;jb 33104"#ndim"f;"\
    COMPUTE_m2(ndim) "subq $2,%7;"\
    "33104"#ndim":\n\t"\
    "testq %7,%7;jz 33105"#ndim"f;"\
    COMPUTE_m1(ndim)\
    "33105"#ndim":\n\t"\
    GENERAL_RECOVER\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(c_store),"+r"(ldc_in_bytes),"+r"(constval),"+r"(K),"+r"(M),"+r"(b_pref),"+Yk"(k2)\
    ::"r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14",\
    "zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",\
    "cc","memory");\
    a_pointer -= M * K; b_pointer += ndim * K; c_pointer += (LDC * ndim - M) * 2;\
}

int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, double alphar, double alphai, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, BLASLONG LDC)
{
    if(m==0||n==0||k==0) return 0;
    int64_t ldc_in_bytes = (int64_t)LDC * sizeof(double) * 2; double const_val[2] = {alphar, alphai};
    int64_t M = (int64_t)m, K = (int64_t)k;
    __mmask16 k2 = 0xcc;
    BLASLONG n_count = n;
    double *a_pointer = A,*b_pointer = B,*c_pointer = C,*c_store = C,*constval = const_val,*b_pref = B;
    for(;n_count>15;n_count-=16) COMPUTE(16)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
    return 0;
}
