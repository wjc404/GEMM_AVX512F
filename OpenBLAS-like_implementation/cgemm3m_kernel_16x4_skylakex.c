/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 for k_count, %5 for c_store */
/* r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const), r15 = %1 + 3r12 */

#include "common.h"
#include <stdint.h>

/* m = 16 */ /* zmm8-zmm31 for accumulators, zmm3-zmm7 for temporary use, zmm0 for alpha, zmm1-zmm2 for control words of permutations */
#define KERNEL_k1m16n1 \
    "vmovups (%0),%%zmm4; addq $64,%0;"\
    "vbroadcastss (%1),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,%%zmm8;"\
    "addq $4,%1;"
#define KERNEL_h_k1m16n2 \
    "vmovsldup (%0),%%zmm4; vmovshdup (%0),%%zmm5; prefetcht0 512(%0); addq $64,%0;"\
    "vbroadcastsd (%1),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,%%zmm8; vfmadd231ps %%zmm5,%%zmm6,%%zmm9;"
#define KERNEL_k1m16n2 KERNEL_h_k1m16n2 "addq $8,%1;"
#define KERNEL_h_k1m16n4 KERNEL_h_k1m16n2 "vbroadcastsd 8(%1),%%zmm7; vfmadd231ps %%zmm4,%%zmm7,%%zmm10; vfmadd231ps %%zmm5,%%zmm7,%%zmm11;"
#define KERNEL_k1m16n4 KERNEL_h_k1m16n4 "addq $16,%1;"
#define unit_kernel_k1m16n4(c1,c2,c3,c4, ...) \
    "vbroadcastsd  ("#__VA_ARGS__"),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,"#c1"; vfmadd231ps %%zmm5,%%zmm6,"#c2";"\
    "vbroadcastsd 8("#__VA_ARGS__"),%%zmm7; vfmadd231ps %%zmm4,%%zmm7,"#c3"; vfmadd231ps %%zmm5,%%zmm7,"#c4";"
#define KERNEL_h_k1m16n8 KERNEL_h_k1m16n4 unit_kernel_k1m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15,%1,%%r12,1)
#define KERNEL_k1m16n8 KERNEL_h_k1m16n8 "addq $16,%1;"
#define KERNEL_h_k1m16n12 KERNEL_h_k1m16n8 unit_kernel_k1m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19,%1,%%r12,2)
#define KERNEL_k1m16n12 KERNEL_h_k1m16n12 "addq $16,%1;"
#define KERNEL_h_k1m16n16 KERNEL_k1m16n12 unit_kernel_k1m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23,%%r15)
#define KERNEL_k1m16n16 KERNEL_h_k1m16n16 "addq $16,%%r15;"
#define KERNEL_h_k1m16n20 KERNEL_h_k1m16n16 unit_kernel_k1m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27,%%r15,%%r12,1)
#define KERNEL_k1m16n20 KERNEL_h_k1m16n20 "addq $16,%%r15;"
#define KERNEL_h_k1m16n24 KERNEL_h_k1m16n20 unit_kernel_k1m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31,%%r15,%%r12,2)
#define KERNEL_k1m16n24 KERNEL_h_k1m16n24 "addq $16,%%r15;"
#define INIT_m16n1 "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define INIT_m16n2 INIT_m16n1 "vpxorq %%zmm9,%%zmm9,%%zmm9;"
#define INIT_m16n4 INIT_m16n2 "vpxorq %%zmm10,%%zmm10,%%zmm10;vpxorq %%zmm11,%%zmm11,%%zmm11;"
#define unit_init_m16n4(c1,c2,c3,c4) \
    "vpxorq "#c1","#c1","#c1";vpxorq "#c2","#c2","#c2";vpxorq "#c3","#c3","#c3";vpxorq "#c4","#c4","#c4";"
#define INIT_m16n8 INIT_m16n4 unit_init_m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15)
#define INIT_m16n12 INIT_m16n8 unit_init_m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19)
#define INIT_m16n16 INIT_m16n12 unit_init_m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23)
#define INIT_m16n20 INIT_m16n16 unit_init_m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27)
#define INIT_m16n24 INIT_m16n20 unit_init_m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31)
#define SAVE_h_m16n1 \
    "vmovsldup %%zmm8,%%zmm3; vmovshdup %%zmm8,%%zmm4; vmovaps %%zmm3,%%zmm5;"\
    "vpermt2pd %%zmm4,%%zmm1,%%zmm3; vpermt2pd %%zmm5,%%zmm2,%%zmm4;"\
    "vfmadd213ps (%2),%%zmm0,%%zmm3; vmovups %%zmm3,(%2);"\
    "vfmadd213ps 64(%2),%%zmm0,%%zmm4; vmovups %%zmm4,64(%2);"
#define unit_save_m16n2(c1,c2) \
    "vmovapd "#c1",%%zmm4; vpermt2pd "#c2",%%zmm1,"#c1"; vpermt2pd %%zmm4,%%zmm2,"#c2";"\
    "vmovsldup "#c1",%%zmm4; vmovsldup "#c2",%%zmm5; vmovshdup "#c1",%%zmm6; vmovshdup "#c2",%%zmm7;"\
    "vfmadd213ps (%5),%%zmm0,%%zmm4;vfmadd213ps 64(%5),%%zmm0,%%zmm5;vfmadd213ps (%5,%3,1),%%zmm0,%%zmm6;vfmadd213ps 64(%5,%3,1),%%zmm0,%%zmm7;"\
    "vmovups %%zmm4,(%5); vmovups %%zmm5,64(%5); vmovups %%zmm6,(%5,%3,1); vmovups %%zmm7,64(%5,%3,1); leaq (%5,%3,2),%5;"
#define SAVE_h_m16n2 "movq %2,%5;" unit_save_m16n2(%%zmm8,%%zmm9)
#define SAVE_h_m16n4  SAVE_h_m16n2  unit_save_m16n2(%%zmm10,%%zmm11)
#define SAVE_h_m16n8  SAVE_h_m16n4  unit_save_m16n2(%%zmm12,%%zmm13) unit_save_m16n2(%%zmm14,%%zmm15)
#define SAVE_h_m16n12 SAVE_h_m16n8  unit_save_m16n2(%%zmm16,%%zmm17) unit_save_m16n2(%%zmm18,%%zmm19)
#define SAVE_h_m16n16 SAVE_h_m16n12 unit_save_m16n2(%%zmm20,%%zmm21) unit_save_m16n2(%%zmm22,%%zmm23)
#define SAVE_h_m16n20 SAVE_h_m16n16 unit_save_m16n2(%%zmm24,%%zmm25) unit_save_m16n2(%%zmm26,%%zmm27)
#define SAVE_h_m16n24 SAVE_h_m16n20 unit_save_m16n2(%%zmm28,%%zmm29) unit_save_m16n2(%%zmm30,%%zmm31)
#define SAVE_m16(ndim) SAVE_h_m16n##ndim "addq $128,%2;"
#define COMPUTE_m16(ndim) \
    INIT_m16n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15; movq %2,%5;"\
    "cmpq $30,%4; jb "#ndim"016162f;"\
    #ndim"016161:\n\t"\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    "prefetcht1 (%5); prefetcht1 64(%5); prefetcht1 127(%5); addq %3,%5;"\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    "prefetcht1 (%8); addq $32,%8;"\
    "subq $6,%4; cmpq $30,%4; jnb "#ndim"016161b;"\
    "movq %2,%5;"\
    #ndim"016162:\n\t"\
    "testq %4,%4; jz "#ndim"016163f;"\
    "prefetcht0 (%5); prefetcht0 64(%5); prefetcht0 127(%5); addq %3,%5;"\
    KERNEL_k1m16n##ndim\
    "decq %4; jmp "#ndim"016162b;"\
    #ndim"016163:\n\t"\
    "prefetcht0 (%%r14); prefetcht0 64(%%r14);"\
    SAVE_m16(ndim)

/* m = 8 *//* ymm0 for alpha, ymm1-ymm3 for temporary use, ymm4-ymm15 for accumulators */
#define KERNEL_k1m8n1(b_addr) \
    "vmovups (%0),%%ymm1; addq $32,%0;"\
    "vbroadcastss ("#b_addr"),%%ymm2; vfmadd231ps %%ymm1,%%ymm2,%%ymm4;"\
    "addq $4,"#b_addr";"
#define KERNEL_h_k1m8n2(b_addr) \
    "vmovsldup (%0),%%ymm1; vmovshdup (%0),%%ymm2; addq $32,%0;"\
    "vbroadcastsd ("#b_addr"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,%%ymm4; vfmadd231ps %%ymm2,%%ymm3,%%ymm5;"
#define KERNEL_k1m8n2(b_addr) KERNEL_h_k1m8n2(b_addr) "addq $8,"#b_addr";"
#define KERNEL_h_k1m8n4(b_addr) \
    KERNEL_h_k1m8n2(b_addr) "vbroadcastsd 8("#b_addr"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,%%ymm6; vfmadd231ps %%ymm2,%%ymm3,%%ymm7;"
#define KERNEL_k1m8n4(b_addr) KERNEL_h_k1m8n4(b_addr) "addq $16,"#b_addr";"
#define unit_kernel_k1m8n4(c1,c2,c3,c4,...) \
    "vbroadcastsd  ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,"#c1"; vfmadd231ps %%ymm2,%%ymm3,"#c2";"\
    "vbroadcastsd 8("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,"#c3"; vfmadd231ps %%ymm2,%%ymm3,"#c4";"
#define KERNEL_h_k1m8n8(b_addr) KERNEL_h_k1m8n4(b_addr) unit_kernel_k1m8n4(%%ymm8,%%ymm9,%%ymm10,%%ymm11,b_addr,%%r12,1)
#define KERNEL_k1m8n8(b_addr) KERNEL_h_k1m8n8(b_addr) "addq $16,"#b_addr";"
#define KERNEL_h_k1m8n12(b_addr) KERNEL_h_k1m8n8(b_addr) unit_kernel_k1m8n4(%%ymm12,%%ymm13,%%ymm14,%%ymm15,b_addr,%%r12,2)
#define KERNEL_k1m8n12(b_addr) KERNEL_h_k1m8n12(b_addr) "addq $16,"#b_addr";"
#define INIT_m8n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define INIT_m8n2 INIT_m8n1 "vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m8n4 INIT_m8n2 "vpxor %%ymm6,%%ymm6,%%ymm6;vpxor %%ymm7,%%ymm7,%%ymm7;"
#define unit_init_m8n4(c1,c2,c3,c4) \
    "vpxor "#c1","#c1","#c1";vpxor "#c2","#c2","#c2";vpxor "#c3","#c3","#c3";vpxor "#c4","#c4","#c4";"
#define INIT_m8n8  INIT_m8n4 unit_init_m8n4(%%ymm8,%%ymm9,%%ymm10,%%ymm11)
#define INIT_m8n12 INIT_m8n8 unit_init_m8n4(%%ymm12,%%ymm13,%%ymm14,%%ymm15)
#define SAVE_L_m8n1 \
    "vunpcklps %%ymm4,%%ymm4,%%ymm2; vunpckhps %%ymm4,%%ymm4,%%ymm3;"\
    "vperm2f128 $2,%%ymm2,%%ymm3,%%ymm1; vperm2f128 $19,%%ymm2,%%ymm3,%%ymm2;"\
    "vfmadd213ps (%2),%%ymm0,%%ymm1; vfmadd213ps 32(%2),%%ymm0,%%ymm2; vmovups %%ymm1,(%2); vmovups %%ymm2,32(%2);"
#define unit_save_m8n2(c1,c2) \
    "vunpcklpd "#c2","#c1",%%ymm2; vunpckhpd "#c2","#c1",%%ymm3;"\
    "vperm2f128 $2,%%ymm2,%%ymm3,"#c1"; vperm2f128 $19,%%ymm2,%%ymm3,"#c2";"\
    "vmovsldup "#c1",%%ymm2; vmovsldup "#c2",%%ymm3;"\
    "vfmadd213ps (%5),%%ymm0,%%ymm2; vfmadd213ps 32(%5),%%ymm0,%%ymm3; vmovups %%ymm2,(%5); vmovups %%ymm3,32(%5);"\
    "vmovshdup "#c1",%%ymm2; vmovshdup "#c2",%%ymm3;"\
    "vfmadd213ps (%5,%3,1),%%ymm0,%%ymm2; vfmadd213ps 32(%5,%3,1),%%ymm0,%%ymm3; vmovups %%ymm2,(%5,%3,1); vmovups %%ymm3,32(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_L_m8n2 "movq %2,%5;" unit_save_m8n2(%%ymm4,%%ymm5)
#define SAVE_L_m8n4  SAVE_L_m8n2  unit_save_m8n2(%%ymm6,%%ymm7)
#define SAVE_L_m8n8  SAVE_L_m8n4  unit_save_m8n2(%%ymm8,%%ymm9)   unit_save_m8n2(%%ymm10,%%ymm11)
#define SAVE_L_m8n12 SAVE_L_m8n8  unit_save_m8n2(%%ymm12,%%ymm13) unit_save_m8n2(%%ymm14,%%ymm15)
#define SAVE_R_m8n4               unit_save_m8n2(%%ymm4,%%ymm5)   unit_save_m8n2(%%ymm6,%%ymm7)
#define SAVE_R_m8n8  SAVE_R_m8n4  unit_save_m8n2(%%ymm8,%%ymm9)   unit_save_m8n2(%%ymm10,%%ymm11)
#define SAVE_R_m8n12 SAVE_R_m8n8  unit_save_m8n2(%%ymm12,%%ymm13) unit_save_m8n2(%%ymm14,%%ymm15)
#define COMPUTE_L_m8(ndim,sim) \
    INIT_m8n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"882:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"883f;"\
    KERNEL_k1m8n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"882b;"\
    #ndim""#sim"883:\n\t"\
    SAVE_L_m8n##ndim "addq $64,%2;"
#define COMPUTE_R_m8(ndim,sim) \
    "subq %%r12,%0; subq %%r12,%0;"\
    INIT_m8n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"882:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"883f;"\
    KERNEL_k1m8n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"882b;"\
    #ndim""#sim"883:\n\t"\
    SAVE_R_m8n##ndim
#define COMPUTE_m8_n1  COMPUTE_L_m8(1,33833)
#define COMPUTE_m8_n2  COMPUTE_L_m8(2,33833)
#define COMPUTE_m8_n4  COMPUTE_L_m8(4,33833)
#define COMPUTE_m8_n8  COMPUTE_L_m8(8,33833)
#define COMPUTE_m8_n12 COMPUTE_L_m8(12,33833)
#define COMPUTE_m8_n16 COMPUTE_L_m8(12,33733) COMPUTE_R_m8(4,33933)
#define COMPUTE_m8_n20 COMPUTE_L_m8(12,33633) COMPUTE_R_m8(8,33933)
#define COMPUTE_m8_n24 COMPUTE_L_m8(12,33533) COMPUTE_R_m8(12,33933)
#define COMPUTE_m8(ndim) COMPUTE_m8_n##ndim

/* m = 4 *//* xmm0 for alpha, xmm1-xmm3 for temporary use, xmm4-xmm15 for accumulators */
#define KERNEL_k1m4n1(b_addr) \
    "vmovups (%0),%%xmm1; addq $16,%0;"\
    "vbroadcastss ("#b_addr"),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,"#b_addr";"
#define KERNEL_h_k1m4n2(b_addr) \
    "vmovsldup (%0),%%xmm1; vmovshdup (%0),%%xmm2; addq $16,%0;"\
    "vmovddup ("#b_addr"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm4; vfmadd231ps %%xmm2,%%xmm3,%%xmm5;"
#define KERNEL_k1m4n2(b_addr) KERNEL_h_k1m4n2(b_addr) "addq $8,"#b_addr";"
#define KERNEL_h_k1m4n4(b_addr) \
    KERNEL_h_k1m4n2(b_addr) "vmovddup 8("#b_addr"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm6; vfmadd231ps %%xmm2,%%xmm3,%%xmm7;"
#define KERNEL_k1m4n4(b_addr) KERNEL_h_k1m4n4(b_addr) "addq $16,"#b_addr";"
#define unit_kernel_k1m4n4(c1,c2,c3,c4,...) \
    "vmovddup  ("#__VA_ARGS__"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,"#c1"; vfmadd231ps %%xmm2,%%xmm3,"#c2";"\
    "vmovddup 8("#__VA_ARGS__"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,"#c3"; vfmadd231ps %%xmm2,%%xmm3,"#c4";"
#define KERNEL_h_k1m4n8(b_addr) KERNEL_h_k1m4n4(b_addr) unit_kernel_k1m4n4(%%xmm8,%%xmm9,%%xmm10,%%xmm11,b_addr,%%r12,1)
#define KERNEL_k1m4n8(b_addr) KERNEL_h_k1m4n8(b_addr) "addq $16,"#b_addr";"
#define KERNEL_h_k1m4n12(b_addr) KERNEL_h_k1m4n8(b_addr) unit_kernel_k1m4n4(%%xmm12,%%xmm13,%%xmm14,%%xmm15,b_addr,%%r12,2)
#define KERNEL_k1m4n12(b_addr) KERNEL_h_k1m4n12(b_addr) "addq $16,"#b_addr";"
#define INIT_m4n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m4n2 INIT_m4n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m4n4 INIT_m4n2 "vpxor %%xmm6,%%xmm6,%%xmm6;vpxor %%xmm7,%%xmm7,%%xmm7;"
#define unit_init_m4n4(c1,c2,c3,c4) \
    "vpxor "#c1","#c1","#c1";vpxor "#c2","#c2","#c2";vpxor "#c3","#c3","#c3";vpxor "#c4","#c4","#c4";"
#define INIT_m4n8  INIT_m4n4 unit_init_m4n4(%%xmm8,%%xmm9,%%xmm10,%%xmm11)
#define INIT_m4n12 INIT_m4n8 unit_init_m4n4(%%xmm12,%%xmm13,%%xmm14,%%xmm15)
#define SAVE_L_m4n1 \
    "vunpcklps %%xmm4,%%xmm4,%%xmm2; vunpckhps %%xmm4,%%xmm4,%%xmm3;"\
    "vfmadd213ps (%2),%%xmm0,%%xmm2; vfmadd213ps 16(%2),%%xmm0,%%xmm3; vmovups %%xmm2,(%2); vmovups %%xmm3,16(%2);"
#define unit_save_m4n2(c1,c2) \
    "vunpcklpd "#c2","#c1",%%xmm2; vunpckhpd "#c2","#c1","#c2"; vmovapd %%xmm2,"#c1";"\
    "vmovsldup "#c1",%%xmm2; vmovsldup "#c2",%%xmm3;"\
    "vfmadd213ps (%5),%%xmm0,%%xmm2; vfmadd213ps 16(%5),%%xmm0,%%xmm3; vmovups %%xmm2,(%5); vmovups %%xmm3,16(%5);"\
    "vmovshdup "#c1",%%xmm2; vmovshdup "#c2",%%xmm3;"\
    "vfmadd213ps (%5,%3,1),%%xmm0,%%xmm2; vfmadd213ps 16(%5,%3,1),%%xmm0,%%xmm3; vmovups %%xmm2,(%5,%3,1); vmovups %%xmm3,16(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_L_m4n2 "movq %2,%5;" unit_save_m4n2(%%xmm4,%%xmm5)
#define SAVE_L_m4n4  SAVE_L_m4n2  unit_save_m4n2(%%xmm6,%%xmm7)
#define SAVE_L_m4n8  SAVE_L_m4n4  unit_save_m4n2(%%xmm8,%%xmm9)   unit_save_m4n2(%%xmm10,%%xmm11)
#define SAVE_L_m4n12 SAVE_L_m4n8  unit_save_m4n2(%%xmm12,%%xmm13) unit_save_m4n2(%%xmm14,%%xmm15)
#define SAVE_R_m4n4               unit_save_m4n2(%%xmm4,%%xmm5)   unit_save_m4n2(%%xmm6,%%xmm7)
#define SAVE_R_m4n8  SAVE_R_m4n4  unit_save_m4n2(%%xmm8,%%xmm9)   unit_save_m4n2(%%xmm10,%%xmm11)
#define SAVE_R_m4n12 SAVE_R_m4n8  unit_save_m4n2(%%xmm12,%%xmm13) unit_save_m4n2(%%xmm14,%%xmm15)
#define COMPUTE_L_m4(ndim,sim) \
    INIT_m4n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"442:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"443f;"\
    KERNEL_k1m4n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"442b;"\
    #ndim""#sim"443:\n\t"\
    SAVE_L_m4n##ndim "addq $32,%2;"
#define COMPUTE_R_m4(ndim,sim) \
    "subq %%r12,%0;"\
    INIT_m4n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"442:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"443f;"\
    KERNEL_k1m4n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"442b;"\
    #ndim""#sim"443:\n\t"\
    SAVE_R_m4n##ndim
#define COMPUTE_m4_n1  COMPUTE_L_m4(1,55855)
#define COMPUTE_m4_n2  COMPUTE_L_m4(2,55855)
#define COMPUTE_m4_n4  COMPUTE_L_m4(4,55855)
#define COMPUTE_m4_n8  COMPUTE_L_m4(8,55855)
#define COMPUTE_m4_n12 COMPUTE_L_m4(12,55855)
#define COMPUTE_m4_n16 COMPUTE_L_m4(12,55755) COMPUTE_R_m4(4,55955)
#define COMPUTE_m4_n20 COMPUTE_L_m4(12,55655) COMPUTE_R_m4(8,55955)
#define COMPUTE_m4_n24 COMPUTE_L_m4(12,55555) COMPUTE_R_m4(12,55955)
#define COMPUTE_m4(ndim) COMPUTE_m4_n##ndim

/* m = 2 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm9 for accumulators */
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m2n1(b_addr) \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss ("#b_addr"),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,"#b_addr";"
#define SAVE_L_m2n1 \
    "vunpcklps %%xmm4,%%xmm4,%%xmm1; vfmadd213ps (%2),%%xmm0,%%xmm1; vmovups %%xmm1,(%2);"
#define INIT_m2n2 INIT_m2n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define KERNEL_k1m2n2(b_addr) \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss  ("#b_addr"),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "vbroadcastss 4("#b_addr"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm5;"\
    "addq $8,"#b_addr";"
#define SAVE_L_m2n2 SAVE_L_m2n1 \
    "vunpcklps %%xmm5,%%xmm5,%%xmm1; vfmadd213ps (%2,%3,1),%%xmm0,%%xmm1; vmovups %%xmm1,(%2,%3,1);"
#define INIT_m2n4  INIT_m2n2
#define INIT_m2n8  INIT_m2n4 "vpxor %%xmm6,%%xmm6,%%xmm6; vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m2n12 INIT_m2n8 "vpxor %%xmm8,%%xmm8,%%xmm8; vpxor %%xmm9,%%xmm9,%%xmm9;"
#define KERNEL_k1m2n4(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "vbroadcastss 4(%0),%%xmm2; vfmadd231ps %%xmm3,%%xmm2,%%xmm5;"\
    "addq $8,%0;"
#define KERNEL_k1m2n8(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4; vfmadd231ps %%xmm2,%%xmm1,%%xmm6;"\
    "vbroadcastss 4(%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm5; vfmadd231ps %%xmm2,%%xmm1,%%xmm7;"\
    "addq $8,%0;"
#define KERNEL_k1m2n12(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; vmovups ("#b_addr",%%r12,2),%%xmm1; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm4; vfmadd231ps %%xmm2,%%xmm10,%%xmm6; vfmadd231ps %%xmm1,%%xmm10,%%xmm8;"\
    "vbroadcastss 4(%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm5; vfmadd231ps %%xmm2,%%xmm10,%%xmm7; vfmadd231ps %%xmm1,%%xmm10,%%xmm9;"\
    "addq $8,%0;"
#define unit_save_m2n4(c1,c2) \
    "vunpcklpd "#c2","#c1",%%xmm1; vunpckhpd "#c2","#c1",%%xmm2;"\
    "vmovsldup %%xmm1,%%xmm3; vfmadd213ps (%5),%%xmm0,%%xmm3; vmovups %%xmm3,(%5);"\
    "vmovshdup %%xmm1,%%xmm3; vfmadd213ps (%5,%3,1),%%xmm0,%%xmm3; vmovups %%xmm3,(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"\
    "vmovsldup %%xmm2,%%xmm3; vfmadd213ps (%5),%%xmm0,%%xmm3; vmovups %%xmm3,(%5);"\
    "vmovshdup %%xmm2,%%xmm3; vfmadd213ps (%5,%3,1),%%xmm0,%%xmm3; vmovups %%xmm3,(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_L_m2n4  "movq %2,%5;" unit_save_m2n4(%%xmm4,%%xmm5)
#define SAVE_L_m2n8  SAVE_L_m2n4   unit_save_m2n4(%%xmm6,%%xmm7)
#define SAVE_L_m2n12 SAVE_L_m2n8   unit_save_m2n4(%%xmm8,%%xmm9)
#define SAVE_R_m2n4                unit_save_m2n4(%%xmm4,%%xmm5)
#define SAVE_R_m2n8  SAVE_R_m2n4   unit_save_m2n4(%%xmm6,%%xmm7)
#define SAVE_R_m2n12 SAVE_R_m2n8   unit_save_m2n4(%%xmm8,%%xmm9)
#define COMPUTE_L_m2(ndim,sim) \
    INIT_m2n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"222:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"223f;"\
    KERNEL_k1m2n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"222b;"\
    #ndim""#sim"223:\n\t"\
    SAVE_L_m2n##ndim "addq $16,%2;"
#define COMPUTE_R_m2(ndim,sim) \
    "salq $3,%%r13;subq %%r13,%0;sarq $3,%%r13;"\
    INIT_m2n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"222:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"223f;"\
    KERNEL_k1m2n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"222b;"\
    #ndim""#sim"223:\n\t"\
    SAVE_R_m2n##ndim
#define COMPUTE_m2_n1  COMPUTE_L_m2(1,77877)
#define COMPUTE_m2_n2  COMPUTE_L_m2(2,77877)
#define COMPUTE_m2_n4  COMPUTE_L_m2(4,77877)
#define COMPUTE_m2_n8  COMPUTE_L_m2(8,77877)
#define COMPUTE_m2_n12 COMPUTE_L_m2(12,77877)
#define COMPUTE_m2_n16 COMPUTE_L_m2(12,77777) COMPUTE_R_m2(4,77977)
#define COMPUTE_m2_n20 COMPUTE_L_m2(12,77677) COMPUTE_R_m2(8,77977)
#define COMPUTE_m2_n24 COMPUTE_L_m2(12,77577) COMPUTE_R_m2(12,77977)
#define COMPUTE_m2(ndim) COMPUTE_m2_n##ndim

/* m = 1 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm6 for accumulators */
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m1n1(b_addr) \
    "vmovss ("#b_addr"),%%xmm3; addq $4,"#b_addr";"\
    "vmovss (%0),%%xmm1; vfmadd231ss %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_L_m1n1 \
    "vunpcklps %%xmm4,%%xmm4,%%xmm4; vmovsd (%2),%%xmm1; vfmadd213ps %%xmm1,%%xmm0,%%xmm4; vmovsd %%xmm4,(%2);"
#define INIT_m1n2 INIT_m1n1
#define KERNEL_k1m1n2(b_addr) \
    "vmovsd ("#b_addr"),%%xmm3; addq $8,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_L_m1n2 \
    "vunpcklps %%xmm4,%%xmm4,%%xmm4; vmovsd (%2),%%xmm3; vmovhpd (%2,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm4;"\
    "vmovsd %%xmm4,(%2); vmovhpd %%xmm4,(%2,%3,1);"
#define INIT_m1n4  INIT_m1n2
#define INIT_m1n8  INIT_m1n4 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m1n12 INIT_m1n8 "vpxor %%xmm6,%%xmm6,%%xmm6;"
#define KERNEL_k1m1n4(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define KERNEL_k1m1n8(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4; vfmadd231ps %%xmm2,%%xmm1,%%xmm5;"\
    "addq $4,%0;"
#define KERNEL_k1m1n12(b_addr) \
    "vmovups ("#b_addr"),%%xmm3; vmovups ("#b_addr",%%r12,1),%%xmm2; vmovups ("#b_addr",%%r12,2),%%xmm1; addq $16,"#b_addr";"\
    "vbroadcastss  (%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm4; vfmadd231ps %%xmm2,%%xmm10,%%xmm5; vfmadd231ps %%xmm1,%%xmm10,%%xmm6;"\
    "addq $4,%0;"
#define unit_save_m1n4(c1) \
    "vunpcklps "#c1","#c1",%%xmm1; vunpckhps "#c1","#c1",%%xmm2;"\
    "vmovsd (%5),%%xmm3; vmovhpd (%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm1;"\
    "vmovsd %%xmm1,(%5); vmovhpd %%xmm1,(%5,%3,1); leaq (%5,%3,2),%5;"\
    "vmovsd (%5),%%xmm3; vmovhpd (%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm2;"\
    "vmovsd %%xmm2,(%5); vmovhpd %%xmm2,(%5,%3,1); leaq (%5,%3,2),%5;"
#define SAVE_L_m1n4 "movq %2,%5;" unit_save_m1n4(%%xmm4)
#define SAVE_L_m1n8  SAVE_L_m1n4  unit_save_m1n4(%%xmm5)
#define SAVE_L_m1n12 SAVE_L_m1n8  unit_save_m1n4(%%xmm6)
#define SAVE_R_m1n4               unit_save_m1n4(%%xmm4)
#define SAVE_R_m1n8  SAVE_R_m1n4  unit_save_m1n4(%%xmm5)
#define SAVE_R_m1n12 SAVE_R_m1n8  unit_save_m1n4(%%xmm6)
#define COMPUTE_L_m1(ndim,sim) \
    INIT_m1n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim""#sim"112:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"113f;"\
    KERNEL_k1m1n##ndim(%1)\
    "decq %4; jmp "#ndim""#sim"112b;"\
    #ndim""#sim"113:\n\t"\
    SAVE_L_m1n##ndim "addq $8,%2;"
#define COMPUTE_R_m1(ndim,sim) \
    "salq $2,%%r13;subq %%r13,%0;sarq $2,%%r13;"\
    INIT_m1n##ndim\
    "movq %%r13,%4; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
    #ndim""#sim"112:\n\t"\
    "testq %4,%4; jz "#ndim""#sim"113f;"\
    KERNEL_k1m1n##ndim(%%r15)\
    "decq %4; jmp "#ndim""#sim"112b;"\
    #ndim""#sim"113:\n\t"\
    SAVE_R_m1n##ndim
#define COMPUTE_m1_n1  COMPUTE_L_m1(1,99899)
#define COMPUTE_m1_n2  COMPUTE_L_m1(2,99899)
#define COMPUTE_m1_n4  COMPUTE_L_m1(4,99899)
#define COMPUTE_m1_n8  COMPUTE_L_m1(8,99899)
#define COMPUTE_m1_n12 COMPUTE_L_m1(12,99899)
#define COMPUTE_m1_n16 COMPUTE_L_m1(12,99799) COMPUTE_R_m1(4,99999)
#define COMPUTE_m1_n20 COMPUTE_L_m1(12,99699) COMPUTE_R_m1(8,99999)
#define COMPUTE_m1_n24 COMPUTE_L_m1(12,99599) COMPUTE_R_m1(12,99999)
#define COMPUTE_m1(ndim) COMPUTE_m1_n##ndim

/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 = "+r"(K), %5 = "+r"(ctemp) */
/* %6 = "+r"(&alpha), %7 = "+r"(M), %8 = "+r"(next_b) */
/* r11 = m(const), r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const), r15 = %1 + 3r12 */

#define COMPUTE(ndim) {\
    next_b = b_pointer + ndim * K;\
    __asm__ __volatile__(\
    "vbroadcastsd (%6),%%zmm0; vmovups 8(%6),%%zmm1; vmovups 72(%6),%%zmm2;"\
    "movq %4,%%r13; movq %4,%%r12; salq $4,%%r12; movq %1,%%r14; movq %7,%%r11;"\
    "cmpq $16,%7;jb 33101"#ndim"f;"\
    "33109"#ndim":\n\t"\
    COMPUTE_m16(ndim)\
    "subq $16,%7;cmpq $16,%7;jnb 33109"#ndim"b;"\
    "33101"#ndim":\n\t"\
    "cmpq $8,%7;jb 33102"#ndim"f;"\
    COMPUTE_m8(ndim)\
    "subq $8,%7;"\
    "33102"#ndim":\n\t"\
    "cmpq $4,%7;jb 33103"#ndim"f;"\
    COMPUTE_m4(ndim)\
    "subq $4,%7;"\
    "33103"#ndim":\n\t"\
    "cmpq $2,%7;jb 33104"#ndim"f;"\
    COMPUTE_m2(ndim)\
    "subq $2,%7;"\
    "33104"#ndim":\n\t"\
    "testq %7,%7;jz 33105"#ndim"f;"\
    COMPUTE_m1(ndim)\
    "33105"#ndim":\n\t"\
    "movq %%r13,%4; movq %%r14,%1; movq %%r11,%7;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(ldc_in_bytes),"+r"(K),"+r"(ctemp),"+r"(const_val),"+r"(M),"+r"(next_b)\
    ::"r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14",\
    "zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",\
    "cc","memory");\
    a_pointer -= M * K; b_pointer += ndim * K; c_pointer += 2*(LDC * ndim - M);\
}

int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, float alphar, float alphai, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, BLASLONG LDC)
{
    if(m==0||n==0||k==0) return 0;
    int64_t ldc_in_bytes = (int64_t)LDC * sizeof(float) * 2;
    float constval[34]; constval[0] = alphar; constval[1] = alphai;
    float *const_val=constval;
    __asm__ __volatile__(
    "movq $0,8(%0);   movq $8,16(%0); movq $1,24(%0);  movq $9,32(%0);"
    "movq $2,40(%0);  movq $10,48(%0);movq $3,56(%0);  movq $11,64(%0);"
    "movq $12,72(%0); movq $4,80(%0); movq $13,88(%0); movq $5,96(%0);"
    "movq $14,104(%0);movq $6,112(%0);movq $15,120(%0);movq $7,128(%0);"
    ::"r"(const_val):"memory");//control words for permutations(vpermt2pd) before saving vectors of C elements
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    float *a_pointer = A,*b_pointer = B,*c_pointer = C,*ctemp = C,*next_b = B;
    for(;n_count>23;n_count-=24) COMPUTE(24)
    for(;n_count>19;n_count-=20) COMPUTE(20)
    for(;n_count>15;n_count-=16) COMPUTE(16)
    for(;n_count>11;n_count-=12) COMPUTE(12)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
    return 0;
}
