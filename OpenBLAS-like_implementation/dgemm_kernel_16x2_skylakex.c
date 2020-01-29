
//16x2
#define KERNEL_h_k1m16n1 \
  "vmovupd (%0),%%zmm1; vmovupd 64(%0),%%zmm2; addq $128,%0;"\
  "vbroadcastsd (%1),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm8; vfmadd231pd %%zmm2,%%zmm3,%%zmm9;"
#define KERNEL_k1m16n1 KERNEL_h_k1m16n1 "addq $8,%1;"
#define KERNEL_h_k1m16n2 KERNEL_h_k1m16n1\
  "vbroadcastsd 8(%1),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm10; vfmadd231pd %%zmm2,%%zmm4,%%zmm11;"
#define KERNEL_k1m16n2 KERNEL_h_k1m16n2 "addq $16,%1;"
#define unit_acc_m16n2(c1_no,c2_no,c3_no,c4_no,...)\
  "vbroadcastsd ("#__VA_ARGS__"),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm"#c1_no"; vfmadd231pd %%zmm2,%%zmm3,%%zmm"#c2_no";"\
  "vbroadcastsd 8("#__VA_ARGS__"),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm"#c3_no"; vfmadd231pd %%zmm2,%%zmm4,%%zmm"#c4_no";"
#define KERNEL_h_k1m16n4 KERNEL_h_k1m16n2 "prefetcht0 384(%0);" unit_acc_m16n2(12,13,14,15,%1,%%r12,1)
#define KERNEL_k1m16n4 KERNEL_h_k1m16n4 "addq $16,%1;"
#define KERNEL_k1m16n6 KERNEL_h_k1m16n4 unit_acc_m16n2(16,17,18,19,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m16n8 KERNEL_k1m16n6 "prefetcht0 448(%0);" unit_acc_m16n2(20,21,22,23,%%r15)
#define KERNEL_k1m16n8 KERNEL_h_k1m16n8 "addq $16,%1;"
#define KERNEL_h_k1m16n10 KERNEL_h_k1m16n8 unit_acc_m16n2(24,25,26,27,%%r15,%%r12,1)
#define KERNEL_k1m16n10 KERNEL_h_k1m16n10 "addq $16,%%r15;"
#define KERNEL_h_k1m16n12 KERNEL_h_k1m16n10 unit_acc_m16n2(28,29,30,31,%%r15,%%r12,2)
#define KERNEL_k1m16n12 KERNEL_h_k1m16n12 "addq $16,%%r15;"
#define save_init_m16 "movq %2,%3; addq $128,%2;"
#define SAVE_m16n1 "vfmadd213pd (%2),%%zmm0,%%zmm8; vmovupd %%zmm8,(%2); vfmadd213pd 64(%2),%%zmm0,%%zmm9; vmovupd %%zmm9,64(%2); addq $128,%2;"
#define unit_save_m16n2(c1_no,c2_no,c3_no,c4_no)\
  "vfmadd213pd (%3),%%zmm0,%%zmm"#c1_no"; vmovupd %%zmm"#c1_no",(%3); vfmadd213pd 64(%3),%%zmm0,%%zmm"#c2_no"; vmovupd %%zmm"#c2_no",64(%3);"\
  "vfmadd213pd (%3,%4,1),%%zmm0,%%zmm"#c3_no"; vmovupd %%zmm"#c3_no",(%3,%4,1); vfmadd213pd 64(%3,%4,1),%%zmm0,%%zmm"#c4_no"; vmovupd %%zmm"#c4_no",64(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m16n2 save_init_m16 unit_save_m16n2(8,9,10,11)
#define SAVE_m16n4 SAVE_m16n2 unit_save_m16n2(12,13,14,15)
#define SAVE_m16n6 SAVE_m16n4 unit_save_m16n2(16,17,18,19)
#define SAVE_m16n8 SAVE_m16n6 unit_save_m16n2(20,21,22,23)
#define SAVE_m16n10 SAVE_m16n8 unit_save_m16n2(24,25,26,27)
#define SAVE_m16n12 SAVE_m16n10 unit_save_m16n2(28,29,30,31)
#define unit_init_2zmm(c1_no,c2_no) "vpxorq %%zmm"#c1_no",%%zmm"#c1_no",%%zmm"#c1_no"; vpxorq %%zmm"#c2_no",%%zmm"#c2_no",%%zmm"#c2_no";"
#define unit_init_4zmm(c1_no,c2_no,c3_no,c4_no) unit_init_2zmm(c1_no,c2_no) unit_init_2zmm(c3_no,c4_no)
#define INIT_m16n1 unit_init_2zmm(8,9)
#define INIT_m16n2 unit_init_4zmm(8,9,10,11)
#define INIT_m16n4 INIT_m16n2 unit_init_4zmm(12,13,14,15)
#define INIT_m16n6 INIT_m16n4 unit_init_4zmm(16,17,18,19)
#define INIT_m16n8 INIT_m16n6 unit_init_4zmm(20,21,22,23)
#define INIT_m16n10 INIT_m16n8 unit_init_4zmm(24,25,26,27)
#define INIT_m16n12 INIT_m16n10 unit_init_4zmm(28,29,30,31)

#define KERNEL_k1m8n1 \
  "vbroadcastsd (%1),%%zmm1; addq $8,%1;"\
  "vfmadd231pd (%0),%%zmm1,%%zmm8; addq $64,%0;"
#define unit_acc_m8n2(c1_no,c2_no,...)\
  "vbroadcastf32x4 ("#__VA_ARGS__"),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm"#c1_no"; vfmadd231pd %%zmm2,%%zmm3,%%zmm"#c2_no";"
#define KERNEL_h_k1m8n2 \
  "vmovddup (%0),%%zmm1; vmovddup 8(%0),%%zmm2; addq $64,%0;" unit_acc_m8n2(8,9,%1)
#define KERNEL_k1m8n2 KERNEL_h_k1m8n2 "addq $16,%1;"
#define KERNEL_h_k1m8n4 KERNEL_h_k1m8n2 unit_acc_m8n2(10,11,%1,%%r12,1)
#define KERNEL_k1m8n4 KERNEL_h_k1m8n4 "addq $16,%1;"
#define KERNEL_k1m8n6 KERNEL_h_k1m8n4 unit_acc_m8n2(12,13,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m8n8 KERNEL_k1m8n6 unit_acc_m8n2(14,15,%%r15)
#define KERNEL_k1m8n8 KERNEL_h_k1m8n8 "addq $16,%%r15;"
#define KERNEL_h_k1m8n10 KERNEL_h_k1m8n8 unit_acc_m8n2(16,17,%%r15,%%r12,1)
#define KERNEL_k1m8n10 KERNEL_h_k1m8n10 "addq $16,%%r15;"
#define KERNEL_h_k1m8n12 KERNEL_h_k1m8n10 unit_acc_m8n2(18,19,%%r15,%%r12,2)
#define KERNEL_k1m8n12 KERNEL_h_k1m8n12 "addq $16,%%r15;"
#define save_init_m8 "movq %2,%3; addq $64,%2;"
#define SAVE_m8n1 "vfmadd213pd (%2),%%zmm0,%%zmm8; vmovupd %%zmm8,(%2); addq $64,%2;"
#define unit_save_m8n2(c1_no,c2_no)\
  "vunpcklpd %%zmm"#c2_no",%%zmm"#c1_no",%%zmm1; vfmadd213pd (%3),%%zmm0,%%zmm1; vmovupd %%zmm1,(%3);"\
  "vunpckhpd %%zmm"#c2_no",%%zmm"#c1_no",%%zmm2; vfmadd213pd (%3,%4,1),%%zmm0,%%zmm2; vmovupd %%zmm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m8n2 save_init_m8 unit_save_m8n2(8,9)
#define SAVE_m8n4 SAVE_m8n2 unit_save_m8n2(10,11)
#define SAVE_m8n6 SAVE_m8n4 unit_save_m8n2(12,13)
#define SAVE_m8n8 SAVE_m8n6 unit_save_m8n2(14,15)
#define SAVE_m8n10 SAVE_m8n8 unit_save_m8n2(16,17)
#define SAVE_m8n12 SAVE_m8n10 unit_save_m8n2(18,19)
#define INIT_m8n1 "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define INIT_m8n2 unit_init_2zmm(8,9)
#define INIT_m8n4 INIT_m8n2 unit_init_2zmm(10,11)
#define INIT_m8n6 INIT_m8n4 unit_init_2zmm(12,13)
#define INIT_m8n8 INIT_m8n6 unit_init_2zmm(14,15)
#define INIT_m8n10 INIT_m8n8 unit_init_2zmm(16,17)
#define INIT_m8n12 INIT_m8n10 unit_init_2zmm(18,19)

#define KERNEL_k1m4n1 \
  "vbroadcastsd (%1),%%ymm1; addq $8,%1;"\
  "vfmadd231pd (%0),%%ymm1,%%ymm4; addq $32,%0;"
#define unit_acc_m4n2(c1_no,c2_no,...)\
  "vbroadcastf128 ("#__VA_ARGS__"),%%ymm3; vfmadd231pd %%ymm1,%%ymm3,%%ymm"#c1_no"; vfmadd231pd %%ymm2,%%ymm3,%%ymm"#c2_no";"
#define KERNEL_h_k1m4n2 \
  "vmovddup (%0),%%ymm1; vmovddup 8(%0),%%ymm2; addq $32,%0;" unit_acc_m4n2(4,5,%1)
#define KERNEL_k1m4n2 KERNEL_h_k1m4n2 "addq $16,%1;"
#define KERNEL_h_k1m4n4 KERNEL_h_k1m4n2 unit_acc_m4n2(6,7,%1,%%r12,1)
#define KERNEL_k1m4n4 KERNEL_h_k1m4n4 "addq $16,%1;"
#define KERNEL_k1m4n6 KERNEL_h_k1m4n4 unit_acc_m4n2(8,9,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m4n8 KERNEL_k1m4n6 unit_acc_m4n2(10,11,%%r15)
#define KERNEL_k1m4n8 KERNEL_h_k1m4n8 "addq $16,%%r15;"
#define KERNEL_h_k1m4n10 KERNEL_h_k1m4n8 unit_acc_m4n2(12,13,%%r15,%%r12,1)
#define KERNEL_k1m4n10 KERNEL_h_k1m4n10 "addq $16,%%r15;"
#define KERNEL_h_k1m4n12 KERNEL_h_k1m4n10 unit_acc_m4n2(14,15,%%r15,%%r12,2)
#define KERNEL_k1m4n12 KERNEL_h_k1m4n12 "addq $16,%%r15;"
#define save_init_m4 "movq %2,%3; addq $32,%2;"
#define SAVE_m4n1 "vfmadd213pd (%2),%%ymm0,%%ymm4; vmovupd %%ymm4,(%2); addq $32,%2;"
#define unit_save_m4n2(c1_no,c2_no)\
  "vunpcklpd %%ymm"#c2_no",%%ymm"#c1_no",%%ymm1; vfmadd213pd (%3),%%ymm0,%%ymm1; vmovupd %%ymm1,(%3);"\
  "vunpckhpd %%ymm"#c2_no",%%ymm"#c1_no",%%ymm2; vfmadd213pd (%3,%4,1),%%ymm0,%%ymm2; vmovupd %%ymm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m4n2 save_init_m4 unit_save_m4n2(4,5)
#define SAVE_m4n4 SAVE_m4n2 unit_save_m4n2(6,7)
#define SAVE_m4n6 SAVE_m4n4 unit_save_m4n2(8,9)
#define SAVE_m4n8 SAVE_m4n6 unit_save_m4n2(10,11)
#define SAVE_m4n10 SAVE_m4n8 unit_save_m4n2(12,13)
#define SAVE_m4n12 SAVE_m4n10 unit_save_m4n2(14,15)
#define INIT_m4n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define unit_init_2ymm(c1_no,c2_no) "vpxor %%ymm"#c1_no",%%ymm"#c1_no",%%ymm"#c1_no"; vpxor %%ymm"#c2_no",%%ymm"#c2_no",%%ymm"#c2_no";"
#define INIT_m4n2 unit_init_2ymm(4,5)
#define INIT_m4n4 INIT_m4n2 unit_init_2ymm(6,7)
#define INIT_m4n6 INIT_m4n4 unit_init_2ymm(8,9)
#define INIT_m4n8 INIT_m4n6 unit_init_2ymm(10,11)
#define INIT_m4n10 INIT_m4n8 unit_init_2ymm(12,13)
#define INIT_m4n12 INIT_m4n10 unit_init_2ymm(14,15)

#define KERNEL_k1m2n1 \
  "vmovddup (%1),%%xmm1; addq $8,%1;"\
  "vfmadd231pd (%0),%%xmm1,%%xmm4; addq $16,%0;"
#define unit_acc_m2n2(c1_no,c2_no,...)\
  "vmovupd ("#__VA_ARGS__"),%%xmm3; vfmadd231pd %%xmm1,%%xmm3,%%xmm"#c1_no"; vfmadd231pd %%xmm2,%%xmm3,%%xmm"#c2_no";"
#define KERNEL_h_k1m2n2 \
  "vmovddup (%0),%%xmm1; vmovddup 8(%0),%%xmm2; addq $16,%0;" unit_acc_m2n2(4,5,%1)
#define KERNEL_k1m2n2 KERNEL_h_k1m2n2 "addq $16,%1;"
#define KERNEL_h_k1m2n4 KERNEL_h_k1m2n2 unit_acc_m2n2(6,7,%1,%%r12,1)
#define KERNEL_k1m2n4 KERNEL_h_k1m2n4 "addq $16,%1;"
#define KERNEL_k1m2n6 KERNEL_h_k1m2n4 unit_acc_m2n2(8,9,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m2n8 KERNEL_k1m2n6 unit_acc_m2n2(10,11,%%r15)
#define KERNEL_k1m2n8 KERNEL_h_k1m2n8 "addq $16,%%r15;"
#define KERNEL_h_k1m2n10 KERNEL_h_k1m2n8 unit_acc_m2n2(12,13,%%r15,%%r12,1)
#define KERNEL_k1m2n10 KERNEL_h_k1m2n10 "addq $16,%%r15;"
#define KERNEL_h_k1m2n12 KERNEL_h_k1m2n10 unit_acc_m2n2(14,15,%%r15,%%r12,2)
#define KERNEL_k1m2n12 KERNEL_h_k1m2n12 "addq $16,%%r15;"
#define save_init_m2 "movq %2,%3; addq $16,%2;"
#define SAVE_m2n1 "vfmadd213pd (%2),%%xmm0,%%xmm4; vmovupd %%xmm4,(%2); addq $16,%2;"
#define unit_save_m2n2(c1_no,c2_no)\
  "vunpcklpd %%xmm"#c2_no",%%xmm"#c1_no",%%xmm1; vfmadd213pd (%3),%%xmm0,%%xmm1; vmovupd %%xmm1,(%3);"\
  "vunpckhpd %%xmm"#c2_no",%%xmm"#c1_no",%%xmm2; vfmadd213pd (%3,%4,1),%%xmm0,%%xmm2; vmovupd %%xmm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m2n2 save_init_m2 unit_save_m2n2(4,5)
#define SAVE_m2n4 SAVE_m2n2 unit_save_m2n2(6,7)
#define SAVE_m2n6 SAVE_m2n4 unit_save_m2n2(8,9)
#define SAVE_m2n8 SAVE_m2n6 unit_save_m2n2(10,11)
#define SAVE_m2n10 SAVE_m2n8 unit_save_m2n2(12,13)
#define SAVE_m2n12 SAVE_m2n10 unit_save_m2n2(14,15)
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define unit_init_2xmm(c1_no,c2_no) "vpxor %%xmm"#c1_no",%%xmm"#c1_no",%%xmm"#c1_no"; vpxor %%xmm"#c2_no",%%xmm"#c2_no",%%xmm"#c2_no";"
#define INIT_m2n2 unit_init_2xmm(4,5)
#define INIT_m2n4 INIT_m2n2 unit_init_2xmm(6,7)
#define INIT_m2n6 INIT_m2n4 unit_init_2xmm(8,9)
#define INIT_m2n8 INIT_m2n6 unit_init_2xmm(10,11)
#define INIT_m2n10 INIT_m2n8 unit_init_2xmm(12,13)
#define INIT_m2n12 INIT_m2n10 unit_init_2xmm(14,15)

#define KERNEL_k1m1n1 \
  "vmovsd (%1),%%xmm1; addq $8,%1;"\
  "vfmadd231sd (%0),%%xmm1,%%xmm4; addq $8,%0;"
#define KERNEL_h_k1m1n2 \
  "vmovddup (%0),%%xmm1; addq $8,%0;"\
  "vfmadd231pd (%1),%%xmm1,%%xmm4;"
#define KERNEL_k1m1n2 KERNEL_h_k1m1n2 "addq $16,%1;"
#define KERNEL_h_k1m1n4 KERNEL_h_k1m1n2 "vfmadd231pd (%1,%%r12,1),%%xmm1,%%xmm5;"
#define KERNEL_k1m1n4 KERNEL_h_k1m1n4 "addq $16,%1;"
#define KERNEL_k1m1n6 KERNEL_h_k1m1n4 "vfmadd231pd (%1,%%r12,2),%%xmm1,%%xmm6; addq $16,%1;"
#define KERNEL_h_k1m1n8 KERNEL_k1m1n6 "vfmadd231pd (%%r15),%%xmm1,%%xmm7;"
#define KERNEL_k1m1n8 KERNEL_h_k1m1n8 "addq $16,%%r15;"
#define KERNEL_h_k1m1n10 KERNEL_h_k1m1n8 "vfmadd231pd (%%r15,%%r12,1),%%xmm1,%%xmm8;"
#define KERNEL_k1m1n10 KERNEL_h_k1m1n10 "addq $16,%%r15;"
#define KERNEL_h_k1m1n12 KERNEL_h_k1m1n10 "vfmadd231pd (%%r15,%%r12,2),%%xmm1,%%xmm9;"
#define KERNEL_k1m1n12 KERNEL_h_k1m1n12 "addq $16,%%r15;"
#define save_init_m1 "movq %2,%3; addq $8,%2;"
#define SAVE_m1n1 "vfmadd213sd (%2),%%xmm0,%%xmm4; vmovsd %%xmm4,(%2); addq $8,%2;"
#define unit_save_m1n2(c1_no)\
  "vmovsd (%3),%%xmm2; vmovhpd (%3,%4,1),%%xmm2,%%xmm2; vfmadd231pd %%xmm"#c1_no",%%xmm0,%%xmm2; vmovsd %%xmm2,(%3); vmovhpd %%xmm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m1n2 save_init_m1 unit_save_m1n2(4)
#define SAVE_m1n4 SAVE_m1n2 unit_save_m1n2(5)
#define SAVE_m1n6 SAVE_m1n4 unit_save_m1n2(6)
#define SAVE_m1n8 SAVE_m1n6 unit_save_m1n2(7)
#define SAVE_m1n10 SAVE_m1n8 unit_save_m1n2(8)
#define SAVE_m1n12 SAVE_m1n10 unit_save_m1n2(9)
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m1n2 INIT_m1n1
#define INIT_m1n4 INIT_m1n2 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m1n6 INIT_m1n4 "vpxor %%xmm6,%%xmm6,%%xmm6;"
#define INIT_m1n8 INIT_m1n6 "vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m1n10 INIT_m1n8 "vpxor %%xmm8,%%xmm8,%%xmm8;"
#define INIT_m1n12 INIT_m1n10 "vpxor %%xmm9,%%xmm9,%%xmm9;"

#define COMPUTE_SIMPLE(mdim,ndim)\
  INIT_m##mdim##n##ndim "testq %%r13,%%r13; jz 7"#mdim"7"#ndim"9f;"\
  "movq %%r13,%5; movq %%r14,%1; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
  "7"#mdim"7"#ndim"1:\n\t"\
  KERNEL_k1m##mdim##n##ndim "decq %5; jnz 7"#mdim"7"#ndim"1b;"\
  "7"#mdim"7"#ndim"9:\n\t"\
  SAVE_m##mdim##n##ndim
#define COMPUTE_m16n1 COMPUTE_SIMPLE(16,1)
#define COMPUTE_m16n2 COMPUTE_SIMPLE(16,2)
#define COMPUTE_m16n4 COMPUTE_SIMPLE(16,4)
#define COMPUTE_m16n6 COMPUTE_SIMPLE(16,6)
#define COMPUTE_m16n8 COMPUTE_SIMPLE(16,8)
#define COMPUTE_m16n10 COMPUTE_SIMPLE(16,10)
#define COMPUTE_m16n12 \
  INIT_m16n12 "movq %%r13,%5; movq %%r14,%1; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15; movq %2,%3;"\
  "cmpq $16,%5; jb 7167123f; movq $16,%5;"\
  "7167121:\n\t"\
  KERNEL_k1m16n12 "addq $4,%5; testq $12,%5; movq $172,%%r10; cmovz %4,%%r10;"\
  KERNEL_k1m16n12 "prefetcht1 (%3); subq $129,%3; addq %%r10,%3;"\
  KERNEL_k1m16n12 "prefetcht1 (%6); addq $32,%6;"\
  KERNEL_k1m16n12 "cmpq %5,%%r13; jnb 7167121b;"\
  "movq %2,%3; negq %5; leaq 16(%%r13,%5,1),%5;"\
  "7167123:\n\t"\
  "testq %5,%5; jz 7167129f;"\
  "7167125:\n\t"\
  "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 127(%3);"\
  KERNEL_k1m16n12 "addq %4,%3; decq %5;jnz 7167125b;"\
  "7167129:\n\t"\
  "prefetcht0 (%%r14);" SAVE_m16n12
#define COMPUTE(ndim) {\
  b_pref = b_ptr + ndim * K;\
  __asm__ __volatile__(\
    "vbroadcastsd %8,%%zmm0; movq %7,%%r11; movq %1,%%r14; movq %5,%%r13; movq %5,%%r12; salq $4,%%r12;"\
    "cmpq $16,%%r11; jb "#ndim"33102f;"\
    #ndim"33101:\n\t"\
    COMPUTE_m16n##ndim "subq $16,%%r11; cmpq $16,%%r11; jnb "#ndim"33101b;"\
    #ndim"33102:\n\t"\
    "cmpq $8,%%r11; jb "#ndim"33103f;"\
    COMPUTE_SIMPLE(8,ndim) "subq $8,%%r11;"\
    #ndim"33103:\n\t"\
    "cmpq $4,%%r11; jb "#ndim"33104f;"\
    COMPUTE_SIMPLE(4,ndim) "subq $4,%%r11;"\
    #ndim"33104:\n\t"\
    "cmpq $2,%%r11; jb "#ndim"33105f;"\
    COMPUTE_SIMPLE(2,ndim) "subq $2,%%r11;"\
    #ndim"33105:\n\t"\
    "testq %%r11,%%r11; jz "#ndim"33106f;"\
    COMPUTE_SIMPLE(1,ndim) "subq $1,%%r11;"\
    #ndim"33106:\n\t"\
    "movq %%r14,%1; movq %%r13,%5;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(K),"+r"(b_pref):"m"(M),"m"(ALPHA):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += ndim * K; c_ptr += ndim * ldc - M;\
}

//#include "common.h"
#include <stdint.h>
#include <stdio.h>//debug
#include <stdlib.h>//debug
#define BLASLONG int//debug
int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, double alpha, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, BLASLONG ldc)
{
    if(m==0||n==0||k==0||alpha==0.0) return 0;
    int64_t ldc_in_bytes = (int64_t)ldc * sizeof(double); double ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    double *a_ptr = A,*b_ptr = B,*c_ptr = C,*c_tmp = C,*b_pref = B;
    for(;n_count>11;n_count-=12) COMPUTE(12)
    for(;n_count>9;n_count-=10) COMPUTE(10)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>5;n_count-=6) COMPUTE(6)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
    return 0;
}

/* test zone */
static void dgemm_tcopy_2(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    double *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 2;
      for(count_first=dim_first;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void dgemm_ncopy_2(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2;
    double *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;
    tosrc_inc=2*lead_dim-dim_first;
    for(count_second=dim_second;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    tosrc_inc-=lead_dim;
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
static void dgemm_tcopy_16(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    double *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 16;
      for(count_first=dim_first;count_first>15;count_first-=16){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        todst[4]=tosrc[4];todst[5]=tosrc[5];todst[6]=tosrc[6];todst[7]=tosrc[7];
        todst[8]=tosrc[8];todst[9]=tosrc[9];todst[10]=tosrc[10];todst[11]=tosrc[11];
        todst[12]=tosrc[12];todst[13]=tosrc[13];todst[14]=tosrc[14];todst[15]=tosrc[15];
        tosrc+=16;todst+=16*dim_second;
      }
      todst -= count_second * 8;
      for(;count_first>7;count_first-=8){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        todst[4]=tosrc[4];todst[5]=tosrc[5];todst[6]=tosrc[6];todst[7]=tosrc[7];
        tosrc+=8;todst+=8*dim_second;
      }
      todst -= count_second * 4;
      for(;count_first>3;count_first-=4){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void dgemm_ncopy_16(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8;
    double *tosrc9,*tosrc10,*tosrc11,*tosrc12,*tosrc13,*tosrc14,*tosrc15,*tosrc16;
    double *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc5=tosrc4+lead_dim;tosrc6=tosrc5+lead_dim;tosrc7=tosrc6+lead_dim;tosrc8=tosrc7+lead_dim;
    tosrc9=tosrc8+lead_dim;tosrc10=tosrc9+lead_dim;tosrc11=tosrc10+lead_dim;tosrc12=tosrc11+lead_dim;
    tosrc13=tosrc12+lead_dim;tosrc14=tosrc13+lead_dim;tosrc15=tosrc14+lead_dim;tosrc16=tosrc15+lead_dim;
    tosrc_inc=16*lead_dim-dim_first;
    for(count_second=dim_second;count_second>15;count_second-=16){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst[8]=*tosrc9;tosrc9++;todst[9]=*tosrc10;tosrc10++;
        todst[10]=*tosrc11;tosrc11++;todst[11]=*tosrc12;tosrc12++;
        todst[12]=*tosrc13;tosrc13++;todst[13]=*tosrc14;tosrc14++;
        todst[14]=*tosrc15;tosrc15++;todst[15]=*tosrc16;tosrc16++;
        todst+=16;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
      tosrc9+=tosrc_inc;tosrc10+=tosrc_inc;tosrc11+=tosrc_inc;tosrc12+=tosrc_inc;
      tosrc13+=tosrc_inc;tosrc14+=tosrc_inc;tosrc15+=tosrc_inc;tosrc16+=tosrc_inc;
    }
    tosrc_inc-=8*lead_dim;
    for(;count_second>7;count_second-=8){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst+=8;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
    }
    tosrc_inc-=4*lead_dim;
    for(;count_second>3;count_second-=4){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst+=4;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
    }
    tosrc_inc-=2*lead_dim;
    for(;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
static void SCALE_MULT(double *dat,double *sca, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0 || (*sca)==1.0) return;
    double scale = *sca; double *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale; current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
#define BLOCKDIM_K 320 //GEMM_Q in OpenBLAS
#define BLOCKDIM_M 320 //GEMM_P in OpenBLAS
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
//gcc-8 -march=skylake-avx512 --shared -fPIC -O2 dgemm_kernel_16x2_skylakex.c -o dgemm_16x2.so
void dgemm_(char *transa,char *transb,BLASLONG *m,BLASLONG *n,BLASLONG *k,double *alpha,double *a,BLASLONG *lda,double *b,BLASLONG *ldb,double *beta,double *c,BLASLONG *ldc){
    if((*m)==0||(*n)==0) return;
    if((*beta)!=1.0) SCALE_MULT(c,beta,*ldc,*m,*n);
    if((*alpha)==0.0||(*k)==0) return;
/* start main calculation here */
    double *b_buffer = (double *)aligned_alloc(64,BLOCKDIM_K*(*n)*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,BLOCKDIM_K*BLOCKDIM_M*sizeof(double));
    double *a_current_pos,*b_current_pos;
    BLASLONG m_count,n_count,k_count,k_subdim,m_subdim;
    b_current_pos = b;
    for(k_count=0;k_count<(*k);k_count+=BLOCKDIM_K){
      k_subdim = (*k)-k_count;
      if(k_subdim > BLOCKDIM_K) k_subdim = BLOCKDIM_K;
      if(NOTRANSB) { dgemm_ncopy_2(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { dgemm_tcopy_2(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { dgemm_tcopy_16(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { dgemm_ncopy_16(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        CNAME(m_subdim,*n,k_subdim,*alpha,a_buffer,b_buffer,c+m_count,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}
