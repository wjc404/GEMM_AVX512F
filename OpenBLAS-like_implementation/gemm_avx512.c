//gcc -march=skylake-avx512 --shared -fPIC -O2 kernel_avx512.c -o DGEMMv0.so


# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h>
/* row-major c_block */
/* 64-bit pointer registers: a_block_pointer,b_block_pointer,c_pointer;*/
#define INNER_KERNEL_k1m1n8 \
    "prefetcht0 384(%1);"\
    "prefetcht0 768(%0); vmovupd (%1),%%zmm5; addq $64,%1;"\
    "vbroadcastsd   (%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm8;"

#define INNER_KERNEL_k1m2n8 \
    INNER_KERNEL_k1m1n8\
    "vbroadcastsd  8(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm9;"

#define INNER_KERNEL_k1m4n8 \
    INNER_KERNEL_k1m2n8\
    "vbroadcastsd 16(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm10;"\
    "vbroadcastsd 24(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm11;"

#define INNER_KERNEL_k1m8n8 \
    INNER_KERNEL_k1m4n8\
    "vbroadcastsd 32(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm12;"\
    "vbroadcastsd 40(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm13;"\
    "vbroadcastsd 48(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm14;"\
    "vbroadcastsd 56(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm15;"

#define INNER_KERNEL_k1m1n16 \
    "prefetcht0 384(%1); prefetcht0 448(%1);"\
    "prefetcht0 768(%0); vmovupd (%1),%%zmm5; vmovupd 64(%1),%%zmm6; addq $128,%1;"\
    "vbroadcastsd   (%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm8; vfmadd231pd %%zmm6,%%zmm4,%%zmm9;"

#define INNER_KERNEL_k1m2n16 \
    INNER_KERNEL_k1m1n16\
    "vbroadcastsd  8(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm10;vfmadd231pd %%zmm6,%%zmm4,%%zmm11;"

#define INNER_KERNEL_k1m4n16 \
    INNER_KERNEL_k1m2n16\
    "vbroadcastsd 16(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm12;vfmadd231pd %%zmm6,%%zmm4,%%zmm13;"\
    "vbroadcastsd 24(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm14;vfmadd231pd %%zmm6,%%zmm4,%%zmm15;"

#define INNER_KERNEL_k1m8n16 \
    INNER_KERNEL_k1m4n16\
    "vbroadcastsd 32(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm16;vfmadd231pd %%zmm6,%%zmm4,%%zmm17;"\
    "vbroadcastsd 40(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm18;vfmadd231pd %%zmm6,%%zmm4,%%zmm19;"\
    "vbroadcastsd 48(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm20;vfmadd231pd %%zmm6,%%zmm4,%%zmm21;"\
    "vbroadcastsd 56(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm22;vfmadd231pd %%zmm6,%%zmm4,%%zmm23;"

#define INNER_KERNEL_k1m1n24 \
    "prefetcht0 384(%1); prefetcht0 448(%1); prefetcht0 512(%1);"\
    "prefetcht0 768(%0); vmovupd (%1),%%zmm5; vmovupd 64(%1),%%zmm6; vmovupd 128(%1),%%zmm7; addq $192,%1;"\
    "vbroadcastsd   (%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm8; vfmadd231pd %%zmm6,%%zmm4,%%zmm9; vfmadd231pd %%zmm7,%%zmm4,%%zmm10;"

#define INNER_KERNEL_k1m2n24 \
    INNER_KERNEL_k1m1n24\
    "vbroadcastsd  8(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm11;vfmadd231pd %%zmm6,%%zmm4,%%zmm12;vfmadd231pd %%zmm7,%%zmm4,%%zmm13;"

#define INNER_KERNEL_k1m4n24 \
    INNER_KERNEL_k1m2n24\
    "vbroadcastsd 16(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm14;vfmadd231pd %%zmm6,%%zmm4,%%zmm15;vfmadd231pd %%zmm7,%%zmm4,%%zmm16;"\
    "vbroadcastsd 24(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm17;vfmadd231pd %%zmm6,%%zmm4,%%zmm18;vfmadd231pd %%zmm7,%%zmm4,%%zmm19;"

#define INNER_KERNEL_k1m8n24 \
    INNER_KERNEL_k1m4n24\
    "vbroadcastsd 32(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm20;vfmadd231pd %%zmm6,%%zmm4,%%zmm21;vfmadd231pd %%zmm7,%%zmm4,%%zmm22;"\
    "vbroadcastsd 40(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm23;vfmadd231pd %%zmm6,%%zmm4,%%zmm24;vfmadd231pd %%zmm7,%%zmm4,%%zmm25;"\
    "vbroadcastsd 48(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm26;vfmadd231pd %%zmm6,%%zmm4,%%zmm27;vfmadd231pd %%zmm7,%%zmm4,%%zmm28;"\
    "vbroadcastsd 56(%0),%%zmm4;vfmadd231pd %%zmm5,%%zmm4,%%zmm29;vfmadd231pd %%zmm6,%%zmm4,%%zmm30;vfmadd231pd %%zmm7,%%zmm4,%%zmm31;"

#define INNER_KERNELm1(nn) \
    "cmpq $1,%2;jb "#nn"3f;"\
    #nn"4:\n\t"\
    INNER_KERNEL_k1m1n##nn "addq $8,%0;"\
    "decq %2;cmpq $1,%2;jnb "#nn"4b;"\
    #nn"3:\n\t"

#define INNER_KERNELm2(nn) \
    "cmpq $1,%2;jb "#nn"0f;"\
    #nn"1:\n\t"\
    INNER_KERNEL_k1m2n##nn "addq $16,%0;"\
    "decq %2;cmpq $1,%2;jnb "#nn"1b;"\
    #nn"0:\n\t"

#define INNER_KERNELm4(nn) \
    "cmpq $1,%2;jb "#nn"00f;"\
    #nn"01:\n\t"\
    INNER_KERNEL_k1m4n##nn "addq $32,%0;"\
    "decq %2;cmpq $1,%2;jnb "#nn"01b;"\
    #nn"00:\n\t"

#define INNER_KERNELm8(nn) \
    "cmpq $8,%2;jb "#nn"001f;"\
    #nn"008:\n\t"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    "subq $8,%2;cmpq $8,%2;jnb "#nn"008b;"\
    #nn"001:\n\t"\
    "cmpq $1,%2;jb "#nn"000f;"\
    INNER_KERNEL_k1m8n##nn "addq $64,%0;"\
    "decq %2;cmpq $1,%2;jnb "#nn"001b;"\
    ""#nn"000:\n\t"

#define INNER_INIT_m1n8 \
    "vpxorq %%zmm8, %%zmm8, %%zmm8;"

#define INNER_INIT_m2n8 \
    "vpxorq %%zmm8, %%zmm8, %%zmm8; vpxorq %%zmm9, %%zmm9, %%zmm9;"

#define INNER_INIT_m4n8 \
    "vpxorq %%zmm8, %%zmm8, %%zmm8; vpxorq %%zmm9, %%zmm9, %%zmm9; vpxorq %%zmm10,%%zmm10,%%zmm10;vpxorq %%zmm11,%%zmm11,%%zmm11;"

#define INNER_INIT_m8n8 \
    INNER_INIT_m4n8\
    "vpxorq %%zmm12,%%zmm12,%%zmm12;vpxorq %%zmm13,%%zmm13,%%zmm13;vpxorq %%zmm14,%%zmm14,%%zmm14;vpxorq %%zmm15,%%zmm15,%%zmm15;"

#define INNER_INIT_m1n16 INNER_INIT_m2n8

#define INNER_INIT_m2n16 INNER_INIT_m4n8

#define INNER_INIT_m4n16 INNER_INIT_m8n8

#define INNER_INIT_m8n16 \
    INNER_INIT_m8n8\
    "vpxorq %%zmm16,%%zmm16,%%zmm16;vpxorq %%zmm17,%%zmm17,%%zmm17;vpxorq %%zmm18,%%zmm18,%%zmm18;vpxorq %%zmm19,%%zmm19,%%zmm19;"\
    "vpxorq %%zmm20,%%zmm20,%%zmm20;vpxorq %%zmm21,%%zmm21,%%zmm21;vpxorq %%zmm22,%%zmm22,%%zmm22;vpxorq %%zmm23,%%zmm23,%%zmm23;"

#define INNER_INIT_m1n24 \
    "vpxorq %%zmm8, %%zmm8, %%zmm8; vpxorq %%zmm9, %%zmm9, %%zmm9; vpxorq %%zmm10,%%zmm10,%%zmm10;"

#define INNER_INIT_m2n24 \
    INNER_INIT_m1n24\
    "vpxorq %%zmm11,%%zmm11,%%zmm11; vpxorq %%zmm12,%%zmm12,%%zmm12; vpxorq %%zmm13,%%zmm13,%%zmm13;"

#define INNER_INIT_m4n24 \
    INNER_INIT_m4n16\
    "vpxorq %%zmm16,%%zmm16,%%zmm16;vpxorq %%zmm17,%%zmm17,%%zmm17;vpxorq %%zmm18,%%zmm18,%%zmm18;vpxorq %%zmm19,%%zmm19,%%zmm19;"

#define INNER_INIT_m8n24 \
    INNER_INIT_m8n16\
    "vpxorq %%zmm24,%%zmm24,%%zmm24;vpxorq %%zmm25,%%zmm25,%%zmm25;vpxorq %%zmm26,%%zmm26,%%zmm26;vpxorq %%zmm27,%%zmm27,%%zmm27;"\
    "vpxorq %%zmm28,%%zmm28,%%zmm28;vpxorq %%zmm29,%%zmm29,%%zmm29;vpxorq %%zmm30,%%zmm30,%%zmm30;vpxorq %%zmm31,%%zmm31,%%zmm31;"

#define INNER_SETINDEX \
    "vpinsrq $0,%4,%%xmm4,%%xmm4; vbroadcastsd %%xmm4,%%zmm4;"\
    "kxnorw %%k1,%%k1,%%k1; kshiftlw $1,%%k1,%%k1; vpxorq %%zmm6,%%zmm6,%%zmm6; vmovapd %%zmm4,%%zmm6%{%%k1%};"\
    "kshiftlw $1,%%k1,%%k1; vpaddq %%zmm4,%%zmm6,%%zmm6%{%%k1%};"\
    "kshiftlw $1,%%k1,%%k1; vpaddq %%zmm4,%%zmm6,%%zmm6%{%%k1%};"\
    "kshiftlw $1,%%k1,%%k1; vpaddq %%zmm4,%%zmm6,%%zmm6%{%%k1%};"\
    "kshiftlw $1,%%k1,%%k1; vpaddq %%zmm4,%%zmm6,%%zmm6%{%%k1%};"\
    "kshiftlw $1,%%k1,%%k1; vpaddq %%zmm4,%%zmm6,%%zmm6%{%%k1%};"\
    "kshiftlw $1,%%k1,%%k1; vpaddq %%zmm4,%%zmm6,%%zmm6%{%%k1%};"

#define INNER_STORE_m1n8(c1,disp) \
    "kxnorw %%k1,%%k1,%%k1;"\
    "vgatherqpd "#disp"(%3,%%zmm6,1), %%zmm7 %{%%k1%};"\
    "vaddpd %%zmm7,"#c1","#c1";"\
    "kxnorw %%k1,%%k1,%%k1;"\
    "vscatterqpd "#c1", "#disp"(%3,%%zmm6,1) %{%%k1%};"

#define INNER_SAVE_m1n8 \
    INNER_SETINDEX\
    INNER_STORE_m1n8(%%zmm8,0)

#define INNER_SAVE_m1n16 \
    INNER_SAVE_m1n8\
    "leaq (%3,%4,8),%3;"\
    INNER_STORE_m1n8(%%zmm9,0)

#define INNER_SAVE_m1n24 \
    INNER_SAVE_m1n16\
    "leaq (%3,%4,8),%3;"\
    INNER_STORE_m1n8(%%zmm10,0)

#define INNER_SAVE_m2n8 \
    INNER_SETINDEX\
    INNER_STORE_m1n8(%%zmm8,0)\
    INNER_STORE_m1n8(%%zmm9,8)

#define INNER_SAVE_m2n16 \
    INNER_SETINDEX\
    INNER_STORE_m1n8(%%zmm8,0)\
    INNER_STORE_m1n8(%%zmm10,8)\
    "leaq (%3,%4,8),%3;"\
    INNER_STORE_m1n8(%%zmm9,0)\
    INNER_STORE_m1n8(%%zmm11,8)

#define INNER_SAVE_m2n24 \
    INNER_SETINDEX\
    INNER_STORE_m1n8(%%zmm8,0)\
    INNER_STORE_m1n8(%%zmm11,8)\
    "leaq (%3,%4,8),%3;"\
    INNER_STORE_m1n8(%%zmm9,0)\
    INNER_STORE_m1n8(%%zmm12,8)\
    "leaq (%3,%4,8),%3;"\
    INNER_STORE_m1n8(%%zmm10,0)\
    INNER_STORE_m1n8(%%zmm13,8)

#define INNER_PREF_8x8 \
    "prefetcht0 (%3); prefetcht0 56(%3); prefetcht0 (%3,%4,1); prefetcht0 56(%3,%4,1); prefetcht0 (%3,%4,2); prefetcht0 56(%3,%4,2);"\
    "prefetcht0 (%3,%4,4); prefetcht0 56(%3,%4,4); leaq (%3,%4,2),%3;"\
    "prefetcht0 (%3,%4,1); prefetcht0 56(%3,%4,1); prefetcht0 (%3,%4,4); prefetcht0 56(%3,%4,4); leaq (%3,%4,1),%3;"\
    "prefetcht0 (%3,%4,2); prefetcht0 56(%3,%4,2); prefetcht0 (%3,%4,4); prefetcht0 56(%3,%4,4);"\
    "subq %4,%3; subq %4,%3; subq %4,%3;"

#define INNER_TRANS_4x8(c1,c2,c3,c4) \
    "vunpcklpd "#c2","#c1",%%zmm4;vunpckhpd "#c2","#c1",%%zmm5;vunpcklpd "#c4","#c3",%%zmm6;vunpckhpd "#c4","#c3",%%zmm7;"\
    "vblendmpd %%zmm6,%%zmm4,"#c1"%{%6%};vblendmpd %%zmm7,%%zmm5,"#c3"%{%6%};"\
    "vshuff64x2 $0xb1,"#c1","#c1","#c1";vshuff64x2 $0xb1,"#c3","#c3","#c3";"\
    "vblendmpd %%zmm4,"#c1",%%zmm4%{%6%};vblendmpd %%zmm5,"#c3","#c2"%{%6%};"\
    "vblendmpd "#c1",%%zmm6,%%zmm6%{%6%};vblendmpd "#c3",%%zmm7,"#c4"%{%6%};"\
    "vmovapd %%zmm4,"#c1"; vmovapd %%zmm6,"#c3";"

#define INNER_TRANS_8x8(c1,c2,c3,c4,c5,c6,c7,c8) \
    INNER_TRANS_4x8(c1,c2,c3,c4)\
    INNER_TRANS_4x8(c5,c6,c7,c8)\
    "vblendmpd "#c5","#c1",%%zmm4%{%5%};vshuff64x2 $0x4e,%%zmm4,%%zmm4,%%zmm4;"\
    "vblendmpd "#c1",%%zmm4,"#c1"%{%5%};vblendmpd  %%zmm4,"#c5","#c5"%{%5%};"\
    "vblendmpd "#c6","#c2",%%zmm5%{%5%};vshuff64x2 $0x4e,%%zmm5,%%zmm5,%%zmm5;"\
    "vblendmpd "#c2",%%zmm5,"#c2"%{%5%};vblendmpd  %%zmm5,"#c6","#c6"%{%5%};"\
    "vblendmpd "#c7","#c3",%%zmm6%{%5%};vshuff64x2 $0x4e,%%zmm6,%%zmm6,%%zmm6;"\
    "vblendmpd "#c3",%%zmm6,"#c3"%{%5%};vblendmpd  %%zmm6,"#c7","#c7"%{%5%};"\
    "vblendmpd "#c8","#c4",%%zmm7%{%5%};vshuff64x2 $0x4e,%%zmm7,%%zmm7,%%zmm7;"\
    "vblendmpd "#c4",%%zmm7,"#c4"%{%5%};vblendmpd  %%zmm7,"#c8","#c8"%{%5%};"

//%7 for k01(input) only when m=4
#define INNER_STORE_4x8(c1,c2,c3,c4) \
    "vmovupd (%3),%%zmm4%{%5%};vmovupd -32(%3,%4,4),%%zmm4%{%7%};vaddpd %%zmm4,"#c1","#c1";"\
    "vmovupd "#c1",(%3)%{%5%}; vmovupd "#c1",-32(%3,%4,4)%{%7%}; leaq (%3,%4,1),%3;"\
    "vmovupd (%3),%%zmm5%{%5%};vmovupd -32(%3,%4,4),%%zmm5%{%7%};vaddpd %%zmm5,"#c2","#c2";"\
    "vmovupd "#c2",(%3)%{%5%}; vmovupd "#c2",-32(%3,%4,4)%{%7%}; leaq (%3,%4,1),%3;"\
    "vmovupd (%3),%%zmm6%{%5%};vmovupd -32(%3,%4,4),%%zmm6%{%7%};vaddpd %%zmm6,"#c3","#c3";"\
    "vmovupd "#c3",(%3)%{%5%}; vmovupd "#c3",-32(%3,%4,4)%{%7%}; leaq (%3,%4,1),%3;"\
    "vmovupd (%3),%%zmm7%{%5%};vmovupd -32(%3,%4,4),%%zmm7%{%7%};vaddpd %%zmm7,"#c4","#c4";"\
    "vmovupd "#c4",(%3)%{%5%}; vmovupd "#c4",-32(%3,%4,4)%{%7%}; leaq (%3,%4,1),%3;"\
    "leaq (%3,%4,4),%3;"

#define INNER_STORE_8x8(c1,c2,c3,c4,c5,c6,c7,c8) \
    "prefetcht1 120(%3); prefetcht1 120(%3,%4,1);"\
    "vaddpd (%3),"#c1","#c1"; vmovupd "#c1",(%3); vaddpd (%3,%4,1),"#c2","#c2"; vmovupd "#c2",(%3,%4,1); leaq (%3,%4,2),%3;"\
    "prefetcht1 120(%3); prefetcht1 120(%3,%4,1);"\
    "vaddpd (%3),"#c3","#c3"; vmovupd "#c3",(%3); vaddpd (%3,%4,1),"#c4","#c4"; vmovupd "#c4",(%3,%4,1); leaq (%3,%4,2),%3;"\
    "prefetcht1 120(%3); prefetcht1 120(%3,%4,1);"\
    "vaddpd (%3),"#c5","#c5"; vmovupd "#c5",(%3); vaddpd (%3,%4,1),"#c6","#c6"; vmovupd "#c6",(%3,%4,1); leaq (%3,%4,2),%3;"\
    "prefetcht1 120(%3); prefetcht1 120(%3,%4,1);"\
    "vaddpd (%3),"#c7","#c7"; vmovupd "#c7",(%3); vaddpd (%3,%4,1),"#c8","#c8"; vmovupd "#c8",(%3,%4,1); leaq (%3,%4,2),%3;"

#define INNER_SAVE_m4n8 \
    INNER_TRANS_4x8(%%zmm8,%%zmm9,%%zmm10,%%zmm11)\
    INNER_STORE_4x8(%%zmm8,%%zmm9,%%zmm10,%%zmm11)

#define INNER_SAVE_m4n16 \
    INNER_TRANS_4x8(%%zmm8,%%zmm10,%%zmm12,%%zmm14)\
    INNER_STORE_4x8(%%zmm8,%%zmm10,%%zmm12,%%zmm14)\
    INNER_TRANS_4x8(%%zmm9,%%zmm11,%%zmm13,%%zmm15)\
    INNER_STORE_4x8(%%zmm9,%%zmm11,%%zmm13,%%zmm15)

#define INNER_SAVE_m4n24 \
    INNER_TRANS_4x8(%%zmm8,%%zmm11,%%zmm14,%%zmm17)\
    INNER_STORE_4x8(%%zmm8,%%zmm11,%%zmm14,%%zmm17)\
    INNER_TRANS_4x8(%%zmm9,%%zmm12,%%zmm15,%%zmm18)\
    INNER_STORE_4x8(%%zmm9,%%zmm12,%%zmm15,%%zmm18)\
    INNER_TRANS_4x8(%%zmm10,%%zmm13,%%zmm16,%%zmm19)\
    INNER_STORE_4x8(%%zmm10,%%zmm13,%%zmm16,%%zmm19)

#define INNER_SAVE_m8n8 \
    INNER_PREF_8x8\
    INNER_TRANS_8x8(%%zmm8,%%zmm9,%%zmm10,%%zmm11,%%zmm12,%%zmm13,%%zmm14,%%zmm15)\
    INNER_STORE_8x8(%%zmm8,%%zmm9,%%zmm10,%%zmm11,%%zmm12,%%zmm13,%%zmm14,%%zmm15)

#define INNER_SAVE_m8n16 \
    INNER_PREF_8x8\
    INNER_TRANS_8x8(%%zmm8,%%zmm10,%%zmm12,%%zmm14,%%zmm16,%%zmm18,%%zmm20,%%zmm22)\
    INNER_STORE_8x8(%%zmm8,%%zmm10,%%zmm12,%%zmm14,%%zmm16,%%zmm18,%%zmm20,%%zmm22)\
    INNER_PREF_8x8\
    INNER_TRANS_8x8(%%zmm9,%%zmm11,%%zmm13,%%zmm15,%%zmm17,%%zmm19,%%zmm21,%%zmm23)\
    INNER_STORE_8x8(%%zmm9,%%zmm11,%%zmm13,%%zmm15,%%zmm17,%%zmm19,%%zmm21,%%zmm23)

#define INNER_SAVE_m8n24 \
    INNER_PREF_8x8\
    INNER_TRANS_8x8(%%zmm8,%%zmm11,%%zmm14,%%zmm17,%%zmm20,%%zmm23,%%zmm26,%%zmm29)\
    INNER_STORE_8x8(%%zmm8,%%zmm11,%%zmm14,%%zmm17,%%zmm20,%%zmm23,%%zmm26,%%zmm29)\
    INNER_PREF_8x8\
    INNER_TRANS_8x8(%%zmm9,%%zmm12,%%zmm15,%%zmm18,%%zmm21,%%zmm24,%%zmm27,%%zmm30)\
    INNER_STORE_8x8(%%zmm9,%%zmm12,%%zmm15,%%zmm18,%%zmm21,%%zmm24,%%zmm27,%%zmm30)\
    INNER_PREF_8x8\
    INNER_TRANS_8x8(%%zmm10,%%zmm13,%%zmm16,%%zmm19,%%zmm22,%%zmm25,%%zmm28,%%zmm31)\
    INNER_STORE_8x8(%%zmm10,%%zmm13,%%zmm16,%%zmm19,%%zmm22,%%zmm25,%%zmm28,%%zmm31)

#define COMPUTE_m1n8 {\
    __asm__ __volatile__(\
    INNER_INIT_m1n8\
    INNER_KERNELm1(8)\
    INNER_SAVE_m1n8\
    :"+r"(a_block_pointer):"r"(packed_b_pointer),"r"((int64_t)k),"r"(c_pointer),"r"(ldc_in_bytes)\
    :"zmm4","zmm5","zmm6","zmm7","zmm8","cc","memory","k1");\
    c_pointer += 1;\
}
#define COMPUTE_m2n8 {\
    __asm__ __volatile__(\
    INNER_INIT_m2n8\
    INNER_KERNELm2(8)\
    INNER_SAVE_m2n8\
    :"+r"(a_block_pointer):"r"(packed_b_pointer),"r"((int64_t)k),"r"(c_pointer),"r"(ldc_in_bytes)\
    :"zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","cc","memory","k1");\
    c_pointer += 2;\
}
#define COMPUTE_m4n8 {\
    __asm__ __volatile__(\
    INNER_INIT_m4n8\
    INNER_KERNELm4(8)\
    INNER_SAVE_m4n8\
    :"+r"(a_block_pointer):"r"(packed_b_pointer),"r"((int64_t)k),"r"(c_pointer),"r"(ldc_in_bytes),"Yk"(k02),"Yk"(k03),"Yk"(k01)\
    :"zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","cc","memory");\
    c_pointer += 4;\
}
#define COMPUTE_m8n8 {\
    __asm__ __volatile__(\
    INNER_INIT_m8n8\
    INNER_KERNELm8(8)\
    INNER_SAVE_m8n8\
    :"+r"(a_block_pointer):"r"(packed_b_pointer),"r"((int64_t)k),"r"(c_pointer),"r"(ldc_in_bytes),"Yk"(k02),"Yk"(k03)\
    :"zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","cc","memory");\
    c_pointer += 8;\
}


//%8 = m(input,r), the rest retain
#define COMPUTE_n8 {\
    __asm__ __volatile__(\
    "movq %8,%%r14;movq %2,%%r13;"\
    "cmpq $8,%8; jb 42222f;"\
    "42221:\n\t"\
    INNER_INIT_m8n8\
    INNER_KERNELm8(8)\
    INNER_SAVE_m8n8\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shrq $6,%2;"\
    "shlq $3,%4;subq %4,%3;shrq $3,%4;addq $64,%3;"\
    "subq $8,%8; cmpq $8,%8; jnb 42221b;"\
    "42222:\n\t"\
    "cmpq $4,%8; jb 42223f;"\
    INNER_INIT_m4n8\
    INNER_KERNELm4(8)\
    INNER_SAVE_m4n8\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shrq $6,%2;"\
    "shlq $3,%4;subq %4,%3;shrq $3,%4;addq $32,%3;"\
    "subq $4,%8;"\
    "42223:\n\t"\
    "cmpq $2,%8; jb 42224f;"\
    INNER_INIT_m2n8\
    INNER_KERNELm2(8)\
    INNER_SAVE_m2n8\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shrq $6,%2;"\
    "addq $16,%3;"\
    "subq $2,%8;"\
    "42224:\n\t"\
    "cmpq $1,%8; jb 42225f;"\
    INNER_INIT_m1n8\
    INNER_KERNELm1(8)\
    INNER_SAVE_m1n8\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shrq $6,%2;"\
    "addq $8,%3;"\
    "42225:\n\t"\
    "movq %%r14,%8;shlq $3,%8;subq %8,%3;shrq $3,%8;"\
    "shlq $3,%4;addq %4,%3;shrq $3,%4;"\
    :"+r"(a_block_pointer),"+r"(packed_b_pointer),"+r"(K),"+r"(c_pointer),"+r"(ldc_in_bytes),"+Yk"(k02),"+Yk"(k03),"+Yk"(k01),"+r"(M)\
    ::"zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","cc","memory","k1","r13","r14");\
    a_block_pointer -= M * K;\
}
#define COMPUTE_n16 {\
    __asm__ __volatile__(\
    "movq %8,%%r14;movq %2,%%r13;"\
    "cmpq $8,%8; jb 32222f;"\
    "32221:\n\t"\
    INNER_INIT_m8n16\
    INNER_KERNELm8(16)\
    INNER_SAVE_m8n16\
    "movq %%r13,%2; shlq $7,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $4,%4;subq %4,%3;shrq $4,%4;addq $64,%3;"\
    "subq $8,%8; cmpq $8,%8; jnb 32221b;"\
    "32222:\n\t"\
    "cmpq $4,%8; jb 32223f;"\
    INNER_INIT_m4n16\
    INNER_KERNELm4(16)\
    INNER_SAVE_m4n16\
    "movq %%r13,%2; shlq $7,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $4,%4;subq %4,%3;shrq $4,%4;addq $32,%3;"\
    "subq $4,%8;"\
    "32223:\n\t"\
    "cmpq $2,%8; jb 32224f;"\
    INNER_INIT_m2n16\
    INNER_KERNELm2(16)\
    INNER_SAVE_m2n16\
    "movq %%r13,%2; shlq $7,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $3,%4;subq %4,%3;shrq $3,%4;addq $16,%3;"\
    "subq $2,%8;"\
    "32224:\n\t"\
    "cmpq $1,%8; jb 32225f;"\
    INNER_INIT_m1n16\
    INNER_KERNELm1(16)\
    INNER_SAVE_m1n16\
    "movq %%r13,%2; shlq $7,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $3,%4;subq %4,%3;shrq $3,%4;addq $8,%3;"\
    "32225:\n\t"\
    "movq %%r14,%8;shlq $3,%8;subq %8,%3;shrq $3,%8;"\
    "shlq $4,%4;addq %4,%3;shrq $4,%4;"\
    :"+r"(a_block_pointer),"+r"(b_scratch),"+r"(K),"+r"(c_pointer),"+r"(ldc_in_bytes),"+Yk"(k02),"+Yk"(k03),"+Yk"(k01),"+r"(M)\
    ::"zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17",\
    "zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","cc","memory","k1","r13","r14");\
    a_block_pointer -= M * K;\
}
#define COMPUTE_n24 {\
    __asm__ __volatile__(\
    "movq %8,%%r14;movq %2,%%r13;"\
    "cmpq $8,%8; jb 22222f;"\
    "22221:\n\t"\
    INNER_INIT_m8n24\
    INNER_KERNELm8(24)\
    INNER_SAVE_m8n24\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shlq $1,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $3,%4;subq %4,%3;shlq $1,%4;subq %4,%3;shrq $4,%4;addq $64,%3;"\
    "subq $8,%8; cmpq $8,%8; jnb 22221b;"\
    "22222:\n\t"\
    "cmpq $4,%8; jb 22223f;"\
    INNER_INIT_m4n24\
    INNER_KERNELm4(24)\
    INNER_SAVE_m4n24\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shlq $1,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $3,%4;subq %4,%3;shlq $1,%4;subq %4,%3;shrq $4,%4;addq $32,%3;"\
    "subq $4,%8;"\
    "22223:\n\t"\
    "cmpq $2,%8; jb 22224f;"\
    INNER_INIT_m2n24\
    INNER_KERNELm2(24)\
    INNER_SAVE_m2n24\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shlq $1,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $4,%4;subq %4,%3;shrq $4,%4;addq $16,%3;"\
    "subq $2,%8;"\
    "22224:\n\t"\
    "cmpq $1,%8; jb 22225f;"\
    INNER_INIT_m1n24\
    INNER_KERNELm1(24)\
    INNER_SAVE_m1n24\
    "movq %%r13,%2; shlq $6,%2;subq %2,%1;shlq $1,%2;subq %2,%1;shrq $7,%2;"\
    "shlq $4,%4;subq %4,%3;shrq $4,%4;addq $8,%3;"\
    "22225:\n\t"\
    "movq %%r14,%8;shlq $3,%8;subq %8,%3;shrq $3,%8;"\
    "shlq $3,%4;addq %4,%3;shlq $1,%4;addq %4,%3;shrq $4,%4;"\
    :"+r"(a_block_pointer),"+r"(b_scratch),"+r"(K),"+r"(c_pointer),"+r"(ldc_in_bytes),"+Yk"(k02),"+Yk"(k03),"+Yk"(k01),"+r"(M)\
    ::"zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19",\
    "zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31","cc","memory","k1","r13","r14");\
    a_block_pointer -= M * K;\
}


#define BLASLONG int //debug

static void __attribute__ ((noinline)) KERNEL_MAIN(double *packed_a, double *packed_b, BLASLONG m, BLASLONG ndiv8, BLASLONG k, BLASLONG LDC, double *c){//icopy=8,ocopy=8
//perform C += A<pack> B<pack>
    if(k==0 || m==0 || ndiv8==0) return;
    int64_t ldc_in_bytes = (int64_t)LDC * sizeof(double);
    int64_t K = (int64_t)k; int64_t M = (int64_t)m;
    double *a_block_pointer,*b_block_pointer;
    double *c_pointer = c;
    __mmask16 k01 = 0x00f0,k02 = 0x000f,k03 = 0x0033;
    BLASLONG m_count,ndiv8_count,k_count;
    double *b_scratch = (double *)aligned_alloc(64,192*k);
    double *packed_b_pointer = packed_b;
    a_block_pointer = packed_a;
    for(ndiv8_count=ndiv8;ndiv8_count>2;ndiv8_count-=3){
      __asm__ __volatile__ (
        "testq %2,%2; jz 100002f;movq %2,%%r13;shlq $6,%%r13;"
        "100001:\n\t"
        "vmovupd (%0),%%zmm5; vmovupd (%0,%%r13,1),%%zmm6; vmovupd (%0,%%r13,2),%%zmm7; addq $64,%0;"
        "vmovupd %%zmm5,(%1); vmovupd %%zmm6,64(%1); vmovupd %%zmm7,128(%1); addq $192,%1;"
        "decq %2; testq %2,%2; jnz 100001b;"
        "100002:\n\t"
        "movq %%r13,%2;shrq $6,%2;leaq (%0,%%r13,2),%0;subq %%r13,%1;subq %%r13,%1;subq %%r13,%1;"
        :"+r"(packed_b_pointer),"+r"(b_scratch),"+r"(K)::"r13","cc","memory","zmm5","zmm6","zmm7");
      COMPUTE_n24
    }
    for(;ndiv8_count>1;ndiv8_count-=2){
      __asm__ __volatile__ (
        "testq %2,%2; jz 1000002f;movq %2,%%r13;shlq $6,%%r13;"
        "1000001:\n\t"
        "vmovupd (%0),%%zmm5; vmovupd (%0,%%r13,1),%%zmm6; addq $64,%0;"
        "vmovupd %%zmm5,(%1); vmovupd %%zmm6,64(%1); addq $128,%1;"
        "decq %2; testq %2,%2; jnz 1000001b;"
        "1000002:\n\t"
        "movq %%r13,%2;shrq $6,%2;leaq (%0,%%r13,1),%0;subq %%r13,%1;subq %%r13,%1;"
        :"+r"(packed_b_pointer),"+r"(b_scratch),"+r"(K)::"r13","cc","memory","zmm5","zmm6");
      COMPUTE_n16
    }
    if(ndiv8_count>0){
      COMPUTE_n8
    }
    free(b_scratch);b_scratch=NULL;
}

/* __m512d accumulators: zc1-zc4; temporary variables: za1,zb1-zb2 */
/* __m256d accumulators: yc1-yc4; temporary variables: ya1,yb1-yb2 */
/* __m128d accumulators: xc1-xc4; temporary variables: xa1,xb1-xb2 */
/*  double accumulator:  sc1;     temporary variables: sa1,sb1 */
/* column-major c_block */
#define KERNEL_m8n4k1 {\
    __asm__ __volatile__(\
    "vmovupd (%0),%2; addq $64,%0;"\
    "vbroadcastsd   (%1),%3; vfmadd231pd %2,%3,%5; "\
    "vbroadcastsd  8(%1),%4; vfmadd231pd %2,%4,%6; "\
    "vbroadcastsd 16(%1),%3; vfmadd231pd %2,%3,%7; "\
    "vbroadcastsd 24(%1),%4; vfmadd231pd %2,%4,%8; "\
    "addq $32,%1;"\
    :"+r"(a_block_pointer),"+r"(b_block_pointer),"+v"(za1),"+v"(zb1),"+v"(zb2),"+v"(zc1),"+v"(zc2),"+v"(zc3),"+v"(zc4)::"cc","memory");\
}
#define KERNEL_m8n2k1 {\
    __asm__ __volatile__(\
    "vmovupd (%0),%2; addq $64,%0;"\
    "vbroadcastsd   (%1),%3; vfmadd231pd %2,%3,%5; "\
    "vbroadcastsd  8(%1),%4; vfmadd231pd %2,%4,%6; "\
    "addq $16,%1;"\
    :"+r"(a_block_pointer),"+r"(b_block_pointer),"+v"(za1),"+v"(zb1),"+v"(zb2),"+v"(zc1),"+v"(zc2)::"cc","memory");\
}
#define KERNEL_m8n1k1 {\
    __asm__ __volatile__(\
    "vmovupd (%0),%2; addq $64,%0;"\
    "vbroadcastsd   (%1),%3; vfmadd231pd %2,%3,%4; "\
    "addq $8,%1;"\
    :"+r"(a_block_pointer),"+r"(b_block_pointer),"+v"(za1),"+v"(zb1),"+v"(zc1)::"cc","memory");\
}
#define INIT_m8n1 zc1=_mm512_setzero_pd();
#define INIT_m8n2 zc2=INIT_m8n1
#define INIT_m8n4 zc4=zc3=INIT_m8n2
#define SAVE_m8n1 {\
    za1 = _mm512_loadu_pd(c_pointer);\
    zc1 = _mm512_add_pd(zc1,za1);\
    _mm512_storeu_pd(c_pointer,zc1);\
    c_pointer += 8;\
}
#define SAVE_m8n2 {\
    zb1 = _mm512_loadu_pd(c_pointer); zb2 = _mm512_loadu_pd(c_pointer+LDC);\
    zc1 = _mm512_add_pd(zc1,zb1); zc2 = _mm512_add_pd(zc2,zb2);\
    _mm512_storeu_pd(c_pointer,zc1); _mm512_storeu_pd(c_pointer+LDC,zc2);\
    c_pointer += 8;\
}
#define SAVE_m8n4 {\
    zb1 = _mm512_loadu_pd(c_pointer); zb2 = _mm512_loadu_pd(c_pointer+LDC);\
    zc1 = _mm512_add_pd(zc1,zb1); zc2 = _mm512_add_pd(zc2,zb2);\
    _mm512_storeu_pd(c_pointer,zc1); _mm512_storeu_pd(c_pointer+LDC,zc2);\
    c_pointer += LDC*2;\
    zb1 = _mm512_loadu_pd(c_pointer); zb2 = _mm512_loadu_pd(c_pointer+LDC);\
    zc3 = _mm512_add_pd(zc3,zb1); zc4 = _mm512_add_pd(zc4,zb2);\
    _mm512_storeu_pd(c_pointer,zc3); _mm512_storeu_pd(c_pointer+LDC,zc4);\
    c_pointer += 8-LDC*2;\
}
#define KERNEL_m4n4k1 {\
    ya1 = _mm256_loadu_pd(a_block_pointer);a_block_pointer+=4;\
    yb1 = _mm256_broadcast_sd(b_block_pointer);   yc1 = _mm256_fmadd_pd(ya1,yb1,yc1);\
    yb2 = _mm256_broadcast_sd(b_block_pointer+1); yc2 = _mm256_fmadd_pd(ya1,yb2,yc2);\
    yb1 = _mm256_broadcast_sd(b_block_pointer+2); yc3 = _mm256_fmadd_pd(ya1,yb1,yc3);\
    yb2 = _mm256_broadcast_sd(b_block_pointer+3); yc4 = _mm256_fmadd_pd(ya1,yb2,yc4);\
    b_block_pointer+=4;\
}
#define KERNEL_m4n2k1 {\
    ya1 = _mm256_loadu_pd(a_block_pointer);a_block_pointer+=4;\
    yb1 = _mm256_broadcast_sd(b_block_pointer);   yc1 = _mm256_fmadd_pd(ya1,yb1,yc1);\
    yb2 = _mm256_broadcast_sd(b_block_pointer+1); yc2 = _mm256_fmadd_pd(ya1,yb2,yc2);\
    b_block_pointer+=2;\
}
#define KERNEL_m4n1k1 {\
    ya1 = _mm256_loadu_pd(a_block_pointer);a_block_pointer+=4;\
    yb1 = _mm256_broadcast_sd(b_block_pointer);   yc1 = _mm256_fmadd_pd(ya1,yb1,yc1);\
    b_block_pointer++;\
}
#define INIT_m4n1 yc1=_mm256_setzero_pd();
#define INIT_m4n2 yc2=INIT_m4n1
#define INIT_m4n4 yc4=yc3=INIT_m4n2
#define SAVE_m4n1 {\
    ya1 = _mm256_loadu_pd(c_pointer);\
    yc1 = _mm256_add_pd(yc1,ya1);\
    _mm256_storeu_pd(c_pointer,yc1);\
    c_pointer += 4;\
}
#define SAVE_m4n2 {\
    yb1 = _mm256_loadu_pd(c_pointer); yb2 = _mm256_loadu_pd(c_pointer+LDC);\
    yc1 = _mm256_add_pd(yc1,yb1); yc2 = _mm256_add_pd(yc2,yb2);\
    _mm256_storeu_pd(c_pointer,yc1); _mm256_storeu_pd(c_pointer+LDC,yc2);\
    c_pointer += 4;\
}
#define SAVE_m4n4 {\
    yb1 = _mm256_loadu_pd(c_pointer); yb2 = _mm256_loadu_pd(c_pointer+LDC);\
    yc1 = _mm256_add_pd(yc1,yb1); yc2 = _mm256_add_pd(yc2,yb2);\
    _mm256_storeu_pd(c_pointer,yc1); _mm256_storeu_pd(c_pointer+LDC,yc2);\
    c_pointer += LDC*2;\
    yb1 = _mm256_loadu_pd(c_pointer); yb2 = _mm256_loadu_pd(c_pointer+LDC);\
    yc3 = _mm256_add_pd(yc3,yb1); yc4 = _mm256_add_pd(yc4,yb2);\
    _mm256_storeu_pd(c_pointer,yc3); _mm256_storeu_pd(c_pointer+LDC,yc4);\
    c_pointer += 4-LDC*2;\
}
#define KERNEL_m2n2k1 {\
    xa1 = _mm_loadu_pd(a_block_pointer); a_block_pointer+=2;\
    xb1 = _mm_loaddup_pd(b_block_pointer);   xc1 = _mm_fmadd_pd(xa1,xb1,xc1);\
    xb2 = _mm_loaddup_pd(b_block_pointer+1); xc2 = _mm_fmadd_pd(xa1,xb2,xc2);\
    b_block_pointer += 2;\
}
#define KERNEL_m2n1k1 {\
    xa1 = _mm_loadu_pd(a_block_pointer); a_block_pointer+=2;\
    xb1 = _mm_loaddup_pd(b_block_pointer);   xc1 = _mm_fmadd_pd(xa1,xb1,xc1);\
    b_block_pointer ++;\
}
#define INIT_m2n1 xc1=_mm_setzero_pd();
#define INIT_m2n2 xc2=INIT_m2n1
#define SAVE_m2n1 {\
    xa1 = _mm_loadu_pd(c_pointer);\
    xc1 = _mm_add_pd(xc1,xa1);\
    _mm_storeu_pd(c_pointer,xc1);\
    c_pointer += 2;\
}
#define SAVE_m2n2 {\
    xb1 = _mm_loadu_pd(c_pointer); xb2 = _mm_loadu_pd(c_pointer+LDC);\
    xc1 = _mm_add_pd(xc1,xb1); xc2 = _mm_add_pd(xc2,xb2);\
    _mm_storeu_pd(c_pointer,xc1); _mm_storeu_pd(c_pointer+LDC,xc2);\
    c_pointer += 2;\
}
#define KERNEL_m1n1k1 {\
    sa1 = *a_block_pointer; a_block_pointer++;\
    sb1 = *b_block_pointer; sc1 += sa1 * sb1;\
    b_block_pointer ++;\
}
#define INIT_m1n1 sc1=0.0;
#define SAVE_m1n1 {\
    *c_pointer += sc1;\
    c_pointer++;\
}

/* row-major c_block */
#define KERNEL_m2n4k1 {\
    yb1 = _mm256_loadu_pd(b_block_pointer);b_block_pointer+=4;\
    ya1 = _mm256_broadcast_sd(a_block_pointer);  yc1 = _mm256_fmadd_pd(ya1,yb1,yc1);\
    ya1 = _mm256_broadcast_sd(a_block_pointer+1);yc2 = _mm256_fmadd_pd(ya1,yb1,yc2);\
    a_block_pointer += 2;\
}
#define KERNEL_m1n4k1 {\
    yb1 = _mm256_loadu_pd(b_block_pointer);b_block_pointer+=4;\
    ya1 = _mm256_broadcast_sd(a_block_pointer);  yc1 = _mm256_fmadd_pd(ya1,yb1,yc1);\
    a_block_pointer ++;\
}
#define KERNEL_m1n2k1 {\
    xb1 = _mm_loadu_pd(b_block_pointer);b_block_pointer+=2;\
    xa1 = _mm_loaddup_pd(a_block_pointer); xc1 = _mm_fmadd_pd(xa1,xb1,xc1);\
    a_block_pointer ++;\
}
#define INIT_m1n2 INIT_m2n1
#define INIT_m1n4 INIT_m4n1
#define INIT_m2n4 INIT_m4n2
#define SAVE_m2n4 {\
    yb1 = _mm256_unpacklo_pd(yc1,yc2);\
    yb2 = _mm256_unpackhi_pd(yc1,yc2);\
    xb1 = _mm_add_pd(_mm_loadu_pd(c_pointer),_mm256_extractf128_pd(yb1,0));\
    xb2 = _mm_add_pd(_mm_loadu_pd(c_pointer+LDC),_mm256_extractf128_pd(yb2,0));\
    _mm_storeu_pd(c_pointer,xb1);\
    _mm_storeu_pd(c_pointer+LDC,xb2);\
    xb1 = _mm_add_pd(_mm_loadu_pd(c_pointer+2*LDC),_mm256_extractf128_pd(yb1,1));\
    xb2 = _mm_add_pd(_mm_loadu_pd(c_pointer+3*LDC),_mm256_extractf128_pd(yb2,1));\
    _mm_storeu_pd(c_pointer+2*LDC,xb1);\
    _mm_storeu_pd(c_pointer+3*LDC,xb2);\
    c_pointer += 2;\
}
#define SAVE_m1n2 {\
    *c_pointer += _mm_cvtsd_f64(xc1);\
    xa1 = _mm_unpackhi_pd(xc1,xc1);\
    c_pointer[LDC]+= _mm_cvtsd_f64(xa1);\
    c_pointer ++;\
}
#define SAVE_m1n4 {\
    *c_pointer += _mm256_cvtsd_f64(yc1);\
    ya1 = _mm256_unpackhi_pd(yc1,yc1);\
    c_pointer[LDC] += _mm256_cvtsd_f64(ya1);\
    xb1 = _mm256_extractf128_pd(yc1,1);\
    c_pointer[LDC*2] += _mm_cvtsd_f64(xb1);\
    xb2 = _mm_unpackhi_pd(xb1,xb1);\
    c_pointer[LDC*3] += _mm_cvtsd_f64(xb2);\
    c_pointer ++;\
}

static void __attribute__ ((noinline)) KERNEL_EDGE(double *packed_a, double *packed_b, BLASLONG m, BLASLONG edge_n, BLASLONG k, BLASLONG LDC, double *c){//icopy=8,ocopy=8
//perform C += A<pack> B<pack> , edge_n<8 must be satisfied !
    if(k==0 || m==0 || edge_n==0) return;
    double *a_block_pointer,*b_block_pointer,*b_base_pointer;
    double *c_pointer = c;
    __m512d zb1,zb2,za1,zc1,zc2,zc3,zc4;
    __m256d yc1,yc2,yc3,yc4,ya1,yb1,yb2;
    __m128d xc1,xc2,xc3,xc4,xa1,xb1,xb2;
    double sc1,sa1,sb1;
    BLASLONG m_count,n_count,k_count;
    b_base_pointer = packed_b;
//now start calculation of the edge part
    for(n_count=edge_n;n_count>3;n_count-=4){
      a_block_pointer = packed_a;
      for(m_count=m;m_count>7;m_count-=8){
        b_block_pointer = b_base_pointer;
        INIT_m8n4
        for(k_count=0;k_count<k;k_count++) KERNEL_m8n4k1
        SAVE_m8n4
      }
      for(;m_count>3;m_count-=4){
        b_block_pointer = b_base_pointer;
        INIT_m4n4
        for(k_count=0;k_count<k;k_count++) KERNEL_m4n4k1
        SAVE_m4n4
      }
      for(;m_count>1;m_count-=2){
        b_block_pointer = b_base_pointer;
        INIT_m2n4
        for(k_count=0;k_count<k;k_count++) KERNEL_m2n4k1
        SAVE_m2n4
      }
      if(m_count>0){
        b_block_pointer = b_base_pointer;
        INIT_m1n4
        for(k_count=0;k_count<k;k_count++) KERNEL_m1n4k1
        SAVE_m1n4
      }
      b_base_pointer += 4*k;
      c_pointer += 4 * LDC - m;
    }
    for(;n_count>1;n_count-=2){
      a_block_pointer = packed_a;
      for(m_count=m;m_count>7;m_count-=8){
        b_block_pointer = b_base_pointer;
        INIT_m8n2
        for(k_count=0;k_count<k;k_count++) KERNEL_m8n2k1
        SAVE_m8n2
      }
      for(;m_count>3;m_count-=4){
        b_block_pointer = b_base_pointer;
        INIT_m4n2
        for(k_count=0;k_count<k;k_count++) KERNEL_m4n2k1
        SAVE_m4n2
      }
      for(;m_count>1;m_count-=2){
        b_block_pointer = b_base_pointer;
        INIT_m2n2
        for(k_count=0;k_count<k;k_count++) KERNEL_m2n2k1
        SAVE_m2n2
      }
      if(m_count>0){
        b_block_pointer = b_base_pointer;
        INIT_m1n2
        for(k_count=0;k_count<k;k_count++) KERNEL_m1n2k1
        SAVE_m1n2
      }
      b_base_pointer += 2*k;
      c_pointer += 2 * LDC - m;
    }
    if(n_count>0){
      a_block_pointer = packed_a;
      for(m_count=m;m_count>7;m_count-=8){
        b_block_pointer = b_base_pointer;
        INIT_m8n1
        for(k_count=0;k_count<k;k_count++) KERNEL_m8n1k1
        SAVE_m8n1
      }
      for(;m_count>3;m_count-=4){
        b_block_pointer = b_base_pointer;
        INIT_m4n1
        for(k_count=0;k_count<k;k_count++) KERNEL_m4n1k1
        SAVE_m4n1
      }
      for(;m_count>1;m_count-=2){
        b_block_pointer = b_base_pointer;
        INIT_m2n1
        for(k_count=0;k_count<k;k_count++) KERNEL_m2n1k1
        SAVE_m2n1
      }
      if(m_count>0){
        b_block_pointer = b_base_pointer;
        INIT_m1n1
        for(k_count=0;k_count<k;k_count++) KERNEL_m1n1k1
        SAVE_m1n1
      }
    }
}
/* debug zone start here */
static void KERNEL_OPERATION(double *packed_a, double *packed_b, double *c, BLASLONG m, BLASLONG n, BLASLONG k, BLASLONG LDC){
    if(m==0 || n==0 || k==0) return;
    BLASLONG ndiv8 = n/8;
    if(ndiv8>0) KERNEL_MAIN(packed_a,packed_b,m,ndiv8,k,LDC,c);
    if(n>ndiv8*8) KERNEL_EDGE(packed_a,packed_b+(int64_t)k*(int64_t)ndiv8*8,m,n-ndiv8*8,k,LDC,c+(int64_t)LDC*(int64_t)ndiv8*8);
}
static void SCALE_MULT(double *dat, double *sca, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0) return;
    double scale = *sca;double *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale;current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
static void dgemm_tcopy_8(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    double *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 8;
      for(count_first=dim_first;count_first>7;count_first-=8){
        _mm512_storeu_pd(todst,_mm512_loadu_pd(tosrc));
        tosrc+=8;todst+=8*dim_second;
      }
      todst -= count_second * 4;
      for(;count_first>3;count_first-=4){
        _mm256_storeu_pd(todst,_mm256_loadu_pd(tosrc));
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        _mm_storeu_pd(todst,_mm_loadu_pd(tosrc));
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void dgemm_ncopy_8(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8,*todst;
    todst=dst;tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc5=tosrc4+lead_dim;tosrc6=tosrc5+lead_dim;tosrc7=tosrc6+lead_dim;tosrc8=tosrc7+lead_dim;tosrc_inc=8*lead_dim-dim_first;
    for(count_second=dim_second;count_second>7;count_second-=8){
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
#define BLOCKDIM_K 128 //GEMM_Q in OpenBLAS
#define BLOCKDIM_M 512 //GEMM_P in OpenBLAS
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
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
      if(NOTRANSB) { dgemm_ncopy_8(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { dgemm_tcopy_8(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { dgemm_tcopy_8(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { dgemm_ncopy_8(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        if((*alpha)!=1.0) SCALE_MULT(a_buffer,alpha,k_subdim,k_subdim,m_subdim);
        KERNEL_OPERATION(a_buffer,b_buffer,c+m_count,m_subdim,*n,k_subdim,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}
