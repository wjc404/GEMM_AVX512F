//zmm0-zmm7 for temp; zmm8-zmm31 for accumulators.
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

# if GEMM_UNROLL_M_VEC == 2
 # include "gemm_kernel_unroll2.S"
# endif
# if GEMM_UNROLL_M_VEC == 3
 # include "gemm_kernel_unroll3.S"
# endif
# if GEMM_UNROLL_M_VEC == 4
 # include "gemm_kernel_unroll4.S"
# endif
# if GEMM_UNROLL_M_VEC == 6
 # include "gemm_kernel_unroll6.S"
# endif

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
    addq $(GEMM_BLOCK_L1DIM_K-4)*BYTES_PER_COL,%r9 //Arefpos
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
    prefetcht2 (CS,%r13,1)
    xorq %r11,%r11
.Linner_gemmblkregccc:
    KERNEL_4 %r9,%r8
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
    KERNEL_2 %r9
    cmpq $GEMM_LOOP_TIMES_K*2,%r11
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
    addq $(GEMM_BLOCK_L1DIM_K-4)*BYTES_PER_COL,%r9 //Arefpos
    movq $(-GEMM_BLOCK_L1DIM_K)*BYTES_PER_COL,%r8 //Areset
.Louter_tail:
    xorq AD,AD
    UPDATECBLK_1col
    PREF_1_col prefetcht0,CS
    xorq %r11,%r11
.Linner_tail:
    KERNEL_4 %r9,%r8
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

.macro PREF_C_1col pref_c_offset
    PREF_1_col prefetcht0,CS
    prefetcht2 (CS,\pref_c_offset,1)
.endm

.macro PREF_C_ncol pref_c_offset
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
    PREF_1_col prefetcht0,CL
    prefetcht2 (CL,\pref_c_offset,1)
    addq LDC,CL
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
    KERNEL_4_edgen
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
    PREF_C_1col %r9
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
    PREF_C_1col %r9
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
    cmpq $2000,%r11
    jb .Ltimedelay

    vzeroupper
    retq
    
