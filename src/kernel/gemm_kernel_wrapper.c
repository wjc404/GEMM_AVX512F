/* 5 kernel functions defined in gemm_kernel.S, with AMD64 ABI. */
extern void unit_gemmblkregccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int prefcoffset);
extern void unit_gemmblktailccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int mdim);
extern void unit_gemmblkirregkccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int kdim,int prefcoffset);
extern void unit_gemmblkirregnccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int ndim,int prefcoffset);
extern void unit_gemmblkirregccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int64_t *dim3,int prefcoffset);//dim3[0]=mdim,dim3[1]=ndim,dim3[2]=kdim.

/* wrapper caller functions */
static void gemmblkregccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc){
# ifdef NO_REPEAT_C_BLOCK
  unit_gemmblkregccc(abufferctpos,bblk,cstartpos,ldc,0);
# else
  int cnt;FLOAT *acpos,*bcpos;
  for(cnt=0;cnt<GEMM_UNROLL_M_VEC;cnt++){
    acpos=abufferctpos+GEMM_BLOCK_L1DIM_K*GEMM_BLOCK_DIM_M*cnt;
    bcpos=bblk+GEMM_BLOCK_L1DIM_K*GEMM_BLOCK_DIM_N*cnt;
    unit_gemmblkregccc(acpos,bcpos,cstartpos,ldc,(cnt+GEMM_UNROLL_M_VEC+1)*64-1);
  }
# endif
}
static void gemmblktailccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int mdim){
# ifdef NO_REPEAT_C_BLOCK
  unit_gemmblktailccc(abufferctpos,bblk,cstartpos,ldc,mdim);
# else
  int cnt;FLOAT *acpos,*bcpos;
  for(cnt=0;cnt<GEMM_UNROLL_M_VEC;cnt++){
    acpos=abufferctpos+GEMM_BLOCK_L1DIM_K*GEMM_BLOCK_DIM_M*cnt;
    bcpos=bblk+GEMM_BLOCK_L1DIM_K*GEMM_BLOCK_DIM_N*cnt;
    unit_gemmblktailccc(acpos,bcpos,cstartpos,ldc,mdim);
  }
# endif
}
static void gemmblkirregkccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int kdim){
# ifdef NO_REPEAT_C_BLOCK
  unit_gemmblkirregkccc(abufferctpos,bblk,cstartpos,ldc,kdim,0);
# else
  int kcnt,subkdim;FLOAT *acpos,*bcpos;
  int prefpos=(GEMM_UNROLL_M_VEC+1)*64-1;
  for(kcnt=0;kcnt<kdim;kcnt+=GEMM_BLOCK_L1DIM_K){
    acpos=abufferctpos+kcnt*GEMM_BLOCK_DIM_M;
    bcpos=bblk+kcnt*GEMM_BLOCK_DIM_N;
    subkdim=kdim-kcnt;
    if(subkdim>=GEMM_BLOCK_L1DIM_K){
      subkdim=GEMM_BLOCK_L1DIM_K;
      unit_gemmblkregccc(acpos,bcpos,cstartpos,ldc,prefpos);
    }
    else unit_gemmblkirregkccc(acpos,bcpos,cstartpos,ldc,subkdim,prefpos);
    prefpos+=64;
  }
# endif
}
static void gemmblkirregnccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int ndim){
# ifdef NO_REPEAT_C_BLOCK
  unit_gemmblkirregnccc(abufferctpos,bblk,cstartpos,ldc,ndim,0);
# else
  int cnt;FLOAT *acpos,*bcpos;
  for(cnt=0;cnt<GEMM_UNROLL_M_VEC;cnt++){
    acpos=abufferctpos+GEMM_BLOCK_L1DIM_K*GEMM_BLOCK_DIM_M*cnt;
    bcpos=bblk+GEMM_BLOCK_L1DIM_K*ndim*cnt;
    unit_gemmblkirregnccc(acpos,bcpos,cstartpos,ldc,ndim,(cnt+GEMM_UNROLL_M_VEC+1)*64-1);
  }
# endif
}
static void gemmblkirregccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int mdim,int ndim,int kdim){
  int64_t dim3[3];dim3[0]=(int64_t)mdim;dim3[1]=(int64_t)ndim;
# ifdef NO_REPEAT_C_BLOCK
  dim3[2]=(int64_t)kdim;
  unit_gemmblkirregccc(abufferctpos,bblk,cstartpos,ldc,dim3,0);
# else
  int kcnt,subkdim;FLOAT *acpos,*bcpos;
  int prefpos=(GEMM_UNROLL_M_VEC+1)*64-1;
  for(kcnt=0;kcnt<kdim;kcnt+=GEMM_BLOCK_L1DIM_K){
    subkdim=kdim-kcnt;
    if(subkdim>GEMM_BLOCK_L1DIM_K) subkdim=GEMM_BLOCK_L1DIM_K;
    dim3[2]=(int64_t)subkdim;
    acpos=abufferctpos+kcnt*GEMM_BLOCK_DIM_M;
    bcpos=bblk+kcnt*ndim;
    if(subkdim==GEMM_BLOCK_L1DIM_K && ndim==GEMM_BLOCK_DIM_N) unit_gemmblktailccc(acpos,bcpos,cstartpos,ldc,mdim);
    else unit_gemmblkirregccc(acpos,bcpos,cstartpos,ldc,dim3,prefpos);
    prefpos+=64;
  }
# endif
}

