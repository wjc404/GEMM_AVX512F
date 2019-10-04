# define ZERO_VALUE 5.0

/* functions for packing A, which are normal */
static void load_irreg_a_c(const FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;const FLOAT *aread;FLOAT *awrite;
  aread=astartpos;awrite=ablk;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      *(awrite+arow)=*(aread+arow);
    }
    for(;arow<GEMM_BLOCK_DIM_M;arow++){
      *(awrite+arow)=ZERO_VALUE;
    }
    aread+=lda;awrite+=GEMM_BLOCK_DIM_M;
  }
}
static void load_irreg_a_r(const FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;const FLOAT *aread;FLOAT *awrite;
  aread=astartpos;awrite=ablk;
  for(arow=0;arow<mdim;arow++){
    for(acol=0;acol<kdim;acol++){
      *(awrite+acol*GEMM_BLOCK_DIM_M)=*(aread+acol);
    }
    aread+=lda;awrite++;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<GEMM_BLOCK_DIM_M-mdim;arow++){
      *(awrite+arow)=ZERO_VALUE;
    }
    awrite+=GEMM_BLOCK_DIM_M;
  }
}
static void load_reg_a_c(const FLOAT *astartpos,FLOAT *ablk,int lda){load_irreg_a_c(astartpos,ablk,lda,GEMM_BLOCK_DIM_M,GEMM_BLOCK_DIM_K);}
static void load_reg_a_r(const FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow;const FLOAT *ar1,*ar2,*ar3,*ar4;FLOAT *awrite;
  for(arow=0;arow<GEMM_BLOCK_DIM_M;arow+=4){
    ar1=astartpos+arow*lda;
    ar2=ar1+lda;ar3=ar2+lda;ar4=ar3+lda;
    awrite=ablk+arow;
    for(acol=0;acol<GEMM_BLOCK_DIM_K;acol++){
      *(awrite+0)=*(ar1+acol);
      *(awrite+1)=*(ar2+acol);
      *(awrite+2)=*(ar3+acol);
      *(awrite+3)=*(ar4+acol);
      awrite+=GEMM_BLOCK_DIM_M;
    }
  }
}
static void load_tail_a_c(const FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_c(astartpos,ablk,lda,mdim,GEMM_BLOCK_DIM_K);}
static void load_tail_a_r(const FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_r(astartpos,ablk,lda,mdim,GEMM_BLOCK_DIM_K);}
static void load_irregk_a_c(const FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_c(astartpos,ablk,lda,GEMM_BLOCK_DIM_M,kdim);}
static void load_irregk_a_r(const FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_r(astartpos,ablk,lda,GEMM_BLOCK_DIM_M,kdim);}

/* functions for packing B, a little complicated */
static void unit_load_reg_b_c(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,const FLOAT * __restrict__ alpha){
  const FLOAT *inb[GEMM_UNROLL_N];FLOAT *outb;const FLOAT ALPHA=*alpha;
  int bcol,brow,bsub1,bsub2;
  outb=bblk;
  inb[0]=bstartpos;
  for(bsub1=1;bsub1<GEMM_UNROLL_N;bsub1++) inb[bsub1]=inb[bsub1-1]+ldb;
  for(bcol=0;bcol<GEMM_LOOP_TIMES_N;bcol++){
    for(bsub1=GEMM_UNROLL_N-1;bsub1>=0;bsub1--){
      for(brow=0;brow<GEMM_LOOP_TIMES_K*4;brow++){
        for(bsub2=0;bsub2<GEMM_UNROLL_N;bsub2++){outb[bsub2]=(*inb[bsub2])*ALPHA;inb[bsub2]++;}
        outb+=GEMM_UNROLL_N;
      }
      for(bsub2=0;bsub2<GEMM_UNROLL_N;bsub2++) inb[bsub2]+=ldb;
      if(bsub1==0) for(bsub2=0;bsub2<GEMM_UNROLL_N;bsub2++) inb[bsub2]-=GEMM_BLOCK_L1DIM_K;
      else inb[bsub1]-=(bcol==GEMM_LOOP_TIMES_N-1)*((int64_t)ldb*(int64_t)GEMM_BLOCK_DIM_N);
    }
  }
}
static void load_reg_b_c(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,const FLOAT * __restrict__ alpha){
  int cnt;const FLOAT *bread = bstartpos;FLOAT *bwrite = bblk;
  for(cnt=0;cnt<GEMM_UNROLL_M_VEC;cnt++){
    unit_load_reg_b_c(bread,bwrite,ldb,alpha);
    bread += GEMM_BLOCK_L1DIM_K;
    bwrite += GEMM_BLOCK_L1DIM_K * GEMM_BLOCK_DIM_N;
  }
}
#define bcopy_4row(num_of_columns) {\
  for(bsub2=0;bsub2<num_of_columns;bsub2++){\
    bout[bsub2+0*GEMM_UNROLL_N]=(*bin1)*ALPHA;bin1++;\
    bout[bsub2+1*GEMM_UNROLL_N]=(*bin2)*ALPHA;bin2++;\
    bout[bsub2+2*GEMM_UNROLL_N]=(*bin3)*ALPHA;bin3++;\
    bout[bsub2+3*GEMM_UNROLL_N]=(*bin4)*ALPHA;bin4++;\
  }\
}
static void unit_load_reg_b_r(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,const FLOAT * __restrict__ alpha){
  const FLOAT *bin1,*bin2,*bin3,*bin4;FLOAT *bout;int bcol,brow,bsub1,bsub2;const FLOAT ALPHA=*alpha;int64_t bshift=(int64_t)4*(int64_t)ldb-GEMM_BLOCK_DIM_N;
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;
  for(brow=0;brow<GEMM_LOOP_TIMES_K*4;brow+=4){
    bout=bblk+brow*GEMM_UNROLL_N;
    for(bcol=0;bcol<GEMM_LOOP_TIMES_N;bcol++){
      bcopy_4row(GEMM_UNROLL_N)
      bout+=GEMM_UNROLL_N*GEMM_BLOCK_L1DIM_K;
    }
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(bsub1=GEMM_UNROLL_N-1;bsub1>0;bsub1--){
    for(;brow<GEMM_LOOP_TIMES_K*4*(GEMM_UNROLL_N+1-bsub1);brow+=4){
      bout=bblk+brow*GEMM_UNROLL_N+(GEMM_LOOP_TIMES_N-1)*GEMM_UNROLL_N*GEMM_BLOCK_L1DIM_K+bsub1;
      bcopy_4row(GEMM_UNROLL_N-bsub1)
      bout=bblk+brow*GEMM_UNROLL_N;
      for(bcol=1;bcol<GEMM_LOOP_TIMES_N;bcol++){
        bcopy_4row(GEMM_UNROLL_N)
        bout+=GEMM_UNROLL_N*GEMM_BLOCK_L1DIM_K;
      }
      bcopy_4row(bsub1)
      bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
    }
  }
}
static void load_reg_b_r(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,const FLOAT * __restrict__ alpha){
  int cnt;const FLOAT *bread = bstartpos;FLOAT *bwrite = bblk;
  for(cnt=0;cnt<GEMM_UNROLL_M_VEC;cnt++){
    unit_load_reg_b_r(bread,bwrite,ldb,alpha);
    bread += (int64_t)GEMM_BLOCK_L1DIM_K * (int64_t)ldb;
    bwrite += GEMM_BLOCK_L1DIM_K * GEMM_BLOCK_DIM_N;
  }
}
static void sub_load_irreg_b_c(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,const FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  const FLOAT *bin[GEMM_UNROLL_N];FLOAT *bout;int bcol,brow,bsub;const FLOAT ALPHA=*alpha;
  bin[0]=bstartpos;
  for(bsub=1;bsub<GEMM_UNROLL_N;bsub++) bin[bsub]=bin[bsub-1]+ldb;
  bout=bblk;
  for(bcol=0;bcol<=ndim-GEMM_UNROLL_N;bcol+=GEMM_UNROLL_N){
    for(brow=0;brow<kdim;brow++){
      for(bsub=0;bsub<GEMM_UNROLL_N;bsub++){
        *bout=(*bin[bsub])*ALPHA;
        bin[bsub]++;bout++;
      }
    }
    for(bsub=0;bsub<GEMM_UNROLL_N;bsub++) bin[bsub]+=(int64_t)GEMM_UNROLL_N*(int64_t)ldb-(int64_t)kdim;
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      *bout=(*bin[0])*ALPHA;bin[0]++;bout++;
    }
    bin[0]+=ldb-kdim;
  }
}
static void sub_load_irreg_b_r(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,const FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  const FLOAT *bin;FLOAT *bout;int bcol,brow,bsub;const FLOAT ALPHA=*alpha;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*GEMM_UNROLL_N;
    for(bcol=0;bcol<=ndim-GEMM_UNROLL_N;bcol+=GEMM_UNROLL_N){
      for(bsub=0;bsub<GEMM_UNROLL_N;bsub++) bout[bsub]=bin[bsub]*ALPHA;
      bin+=GEMM_UNROLL_N;bout+=GEMM_UNROLL_N*kdim;
    }
    bout-=(GEMM_UNROLL_N-1)*brow;
    for(;bcol<ndim;bcol++){
      *bout=(*bin)*ALPHA;bin++;bout+=kdim;
    }
    bin+=ldb-ndim;
  }
}
static void load_irreg_b_c(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,const FLOAT * __restrict__ alpha){
  int brow,subbr;
  for(brow=0;brow<kdim;brow+=GEMM_BLOCK_L1DIM_K){
    subbr=kdim-brow;
    if(subbr>=GEMM_BLOCK_L1DIM_K){
      subbr=GEMM_BLOCK_L1DIM_K;
      if(ndim==GEMM_BLOCK_DIM_N) unit_load_reg_b_c(bstartpos+brow,bblk+(int64_t)brow*(int64_t)ndim,ldb,alpha);
      else sub_load_irreg_b_c(bstartpos+brow,bblk+(int64_t)brow*(int64_t)ndim,ldb,ndim,subbr,alpha);
    }
    else sub_load_irreg_b_c(bstartpos+brow,bblk+(int64_t)brow*(int64_t)ndim,ldb,ndim,subbr,alpha);
  }
}
static void load_irreg_b_r(const FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,const FLOAT * __restrict__ alpha){
  int brow,subbr;
  for(brow=0;brow<kdim;brow+=GEMM_BLOCK_L1DIM_K){
    subbr=kdim-brow;
    if(subbr>=GEMM_BLOCK_L1DIM_K){
      subbr=GEMM_BLOCK_L1DIM_K;
      if(ndim==GEMM_BLOCK_DIM_N) unit_load_reg_b_r(bstartpos+(int64_t)brow*(int64_t)ldb,bblk+(int64_t)brow*(int64_t)ndim,ldb,alpha);
      else sub_load_irreg_b_r(bstartpos+(int64_t)brow*(int64_t)ldb,bblk+(int64_t)brow*(int64_t)ndim,ldb,ndim,subbr,alpha);
    }
    else sub_load_irreg_b_r(bstartpos+(int64_t)brow*(int64_t)ldb,bblk+(int64_t)brow*(int64_t)ndim,ldb,ndim,subbr,alpha);
  }
}
