# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h>
# include <omp.h>
//Tuning parameters: GEMM_UNROLL_N, GEMM_LOOP_TIMES_N, GEMM_LOOP_TIMES_K, PREF_CYCLES_PACKED_A, PREF_CYCLES_PACKED_B
# include "gemm_set_parameters.h"

# ifdef DOUBLE
 # define FLOAT double
 # define GEMM_DRIVER dgemm_
# else
 # define FLOAT float
 # define GEMM_DRIVER sgemm_
# endif
# pragma GCC visibility push(hidden)
# include "gemm_copy.c" //functions for packing matrix blocks
# include "kernel/gemm_kernel_wrapper.c"
# define NOTRANSB (TRANSB=='N'||TRANSB=='n')
# define NOTRANSA (TRANSA=='N'||TRANSA=='n')
extern void timedelay();//produce nothing besides a delay(~3 us), with no system calls
static void SYNC_ACCESS_ABUFFER(int tid,int threads,int *workprogress){//this function is for synchronization of threads before/after pack_abuffer; workprogress[] must be shared among all threads
  int waitothers,ctid,temp;
  workprogress[16*tid]++;
  temp=workprogress[16*tid];
  for(waitothers=1;waitothers;timedelay()){
    waitothers=0;
    for(ctid=0;ctid<threads;ctid++){
      if(workprogress[16*ctid]<temp) waitothers = 1;
    }
  }
}
static void GEMM_COPY_A(char TRANSA,const FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BLOCK_NUM_M,int GEMM_EDGE_DIM_M){//similar to GEMM_OCOPY in OpenBLAS
  int loop_counter;
  const FLOAT *a_current_pos;FLOAT *abuffer_current_pos;
  a_current_pos=aheadpos;
  abuffer_current_pos=abuffer;
  for(loop_counter=0;loop_counter<BLOCK_NUM_M-1;loop_counter++){
    if(NOTRANSA){
      load_reg_a_c(a_current_pos,abuffer_current_pos,LDA);
      a_current_pos+=GEMM_BLOCK_DIM_M;
    }
    else{
      load_reg_a_r(a_current_pos,abuffer_current_pos,LDA);
      a_current_pos+=(int64_t)GEMM_BLOCK_DIM_M*(int64_t)LDA;
    }
    abuffer_current_pos+=GEMM_BLOCK_DIM_M*GEMM_BLOCK_DIM_K;
  }
  if(NOTRANSA) load_tail_a_c(a_current_pos,abuffer_current_pos,LDA,GEMM_EDGE_DIM_M);
  else         load_tail_a_r(a_current_pos,abuffer_current_pos,LDA,GEMM_EDGE_DIM_M);
}
static void GEMM_COPY_A_EDGE_K(char TRANSA,const FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BLOCK_NUM_M,int GEMM_EDGE_DIM_M,int kdim){
  int loop_counter;
  const FLOAT *a_current_pos;FLOAT *abuffer_current_pos;
  a_current_pos=aheadpos;
  abuffer_current_pos=abuffer;
  for(loop_counter=0;loop_counter<BLOCK_NUM_M-1;loop_counter++){
    if(NOTRANSA){
      load_irregk_a_c(a_current_pos,abuffer_current_pos,LDA,kdim);
      a_current_pos+=GEMM_BLOCK_DIM_M;
    }
    else{
      load_irregk_a_r(a_current_pos,abuffer_current_pos,LDA,kdim);
      a_current_pos+=(int64_t)GEMM_BLOCK_DIM_M*(int64_t)LDA;
    }
    abuffer_current_pos+=GEMM_BLOCK_DIM_M*kdim;
  }
  if(NOTRANSA) load_irreg_a_c(a_current_pos,abuffer_current_pos,LDA,GEMM_EDGE_DIM_M,kdim);
  else         load_irreg_a_r(a_current_pos,abuffer_current_pos,LDA,GEMM_EDGE_DIM_M,kdim);
}
static void GEMM_COPY_B(char TRANSB,const FLOAT *bstartpos,FLOAT *bblk,int LDB,const FLOAT *alpha){//similar to GEMM_ICOPY in OpenBLAS
  if(NOTRANSB) load_reg_b_c(bstartpos,bblk,LDB,alpha);
  else         load_reg_b_r(bstartpos,bblk,LDB,alpha);
}
static void GEMM_COPY_B_EDGE(char TRANSB,const FLOAT *bstartpos,FLOAT *bblk,int LDB,int ndim,int kdim,const FLOAT *alpha){
  if(NOTRANSB) load_irreg_b_c(bstartpos,bblk,LDB,ndim,kdim,alpha);
  else         load_irreg_b_r(bstartpos,bblk,LDB,ndim,kdim,alpha);
}
static void GEMM_CALC_COLUMN(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BLOCK_NUM_M,int GEMM_EDGE_DIM_M,int LDC){//similar to GEMM_KERNEL in OpenBLAS
  int loop_counter;
  FLOAT *c_current_pos, *abuffer_current_pos;
  c_current_pos=cheadpos;
  abuffer_current_pos=abuffer;
  for(loop_counter=0;loop_counter<BLOCK_NUM_M-1;loop_counter++){
    gemmblkregccc(abuffer_current_pos,bblk,c_current_pos,LDC);//SEGFAULT BUG HERE ___
    c_current_pos+=GEMM_BLOCK_DIM_M;
    abuffer_current_pos+=GEMM_BLOCK_DIM_M*GEMM_BLOCK_DIM_K;
  }
  gemmblktailccc(abuffer_current_pos,bblk,c_current_pos,LDC,GEMM_EDGE_DIM_M);
}
static void GEMM_CALC_COLUMN_EDGE(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BLOCK_NUM_M,int GEMM_EDGE_DIM_M,int LDC,int kdim,int ndim){
  int loop_counter;
  FLOAT *c_current_pos, *abuffer_current_pos;
  c_current_pos=cheadpos;
  abuffer_current_pos=abuffer;
  for(loop_counter=0;loop_counter<BLOCK_NUM_M-1;loop_counter++){
    if(kdim==GEMM_BLOCK_DIM_K)      gemmblkirregnccc(abuffer_current_pos,bblk,c_current_pos,LDC,ndim);
    else if(ndim==GEMM_BLOCK_DIM_N) gemmblkirregkccc(abuffer_current_pos,bblk,c_current_pos,LDC,kdim);
    else                             gemmblkirregccc(abuffer_current_pos,bblk,c_current_pos,LDC,GEMM_BLOCK_DIM_M,ndim,kdim);
    c_current_pos+=GEMM_BLOCK_DIM_M;
    abuffer_current_pos+=GEMM_BLOCK_DIM_M*kdim;
  }
  gemmblkirregccc(abuffer_current_pos,bblk,c_current_pos,LDC,GEMM_EDGE_DIM_M,ndim,kdim);
}
static void C_MULT_BETA(FLOAT *c,int ldc,int m,int n,FLOAT beta){
  int i,j;FLOAT *C0,*C;
  C0=c;
  for(i=0;i<n;i++){
    C=C0;
    for(j=0;j<m;j++){
      *C*=beta;C++;
    }
    C0+=ldc;
  }
}
# pragma GCC visibility pop
void GEMM_DRIVER(const char *transa,const char *transb,const int *m,const int *n,const int *k,const FLOAT *alpha,const FLOAT *a,const int *lda,const FLOAT *bstart,const int *ldb,const FLOAT *beta,FLOAT *cstart,const int *ldc){
//assume column-major storage when no transposition is requested.
//a:matrix with m rows and k columns if transa=N
//b:matrix with k rows and n columns if transb=N
//c:product matrix with m rows and n columns
 const int M = *m;const int K = *k;
 const int LDA = *lda;const int LDB = *ldb;const int LDC=*ldc;
 const char TRANSA = *transa;const char TRANSB = *transb;
 const int BLOCK_NUM_M = (M-1)/GEMM_BLOCK_DIM_M+1;const int GEMM_EDGE_DIM_M = M-(BLOCK_NUM_M-1)*GEMM_BLOCK_DIM_M;
 const int BLOCK_NUM_K = (K-1)/GEMM_BLOCK_DIM_K+1;const int GEMM_EDGE_DIM_K = K-(BLOCK_NUM_K-1)*GEMM_BLOCK_DIM_K;
 int *workprogress, *cchunks;const int numthreads=omp_get_max_threads();int i; //variables for parallel execution
 //cchunk[] for dividing tasks, workprogress[] for recording the progresses of all threads and synchronization.
 //synchronization is necessary here since abuffer[] is shared between threads.
 //if abuffer[] is thread-private, the bandwidth of memory will limit the performance.
 FLOAT *abuffer; //store packed matrix tile A
 if((*beta) != 1.0) C_MULT_BETA(cstart,LDC,M,(*n),(*beta));//limited by memory bendwidth so no need for parallel execution
 if((*alpha) != 0.0){//then do C=alpha*AB+beta*C
  abuffer = (FLOAT *)aligned_alloc(64,(GEMM_BLOCK_DIM_M*GEMM_BLOCK_DIM_K*BLOCK_NUM_M)*sizeof(FLOAT));
  workprogress = (int *)calloc(20*numthreads,sizeof(int));
  cchunks = (int *)malloc((numthreads+1)*sizeof(int));
  for(i=0;i<=numthreads;i++) cchunks[i]=(int)((int64_t)(*n)*(int64_t)i/(int64_t)numthreads);
#pragma omp parallel
 {
  int tid = omp_get_thread_num();
  FLOAT *bblk = (FLOAT *)aligned_alloc(64,(GEMM_BLOCK_DIM_N*GEMM_BLOCK_DIM_K)*sizeof(FLOAT)); //thread-private bblk[]
//base pointer to B & C for each thread
  FLOAT *c = cstart + (int64_t)LDC * (int64_t)cchunks[tid];
  const FLOAT *b;
  if(NOTRANSB) b = bstart + (int64_t)LDB * (int64_t)cchunks[tid];
  else b = bstart + cchunks[tid];
//task parameters for each thread
  const int N = cchunks[tid+1]-cchunks[tid];
  const int BLOCK_NUM_N = (N-1)/GEMM_BLOCK_DIM_N+1;
  const int GEMM_EDGE_DIM_N = N-(BLOCK_NUM_N-1)*GEMM_BLOCK_DIM_N;
  int N_BLOCK_COUNTER,K_BLOCK_COUNTER;//loop counters over blocks
//active pointers for each thread
  const FLOAT *a_current_pos,*b_current_pos;FLOAT *c_current_pos;
  a_current_pos=a;b_current_pos=b;c_current_pos=c;
//first calculate from k-edge part of A
  if(tid==0) GEMM_COPY_A_EDGE_K(TRANSA,a_current_pos,abuffer,LDA,BLOCK_NUM_M,GEMM_EDGE_DIM_M,GEMM_EDGE_DIM_K);
  if(NOTRANSA) a_current_pos+=(int64_t)LDA*(int64_t)GEMM_EDGE_DIM_K;
  else a_current_pos+=GEMM_EDGE_DIM_K;
  SYNC_ACCESS_ABUFFER(tid,numthreads,workprogress);
  for(N_BLOCK_COUNTER=0;N_BLOCK_COUNTER<BLOCK_NUM_N-1;N_BLOCK_COUNTER++){
    GEMM_COPY_B_EDGE(TRANSB,b_current_pos,bblk,LDB,GEMM_BLOCK_DIM_N,GEMM_EDGE_DIM_K,alpha);
    GEMM_CALC_COLUMN_EDGE(abuffer,bblk,c_current_pos,BLOCK_NUM_M,GEMM_EDGE_DIM_M,LDC,GEMM_EDGE_DIM_K,GEMM_BLOCK_DIM_N);
    c_current_pos+=(int64_t)LDC*(int64_t)GEMM_BLOCK_DIM_N;
    if(NOTRANSB) b_current_pos+=(int64_t)LDB*(int64_t)GEMM_BLOCK_DIM_N;
    else         b_current_pos+=GEMM_BLOCK_DIM_N;
  }
  GEMM_COPY_B_EDGE(TRANSB,b_current_pos,bblk,LDB,GEMM_EDGE_DIM_N,GEMM_EDGE_DIM_K,alpha);
  GEMM_CALC_COLUMN_EDGE(abuffer,bblk,c_current_pos,BLOCK_NUM_M,GEMM_EDGE_DIM_M,LDC,GEMM_EDGE_DIM_K,GEMM_EDGE_DIM_N);
  SYNC_ACCESS_ABUFFER(tid,numthreads,workprogress);
//then calculate through the rest of A
  for(K_BLOCK_COUNTER=0;K_BLOCK_COUNTER<BLOCK_NUM_K-1;K_BLOCK_COUNTER++){
    if(tid==0) GEMM_COPY_A(TRANSA,a_current_pos,abuffer,LDA,BLOCK_NUM_M,GEMM_EDGE_DIM_M);
    if(NOTRANSA) a_current_pos+=(int64_t)LDA*(int64_t)GEMM_BLOCK_DIM_K;
    else a_current_pos+=GEMM_BLOCK_DIM_K;
    SYNC_ACCESS_ABUFFER(tid,numthreads,workprogress);
    c_current_pos=c;
    if(NOTRANSB) b_current_pos=b+(K_BLOCK_COUNTER*GEMM_BLOCK_DIM_K+GEMM_EDGE_DIM_K);
    else         b_current_pos=b+(int64_t)LDB*(int64_t)(K_BLOCK_COUNTER*GEMM_BLOCK_DIM_K+GEMM_EDGE_DIM_K);
    for(N_BLOCK_COUNTER=0;N_BLOCK_COUNTER<BLOCK_NUM_N-1;N_BLOCK_COUNTER++){
      GEMM_COPY_B(TRANSB,b_current_pos,bblk,LDB,alpha);
      GEMM_CALC_COLUMN(abuffer,bblk,c_current_pos,BLOCK_NUM_M,GEMM_EDGE_DIM_M,LDC);
      c_current_pos+=(int64_t)LDC*(int64_t)GEMM_BLOCK_DIM_N;
      if(NOTRANSB) b_current_pos+=(int64_t)LDB*(int64_t)GEMM_BLOCK_DIM_N;
      else         b_current_pos+=GEMM_BLOCK_DIM_N;
    }
    GEMM_COPY_B_EDGE(TRANSB,b_current_pos,bblk,LDB,GEMM_EDGE_DIM_N,GEMM_BLOCK_DIM_K,alpha);
    GEMM_CALC_COLUMN_EDGE(abuffer,bblk,c_current_pos,BLOCK_NUM_M,GEMM_EDGE_DIM_M,LDC,GEMM_BLOCK_DIM_K,GEMM_EDGE_DIM_N);
    SYNC_ACCESS_ABUFFER(tid,numthreads,workprogress);
  }
  free(bblk);bblk=NULL;
 }//out of openmp region, end of calculation
  free(cchunks);cchunks=NULL;
  free(workprogress);workprogress=NULL;
  free(abuffer);abuffer=NULL;
 }
}
