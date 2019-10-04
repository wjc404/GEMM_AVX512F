# include <stdio.h>
# include <stdlib.h>
# include <sys/time.h>
//gcc -mavx512f AVX512_CPUTEST.c -o AVX512_CPUTEST
int main(){
  int64_t num_loops = 200000000;
  float float1 = 1.0,float2 = 2.0;
  double double1 = 1.0,double2 = 2.0 ;
  int64_t num_operations = num_loops * (int64_t)56;
  float GOPS,seconds_elapsed;
  struct timeval starttime,endtime;
  gettimeofday(&starttime,0);
  {
    __asm__  __volatile__(
        "vbroadcastss %0,%%zmm2;vbroadcastss %1,%%zmm3;xor %%r11,%%r11;"
        "999:\n\t" //loop head;
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfmadd231ps %%zmm2,%%zmm3,%%zmm5;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm6; vfmadd231ps %%zmm2,%%zmm3,%%zmm7;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm8; vfmadd231ps %%zmm2,%%zmm3,%%zmm9;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm10;vfmadd231ps %%zmm2,%%zmm3,%%zmm11;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm12;vfmadd231ps %%zmm2,%%zmm3,%%zmm13;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm14;vfmadd231ps %%zmm2,%%zmm3,%%zmm15;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm16;vfmadd231ps %%zmm2,%%zmm3,%%zmm17;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm18;vfmadd231ps %%zmm2,%%zmm3,%%zmm19;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm20;vfmadd231ps %%zmm2,%%zmm3,%%zmm21;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm22;vfmadd231ps %%zmm2,%%zmm3,%%zmm23;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm24;vfmadd231ps %%zmm2,%%zmm3,%%zmm25;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm26;vfmadd231ps %%zmm2,%%zmm3,%%zmm27;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm28;vfmadd231ps %%zmm2,%%zmm3,%%zmm29;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm30;vfmadd231ps %%zmm2,%%zmm3,%%zmm31;"
        "inc %%r11;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm5;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm6; vfnmadd231ps %%zmm2,%%zmm3,%%zmm7;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm8; vfnmadd231ps %%zmm2,%%zmm3,%%zmm9;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm10;vfnmadd231ps %%zmm2,%%zmm3,%%zmm11;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm12;vfnmadd231ps %%zmm2,%%zmm3,%%zmm13;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm14;vfnmadd231ps %%zmm2,%%zmm3,%%zmm15;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm16;vfnmadd231ps %%zmm2,%%zmm3,%%zmm17;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm18;vfnmadd231ps %%zmm2,%%zmm3,%%zmm19;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm20;vfnmadd231ps %%zmm2,%%zmm3,%%zmm21;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm22;vfnmadd231ps %%zmm2,%%zmm3,%%zmm23;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm24;vfnmadd231ps %%zmm2,%%zmm3,%%zmm25;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm26;vfnmadd231ps %%zmm2,%%zmm3,%%zmm27;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm28;vfnmadd231ps %%zmm2,%%zmm3,%%zmm29;"
        "vfnmadd231ps %%zmm2,%%zmm3,%%zmm30;vfnmadd231ps %%zmm2,%%zmm3,%%zmm31;"
        "cmp %2,%%r11;jb 999b;"
    ::"m"(float1),"m"(float2),"r"(num_loops)
    :"%r11","%zmm2","%zmm3","%zmm4","%zmm5","%zmm6","%zmm7","%zmm8","%zmm9","%zmm10","%zmm11",
    "%zmm12","%zmm13","%zmm14","%zmm15","%zmm16","%zmm17","%zmm18","%zmm19","%zmm20","%zmm21","%zmm22","%zmm23",
    "%zmm24","%zmm25","%zmm26","%zmm27","%zmm28","%zmm29","%zmm30","%zmm31");
  }
  gettimeofday(&endtime,0);
  seconds_elapsed = (float)(endtime.tv_sec - starttime.tv_sec) + (float)(endtime.tv_usec - starttime.tv_usec)/1.0e6;
  GOPS = (float)num_operations / 1.0e9 / seconds_elapsed;
  printf("The maximum 1-core GOPS of independent avx512 float32 fma instructions: %.2f\nDivide this value by CPU frequency (in GHz) you get the avx512 float32 fma throughput (in insns per cycle) per core, i.e. CPU_NUM_512FMA_UNITS\n\n",GOPS);

  gettimeofday(&starttime,0);
  {
    __asm__  __volatile__(
        "vbroadcastsd %0,%%zmm2;vbroadcastsd %1,%%zmm3;xor %%r11,%%r11;"
        "999:\n\t" //loop head;
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfmadd231pd %%zmm2,%%zmm3,%%zmm5;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm6; vfmadd231pd %%zmm2,%%zmm3,%%zmm7;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm8; vfmadd231pd %%zmm2,%%zmm3,%%zmm9;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm10;vfmadd231pd %%zmm2,%%zmm3,%%zmm11;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm12;vfmadd231pd %%zmm2,%%zmm3,%%zmm13;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm14;vfmadd231pd %%zmm2,%%zmm3,%%zmm15;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm16;vfmadd231pd %%zmm2,%%zmm3,%%zmm17;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm18;vfmadd231pd %%zmm2,%%zmm3,%%zmm19;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm20;vfmadd231pd %%zmm2,%%zmm3,%%zmm21;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm22;vfmadd231pd %%zmm2,%%zmm3,%%zmm23;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm24;vfmadd231pd %%zmm2,%%zmm3,%%zmm25;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm26;vfmadd231pd %%zmm2,%%zmm3,%%zmm27;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm28;vfmadd231pd %%zmm2,%%zmm3,%%zmm29;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm30;vfmadd231pd %%zmm2,%%zmm3,%%zmm31;"
        "inc %%r11;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm5;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm6; vfnmadd231pd %%zmm2,%%zmm3,%%zmm7;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm8; vfnmadd231pd %%zmm2,%%zmm3,%%zmm9;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm10;vfnmadd231pd %%zmm2,%%zmm3,%%zmm11;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm12;vfnmadd231pd %%zmm2,%%zmm3,%%zmm13;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm14;vfnmadd231pd %%zmm2,%%zmm3,%%zmm15;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm16;vfnmadd231pd %%zmm2,%%zmm3,%%zmm17;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm18;vfnmadd231pd %%zmm2,%%zmm3,%%zmm19;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm20;vfnmadd231pd %%zmm2,%%zmm3,%%zmm21;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm22;vfnmadd231pd %%zmm2,%%zmm3,%%zmm23;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm24;vfnmadd231pd %%zmm2,%%zmm3,%%zmm25;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm26;vfnmadd231pd %%zmm2,%%zmm3,%%zmm27;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm28;vfnmadd231pd %%zmm2,%%zmm3,%%zmm29;"
        "vfnmadd231pd %%zmm2,%%zmm3,%%zmm30;vfnmadd231pd %%zmm2,%%zmm3,%%zmm31;"
        "cmp %2,%%r11;jb 999b;"
    ::"m"(double1),"m"(double2),"r"(num_loops)
    :"%r11","%zmm2","%zmm3","%zmm4","%zmm5","%zmm6","%zmm7","%zmm8","%zmm9","%zmm10","%zmm11",
    "%zmm12","%zmm13","%zmm14","%zmm15","%zmm16","%zmm17","%zmm18","%zmm19","%zmm20","%zmm21","%zmm22","%zmm23",
    "%zmm24","%zmm25","%zmm26","%zmm27","%zmm28","%zmm29","%zmm30","%zmm31");
  }
  gettimeofday(&endtime,0);
  seconds_elapsed = (float)(endtime.tv_sec - starttime.tv_sec) + (float)(endtime.tv_usec - starttime.tv_usec)/1.0e6;
  GOPS = (float)num_operations / 1.0e9 / seconds_elapsed;
  printf("The maximum 1-core GOPS of independent avx512 float64 fma instructions: %.2f\nDivide this value by CPU frequency (in GHz) you get the avx512 float64 fma throughput (in insns per cycle) per core, i.e. CPU_NUM_512FMA_UNITS\n\n",GOPS);


  num_loops = 20000000;
  num_operations = num_loops * (int64_t)56;
  gettimeofday(&starttime,0);
  {
    __asm__  __volatile__(
        "vbroadcastss %0,%%zmm2;vbroadcastss %1,%%zmm3;xor %%r11,%%r11;"
        "999:\n\t" //loop head;
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "inc %%r11;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231ps %%zmm2,%%zmm3,%%zmm4; vfnmadd231ps %%zmm2,%%zmm3,%%zmm4;"
        "cmp %2,%%r11;jb 999b;"
    ::"m"(float1),"m"(float2),"r"(num_loops)
    :"%r11","%zmm2","%zmm3","%zmm4");
  }
  gettimeofday(&endtime,0);
  seconds_elapsed = (float)(endtime.tv_sec - starttime.tv_sec) + (float)(endtime.tv_usec - starttime.tv_usec)/1.0e6;
  GOPS = (float)num_operations / 1.0e9 / seconds_elapsed;
  printf("The maximum 1-core GOPS of chained avx512 float32 fma instructions: %.3f\nDivide CPU frequency (in GHz) by this value you get the avx512 float32 fma latency in cycles.\n\n",GOPS);

  gettimeofday(&starttime,0);
  {
    __asm__  __volatile__(
        "vbroadcastsd %0,%%zmm2;vbroadcastsd %1,%%zmm3;xor %%r11,%%r11;"
        "999:\n\t" //loop head;
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "inc %%r11;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "vfmadd231pd %%zmm2,%%zmm3,%%zmm4; vfnmadd231pd %%zmm2,%%zmm3,%%zmm4;"
        "cmp %2,%%r11;jb 999b;"
    ::"m"(double1),"m"(double2),"r"(num_loops)
    :"%r11","%zmm2","%zmm3","%zmm4");
  }
  gettimeofday(&endtime,0);
  seconds_elapsed = (float)(endtime.tv_sec - starttime.tv_sec) + (float)(endtime.tv_usec - starttime.tv_usec)/1.0e6;
  GOPS = (float)num_operations / 1.0e9 / seconds_elapsed;
  printf("The maximum 1-core GOPS of chained avx512 float64 fma instructions: %.3f\nDivide CPU frequency (in GHz) by this value you get the avx512 float64 fma latency in cycles.\n\n",GOPS);


  int64_t int1=1,int2=-1;
  num_loops = 80000000;
  num_operations = num_loops * (int64_t)56;
  gettimeofday(&starttime,0);
  {
    __asm__  __volatile__(
        "vbroadcastsd %0,%%zmm2;vbroadcastsd %1,%%zmm3;xor %%r11,%%r11;"
        "999:\n\t" //loop head; 56 chained avx512 instructions per iteration
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "inc %%r11;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "vpaddq %%zmm2,%%zmm4,%%zmm4; vpaddq %%zmm3,%%zmm4,%%zmm4;"
        "cmp %2,%%r11;jb 999b;"
    ::"m"(int1),"m"(int2),"r"(num_loops)
    :"%r11","%zmm2","%zmm3","%zmm4");
  }
  gettimeofday(&endtime,0);
  seconds_elapsed = (float)(endtime.tv_sec - starttime.tv_sec) + (float)(endtime.tv_usec - starttime.tv_usec)/1.0e6;
  GOPS = (float)num_operations / 1.0e9 / seconds_elapsed;
  printf("The maximum 1-core GOPS of chained avx512 int64_add instructions: %.3f\n\n",GOPS);

  return 0;
}
