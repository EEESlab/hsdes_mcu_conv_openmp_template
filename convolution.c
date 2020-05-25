/*
 * Copyright (C) 2015-2020 ETH Zurich and University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pulp.h"
#include "conv_kernel.h"
#include "data.h"
#include <stdio.h>

#define NB_ITER 1
#define NB_CORES 8

static void cl_main (int *err_perf);
static void check_function (int *err_perf);
void __attribute__ ((noinline))  InitData         (uint8_t * __restrict__ Img,    int size);
void __attribute__ ((noinline))  InitZero         (uint8_t * __restrict__ Img,    int size);
void __attribute__ ((noinline))  InitKernel       (uint8_t * __restrict__ Kernel, int size);
int  __attribute__ ((noinline))  checkresult      (uint8_t * __restrict__ Out, uint8_t * __restrict__ OutGold, int N);

int main()
{
  int errors = 0;
  int cycles[8] = {0,0,0,0,0,0,0,0};
  int *err_perf[2];
  err_perf[0] = &errors;
  err_perf[1] = &cycles;

  // Mount the cluster, call the "cl_main" function with NB_CORES, at return unmount the cluster
  rt_cluster_mount(1, 0, 0, NULL);
  rt_cluster_call(NULL, 0, cl_main, err_perf, NULL, 0, 0, NB_CORES, NULL);
  rt_cluster_mount(0, 0, 0, NULL);

  if(get_core_id() == 0) {
    printf("errors=%d\n", errors);
    for(int i=0; i<8; i++) {
      printf("cycles[%d]=%d\n", i, cycles[i]);
    }
    return errors;
  }
}

// These buffers are allocated in L2 memory heap
static uint8_t  __attribute__ ((section(".heapl2ram"))) Out[IMG_DIM];
static uint8_t  __attribute__ ((section(".heapl2ram"))) In[IMG_DIM];
static uint8_t  __attribute__ ((section(".heapl2ram"))) Kernel[FILT_DIM];

// TIP: something to be allocated in L1?

// This is the main function executed on the cluster
static void cl_main(int *err_perf) {
  rt_omp_start();
  check_function(err_perf);
  rt_omp_stop();
}

static void check_function(int *err_perf) {

  int *errors = (int *) err_perf[0];
  int *cycles = (int *) (err_perf[1] + 4*rt_core_id());
  rt_perf_t perf;

  // start benchmark

  printf("2D Convolution WINDOW=%d, DATA FORMAT Q%d.%d\n",FILT_WIN,8-FRACTIONARY_BITS,FRACTIONARY_BITS);
  InitKernel(Kernel,FILT_WIN);
  InitData(In, IMG_DIM);
  InitZero(Out, IMG_DIM);

  rt_perf_reset(&perf);
  rt_perf_start(&perf);
  // TIP: look into the kernel code
  ConvKxK_Naive(In, Out, IMG_ROW, 1, IMG_ROW-1, IMG_COL, Kernel, 3);
  rt_perf_stop(&perf);

  // TIP: do not change the following
  *cycles = rt_perf_read(RT_PERF_CYCLES);
  *errors = checkresult(Out, Gold_Out_Img, IMG_DIM);
}

// load kernel
void __attribute__ ((noinline)) InitKernel(uint8_t * __restrict__ Kernel, int size)
{
  int i;
  int n = size*size;
  for (i=0; i < n; i++) {
      Kernel[i] = Filter_Kern[i];
  }
}

// load input img
void __attribute__ ((noinline)) InitData(uint8_t * __restrict__ Img, int size)
{
  int i;

  for (i=0; i < size; i++) 
      Img[i] = In_Img[i];

}

// load initialize out to 0
void __attribute__ ((noinline)) InitZero(uint8_t * __restrict__ Img, int size)
{
  int i;

  for (i=0; i < size; i++) 
      Img[i] = 0;

}

int  __attribute__ ((noinline)) checkresult(uint8_t * __restrict__ Out, uint8_t * __restrict__ OutGold, int N)
{
  int i;
  int err = 0;

  for (i = 0; i<N; i++) {
    if (Out[i]!=OutGold[i]) {
#ifdef CONV2D_DEBUG
      printf("At index %d: Actual value: %x: Expected: %x\n", i, Out[i],  OutGold[i]);
#endif
      err++;
    }
  }
  return err;
}

void print_image(uint8_t* image, int N, int M) {
  for (int y = 0; y < M; y++) {
    for (int x = 0; x < N; x++)
      printf("%d,", (uint8_t) image[M * y + x]);

    printf("\n");
  }

  printf("\n");
}
