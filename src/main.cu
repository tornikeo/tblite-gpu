#define NDEBUG

#define MAX_THREADS_PER_BLOCK 32
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "device_tensor.h"
#include "types.h"

#define s3 1.73205080757 // sqrt(3)
#define s3_4 (s3 / 2) // sqrt(3)/2
#define sqrtpi3 5.56832799683 // sqrt(pi)**3

template <typename T, int D>
__device__ inline void transform0(const int li, const int lj, const device_tensor2d_t<T> &cart,  device_tensor2d_fixed_t<T, D, D> &sphr)
{
  /* sphr is a larger array. It contains the max size that an integral might need
  so iterate over smaller cart dims instead*/
  if (li <= 1 && lj <= 1)
  {
    // for(int i = 0; i < cart.dim1; i++)
    //   for(int j = 0; j < cart.dim2; ++j)
    const auto total = cart.dim1 * cart.dim2;
    for(int t = threadIdx.x; t < total; t+= blockDim.x)
    {
      const int i = t / cart.dim2; 
      const int j = t % cart.dim2; 
      sphr(i,j) = cart(i,j);
    }
  } 
  else if (li <= 1 && lj == 2)
  {
    for(int i = threadIdx.x + 1; i <= cart.dim1; i += blockDim.x)
    {
      // sphr = matmul(dtrafo, cart)
      // sphr(1, :) = cart(3, :) - 0.5_wp * (cart(1, :) + cart(2, :))
        sphr(1, i, 'f') = cart(3, i, 'f') - 0.5 * (cart(1, i, 'f') + cart(2, i, 'f'));
      // sphr(2, :) = s3 * cart(5, :)
        sphr(2, i, 'f') = s3 * cart(5, i, 'f');
      // sphr(3, :) = s3 * cart(6, :)
        sphr(3, i, 'f') = s3 * cart(6, i, 'f');
      // sphr(4, :) = s3_4 * (cart(1, :) - cart(2, :))
        sphr(4, i, 'f') = s3_4 * (cart(1, i, 'f') - cart(2, i, 'f'));
      // sphr(5, :) = s3 * cart(4, :)
        sphr(5, i, 'f') = s3 * cart(4, i, 'f');
    }
  } 
  else if (li == 2 && lj <= 1) 
  {    
    for(int i = threadIdx.x + 1; i <= cart.dim2; i += blockDim.x)
    {
    // sphr(:, 1) = cart(:, 3) - 0.5_wp * (cart(:, 1) + cart(:, 2))
      sphr(i, 1, 'f') = cart(i, 3, 'f') - 0.5 * (cart(i, 1, 'f') + cart(i, 2, 'f'));
    // sphr(:, 2) = s3 * cart(:, 5)
      sphr(i, 2, 'f') = s3 * cart(i, 5, 'f');
    // sphr(:, 3) = s3 * cart(:, 6)
      sphr(i, 3, 'f') = s3 * cart(i, 6, 'f');
    // sphr(:, 4) = s3_4 * (cart(:, 1) - cart(:, 2))
      sphr(i, 4, 'f') = s3_4 * (cart(i, 1, 'f') - cart(i, 2, 'f'));
    // sphr(:, 5) = s3 * cart(:, 4)
      sphr(i, 5, 'f') = s3 * cart(i, 4, 'f');
    }
  } 
  else if (li == 2 && lj == 2)
  {
    /* REMEMBER 
      i,j -> i-1, j-1, due to Fortran indexing
    */
    // sphr(1, 1) = cart(3, 3) &
    //   & - 0.5_wp * (cart(3, 1) + cart(3, 2) + cart(1, 3) + cart(2, 3)) &
    //   & + 0.25_wp * (cart(1, 1) + cart(1, 2) + cart(2, 1) + cart(2, 2))
    if(threadIdx.x == 0)
    {
    sphr(1,1,'f') = cart(3,3,'f')
      - 0.5 * (cart(3,1,'f') + cart(3,2,'f') + cart(1,3,'f') + cart(2,3,'f'))
      + 0.25 * (cart(1,1,'f') + cart(1,2,'f') + cart(2,1,'f') + cart(2,2,'f'));
    // sphr([2, 3, 5], 1) = s3 * cart([5, 6, 4], 3) &
    //   & - s3_4 * (cart([5, 6, 4], 1) + cart([5, 6, 4], 2))
    sphr(2, 1,'f') = s3 * cart(5, 3,'f') - s3_4 * (cart(5, 1,'f') + cart(5, 2,'f'));
    sphr(3, 1,'f') = s3 * cart(6, 3,'f') - s3_4 * (cart(6, 1,'f') + cart(6, 2,'f'));
    sphr(5, 1,'f') = s3 * cart(4, 3,'f') - s3_4 * (cart(4, 1,'f') + cart(4, 2,'f'));
    // sphr(4, 1) = s3_4 * (cart(1, 3) - cart(2, 3)) &
    //   & - s3 * 0.25_wp * (cart(1, 1) - cart(2, 1) + cart(1, 2) - cart(2, 2))
    sphr(4,1,'f') = s3_4 * (cart(1,3,'f') - cart(2,3,'f'))
      - s3 * 0.25 * (cart(1,1,'f') - cart(2,1,'f') + cart(1,2,'f') - cart(2,2,'f'));
    // sphr(1, 2) = s3 * cart(3, 5) - s3_4 * (cart(1, 5) + cart(2, 5))
    sphr(1,2,'f') = s3 * cart(3,5,'f') - s3_4 * (cart(1,5,'f') + cart(2,5,'f'));
    // sphr([2, 3, 5], 2) = 3 * cart([5, 6, 4], 5)
    sphr(2, 2,'f') = 3 * cart(5, 5,'f');
    sphr(3, 2,'f') = 3 * cart(6, 5,'f');
    sphr(5, 2,'f') = 3 * cart(4, 5,'f');
    // sphr(4, 2) = 1.5_wp * (cart(1, 5) - cart(2, 5))
    sphr(4,2,'f') = 1.5 * (cart(1,5,'f') - cart(2,5,'f'));
    // sphr(1, 3) = s3 * cart(3, 6) - s3_4 * (cart(1, 6) + cart(2, 6))
    sphr(1,3,'f') = s3 * cart(3,6,'f') - s3_4 * (cart(1,6,'f') + cart(2,6,'f'));
    // sphr([2, 3, 5], 3) = 3 * cart([5, 6, 4], 6)
    sphr(2, 3,'f') = 3 * cart(5, 6,'f');
    sphr(3, 3,'f') = 3 * cart(6, 6,'f');
    sphr(5, 3,'f') = 3 * cart(4, 6,'f');
    // sphr(4, 3) = 1.5_wp * (cart(1, 6) - cart(2, 6))
    sphr(4,3,'f') = 1.5 * (cart(1,6,'f') - cart(2,6,'f'));
    // sphr(1, 4) = s3_4 * (cart(3, 1) - cart(3, 2)) &
    //   & - s3 * 0.25_wp * (cart(1, 1) - cart(1, 2) + cart(2, 1) - cart(2, 2))
    sphr(1,4,'f') = s3_4 * (cart(3,1,'f') - cart(3,2,'f'))
      - s3 * 0.25 * (cart(1,1,'f') - cart(1,2,'f') + cart(2,1,'f') - cart(2,2,'f'));
    // sphr([2, 3, 5], 4) = 1.5_wp * (cart([5, 6, 4], 1) - cart([5, 6, 4], 2))
    sphr(2, 4,'f') = 1.5 * (cart(5, 1,'f') - cart(5, 2,'f'));
    sphr(3, 4,'f') = 1.5 * (cart(6, 1,'f') - cart(6, 2,'f'));
    sphr(5, 4,'f') = 1.5 * (cart(4, 1,'f') - cart(4, 2,'f'));
    // sphr(4, 4) = 0.75_wp * (cart(1, 1) - cart(2, 1) - cart(1, 2) + cart(2, 2))
    sphr(4,4,'f') = 0.75 * (cart(1,1,'f') - cart(2,1,'f') - cart(1,2,'f') + cart(2,2,'f'));
    // sphr(1, 5) = s3 * cart(3, 4) - s3_4 * (cart(1, 4) + cart(2, 4))
    sphr(1,5,'f') = s3 * cart(3,4,'f') - s3_4 * (cart(1,4,'f') + cart(2,4,'f'));
    // sphr([2, 3, 5], 5) = 3 * cart([5, 6, 4], 4)
    sphr(2, 5,'f') = 3 * cart(5, 4,'f');
    sphr(3, 5,'f') = 3 * cart(6, 4,'f');
    sphr(5, 5,'f') = 3 * cart(4, 4,'f');
    // sphr(4, 5) = 1.5_wp * (cart(1, 4) - cart(2, 4))
    sphr(4,5,'f') = 1.5 * (cart(1,4,'f') - cart(2,4,'f'));
    }
  } 
  else 
  {
    printf("[Fatal] transform0 not supported for li=%d lj=%d\n", li, lj);
    assert(false);
  }
}

template <typename T, int D, int K>
__device__ inline void transform1(const int li, const int lj, const device_tensor3d_t<T> &cart, device_tensor3d_fixed_t<T, D, D, K> &sphr)
{
  if (li <= 1 && lj <= 1) /* HOT PATH */
  {
    const auto total = cart.dim1 * cart.dim2 * cart.dim3;
    for(int t = threadIdx.x; t < total; t += blockDim.x)
    {
      const int i = t / (cart.dim2 * cart.dim3);
      const int j = (t / cart.dim3) % cart.dim2;
      const int k = t % cart.dim3;
      sphr(i,j,k) = cart(i,j,k);
    }
  }
  else if (li <= 1 && lj == 2)
  {
    const int total = cart.dim1 * cart.dim3;
    for(int t = threadIdx.x; t < total; t+=blockDim.x)
    // for(int i = 1; i <= cart.dim1; ++i)
    {
      // for(int k = 1; k <= cart.dim3; ++k)
      {
        const int i = t / cart.dim3 + 1;
        const int k = t % cart.dim3 + 1;
        // sphr(1, :) = cart(3, :) - 0.5_wp * (cart(1, :) + cart(2, :))
        sphr(k, 1, i, 'f') = cart(k, 3, i, 'f') - 0.5 * (cart(k, 1, i, 'f') + cart(k, 2, i, 'f'));
        // sphr(2, :) = s3 * cart(5, :)
        sphr(k, 2, i, 'f') = s3 * cart(k, 5, i, 'f');
        // sphr(3, :) = s3 * cart(6, :)
        sphr(k, 3, i, 'f') = s3 * cart(k, 6, i, 'f');
        // sphr(4, :) = s3_4 * (cart(1, :) - cart(2, :))
        sphr(k, 4, i, 'f') = s3_4 * (cart(k, 1, i, 'f') - cart(k, 2, i, 'f'));
        // sphr(5, :) = s3 * cart(4, :)
        sphr(k, 5, i, 'f') = s3 * cart(k, 4, i, 'f');
      }
    }
  }
  else if (li == 2 && lj <= 1)
  {
    // if(threadIdx.x > 0) return;
    // sphr(:, 1) = cart(:, 3) - 0.5_wp * (cart(:, 1) + cart(:, 2))
    const int total = cart.dim2 * cart.dim3;
    for(int t = threadIdx.x; t < total; t+=blockDim.x)
    // for(int i = 1; i <= cart.dim2; ++i)
    {
      // for(int k = 1; k <= cart.dim3; ++k)
      {
        const int i = t / cart.dim3 + 1;
        const int k = t % cart.dim3 + 1;
        sphr(k, i, 1, 'f') = cart(k, i, 3, 'f') - 0.5 * (cart(k, i, 1, 'f') + cart(k, i, 2, 'f'));
        // sphr(:, 2) = s3 * cart(:, 5)
        sphr(k, i, 2, 'f') = s3 * cart(k, i, 5, 'f');
        // sphr(:, 3) = s3 * cart(:, 6)
        sphr(k, i, 3, 'f') = s3 * cart(k, i, 6, 'f');
        // sphr(:, 4) = s3_4 * (cart(:, 1) - cart(:, 2))
        sphr(k, i, 4, 'f') = s3_4 * (cart(k, i, 1, 'f') - cart(k, i, 2, 'f'));
        // sphr(:, 5) = s3 * cart(:, 4)
        sphr(k, i, 5, 'f') = s3 * cart(k, i, 4, 'f');
      }
    }
  } 
  else if (li == 2 && lj == 2)
  {
    for(int k = threadIdx.x+1; k <= cart.dim3; k+=blockDim.x)
    {
      // sphr(1, 1) = cart(3, 3) &
      //   & - 0.5_wp * (cart(3, 1) + cart(3, 2) + cart(1, 3) + cart(2, 3)) &
      //   & + 0.25_wp * (cart(1, 1) + cart(1, 2) + cart(2, 1) + cart(2, 2))
      sphr(k, 1, 1, 'f') = cart(k, 3, 3, 'f')
        - 0.5 * (cart(k, 3, 1, 'f') + cart(k, 3, 2, 'f') + cart(k, 1, 3, 'f') + cart(k, 2, 3, 'f'))
        + 0.25 * (cart(k, 1, 1, 'f') + cart(k, 1, 2, 'f') + cart(k, 2, 1, 'f') + cart(k, 2, 2, 'f'));
      // sphr([2, 3, 5], 1) = s3 * cart([5, 6, 4], 3) &
      //   & - s3_4 * (cart([5, 6, 4], 1) + cart([5, 6, 4], 2))
      sphr(k, 2, 1, 'f') = s3 * cart(k, 5, 3, 'f') - s3_4 * (cart(k, 5, 1, 'f') + cart(k, 5, 2, 'f'));
      sphr(k, 3, 1, 'f') = s3 * cart(k, 6, 3, 'f') - s3_4 * (cart(k, 6, 1, 'f') + cart(k, 6, 2, 'f'));
      sphr(k, 5, 1, 'f') = s3 * cart(k, 4, 3, 'f') - s3_4 * (cart(k, 4, 1, 'f') + cart(k, 4, 2, 'f'));
      // sphr(4, 1) = s3_4 * (cart(1, 3) - cart(2, 3)) &
      //   & - s3 * 0.25_wp * (cart(1, 1) - cart(2, 1) + cart(1, 2) - cart(2, 2))
      sphr(k, 4, 1, 'f') = s3_4 * (cart(k, 1, 3, 'f') - cart(k, 2, 3, 'f'))
        - s3 * 0.25 * (cart(k, 1, 1, 'f') - cart(k, 2, 1, 'f') + cart(k, 1, 2, 'f') - cart(k, 2, 2, 'f'));
      // sphr(1, 2) = s3 * cart(3, 5) - s3_4 * (cart(1, 5) + cart(2, 5))
        sphr(k, 1, 2, 'f') = s3 * cart(k, 3, 5, 'f') - s3_4 * (cart(k, 1, 5, 'f') + cart(k, 2, 5, 'f'));
      // sphr([2, 3, 5], 2) = 3 * cart([5, 6, 4], 5)
      sphr(k, 2, 2, 'f') = 3 * cart(k, 5, 5, 'f');
      sphr(k, 3, 2, 'f') = 3 * cart(k, 6, 5, 'f');
      sphr(k, 5, 2, 'f') = 3 * cart(k, 4, 5, 'f');
      // sphr(4, 2) = 1.5_wp * (cart(1, 5) - cart(2, 5))
      sphr(k, 4, 2, 'f') = 1.5 * (cart(k, 1, 5, 'f') - cart(k, 2, 5, 'f'));
      // sphr(1, 3) = s3 * cart(3, 6) - s3_4 * (cart(1, 6) + cart(2, 6))
      sphr(k, 1, 3, 'f') = s3 * cart(k, 3, 6, 'f') - s3_4 * (cart(k, 1, 6, 'f') + cart(k, 2, 6, 'f'));
      // sphr([2, 3, 5], 3) = 3 * cart([5, 6, 4], 6)
      sphr(k, 2, 3, 'f') = 3 * cart(k, 5, 6, 'f');
      sphr(k, 3, 3, 'f') = 3 * cart(k, 6, 6, 'f');
      sphr(k, 5, 3, 'f') = 3 * cart(k, 4, 6, 'f');
      // sphr(4, 3) = 1.5_wp * (cart(1, 6) - cart(2, 6))
      sphr(k, 4, 3, 'f') = 1.5 * (cart(k, 1, 6, 'f') - cart(k, 2, 6, 'f'));
      // sphr(1, 4) = s3_4 * (cart(3, 1) - cart(3, 2)) &
      //   & - s3 * 0.25_wp * (cart(1, 1) - cart(1, 2) + cart(2, 1) - cart(2, 2))
      sphr(k, 1, 4, 'f') = s3_4 * (cart(k, 3, 1, 'f') - cart(k, 3, 2, 'f'))
        - s3 * 0.25 * (cart(k, 1, 1, 'f') - cart(k, 1, 2, 'f') + cart(k, 2, 1, 'f') - cart(k, 2, 2, 'f'));
      // sphr([2, 3, 5], 4) = 1.5_wp * (cart([5, 6, 4], 1) - cart([5, 6, 4], 2))
      sphr(k, 2, 4, 'f') = 1.5 * (cart(k, 5, 1, 'f') - cart(k, 5, 2, 'f'));
      sphr(k, 3, 4, 'f') = 1.5 * (cart(k, 6, 1, 'f') - cart(k, 6, 2, 'f'));
      sphr(k, 5, 4, 'f') = 1.5 * (cart(k, 4, 1, 'f') - cart(k, 4, 2, 'f'));
      // sphr(4, 4) = 0.75_wp * (cart(1, 1) - cart(2, 1) - cart(1, 2) + cart(2, 2))
      sphr(k, 4, 4, 'f') = 0.75 * (cart(k, 1, 1, 'f') - cart(k, 2, 1, 'f') - cart(k, 1, 2, 'f') + cart(k, 2, 2, 'f'));
      // sphr(1, 5) = s3 * cart(3, 4) - s3_4 * (cart(1, 4) + cart(2, 4))
      sphr(k, 1, 5, 'f') = s3 * cart(k, 3, 4, 'f') - s3_4 * (cart(k, 1, 4, 'f') + cart(k, 2, 4, 'f'));
      // sphr([2, 3, 5], 5) = 3 * cart([5, 6, 4], 4)
      sphr(k, 2, 5, 'f') = 3 * cart(k, 5, 4, 'f');
      sphr(k, 3, 5, 'f') = 3 * cart(k, 6, 4, 'f');
      sphr(k, 5, 5, 'f') = 3 * cart(k, 4, 4, 'f');
      // sphr(4, 5) = 1.5_wp * (cart(1, 4) - cart(2, 4))
      sphr(k, 4, 5, 'f') = 1.5 * (cart(k, 1, 4, 'f') - cart(k, 2, 4, 'f'));
    }
  }
  else
  {
    printf("[Fatal] transform1 not supported for li=%i lj=%i\n", li, lj);
    assert(false);
  }
}


template <typename T, typename T2>
__device__ inline void horizontal_shift(const T ae, const int l, T2 * __restrict__ cfs)
{
  switch (l)
  {
  case 0: // s
    break;
  case 1: // p
    cfs[0] += ae * cfs[1];
    break;
  case 2: // d
    cfs[0] += ae * ae * cfs[2];
    cfs[1] += 2 * ae * cfs[2];
    break;
  case 3: // f
    cfs[0] += ae * ae * ae * cfs[3];
    cfs[1] += 3 * ae * ae * cfs[3];
    cfs[2] += 3 * ae * cfs[3];
    break;
  case 4: // g
    cfs[0] += ae * ae * ae * ae * cfs[4];
    cfs[1] += 4 * ae * ae * ae * cfs[4];
    cfs[2] += 6 * ae * ae * cfs[4];
    cfs[3] += 4 * ae * cfs[4];
    break;
  default:
    printf("[Fatal] horizontal_shift not supported for l=%i\n", l);
    assert(false);
    return;
  }
}

template <typename T, typename T2, typename T3>
__device__ inline void form_product(
  const T * __restrict__ a /*[MAXL]*/,
  const T2 * __restrict__ b /*[MAXL]*/,
  const int &la, const int &lb,
  T3 (&d)[MAXL2])
{
  if (la >= 4 || lb >= 4) goto label_40;
  if (la >= 3 || lb >= 3) goto label_30;
  if (la >= 2 || lb >= 2) goto label_20;
  // <s|s> = <s>
  d[0] = a[0] * b[0];
  if (la == 0 && lb == 0) return;
  // <s|p> = <s|*(|s>+|p>)
  //       = <s> + <p>
  d[1] = a[0] * b[1] + a[1] * b[0];
  if (la == 0 || lb == 0) return;
  // <p|p> = (<s|+<p|)*(|s>+|p>)
  //       = <s> + <p> + <d>
  d[2] = a[1] * b[1];
  return;
label_20:
  // <s|d> = <s|*(|s>+|p>+|d>)
  //       = <s> + <p> + <d>
  d[0] = a[0] * b[0];
  d[1] = a[0] * b[1] + a[1] * b[0];
  d[2] = a[0] * b[2] + a[2] * b[0];
  if (la == 0 || lb == 0) return;
  // <p|d> = (<s|+<p|)*(|s>+|p>+|d>)
  //       = <s> + <p> + <d> + <f>
  d[2] += a[1] * b[1];
  d[3] = a[1] * b[2] + a[2] * b[1];
  if (la <= 1 || lb <= 1) return;
  // <d|d> = (<s|+<p|+<d|)*(|s>+|p>+|d>)
  //       = <s> + <p> + <d> + <f> + <g>
  d[4] = a[2] * b[2];
  return;
label_30:
  // <s|f> = <s|*(|s>+|p>+|d>+|f>)
  //       = <s> + <p> + <d> + <f>
  d[0] = a[0] * b[0];
  d[1] = a[0] * b[1] + a[1] * b[0];
  d[2] = a[0] * b[2] + a[2] * b[0];
  d[3] = a[0] * b[3] + a[3] * b[0];
  if (la == 0 || lb == 0) return;
  // <p|f> = (<s|+<p|)*(|s>+|p>+|d>+|f>)
  //       = <s> + <p> + <d> + <f> + <g>
  d[2] += a[1] * b[1];
  d[3] += a[1] * b[2] + a[2] * b[1];
  d[4] = a[1] * b[3] + a[3] * b[1];
  if (la <= 1 || lb <= 1) return;
  // <d|f> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>)
  //       = <s> + <p> + <d> + <f> + <g> + <h>
  d[4] += a[2] * b[2];
  d[5] = a[2] * b[3] + a[3] * b[2];
  if (la <= 2 || lb <= 2) return;
  // <f|f> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>)
  //       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
  d[6] = a[3] * b[3];
  return;
label_40:
  // <s|g> = <s|*(|s>+|p>+|d>+|f>+|g>)
  //       = <s> + <p> + <d> + <f> + <g>
  d[0] = a[0] * b[0];
  d[1] = a[0] * b[1] + a[1] * b[0];
  d[2] = a[0] * b[2] + a[2] * b[0];
  d[3] = a[0] * b[3] + a[3] * b[0];
  d[4] = a[0] * b[4] + a[4] * b[0];
  if (la == 0 || lb == 0) return;
  // <p|g> = (<s|+<p|)*(|s>+|p>+|d>+|f>+|g>)
  //       = <s> + <p> + <d> + <f> + <g> + <h>
  d[2] += a[1] * b[1];
  d[3] += a[1] * b[2] + a[2] * b[1];
  d[4] += a[1] * b[3] + a[3] * b[1];
  d[5] = a[1] * b[4] + a[4] * b[1];
  if (la <= 1 || lb <= 1) return;
  // <d|g> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>+|g>)
  //       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
  d[4] += a[2] * b[2];
  d[5] += a[2] * b[3] + a[3] * b[2];
  d[6] = a[2] * b[4] + a[4] * b[2];
  if (la <= 2 || lb <= 2) return;
  // <f|g> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>+|g>)
  //       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k>
  d[6] += a[3] * b[3];
  d[7] = a[3] * b[4] + a[4] * b[3];
  if (la <= 3 || lb <= 3) return;
  // <g|g> = (<s|+<p|+<d|+<f|+<g|)*(|s>+|p>+|d>+|f>+|g>)
  //       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k> + <l>
  d[8] = a[4] * b[4];
  return;
}

__device__ inline double overlap_1d(const int moment, const double alpha)
{
  // double overlap = 0.0;
  constexpr double dfactorial[/*8*/] = {1.0, 1.0, 3.0, 15.0, /*105.0, 945.0, 10395.0, 135135.0*/};
  assert(moment >= 0 && moment <= 7);
  const auto mul = (moment % 2 == 0);
  // if ()
  // const auto overlap = mul * (pow(0.5 / alpha, moment / 2) * dfactorial[moment / 2]);
  // else
  //   overlap = 0.0;
  // return overlap;
  return mul * (pow(0.5 / alpha, moment / 2) * dfactorial[moment / 2]);
}

__device__ inline void multipole_3d(
  const double (&rpi)[3],
  const double (&rpj)[3],
  const double aj,
  const double ai,
  const int (&lj)[3],
  const int (&li)[3],
  const double (&s1d)[MAXL2],
  double &s3d,
  double (&d3d)[3],
  double (&q3d)[6])
{
  float v1d[3][3] = {0.0};

  #pragma unroll
  for(int k = 0; k < 3; ++k)
  {
    float vi[MAXL] = {0.0};
    float vj[MAXL] = {0.0};
    float vv[MAXL2] = {0.0};
    // #pragma unroll
    // for(int i = 0; i <= MAXL; ++i)
    // {
    //   vi[i] += li[k] == i;
    //   vj[i] += lj[k] == i;
    // }
    vi[li[k]] = 1.0;
    vj[lj[k]] = 1.0;
    horizontal_shift(rpi[k], li[k], vi);
    horizontal_shift(rpj[k], lj[k], vj);
    form_product(vi, vj, li[k], lj[k], vv);
    
    for (int l = 0; l <= li[k] + lj[k]; ++l)
    {
      assert(l < MAXL2 - 1);
      v1d[k][0] += s1d[l] * vv[l];
      v1d[k][1] += (s1d[l + 1] + rpj[k] * s1d[l]) * vv[l];
      v1d[k][2] += (s1d[l + 2] + 2 * rpj[k] * s1d[l + 1] + rpj[k] * rpj[k] * s1d[l]) * vv[l];
    }
  }
  s3d = v1d[0][0] * v1d[1][0] * v1d[2][0];
  
  d3d[0] = v1d[0][1] * v1d[1][0] * v1d[2][0];
  d3d[1] = v1d[0][0] * v1d[1][1] * v1d[2][0];
  d3d[2] = v1d[0][0] * v1d[1][0] * v1d[2][1];

  q3d[0] = v1d[0][2] * v1d[1][0] * v1d[2][0];
  q3d[1] = v1d[0][1] * v1d[1][1] * v1d[2][0];
  q3d[2] = v1d[0][0] * v1d[1][2] * v1d[2][0];
  q3d[3] = v1d[0][1] * v1d[1][0] * v1d[2][1];
  q3d[4] = v1d[0][0] * v1d[1][1] * v1d[2][1];
  q3d[5] = v1d[0][0] * v1d[1][0] * v1d[2][2];
}

template <size_t maxl, int D>
__device__ void multipole_cgto(
  const cgto_type &cgtoj,
  const cgto_type &cgtoi,
  const double r2,
  const double (&vec)[3],
  const double intcut,
  device_tensor2d_fixed_t<double, D, D> &overlap,
  device_tensor3d_fixed_t<double, D, D, 3> &dpint,
  device_tensor3d_fixed_t<double, D, D, 6> &qpint)
{
  constexpr int msao[] = {1, 3, 5, /*7, 9, 11, 13   */}; 
  constexpr int mlao[] = {1, 3, 6, /*10, 15, 21, 28 */};
  constexpr int lmap[] = {0, 1, 4, /*10, 20, 35, 56 */};
  constexpr int lx[/*84*/][3] = {
    {0,0,0,},
    {0,1,0,},
    {0,0,1,},
    {1,0,0,},
    {2,0,0,},
    {0,2,0,},
    {0,0,2,},
    {1,1,0,},
    {1,0,1,},
    {0,1,1,},
    {3,0,0,},
    {0,3,0,},
    {0,0,3,},
    {2,1,0,},
    {2,0,1,},
    {1,2,0,},
    {0,2,1,},
    {1,0,2,},
    {0,1,2,},
    {1,1,1,}//,
    // {4,0,0,},
    // {0,4,0,},
    // {0,0,4,},
    // {3,1,0,},
    // {3,0,1,},
    // {1,3,0,},
    // {0,3,1,},
    // {1,0,3,},
    // {0,1,3,},
    // {2,2,0,},
    // {2,0,2,},
    // {0,2,2,},
    // {2,1,1,},
    // {1,2,1,},
    // {1,1,2,},
    // {5,0,0,},
    // {0,5,0,},
    // {0,0,5,},
    // {3,2,0,},
    // {3,0,2,},
    // {2,3,0,},
    // {2,0,3,},
    // {0,3,2,},
    // {0,2,3,},
    // {4,1,0,},
    // {4,0,1,},
    // {1,4,0,},
    // {0,4,1,},
    // {0,1,4,},
    // {1,0,4,},
    // {1,1,3,},
    // {3,1,1,},
    // {1,3,1,},
    // {2,2,1,},
    // {2,1,2,},
    // {1,2,2,},
    // {6,0,0,},
    // {0,6,0,},
    // {0,0,6,},
    // {3,3,0,},
    // {3,0,3,},
    // {0,3,3,},
    // {5,1,0,},
    // {5,0,1,},
    // {1,0,5,},
    // {0,1,5,},
    // {0,5,1,},
    // {1,5,0,},
    // {4,2,0,},
    // {4,0,2,},
    // {2,0,4,},
    // {0,2,4,},
    // {2,4,0,},
    // {0,4,2,},
    // {3,2,1,},
    // {3,1,2,},
    // {1,3,2,},
    // {2,1,3,},
    // {2,3,1,},
    // {1,2,3,},
    // {4,1,1,},
    // {1,4,1,},
    // {1,1,4,},
    // {2,2,2,},
  };
  constexpr size_t N = mlao[maxl];

  const int iang = cgtoi.ang;
  const int jang = cgtoj.ang;
  /* Initialize spherical integral matrices in shared memory*/
  __shared__ double s3d_[N * N]; //= {0.0};
  __shared__ double d3d_[N * N * 3]; //= {0.0};
  __shared__ double q3d_[N * N * 6]; //= {0.0};
  #pragma unroll
  for (int i = threadIdx.x; i < N * N; i += blockDim.x)
    s3d_[i] = 0.0;
  #pragma unroll
  for (int i = threadIdx.x; i < N * N * 3; i += blockDim.x)
    d3d_[i] = 0.0;
  #pragma unroll
  for (int i = threadIdx.x; i < N * N * 6; i += blockDim.x)
    q3d_[i] = 0.0;
  __syncthreads();
  device_tensor2d_t<double> s3d(mlao[iang], mlao[jang],    &s3d_[0]); 
  device_tensor3d_t<double> d3d(mlao[iang], mlao[jang], 3, &d3d_[0]); 
  device_tensor3d_t<double> q3d(mlao[iang], mlao[jang], 6, &q3d_[0]); 

  {
    const int total = cgtoi.nprim * cgtoj.nprim;
    for(int i = threadIdx.x; i < total; i+=blockDim.x) // 1400 down from 2300
    // for (int ip = 0; ip < cgtoi.nprim; ++ip)
    {
      // for (int jp = 0; jp < cgtoj.nprim; ++jp)
      {
        const int ip = i / cgtoj.nprim;
        const int jp = i % cgtoj.nprim;

        double s1d[MAXL2] = {0.0};
        double rpi[3] = {0.0};
        double rpj[3] = {0.0}; 
        double dip[3] = {0.0}; 
        double quad[6] = {0.0};
        const auto eab = cgtoi.alpha[ip] + cgtoj.alpha[jp];
        const auto oab = 1.0 / eab;
        const auto est = cgtoi.alpha[ip] * cgtoj.alpha[jp] * r2 * oab;

        if (est > intcut) continue;

        const auto pre = exp(-est) * sqrtpi3 * pow(oab, 1.5); /*pow(sqrt(oab), 3)*/;
        const double cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre;

        #pragma unroll
        for (int k = 0; k < 3; ++k)
        {
          rpi[k] = +vec[k] * cgtoi.alpha[ip] * oab;
          rpj[k] = -vec[k] * cgtoj.alpha[jp] * oab;
        }

        for (int l = 0; l <= iang + jang + 2; ++l)
        {
          s1d[l] = overlap_1d(l, eab);
        }

        for (int mli = 0; mli < mlao[iang]; ++mli)
        {
          for (int mlj = 0; mlj < mlao[jang]; ++mlj)
          {
            double val = 0.0;
            multipole_3d(
              rpi, rpj,
              cgtoi.alpha[ip], cgtoj.alpha[jp], 
              lx[mli + lmap[iang]], lx[mlj + lmap[jang]],
              s1d, val, dip, quad);
            
            // s3d(mli, mlj) += cc * val;
            atomicAdd(&s3d(mli, mlj), cc * val);
            #pragma unroll
            for (int k = 0; k < 3; ++k)
            {
              atomicAdd(&d3d(mli, mlj, k), cc * dip[k]);
            }

            // d3d(mli,mlj,k) += cc * dip[k];
            #pragma unroll
            for (int k = 0; k < 6; ++k)
              // q3d(mli,mlj,k) += cc * quad[k];
              atomicAdd(&q3d(mli, mlj, k), cc * quad[k]);
          }
        }
      }
    }
    __syncthreads();
  }

  transform0<double>(iang, jang, s3d, overlap);
  transform1<double>(iang, jang, d3d, dpint);
  transform1<double>(iang, jang, q3d, qpint);
  __syncthreads();

  {
    const int total = msao[iang] * msao[jang];
    // for (int mli = 0; mli < msao[cgtoi.ang]; ++mli)
    #pragma unroll
    for(int i = threadIdx.x; i < total; i+=blockDim.x)
    {
      // for (int mlj = 0; mlj < msao[cgtoj.ang]; ++mlj)
      {
        const int mli = i / msao[jang];
        const int mlj = i % msao[jang];

        double tr = 0.5 * (qpint(mli, mlj, 0) + qpint(mli, mlj, 2) + qpint(mli, mlj, 5));
        qpint(mli, mlj, 0) = 1.5 * qpint(mli, mlj, 0) - tr;
        qpint(mli, mlj, 1) = 1.5 * qpint(mli, mlj, 1);
        qpint(mli, mlj, 2) = 1.5 * qpint(mli, mlj, 2) - tr;
        qpint(mli, mlj, 3) = 1.5 * qpint(mli, mlj, 3);
        qpint(mli, mlj, 4) = 1.5 * qpint(mli, mlj, 4);
        qpint(mli, mlj, 5) = 1.5 * qpint(mli, mlj, 5) - tr;
      }
    }
  }
  __syncthreads();
}

template <typename T, int D>
__device__ inline void shift_operator(
    const int iao, 
    const int jao,
    const T * __restrict__ vec,
    const device_tensor2d_fixed_t<T, D, D> &s,
    const device_tensor3d_fixed_t<T, D, D, 3> &di,
    const device_tensor3d_fixed_t<T, D, D, 6> &qi,
    T (&dj)[3],
    T (&qj)[6])
{
  dj[0] = di(iao, jao, 0) + vec[0] * s(iao, jao);
  dj[1] = di(iao, jao, 1) + vec[1] * s(iao, jao);
  dj[2] = di(iao, jao, 2) + vec[2] * s(iao, jao);
  
  qj[0] = 2 * vec[0] * di(iao, jao, 0) + vec[0] * vec[0] * s(iao, jao);
  qj[2] = 2 * vec[1] * di(iao, jao, 1) + vec[1] * vec[1] * s(iao, jao);
  qj[5] = 2 * vec[2] * di(iao, jao, 2) + vec[2] * vec[2] * s(iao, jao);
  qj[1] = vec[0] * di(iao, jao, 1) + vec[1] * di(iao, jao, 0) + vec[0] * vec[1] * s(iao, jao);
  qj[3] = vec[0] * di(iao, jao, 2) + vec[2] * di(iao, jao, 0) + vec[0] * vec[2] * s(iao, jao);
  qj[4] = vec[1] * di(iao, jao, 2) + vec[2] * di(iao, jao, 1) + vec[1] * vec[2] * s(iao, jao);

  const auto tr = 0.5 * (qj[0] + qj[2] + qj[5]);
  qj[0] = qi(iao, jao, 0) + 1.5 * qj[0] - tr;
  qj[1] = qi(iao, jao, 1) + 1.5 * qj[1];
  qj[2] = qi(iao, jao, 2) + 1.5 * qj[2] - tr;
  qj[3] = qi(iao, jao, 3) + 1.5 * qj[3];
  qj[4] = qi(iao, jao, 4) + 1.5 * qj[4];
  qj[5] = qi(iao, jao, 5) + 1.5 * qj[5] - tr;
}

template <size_t maxl>
__global__ void 
__launch_bounds__(MAX_THREADS_PER_BLOCK)
get_hamiltonian_between_atoms_kernel(
    const __grid_constant__ int batch_size,
    const __grid_constant__ __restrict__ structure_type mol,
    const __grid_constant__ tensor2d_t<const double> trans,
    const __grid_constant__ __restrict__ adjacency_list alist,
    const __grid_constant__ __restrict__ basis_type bas,
    const __grid_constant__ __restrict__ tb_hamiltonian h0,
    const __grid_constant__ tensor1d_t<const double> selfenergy,
    tensor2d_t<double> overlap,
    tensor3d_t<double> dpint,
    tensor3d_t<double> qpint,
    tensor2d_t<double> hamiltonian)
{
  constexpr int msao[] = {1, 3, 5, /*7, 9, 11, 13*/};
  constexpr int N = msao[maxl];

  for(int batch = blockIdx.x; batch < mol.nat * batch_size; batch += gridDim.x)
  {
    const int iat = batch % mol.nat;

    const int total_neig = alist.nnl[iat];
    for (int img = blockIdx.y; img < total_neig; img+=gridDim.y)
    {
      const int inl = alist.inl[iat];
      const int jat = alist.nlat[img + inl];
      const int izp = mol.id[iat];
      const int jzp = mol.id[jat];
      const double total_radii = h0.rad[jzp] + h0.rad[izp];
      const double intcut = bas.intcut;
      const auto total_iters = bas.nsh_id[izp] * bas.nsh_id[jzp];
      for (int total = blockIdx.z; total < total_iters; total += gridDim.z)
      // for (int ish = blockIdx.z; ish < bas.nsh_id[izp]; ish += gridDim.z)
      {
        // for (int jsh = 0; jsh < bas.nsh_id[jzp]; ++jsh)
        {
          /////////////////////////////////////////
          // THIS SECTION IS SINGLE THREAD BLOCK
          /////////////////////////////////////////
          const int ish = total / bas.nsh_id[jzp];
          const int jsh = total % bas.nsh_id[jzp];
          const double ishpoly = h0.shpoly(izp, ish);
          const double jshpoly = h0.shpoly(jzp, jsh);
          const int is = bas.ish_at[iat];
          const int itr = alist.nltr[img + inl];
          const int js = bas.ish_at[jat];
          
          const double scaled_selfenergy = (selfenergy[is + ish] + selfenergy[js + jsh]) *
                h0.hscale(izp, jzp, ish, jsh);

          const int ii = bas.iao_sh[is + ish];
          const int jj = bas.iao_sh[js + jsh];

          __shared__ cgto_type cgtoi, cgtoj; 
          if(threadIdx.x == 0)
          {
            cgtoi.ang = bas.cgto(izp, ish).ang;
            cgtoj.ang = bas.cgto(jzp, jsh).ang;
            cgtoi.nprim = bas.cgto(izp, ish).nprim;
            cgtoj.nprim = bas.cgto(jzp, jsh).nprim;
          }
          #pragma unroll
          for(int i = threadIdx.x; i < MAXG; i+=blockDim.x)
          {
            cgtoi.alpha[i] = bas.cgto(izp, ish).alpha[i];
            cgtoi.coeff[i] = bas.cgto(izp, ish).coeff[i];
            cgtoj.alpha[i] = bas.cgto(jzp, jsh).alpha[i];
            cgtoj.coeff[i] = bas.cgto(jzp, jsh).coeff[i];
          }

          __shared__ double vec[3];   // = {0.0};
          __shared__ double r2, rr;
          
          for (int k = threadIdx.x; k < 3; k+=blockDim.x)
            vec[k] = mol.xyz(iat, k) - mol.xyz(jat, k) - trans(itr, k);
          __syncthreads();
          
          if (threadIdx.x == 0)
          {
            r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
            rr = sqrt(sqrt(r2) / total_radii); //(h0_jrad + h0_irad));
          }
          __syncthreads();

          __shared__ double hij;
          if(threadIdx.x == 0)
          {
            const double shpoly = (1.0 + ishpoly * rr) *
              (1.0 + jshpoly * rr);
            hij = 0.5 * scaled_selfenergy * (1.0 + ishpoly * rr) *
              (1.0 + jshpoly * rr);
          }
          __syncthreads();

          /* Make stmp, dtmpi and qtmpi integral, shared arrays */
          __shared__ double stmp_ [N * N];
          __shared__ double dtmpi_[N * N * 3];
          __shared__ double qtmpi_[N * N * 6];
          for(int k = threadIdx.x; k < N * N; k+=blockDim.x)
            stmp_[k] = 0.0;
          for(int k = threadIdx.x; k < N * N * 3; k+=blockDim.x)
            dtmpi_[k] = 0.0;
          for(int k = threadIdx.x; k < N * N * 6; k+=blockDim.x)  
            qtmpi_[k] = 0.0;
          __syncthreads();
          device_tensor2d_fixed_t<double, N, N   > stmp (&stmp_[0]); 
          device_tensor3d_fixed_t<double, N, N, 3> dtmpi(&dtmpi_[0]); 
          device_tensor3d_fixed_t<double, N, N, 6> qtmpi(&qtmpi_[0]); 
          double dtmpj[3];
          double qtmpj[6];

          /* Read element-specific parameters from global mem */
          const int njao = msao[cgtoj.ang];
          const int niao = msao[cgtoi.ang];

          multipole_cgto<maxl, N>(
            cgtoj, cgtoi, 
            r2, vec, intcut, stmp, dtmpi, qtmpi);
          __syncthreads();
          
          const int total = niao * njao;
          // for(int iao = 0; iao < niao; ++iao)
          for(int i = threadIdx.x; i < total; i+=blockDim.x) // 1800 ms without, 2300 ms with
          {
            // for(int jao = 0; jao < njao; ++jao)
            {
              const int iao = i / njao;
              const int jao = i % njao;
              shift_operator(iao, jao, vec, stmp, dtmpi, qtmpi, dtmpj, qtmpj); 

              // atomicAdd(&overlap(ii + iao, jj + jao), stmp(iao, jao));
              overlap(ii + iao, jj + jao) += stmp(iao, jao);
              
              #pragma unroll
              for (int k = 0; k < 3; ++k)
              {
                // atomicAdd(&dpint(ii + iao, jj + jao, k), dtmpi(iao, jao, k));
                dpint(ii + iao, jj + jao, k) += dtmpi(iao, jao, k);
              }

              #pragma unroll
              for (int k = 0; k < 6; ++k)
              {
                // atomicAdd(&qpint(ii + iao, jj + jao, k), qtmpi(iao, jao, k));
                qpint(ii + iao, jj + jao, k) += qtmpi(iao, jao, k);
              }

              // atomicAdd(&hamiltonian(ii + iao, jj + jao), stmp(iao, jao) * hij);
              hamiltonian(ii + iao, jj + jao) += stmp(iao, jao) * hij;

              /* TODO: This is a symmetrification of these matrices. Maybe this should be
              done in the outside this loop? */
              if (iat != jat) // 2200ms vs 2300ms
              {
                // atomicAdd(&overlap(jj + jao, ii + iao), stmp(iao, jao));
                overlap(jj + jao, ii + iao) += stmp(iao, jao);

                #pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                  // atomicAdd(&dpint(jj + jao, ii + iao,  k), dtmpj[k]);
                  dpint(jj + jao, ii + iao, k) += dtmpj[k];
                }

                #pragma unroll
                for (int k = 0; k < 6; ++k)
                {
                  // atomicAdd(&qpint(jj + jao, ii + iao,  k), qtmpj[k]);
                  qpint(jj + jao, ii + iao, k) += qtmpj[k];
                }
                
                // atomicAdd(&hamiltonian(jj + jao, ii + iao), stmp(iao, jao) * hij);
                hamiltonian(jj + jao, ii + iao) += stmp(iao, jao) * hij;
              }
            }
          }
        }
      }
    }
  }
}

void get_hamiltonian_between_atoms(
  // Batch size is not fully implemented yet. For now, this is only for feasibility testing.
  const int batch_size,
  const structure_type mol,
  const tensor2d_t<const double> trans,
  const adjacency_list alist,
  const basis_type bas,
  const tb_hamiltonian h0,
  const tensor1d_t<const double> selfenergy,
  tensor2d_t<double> overlap,
  tensor3d_t<double> dpint,
  tensor3d_t<double> qpint,
  tensor2d_t<double> hamiltonian)
{
  dim3 dimGrid(batch_size * mol.nat, alist.nnl.max(), bas.nsh_id.max() * bas.nsh_id.max());
  dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);
  switch(bas.maxl)
  {
    case 0: 
      get_hamiltonian_between_atoms_kernel<0><<<dimGrid, dimBlock>>>(batch_size, mol, trans, alist, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian);
      break;
    case 1: 
      get_hamiltonian_between_atoms_kernel<1><<<dimGrid, dimBlock>>>(batch_size, mol, trans, alist, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian);
      break;
    case 2:
      get_hamiltonian_between_atoms_kernel<2><<<dimGrid, dimBlock>>>(batch_size, mol, trans, alist, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian);
      break;
    default:
      printf("[Fatal] get_hamiltonian_between_atoms_kernel not supported for maxl=%d\n", bas.maxl);
      break;
  }
}

template <size_t maxl>
__global__ void 
__launch_bounds__(MAX_THREADS_PER_BLOCK)
get_hamiltonian_in_atoms_kernel(
  const __grid_constant__ __restrict__ structure_type mol,
  const __grid_constant__ tensor2d_t<const double> trans,
  const __grid_constant__ __restrict__ adjacency_list alist,
  const __grid_constant__ __restrict__ basis_type bas,
  const __grid_constant__ __restrict__ tb_hamiltonian h0,
  const __grid_constant__ tensor1d_t<const double> selfenergy,
  tensor2d_t<double> overlap,
  tensor3d_t<double> dpint,
  tensor3d_t<double> qpint,
  tensor2d_t<double> hamiltonian)
{
  constexpr int msao[] = {1, 3, 5, 7, 9, 11, 13};
  constexpr int N = msao[maxl];

  /* Parallelize over the grid x, y, z*/
  for(int iat = blockIdx.x; iat < mol.nat; iat += gridDim.x)
  {
    int izp = mol.id[iat];
    int is = bas.ish_at[iat];
    for(int ish = blockIdx.y; ish < bas.nsh_id[izp]; ish += gridDim.y)
    {
      int ii = bas.iao_sh[is + ish];
      for(int jsh = blockIdx.z; jsh < bas.nsh_id[izp]; jsh += gridDim.z)
      {
        /////////////////////////////////////////
        // THIS SECTION IS SINGLE THREAD BLOCK
        /////////////////////////////////////////
        int jj = bas.iao_sh[is + jsh];

        __shared__ double vec[3];// = {0.0};
        for (int k = threadIdx.x; k < 3; k+=blockDim.x)
          vec[k] = 0.0;
        const double r2 = 0.0;
        const double rr = 0.0; //sqrt(sqrt(r2) / (h0.rad[izp] + h0.rad[izp]));
        
        // load cgtoj and cgtoi to shared memory
        __shared__ cgto_type cgtoi, cgtoj; 
        if(threadIdx.x == 0)
        {
          cgtoi.ang = bas.cgto(izp, ish).ang;
          cgtoi.nprim = bas.cgto(izp, ish).nprim;
          
          cgtoj.ang = bas.cgto(izp, jsh).ang;
          cgtoj.nprim = bas.cgto(izp, jsh).nprim;
        }

        #pragma unroll
        for(int i = threadIdx.x; i < MAXG; i+=blockDim.x)
        {
          cgtoi.alpha[i] = bas.cgto(izp, ish).alpha[i];
          cgtoi.coeff[i] = bas.cgto(izp, ish).coeff[i];

          cgtoj.alpha[i] = bas.cgto(izp, jsh).alpha[i];
          cgtoj.coeff[i] = bas.cgto(izp, jsh).coeff[i];
        }

        /* Make stmp, dtmpi and qtmpi integral, shared arrays */
        __shared__ double stmp_ [N * N];
        __shared__ double dtmpi_[N * N * 3];
        __shared__ double qtmpi_[N * N * 6];
        for(int k = threadIdx.x; k < N * N; k+=blockDim.x)
          stmp_[k] = 0.0;
        for(int k = threadIdx.x; k < N * N * 3; k+=blockDim.x)
          dtmpi_[k] = 0.0;
        for(int k = threadIdx.x; k < N * N * 6; k+=blockDim.x)  
          qtmpi_[k] = 0.0;
        device_tensor2d_fixed_t<double, N,  N   > stmp (&stmp_[0]); 
        device_tensor3d_fixed_t<double, N,  N, 3> dtmpi(&dtmpi_[0]); 
        device_tensor3d_fixed_t<double, N,  N, 6> qtmpi(&qtmpi_[0]); 
        __syncthreads();
        
        multipole_cgto<maxl, N>(//bas.cgto(izp, jsh), bas.cgto(izp, ish), 
          cgtoj, cgtoi,
          r2, vec, bas.intcut, stmp, dtmpi, qtmpi);
        __syncthreads();

        const double shpoly = 1.0 / 1.0; /*(1.0 + h0.shpoly(izp, ish) * rr) *
          (1.0 + h0.shpoly(izp, jsh) * rr);*/
        const double hij = 0.5 * (selfenergy[is + ish] + selfenergy[is + jsh]) *
          shpoly;
        const int jnao = msao[cgtoj.ang];
        const int inao = msao[cgtoi.ang];
        const int total = inao * jnao;
        for(int i = threadIdx.x; i < total; i+=blockDim.x)
        // for(int iao = 0; iao < inao; ++iao)
        {
          // for(int jao = 0; jao < jnao; ++jao)
          {
            const int iao = i / jnao;
            const int jao = i % jnao;
            // atomicAdd(&overlap(ii + iao, jj + jao), stmp(iao, jao));
            overlap(ii + iao, jj + jao) += stmp(iao, jao);
            #pragma unroll
            for(int k = 0; k < 3; ++k)
            {
              dpint(ii + iao, jj + jao, k) += dtmpi(iao, jao, k);
              // atomicAdd(&dpint(ii + iao, jj + jao, k), dtmpi(iao, jao, k));
            }
            #pragma unroll
            for(int k = 0; k < 6; ++k)
            {
              qpint(ii + iao, jj + jao, k) += qtmpi(iao, jao, k);
              // atomicAdd(&qpint(ii + iao, jj + jao, k), qtmpi(iao, jao, k));
            }
            // atomicAdd(&hamiltonian(ii + iao, jj + jao), stmp(iao, jao) * hij);
            hamiltonian(ii + iao, jj + jao) += stmp(iao, jao) * hij;
          }
        }
      }
    }
  }
}

void get_hamiltonian_in_atoms( 
  const structure_type mol,
  const tensor2d_t<const double> trans, 
  const adjacency_list alist,
  const basis_type bas,
  const tb_hamiltonian h0,
  const tensor1d_t<const double> selfenergy,
  tensor2d_t<double> overlap,
  tensor3d_t<double> dpint,
  tensor3d_t<double> qpint,
  tensor2d_t<double> hamiltonian)
{
  dim3 dimGrid(mol.nat, bas.nsh_id.max(), bas.nsh_id.max());
  dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);
  switch(bas.maxl)
  {
    case 0: 
      get_hamiltonian_in_atoms_kernel<0><<<dimGrid, dimBlock>>>(mol, trans, alist, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian);
      break;
    case 1:
      get_hamiltonian_in_atoms_kernel<1><<<dimGrid, dimBlock>>>(mol, trans, alist, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian);
      break;
    case 2:
      get_hamiltonian_in_atoms_kernel<2><<<dimGrid, dimBlock>>>(mol, trans, alist, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian);
      break;
    default:
      printf("[Fatal] get_hamiltonian_in_atoms_kernel not supported for maxl=%d\n", bas.maxl);
      break;
  }
}

extern "C" void cuda_get_hamiltonian_kernel_(
    int batch_size,
    int nao,
    int nelem,

    /* structure_type */
    const int mol_nat,
    const int mol_nid,
    const int mol_nbd,
    const int *mol_id, int mol_id_dim1,
    const int *mol_num, int mol_num_dim1,
    const double *mol_xyz, int mol_xyz_dim1, int mol_xyz_dim2, 
    const int mol_uhf,
    const double mol_charge,
    // const double *mol_lattice, int mol_lattice_dim1, int mol_lattice_dim2,
    // const int *mol_periodic, int mol_periodic_dim1,
    // const int *mol_bond, int mol_bond_dim1, int mol_bond_dim2,

    /* trans for lattice */
    const double *trans, const int trans_dim1, const int trans_dim2,

    /* adjacency_list */
    const int *alist_inl, int alist_inl_dim1,
    const int *alist_nnl, int alist_nnl_dim1,
    const int *alist_nlat, int alist_nlat_dim1,
    const int *alist_nltr, int alist_nltr_dim1,

    /* basis_type */
    const int bas_maxl,
    const int bas_nsh,
    const int bas_nao,
    const double bas_intcut,
    const double bas_min_alpha,
    const int *bas_nsh_id, int bas_nsh_id_dim1,
    const int *bas_nsh_at, int bas_nsh_at_dim1,
    const int *bas_nao_sh, int bas_nao_sh_dim1,
    const int *bas_iao_sh, int bas_iao_sh_dim1,
    const int *bas_ish_at, int bas_ish_at_dim1,
    const int *bas_ao2at, int bas_ao2at_dim1,
    const int *bas_ao2sh, int bas_ao2sh_dim1,
    const int *bas_sh2at, int bas_sh2at_dim1,
    const cgto_type *cgto, int cgto_dim1, int cgto_dim2,

    /* tb_hamiltonian */
    const double *h0_selfenergy, int h0_selfenergy_dim1, int h0_selfenergy_dim2,
    const double *h0_kcn, int h0_kcn_dim1, int h0_kcn_dim2,
    const double *h0_kq1, int h0_kq1_dim1, int h0_kq1_dim2,
    const double *h0_kq2, int h0_kq2_dim1, int h0_kq2_dim2,
    const double *h0_hscale, int h0_hscale_dim1, int h0_hscale_dim2, int h0_hscale_dim3, int h0_hscale_dim4,
    const double *h0_shpoly, int h0_shpoly_dim1, int h0_shpoly_dim2,
    const double *h0_rad, int h0_rad_dim1,
    const double *h0_refocc, int h0_refocc_dim1, int h0_refocc_dim2,

    // Diagonal elememts of the Hamiltonian  (nelem)
    const double *selfenergy,
    // Overlap integral matrix (nao, nao)
    double *overlap,
    // Dipole moment integral matrix (nao, nao, 3)
    double *dpint,
    // Quadrupole moment integral matrix (nao, nao, 6)
    double *qpint,
    // Hamiltonian matrix (nao, nao)
    double *hamiltonian
    // double * time
)
{
  const adjacency_list alist{
      tensor1d_t(alist_inl, alist_inl_dim1),
      tensor1d_t(alist_nnl, alist_nnl_dim1),
      tensor1d_t(alist_nlat, alist_nlat_dim1),
      tensor1d_t(alist_nltr, alist_nltr_dim1)};

  const structure_type mol{
    mol_nat, mol_nid, mol_nbd,
    tensor1d_t(mol_id, mol_id_dim1),
    tensor1d_t(mol_num, mol_num_dim1),
    tensor2d_t(mol_xyz, mol_xyz_dim1, mol_xyz_dim2),
    mol_uhf,
    mol_charge,
  };

  const tb_hamiltonian h0{
      tensor2d_t(h0_selfenergy, h0_selfenergy_dim1, h0_selfenergy_dim2),
      tensor2d_t(h0_kcn, h0_kcn_dim1, h0_kcn_dim2),
      tensor2d_t(h0_kq1, h0_kq1_dim1, h0_kq1_dim2),
      tensor2d_t(h0_kq2, h0_kq2_dim1, h0_kq2_dim2),
      tensor4d_t(h0_hscale, h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4),
      tensor2d_t(h0_shpoly, h0_shpoly_dim1, h0_shpoly_dim2),
      tensor1d_t(h0_rad, h0_rad_dim1),
      tensor2d_t(h0_refocc, h0_refocc_dim1, h0_refocc_dim2)};

  const basis_type bas{
      bas_maxl,
      bas_nsh,
      bas_nao,
      bas_intcut,
      bas_min_alpha,
      tensor1d_t(bas_nsh_id, bas_nsh_id_dim1),
      tensor1d_t(bas_nsh_at, bas_nsh_at_dim1),
      tensor1d_t(bas_nao_sh, bas_nao_sh_dim1),
      tensor1d_t(bas_iao_sh, bas_iao_sh_dim1),
      tensor1d_t(bas_ish_at, bas_ish_at_dim1),
      tensor1d_t(bas_ao2at, bas_ao2at_dim1),
      tensor1d_t(bas_ao2sh, bas_ao2sh_dim1),
      tensor1d_t(bas_sh2at, bas_sh2at_dim1),
      tensor2d_t(cgto, cgto_dim1, cgto_dim2)};
  

  const tensor2d_t<const double> trans_ten(trans, trans_dim1, trans_dim2);
  const tensor1d_t<const double> selfenergy_ten(selfenergy, nelem);
  tensor2d_t<double> overlap_ten(overlap, nao, nao);
  tensor3d_t<double> dpint_ten(dpint, nao, nao, 3);
  tensor3d_t<double> qpint_ten(qpint, nao, nao, 6);
  tensor2d_t<double> hamiltonian_ten(hamiltonian, nao, nao);
  
  /////////////////////////////////////////////
  // Calculate total number of bytes transferred
  /////////////////////////////////////////////
  size_t total_bytes_in = 0;
  size_t total_bytes_out = 0;
  // mol
  total_bytes_in += sizeof(structure_type)
    + sizeof(double) * mol.xyz.size() 
    + sizeof(int) * mol.num.size()
    + sizeof(int) * mol.id.size();
  // alist
  total_bytes_in += sizeof(adjacency_list)
    + sizeof(int) * alist.inl.size()
    + sizeof(int) * alist.nnl.size()
    + sizeof(int) * alist.nlat.size()
    + sizeof(int) * alist.nltr.size();
  // bas
  total_bytes_in += sizeof(basis_type)
    + sizeof(cgto_type) * bas.cgto.size()
    + sizeof(int) * bas.nsh_id.size()
    + sizeof(int) * bas.nsh_at.size()
    + sizeof(int) * bas.nao_sh.size()
    + sizeof(int) * bas.iao_sh.size()
    + sizeof(int) * bas.ish_at.size()
    + sizeof(int) * bas.ao2at.size()
    + sizeof(int) * bas.ao2sh.size()
    + sizeof(int) * bas.sh2at.size();
  // h0
  total_bytes_in += sizeof(tb_hamiltonian)
    + sizeof(double) * h0.selfenergy.size()
    + sizeof(double) * h0.kcn.size()
    + sizeof(double) * h0.kq1.size()
    + sizeof(double) * h0.kq2.size()
    + sizeof(double) * h0.hscale.size()
    + sizeof(double) * h0.shpoly.size()
    + sizeof(double) * h0.rad.size()
    + sizeof(double) * h0.refocc.size();
  // selfenergy
  total_bytes_in += sizeof(double) * selfenergy_ten.size();
  // overlap
  total_bytes_in += sizeof(double) * overlap_ten.size();
  // dpint
  total_bytes_in += sizeof(double) * dpint_ten.size();
  // qpint
  total_bytes_in += sizeof(double) * qpint_ten.size();
  // hamiltonian
  total_bytes_in += sizeof(double) * hamiltonian_ten.size();

  total_bytes_out = sizeof(double) * overlap_ten.size()
    + sizeof(double) * dpint_ten.size()
    + sizeof(double) * qpint_ten.size()
    + sizeof(double) * hamiltonian_ten.size();
  
  printf("gpu_gb_in %f\n", (double)total_bytes_in / 1e9);
  printf("gpu_gb_out %f\n", (double)total_bytes_out / 1e9);
  printf("gpu_gb_total %f\n", (double)(total_bytes_in + total_bytes_out) / 1e9);
  
  ////////////////////////////////////////////
  // Launch kernel part I, between-atom interactions
  ////////////////////////////////////////////
  cudaEvent_t start, stop;
  float total_time = 0;
  float milliseconds = 0;
  {
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    get_hamiltonian_between_atoms(
      batch_size,
      mol,
      trans_ten,
      alist,
      bas,
      h0,
      selfenergy_ten,
      overlap_ten,
      dpint_ten,
      qpint_ten,
      hamiltonian_ten);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("gpu_between_atoms %f\n", milliseconds / batch_size);
    total_time += milliseconds;
    cudaEventDestroy(start); cudaEventDestroy(stop);
  }
  // cudaFuncAttributes attr;
  // Get attributes of the kernel
  // CUDA_CHECK(cudaFuncGetAttributes(&attr, get_hamiltonian_between_atoms_kernel<2>));
  // Print the total shared memory used by the kernel
  // printf("gpu_shmem %d\n", attr.sharedSizeBytes);

  ////////////////////////////////////////////
  // Launch kernel part II, in-atom interactions
  ////////////////////////////////////////////
  {
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    get_hamiltonian_in_atoms(
      mol,
      trans_ten,
      alist,
      bas,
      h0,
      selfenergy_ten,
      overlap_ten,
      dpint_ten,
      qpint_ten,
      hamiltonian_ten);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("gpu_in_atoms %f\n", milliseconds);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    total_time += milliseconds;
  }
  printf("gpu_time %f\n", total_time);
  ////////////////////////////
  // copy data back to host
  ////////////////////////////
  memcpy(overlap, overlap_ten.data, overlap_ten.size() * sizeof(double));
  memcpy(dpint, dpint_ten.data, dpint_ten.size() * sizeof(double));
  memcpy(qpint, qpint_ten.data, qpint_ten.size() * sizeof(double));
  memcpy(hamiltonian, hamiltonian_ten.data, hamiltonian_ten.size() * sizeof(double));

  ///////////////////////////
  // free cuda memory
  //////////////////////////
  cudaFree((void *)mol.num.data);
  cudaFree((void *)mol.xyz.data);

  cudaFree((void *)trans_ten.data);

  cudaFree((void *)alist.inl.data);
  cudaFree((void *)alist.nnl.data);
  cudaFree((void *)alist.nlat.data);
  cudaFree((void *)alist.nltr.data);

  cudaFree((void *)bas.nsh_id.data);
  cudaFree((void *)bas.nsh_at.data);
  cudaFree((void *)bas.nao_sh.data);
  cudaFree((void *)bas.iao_sh.data);
  cudaFree((void *)bas.ish_at.data);
  cudaFree((void *)bas.ao2at.data);
  cudaFree((void *)bas.ao2sh.data);
  cudaFree((void *)bas.sh2at.data);
  cudaFree((void *)bas.cgto.data);

  cudaFree((void *)h0.selfenergy.data);
  cudaFree((void *)h0.kcn.data);
  cudaFree((void *)h0.kq1.data);
  cudaFree((void *)h0.kq2.data);
  cudaFree((void *)h0.hscale.data);
  cudaFree((void *)h0.shpoly.data);
  cudaFree((void *)h0.rad.data);
  cudaFree((void *)h0.refocc.data);

  cudaFree((void *)selfenergy_ten.data);

  cudaFree(overlap_ten.data);
  cudaFree(dpint_ten.data);
  cudaFree(qpint_ten.data);
  cudaFree(hamiltonian_ten.data);
}
