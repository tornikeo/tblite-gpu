#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "device_tensor.h"
#include "types.h"


template <typename T>
__device__ inline void transform0(
  const int lj, const int li, 
  const device_tensor2d_t<T> &cart, 
  device_tensor2d_t<T> &sphr)
{
  constexpr T s3 = 1.7320508075688772; // sqrt(3)
  constexpr T s3_4 = 0.6123724356957945; // sqrt(3)/2
  printf("cart(%d, %d) sphr(%d, %d) %s %s:%d\n", cart.dim1, cart.dim2, sphr.dim1, sphr.dim2, __func__, __FILE__, __LINE__);
  /* sphr is a larger array. It contains the max size that an integral might need
  so iterate over smaller cart dims instead*/
  assert(sphr.dim1 >= cart.dim1); 
  assert(sphr.dim2 >= cart.dim2);
  switch (li)
  {
  case 0:
  case 1:
    switch (lj)
    {
    case 0:
    case 1:
      // Copy cart to sphr
      for (int i = 0; i < cart.dim1; ++i)
      {
        for (int j = 0; j < cart.dim2; ++j)
        {
          sphr(i, j) = cart(i, j);
        }
      }
      printf("sphr = \n");
      for (int i = 0; i < cart.dim1; ++i)
      {
        for (int j = 0; j < cart.dim2; ++j)
        {
          printf("%f, ", sphr(i, j));
        }
        printf("\n");
      }
      break;
    case 2:
      // sphr = matmul(dtrafo, cart)
      for(int i = 0; i < cart.dim1; ++i)
      {
        sphr(i, 0) = cart(i, 2) - 0.5 * (cart(i, 0) + cart(i, 1));
        sphr(i, 1) = s3 * cart(i, 4);
        sphr(i, 2) = s3 * cart(i, 5);
        sphr(i, 3) = s3_4 * (cart(i, 0) - cart(i, 1));
        sphr(i, 4) = s3 * cart(i, 3);
      }
      break;
    
    default:
      printf("[Fatal] moment li=%i lj=%i not supported\n", li, lj);
      assert(false);
      return;
    }
    break;
  default:
    printf("[Fatal] transform0 not supported for li=%d lj=%d\n", li, lj);
    assert(false);
    return;
  }
  /* TODO: Support the rest of the transform cases*/
}

template <typename T>
__device__ inline void transform0(int lj, int li, int k, const device_tensor3d_t<T> &cart, device_tensor3d_t<T> &sphr)
{
  constexpr T s3 = 1.7320508075688772; // sqrt(3)
  constexpr T s3_4 = 0.6123724356957945; // sqrt(3)/2
  switch (li)
  {
  case 0:
  case 1:
    switch (lj)
    {
    case 0:
    case 1:
      // Copy cart to sphr
      for (int i = 0; i < cart.dim2; ++i)
      {
        for (int j = 0; j < cart.dim3; ++j)
        {
          sphr(k, i, j) = cart(k, i, j);
        }
      }
      break;
    case 2:
      // sphr = matmul(dtrafo, cart)
      assert(sphr.dim1 == cart.dim1);
      for(int i = 0; i < cart.dim1; ++i)
      {
        sphr(i, 0, k) = cart(i, 2, k) - 0.5 * (cart(i, 0, k) + cart(i, 1, k));
        sphr(i, 1, k) = s3 * cart(i, 4, k);
        sphr(i, 2, k) = s3 * cart(i, 5, k);
        sphr(i, 3, k) = s3_4 * (cart(i, 0, k) - cart(i, 1, k));
        sphr(i, 4, k) = s3 * cart(i, 3, k);
      }
      break;
    
    default:
      printf("[Fatal] moment li=%i lj=%i not supported\n", li, lj);
      assert(false);
      return;
    }
    break;
  default:
    printf("[Fatal] transform0 not supported for li=%d lj=%d\n", li, lj);
    assert(false);
    return;
  }
  /* TODO: Support the rest of the transform cases*/
}


__device__ inline void horizontal_shift(const double ae, const int l, double (&cfs)[MAXL])
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
    printf("[Fatal] horizontal_shift not supported for l=%d\n", l);
    assert(false);
    return;
  }
}

__device__ inline void form_product(
  const double (&a)[MAXL],
  const double (&b)[MAXL],
  const int &la, const int &lb,
  double (&d)[MAXL2])
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

__device__ inline double overlap_1d(int moment, double alpha)
{
  double overlap = 0.0;
  constexpr double dfactorial[8] = {1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0};
  assert(moment >= 0 && moment <= 7);
  if (moment % 2 == 0)
    overlap = pow(0.5 / alpha, moment / 2) * dfactorial[moment / 2];
  else
    overlap = 0.0;

  return overlap;
}

__device__ inline void multipole_3d(
  const double (&rpj)[3],
  const double (&rpi)[3],
  const double aj,
  const double ai,
  const int (&lj)[3],
  const int (&li)[3],
  const double (&s1d)[MAXL2],
  double &s3d,
  double (&d3d)[3],
  double (&q3d)[6])
{
  double v1d[3][3] = {0.0};

  for(int k = 0; k < 3; ++k)
  {
    double vv[MAXL2] = {0.0};
    double vi[MAXL] = {0.0};
    double vj[MAXL] = {0.0};
    
    vi[li[k]] = 1.0;
    vj[lj[k]] = 1.0;
    horizontal_shift(rpi[k], li[k], vi);
    horizontal_shift(rpj[k], lj[k], vj);
    form_product(vi, vj, li[k], lj[k], vv);
    for (int l = 0; l <= li[k] + lj[k]; ++l)
    {
      v1d[k][0] += s1d[l] * vv[l];
      v1d[k][1] += (s1d[l + 1] + rpi[k] * s1d[l]) * vv[l];
      v1d[k][2] += (s1d[l + 2] + 2 * rpi[k] * s1d[l + 1] + rpi[k] * rpi[k] * s1d[l]) * vv[l];
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

template <typename T>
__device__ inline void transform1(int lj, int li, 
  const device_tensor3d_t<T> &cart, device_tensor3d_t<T> &sphr)
{
  for(int k = 0; k < cart.dim1; ++k)
  {
    transform0(lj, li, k, cart, sphr);
  }
}
__device__ void multipole_cgto(
  const cgto_type &cgtoj,
  const cgto_type &cgtoi,
  const double r2,
  const double (&vec)[3],
  const double intcut,
  device_tensor2d_t<double> &overlap,
  device_tensor3d_t<double> &dpint,
  device_tensor3d_t<double> &qpint)
{

  constexpr int msao[] = {1, 3, 5, 7, 9, 11, 13}; 
  constexpr int mlao[] = {1, 3, 6, 10, 15, 21, 28};
  constexpr int lmap[] = {0, 1, 4, 10, 20, 35, 56};
  constexpr int lx[32][3] = {
    {0, 0, 0}, 
    {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, 
    {0, 0, 2}, {0, 1, 1}, {0, 2, 0}, {1, 0, 1}, {1, 1, 0}, {2, 0, 0}, 
    {0, 0, 3}, {0, 1, 2}, {0, 2, 1}, {0, 3, 0}, {1, 0, 2}, {1, 1, 1}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {3, 0, 0}, 
    {0, 0, 4}, {0, 1, 3}, {0, 2, 2}, {0, 3, 1}, {0, 4, 0}, {1, 0, 3}, {1, 1, 2}, {1, 2, 1}, {1, 3, 0}, {2, 0, 2}, {2, 1, 1}, {2, 2, 0}
  };

  double eab = 0.0, oab = 0.0, est = 0.0, s1d[MAXL2] = {0.0}, rpi[3] = {0.0}, 
    rpj[3] = {0.0}, cc = 0.0, val = 0.0, dip[3] = {0.0}, quad[6] = {0.0}, pre = 0.0, tr = 0.0;
  constexpr double sqrtpi3 = 5.56832799683; // sqrt(pi)**3

  device_tensor2d_t<double> s3d(mlao[cgtoj.ang], mlao[cgtoi.ang]); s3d.fill(0.0);
  device_tensor3d_t<double> d3d(mlao[cgtoj.ang], mlao[cgtoi.ang], 3); d3d.fill(0.0);
  device_tensor3d_t<double> q3d(mlao[cgtoj.ang], mlao[cgtoi.ang], 6); q3d.fill(0.0);
  printf("cgtoj.ang=%d, mlao[cgtoj.ang]=%d\n", cgtoj.ang, mlao[cgtoj.ang]);
  printf("cgtoi.ang=%d, mlao[cgtoi.ang]=%d\n", cgtoi.ang, mlao[cgtoi.ang]);

  for (int ip = 0; ip < cgtoi.nprim; ++ip)
  {
    for (int jp = 0; jp < cgtoj.nprim; ++jp)
    {
      eab = cgtoi.alpha[ip] + cgtoj.alpha[jp];
      oab = 1.0 / eab;
      est = cgtoi.alpha[ip] * cgtoj.alpha[jp] * r2 * oab;

      if (est > intcut) continue;

      pre = exp(-est) * sqrtpi3 * pow(sqrt(oab), 3);
      for (int k = 0; k < 3; ++k)
      {
        rpi[k] = -vec[k] * cgtoj.alpha[jp] * oab;
        rpj[k] = +vec[k] * cgtoi.alpha[ip] * oab;
      }
      for (int l = 0; l <= cgtoi.ang + cgtoj.ang + 2; ++l)
        s1d[l] = overlap_1d(l, eab);
      double cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre;
      for (int mli = 0; mli < mlao[cgtoi.ang]; ++mli)
      {
        for (int mlj = 0; mlj < mlao[cgtoj.ang]; ++mlj)
        {
          multipole_3d(
            rpj, rpi, 
            cgtoj.alpha[jp], cgtoi.alpha[ip], 
            lx[mlj + lmap[cgtoj.ang]], lx[mli + lmap[cgtoi.ang]], 
            s1d, val, dip, quad);
          
          s3d(mlj, mli) += cc * val;
          
          for (size_t k = 0; k < 3; ++k)
            d3d(mlj,mli,k) += cc * dip[k];
          for (size_t k = 0; k < 6; ++k)
            q3d(mlj, mli, k) += cc * quad[k];
        }
      }
    }
  }

  transform0(cgtoj.ang, cgtoi.ang, /*cart=*/s3d, /*sphr=*/overlap);
  transform1(cgtoj.ang, cgtoi.ang, d3d, dpint);
  transform1(cgtoj.ang, cgtoi.ang, q3d, qpint);



  for (int mli = 0; mli < msao[cgtoi.ang]; ++mli)
  {
    for (int mlj = 0; mlj < msao[cgtoj.ang]; ++mlj)
    {
      tr = 0.5 * (qpint(mli, mlj, 0) + qpint(mli, mlj, 2) + qpint(mli, mlj, 5));
      qpint(mli, mlj, 0) = 1.5 * qpint(mli, mlj, 0) - tr;
      qpint(mli, mlj, 1) = 1.5 * qpint(mli, mlj, 1);
      qpint(mli, mlj, 2) = 1.5 * qpint(mli, mlj, 2) - tr;
      qpint(mli, mlj, 3) = 1.5 * qpint(mli, mlj, 3);
      qpint(mli, mlj, 4) = 1.5 * qpint(mli, mlj, 4);
      qpint(mli, mlj, 5) = 1.5 * qpint(mli, mlj, 5) - tr;
    }
  }
}

extern "C"
{
  void get_vec_(
      const double *xyz_iat,
      const double *xyz_jat,
      const double *trans,
      double vec[3])
  {
    printf("C: XYZ (atom iat): %f, %f, %f\n",
           xyz_iat[0],
           xyz_iat[1],
           xyz_iat[2]);
    printf("C: XYZ (atom jat): %f, %f, %f\n",
           xyz_jat[0],
           xyz_jat[1],
           xyz_jat[2]);
    printf("C: TRANS   : %f, %f, %f\n",
           trans[0],
           trans[1],
           trans[2]);
    // for (size_t i = 0; i < 3; i++)
    // {
    //     printf("%d, ", vec[i]);
    // }
    // printf("\n");

    for (size_t k = 0; k < 3; ++k)
    {
      vec[k] = xyz_iat[k] - xyz_jat[k] - trans[k];
    }
    printf("C: Computed vec: %f, %f, %f\n", vec[0], vec[1], vec[2]);
    // printf("C: Result");
    // for (size_t i = 0; i < 3; i++)
    // {
    //     printf("%d, ", vec[i]);
    // }
    // printf("\n");
  }
}

template <typename T>
__device__ inline void shift_operator(
    const int iao, 
    const int jao,
    const T (&vec)[3],
    const device_tensor2d_t<T> &s,
    const device_tensor3d_t<T> &di,
    const device_tensor3d_t<T> &qi,
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

__global__ void get_hamiltonian_inter_atomic(
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
  {
    // printf("mol = \n");
    // printstruct(mol);
    // printf("trans = \n");
    // trans.print();
    // printf("alist = \n");
    // printstruct(alist);
    // printf("bas = \n");
    // printstruct(bas);
    // printf("h0 = \n");
    // printstruct(h0);
    // printf("selfenergy = \n");
    // selfenergy.print();
    // printf("overlap = \n");
    // overlap.print();
    // printf("dpint = \n");
    // dpint.print();
    // printf("qpint = \n");
    // qpint.print();
    // printf("hamiltonian = \n");
    // hamiltonian.print();
    // constants
    // integer, parameter :: msao(0:maxl) = [1, 3, 5, 7, 9, 11, 13]
  }
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int msao[] = {1, 3, 5, 7, 9, 11, 13};
  double rr = 0.0, r2 = 0.0, vec[3] = {0.0}, cutoff2 = 0.0, hij = 0.0, shpoly = 0.0, dtmpj[3] = {0.0}, qtmpj[6] = {0.0};

  device_tensor2d_t<double> stmp(msao[bas.maxl], msao[bas.maxl]); stmp.fill(0.0);
  device_tensor3d_t<double> dtmpi(msao[bas.maxl], msao[bas.maxl], 3); dtmpi.fill(0.0);
  device_tensor3d_t<double> qtmpi(msao[bas.maxl], msao[bas.maxl], 6); qtmpi.fill(0.0);

  int iat = thread_id;
  if (iat >= mol.nat)
    return;
  int izp = mol.id[iat];
  int is = bas.ish_at[iat];
  int inl = alist.inl[iat];
  for (int img = 0; img < alist.nnl[iat]; ++img)
  {
    int jat = alist.nlat[img + inl];
    int itr = alist.nltr[img + inl];
    int jzp = mol.id[jat];
    int js = bas.ish_at[jat];
    for (int k = 0; k < 3; ++k)
    {
      vec[k] = mol.xyz(iat, k) - mol.xyz(jat, k) - trans(itr, k);
    }

    r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    rr = sqrt(sqrt(r2) / (h0.rad[jzp] + h0.rad[izp]));
    for (int ish = 0; ish < bas.nsh_id[izp]; ++ish)
    {
      int ii = bas.iao_sh[is + ish];
      for (int jsh = 0; jsh < bas.nsh_id[jzp]; ++jsh)
      {
        int jj = bas.iao_sh[js + jsh];
        const auto &cgtoj = bas.cgto(jzp, jsh); 
        const auto &cgtoi = bas.cgto(izp, ish);
        multipole_cgto(cgtoj, cgtoi, r2, vec, bas.intcut, /*overlap=*/stmp, dtmpi, qtmpi);

        shpoly = (1.0 + h0.shpoly(izp, ish) * rr) *
                 (1.0 + h0.shpoly(jzp, jsh) * rr);
        hij = 0.5 * (selfenergy[is + ish] + selfenergy[js + jsh]) *
              h0.hscale(izp, jzp, ish, jsh) * shpoly;

        const int nao = msao[bas.cgto(jzp, jsh).ang];
        for(int iao = 0; iao < msao[bas.cgto(izp, ish).ang]; ++iao)
        {
          for(int jao = 0; jao < nao; ++jao)
          {
            shift_operator(iao, jao, vec, stmp, dtmpi, qtmpi, dtmpj, qtmpj); 

            atomicAdd(&overlap(ii + iao, jj + jao), stmp(iao, jao));

            for (int k = 0; k < 3; ++k)
              atomicAdd(&dpint(ii + iao, jj + jao, k), dtmpi(iao, jao, k));

            for (int k = 0; k < 6; ++k)
              atomicAdd(&qpint(ii + iao, jj + jao, k), qtmpi(iao, jao, k));
            atomicAdd(&hamiltonian(ii + iao, jj + jao), stmp(iao, jao) * hij);
            
            /* TODO: This is a symmetrification of these matrices. Maybe this should be
            done in the outside this loop? */
            if (iat != jat) 
            {
              atomicAdd(&overlap(jj + jao, ii + iao), stmp(iao, jao));
              for (int k = 0; k < 3; ++k)
                atomicAdd(&dpint(jj + jao, ii + iao,  k), dtmpj[k]);
              for (int k = 0; k < 6; ++k)
                atomicAdd(&qpint(jj + jao, ii + iao,  k), qtmpj[k]);
              atomicAdd(&hamiltonian(jj + jao, ii + iao), stmp(iao, jao) * hij);
            }
          }
        }
      }
    }
  }
}

__global__ void get_hamiltonian_intra_atomic(
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
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int msao[] = {1, 3, 5, 7, 9, 11, 13};
  device_tensor2d_t<double> stmp(msao[bas.maxl], msao[bas.maxl]); stmp.fill(0.0);
  device_tensor3d_t<double> dtmpi(msao[bas.maxl], msao[bas.maxl], 3); dtmpi.fill(0.0);
  device_tensor3d_t<double> qtmpi(msao[bas.maxl], msao[bas.maxl], 6); qtmpi.fill(0.0);

  int iat = thread_id;
  if (iat >= mol.nat) return;
  int izp = mol.id[iat];
  int is = bas.ish_at[iat];
  double vec[3] = {0.0};
  double r2 = 0.0;
  double rr = sqrt(sqrt(r2) / (h0.rad[izp] + h0.rad[izp]));
  for(int ish = 0; ish < bas.nsh_id[izp]; ++ish)
  {
    int ii = bas.iao_sh[is + ish];
    for(int jsh = 0; jsh < bas.nsh_id[izp]; ++jsh)
    {
      int jj = bas.iao_sh[is + jsh];
      multipole_cgto(bas.cgto(izp, jsh), bas.cgto(izp, ish), 
        r2, vec, bas.intcut, stmp, dtmpi, qtmpi);
      double shpoly = (1.0 + h0.shpoly(izp, ish) * rr) *
        (1.0 + h0.shpoly(izp, jsh) * rr);
      double hij = 0.5 * (selfenergy[is + ish] + selfenergy[is + jsh]) *
        shpoly;
      const int nao = msao[bas.cgto(izp, jsh).ang];
      for(int iao = 0; iao < msao[bas.cgto(izp, ish).ang]; ++iao)
      {
        for(int jao = 0; jao < nao; ++jao)
        {
          overlap(ii + iao, jj + jao) += stmp(iao, jao);
          for(int k = 0; k < 3; ++k)
            dpint(ii + iao, jj + jao, k) += dtmpi(iao, jao, k);
          for(int k = 0; k < 6; ++k)
            qpint(ii + iao, jj + jao, k) += qtmpi(iao, jao, k);
          hamiltonian(ii + iao, jj + jao) += stmp(iao, jao) * hij;
        }
      }
    }
  }
}

void setCudaMallocHeapSizeOnce(size_t size) {
  static bool isHeapSizeSet = false; // Tracks if the limit has already been set
  if (!isHeapSizeSet) {
      CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size));
      isHeapSizeSet = true; // Mark as set
  }
}

extern "C" void cuda_get_hamiltonian_kernel_(
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
    double *hamiltonian)
{
  printf("================= CUDA C/C++ =================\n");

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
  
  
  ////////////////////////////////////////////
  // Launch kernel part I
  ////////////////////////////////////////////
  cudaEvent_t start, stop;
  float milliseconds = 0;

  {
    /*
    NOTE
    cudaLimitMallocHeapSize controls the size in bytes of the heap used by 
    the ::malloc() and ::free() device system calls. Setting ::cudaLimitMallocHeapSize 
    must not be performed after launching any kernel that uses the ::malloc() or ::free() 
    device system calls - in such case ::cudaErrorInvalidValue will be returned
    */
    setCudaMallocHeapSizeOnce(1024 * sizeof(double));
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    get_hamiltonian_inter_atomic<<<1, mol.nat>>>(
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
    printf("Kernel part I execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start); cudaEventDestroy(stop);
  }
  printf("hamiltonian_ten(pre) = \n");
  hamiltonian_ten.print();
  ////////////////////////////////////////////
  // Launch kernel part II
  ////////////////////////////////////////////
  {
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    get_hamiltonian_intra_atomic<<<1, mol.nat>>>(
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
    printf("Kernel part II execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start); cudaEventDestroy(stop);
  }

  ////////////////////////////
  // copy data back to host
  ////////////////////////////
  memcpy(overlap, overlap_ten.data, overlap_ten.size() * sizeof(double));
  memcpy(dpint, dpint_ten.data, dpint_ten.size() * sizeof(double));
  memcpy(qpint, qpint_ten.data, qpint_ten.size() * sizeof(double));
  memcpy(hamiltonian, hamiltonian_ten.data, hamiltonian_ten.size() * sizeof(double));
  
  // printf("overlap_ten = \n");
  // overlap_ten.print();
  // printf("dpint_ten = \n");
  // dpint_ten.print();
  // printf("qpint_ten = \n");
  // qpint_ten.print();
  // printf("hamiltonian_ten = \n");
  // hamiltonian_ten.print();

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
  // printf("Exiting prematurely. Remove this once the kernel is working.\n");
  // exit(EXIT_FAILURE);

  
}
