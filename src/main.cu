#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "device_tensor.h"
#include "types.h"

__global__ void hello_kernel()
{
  printf("%i %i Says Hello!", blockIdx.x, threadIdx.x);
}

// Kernel to test the constants
__global__ void testKernel()
{
  // printf("s3: %f, s3_4: %f, dtrafo[0][2]: %f, ftrafo[0][4]: %f, gtrafo[0][4]: %f\n",
  //        s3, s3_4, dtrafo[0][2], ftrafo[0][4], gtrafo[0][4]);
}


// Assuming constants like s3, s3_4, dtrafo, ftrafo, and gtrafo are already defined as __device__ __constant__

template <typename T>
__device__ inline void transform0(int lj, int li, 
  const device_tensor2d_t<T> &cart, device_tensor2d_t<T> &sphr)
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
      for (int i = 0; i < cart.dim1; ++i)
      {
        for (int j = 0; j < cart.dim2; ++j)
        {
          sphr(i, j) = cart(i, j);
        }
      }
      break;
    case 2:
      // sphr = matmul(dtrafo, cart)
      sphr(0, 0) = cart(2, 0) - 0.5 * (cart(0, 0) + cart(1, 0));
      sphr(1, 0) = s3 * cart(4, 0);
      sphr(2, 0) = s3 * cart(5, 0);
      sphr(3, 0) = s3_4 * (cart(0, 0) - cart(1, 0));
      sphr(4, 0) = s3 * cart(3, 0);
      break;
    
    default:
      printf("[Fatal] moment li=%i lj=%i not supported\n", li, lj);
      assert(false)
      return;
    }
  }
  //   case 3:
  //     // sphr = matmul(ftrafo, cart)
  //     for (int i = 0; i < 7; ++i)
  //     {
  //       sphr(i, 0) = 0.0;
  //       for (int j = 0; j < 10; ++j)
  //       {
  //         sphr(i, 0) += ftrafo[i][j] * cart[j];
  //       }
  //     }
  //     break;
  //   case 4:
  //     // sphr = matmul(gtrafo, cart)
  //     for (int i = 0; i < 9; ++i)
  //     {
  //       sphr[i] = 0.0;
  //       for (int j = 0; j < 15; ++j)
  //       {
  //         sphr[i] += gtrafo[i][j] * cart[j];
  //       }
  //     }
  //     break;
  //   default:
  //     printf("[Fatal] Moments higher than g are not supported\n");
  //     return;
  //   }
  //   break;

  // case 2:
  //   switch (lj)
  //   {
  //   case 0:
  //   case 1:
  //     // sphr = matmul(cart, transpose(dtrafo))
  //     for (int i = 0; i < cart_rows; ++i)
  //     {
  //       sphr[i * 5 + 0] = cart[i * 6 + 2] - 0.5 * (cart[i * 6 + 0] + cart[i * 6 + 1]);
  //       sphr[i * 5 + 1] = s3 * cart[i * 6 + 4];
  //       sphr[i * 5 + 2] = s3 * cart[i * 6 + 5];
  //       sphr[i * 5 + 3] = s3_4 * (cart[i * 6 + 0] - cart[i * 6 + 1]);
  //       sphr[i * 5 + 4] = s3 * cart[i * 6 + 3];
  //     }
  //     break;
  //   case 2:
  //     // sphr = matmul(dtrafo, matmul(cart, transpose(dtrafo)))
  //     // This is a simplified example; the full implementation would require nested loops
  //     printf("[Fatal] Higher-order transformations not implemented\n");
  //     return;
  //   case 3:
  //     // sphr = matmul(ftrafo, matmul(cart, transpose(dtrafo)))
  //     printf("[Fatal] Higher-order transformations not implemented\n");
  //     return;
  //   case 4:
  //     // sphr = matmul(gtrafo, matmul(cart, transpose(dtrafo)))
  //     printf("[Fatal] Higher-order transformations not implemented\n");
  //     return;
  //   default:
  //     printf("[Fatal] Moments higher than g are not supported\n");
  //     return;
  //   }
  //   break;

  // case 3:
  //   switch (lj)
  //   {
  //   case 0:
  //   case 1:
  //     // sphr = matmul(cart, transpose(ftrafo))
  //     for (int i = 0; i < cart_rows; ++i)
  //     {
  //       for (int j = 0; j < 7; ++j)
  //       {
  //         sphr[i * 7 + j] = 0.0;
  //         for (int k = 0; k < 10; ++k)
  //         {
  //           sphr[i * 7 + j] += cart[i * 10 + k] * ftrafo[j][k];
  //         }
  //       }
  //     }
  //     break;
  //   case 2:
  //   case 3:
  //   case 4:
  //     printf("[Fatal] Higher-order transformations not implemented\n");
  //     return;
  //   default:
  //     printf("[Fatal] Moments higher than g are not supported\n");
  //     return;
  //   }
  //   break;

  // case 4:
  //   switch (lj)
  //   {
  //   case 0:
  //   case 1:
  //     // sphr = matmul(cart, transpose(gtrafo))
  //     for (int i = 0; i < cart_rows; ++i)
  //     {
  //       for (int j = 0; j < 9; ++j)
  //       {
  //         sphr[i * 9 + j] = 0.0;
  //         for (int k = 0; k < 15; ++k)
  //         {
  //           sphr[i * 9 + j] += cart[i * 15 + k] * gtrafo[j][k];
  //         }
  //       }
  //     }
  //     break;
  //   case 2:
  //   case 3:
  //   case 4:
  //     printf("[Fatal] Higher-order transformations not implemented\n");
  //     return;
  //   default:
  //     printf("[Fatal] Moments higher than g are not supported\n");
  //     return;
  //   }
  //   break;

  // default:
  //   printf("[Fatal] Moments higher than g are not supported\n");
  //   return;
  // }

}


/*elemental function overlap_1d(moment, alpha) result(overlap)
   integer, intent(in) :: moment
   real(wp), intent(in) :: alpha
   real(wp) :: overlap
   real(wp), parameter :: dfactorial(0:7) = & ! see OEIS A001147
      & [1._wp,1._wp,3._wp,15._wp,105._wp,945._wp,10395._wp,135135._wp]

   if (modulo(moment, 2) == 0) then
      overlap = (0.5_wp/alpha)**(moment/2) * dfactorial(moment/2)
   else
      overlap = 0.0_wp
   end if
end function overlap_1d*/

__device__ inline double overlap_1d(int moment, double alpha)
{
  // integer, intent(in) :: moment
  // real(wp), intent(in) :: alpha
  double overlap = 0.0;
  // real(wp), parameter :: dfactorial(0:7) = & ! see OEIS A001147
  //   & [1._wp,1._wp,3._wp,15._wp,105._wp,945._wp,10395._wp,135135._wp]
  constexpr double dfactorial[8] = {1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0};
  assert(moment >= 0 && moment <= 7);
  if (moment % 2 == 0)
    overlap = pow(0.5 / alpha, moment / 2) * dfactorial[moment / 2];
  else
    overlap = 0.0;

  return overlap;
}

/*pure subroutine multipole_3d(rpj, rpi, aj, ai, lj, li, s1d, s3d, d3d, q3d)
   real(wp), intent(in) :: rpi(3)
   real(wp), intent(in) :: rpj(3)
   real(wp), intent(in) :: ai
   real(wp), intent(in) :: aj
   integer, intent(in) :: li(3)
   integer, intent(in) :: lj(3)
   real(wp), intent(in) :: s1d(0:)
   real(wp), intent(out) :: s3d
   real(wp), intent(out) :: d3d(3)
   real(wp), intent(out) :: q3d(6)
*/

__device__ inline void multipole_3d(
  const double *rpj,
  const double *rpi,
  double aj,
  double ai,
  const int (&lj)[3],
  const int (&li)[3],
  const double (&s1d)[MAXL2],
  double &val,
  double (&dip)[3],
  double (&quad)[6])
{
  // real(wp), intent(in) :: rpi(3)
  // real(wp), intent(in) :: rpj(3)
  // real(wp), intent(in) :: ai
  // real(wp), intent(in) :: aj
  // integer, intent(in) :: li(3)
  // integer, intent(in) :: lj(3)
  // real(wp), intent(in) :: s1d(0:)
  // real(wp), intent(out) :: s3d
  // real(wp), intent(out) :: d3d(3)
  // real(wp), intent(out) :: q3d(6)

  val = s1d[lj[0]] * s1d[li[0]];
  dip[0] = (rpj[0] - rpi[0]) * val;
  dip[1] = (rpj[1] - rpi[1]) * val;
  dip[2] = (rpj[2] - rpi[2]) * val;

  quad[0] = (rpj[0] - rpi[0]) * dip[0];
  quad[1] = (rpj[1] - rpi[1]) * dip[1];
  quad[2] = (rpj[2] - rpi[2]) * dip[2];
}

/*subroutine multipole_cgto(cgtoj, cgtoi, r2, vec, intcut, overlap, dpint, qpint)
   !> Description of contracted Gaussian function on center i
   type(cgto_type), intent(in) :: cgtoi
   !> Description of contracted Gaussian function on center j
   type(cgto_type), intent(in) :: cgtoj
   !> Square distance between center i and j
   real(wp), intent(in) :: r2
   !> Distance vector between center i and j, ri - rj
   real(wp), intent(in) :: vec(3)
   !> Maximum value of integral prefactor to consider
   real(wp), intent(in) :: intcut
   !> Overlap integrals for the given pair i  and j
   real(wp), intent(out) :: overlap(msao(cgtoj%ang), msao(cgtoi%ang))
   !> Dipole moment integrals for the given pair i  and j
   real(wp), intent(out) :: dpint(3, msao(cgtoj%ang), msao(cgtoi%ang))
   !> Quadrupole moment integrals for the given pair i  and j
   real(wp), intent(out) :: qpint(6, msao(cgtoj%ang), msao(cgtoi%ang))*/
__device__ void multipole_cgto(
  const cgto_type &cgtoj,
  const cgto_type &cgtoi,
  const double r2,
  const double *vec,
  double intcut,
  device_tensor2d_t<double> &overlap,
  device_tensor3d_t<double> &dpint,
  device_tensor3d_t<double> &qpint)
{
  printf("================== MULTIPOLE_CGTO =================\n");
  printf("bid %i tid %i: multipole_cgto\n", blockIdx.x, threadIdx.x);
  printf("%s:%d: %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  printf("Parameters\n");
  printf("cgtoj=\n");
  printstruct(cgtoj);
  printf("cgtoi=\n");
  printstruct(cgtoi);
  printf("r2=%f\n", r2);
  printf("vec=%f, %f, %f\n", vec[0], vec[1], vec[2]);
  printf("intcut=%f\n", intcut);
  printf("overlap=\n");
  overlap.print();
  printf("dpint=\n");
  dpint.print();
  printf("qpint=\n");
  qpint.print();
  printf("====================================================\n");

  /*integer, parameter :: msao(0:maxl) = [1, 3, 5, 7, 9, 11, 13]
   integer, parameter :: mlao(0:maxl) = [1, 3, 6, 10, 15, 21, 28]
   integer, parameter :: lmap(0:maxl) = [0, 1, 4, 10, 20, 35, 56]*/
  constexpr int msao[] = {1, 3, 5, 7, 9, 11, 13};
  constexpr int mlao[] = {1, 3, 6, 10, 15, 21, 28};
  constexpr int lmap[] = {0, 1, 4, 10, 20, 35, 56};
  /*   ! x (+1), y (-1), z (0) in [-1, 0, 1] sorting
   integer, parameter :: lx(3, 84) = reshape([&
      & 0, &
      & 0,0,1, &
      & 2,0,0,1,1,0, &
      & 3,0,0,2,2,1,0,1,0,1, &
      & 4,0,0,3,3,1,0,1,0,2,2,0,2,1,1, &
      & 5,0,0,3,3,2,2,0,0,4,4,1,0,0,1,1,3,1,2,2,1, &
      & 6,0,0,3,3,0,5,5,1,0,0,1,4,4,2,0,2,0,3,3,1,2,2,1,4,1,1,2, &
      & 0, &
      & 1,0,0, &
      & 0,2,0,1,0,1, &
      & 0,3,0,1,0,2,2,0,1,1, &
      & 0,4,0,1,0,3,3,0,1,2,0,2,1,2,1, &
      & 0,5,0,2,0,3,0,3,2,1,0,4,4,1,0,1,1,3,2,1,2, &
      & 0,6,0,3,0,3,1,0,0,1,5,5,2,0,0,2,4,4,2,1,3,1,3,2,1,4,1,2, &
      & 0, &
      & 0,1,0, &
      & 0,0,2,0,1,1, &
      & 0,0,3,0,1,0,1,2,2,1, &
      & 0,0,4,0,1,0,1,3,3,0,2,2,1,1,2, &
      & 0,0,5,0,2,0,3,2,3,0,1,0,1,4,4,3,1,1,1,2,2, &
      & 0,0,6,0,3,3,0,1,5,5,1,0,0,2,4,4,0,2,1,2,2,3,1,3,1,1,4,2], &
      & shape(lx), order=[2, 1])
  */
  constexpr int lx[32][3] = {
    {0, 0, 0}, 
    {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, 
    {0, 0, 2}, {0, 1, 1}, {0, 2, 0}, {1, 0, 1}, {1, 1, 0}, {2, 0, 0}, 
    {0, 0, 3}, {0, 1, 2}, {0, 2, 1}, {0, 3, 0}, {1, 0, 2}, {1, 1, 1}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {3, 0, 0}, 
    {0, 0, 4}, {0, 1, 3}, {0, 2, 2}, {0, 3, 1}, {0, 4, 0}, {1, 0, 3}, {1, 1, 2}, {1, 2, 1}, {1, 3, 0}, {2, 0, 2}, {2, 1, 1}, {2, 2, 0}
  };

  /*integer :: ip, jp, mli, mlj, l
  real(wp) :: eab, oab, est, s1d(0:maxl2), rpi(3), rpj(3), cc, val, dip(3), quad(6), pre, tr
  real(wp) :: s3d(mlao(cgtoj%ang), mlao(cgtoi%ang))
  real(wp) :: d3d(3, mlao(cgtoj%ang), mlao(cgtoi%ang))
  real(wp) :: q3d(6, mlao(cgtoj%ang), mlao(cgtoi%ang))

  s3d(:, :) = 0.0_wp
  d3d(:, :, :) = 0.0_wp
  q3d(:, :, :) = 0.0_wp*/
  int ip = 0, jp = 0, mli = 0, mlj = 0, l = 0;
  double eab = 0.0, oab = 0.0, est = 0.0, s1d[MAXL2] = {0.0}, rpi[3] = {0.0}, 
    rpj[3] = {0.0}, cc = 0.0, val = 0.0, dip[3] = {0.0}, quad[6] = {0.0}, pre = 0.0, tr = 0.0;
  constexpr double sqrtpi3 = 5.56832799683; // sqrt(pi)**3

  device_tensor2d_t<double> s3d(mlao[cgtoi.ang], mlao[cgtoj.ang]); s3d.fill(0.0);
  device_tensor3d_t<double> d3d(mlao[cgtoi.ang], mlao[cgtoj.ang], 3); d3d.fill(0.0);
  device_tensor3d_t<double> q3d(mlao[cgtoi.ang], mlao[cgtoj.ang], 6); q3d.fill(0.0);

  /*
  do ip = 1, cgtoi%nprim
      do jp = 1, cgtoj%nprim
         eab = cgtoi%alpha(ip) + cgtoj%alpha(jp)
         oab = 1.0_wp/eab
         est = cgtoi%alpha(ip) * cgtoj%alpha(jp) * r2 * oab
         if (est > intcut) cycle
         pre = exp(-est) * sqrtpi3*sqrt(oab)**3
         rpi = -vec * cgtoj%alpha(jp) * oab
         rpj = +vec * cgtoi%alpha(ip) * oab
         do l = 0, cgtoi%ang + cgtoj%ang + 2
            s1d(l) = overlap_1d(l, eab)
         end do
         cc = cgtoi%coeff(ip) * cgtoj%coeff(jp) * pre
         do mli = 1, mlao(cgtoi%ang)
            do mlj = 1, mlao(cgtoj%ang)
               call multipole_3d(rpj, rpi, cgtoj%alpha(jp), cgtoi%alpha(ip), &
                  & lx(:, mlj+lmap(cgtoj%ang)), lx(:, mli+lmap(cgtoi%ang)), &
                  & s1d, val, dip, quad)
               s3d(mlj, mli) = s3d(mlj, mli) + cc*val
               d3d(:, mlj, mli) = d3d(:, mlj, mli) + cc*dip
               q3d(:, mlj, mli) = q3d(:, mlj, mli) + cc*quad
            end do
         end do
      end do
   end do*/
  for (ip = 0; ip < cgtoi.nprim; ++ip)
  {
    for (jp = 0; jp < cgtoj.nprim; ++jp)
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
      for (l = 0; l <= cgtoi.ang + cgtoj.ang + 2; ++l)
      {
        s1d[l] = overlap_1d(l, eab);
      }
      double cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre;
      for (mli = 0; mli < mlao[cgtoi.ang]; ++mli)
      {
        for (mlj = 0; mlj < mlao[cgtoj.ang]; ++mlj)
        {
          // DEBUG
          printf("================== MULTIPOLE_CGTO INNER LOOP =================\n");
          printf("bid %i tid %i: multipole_cgto inner loop\n", blockIdx.x, threadIdx.x);
          printf("Parameters\n");
          printf("mli=%d, mlj=%d\n", mli, mlj);
          printf("rpi=%f, %f, %f\n", rpi[0], rpi[1], rpi[2]); printf("rpj=%f, %f, %f\n", rpj[0], rpj[1], rpj[2]);
          printf("cgtoj.alpha[%d]=%f, cgtoi.alpha[%d]=%f\n", jp, cgtoj.alpha[jp], ip, cgtoi.alpha[ip]);
          printf("lx[%i][:] = [%i, %i, %i]\n", mlj + lmap[cgtoj.ang], lx[mlj + lmap[cgtoj.ang]][0], lx[mlj + lmap[cgtoj.ang]][1], lx[mlj + lmap[cgtoj.ang]][2]);
          printf("lx[%i][:] = [%i, %i, %i]\n", mli + lmap[cgtoi.ang], lx[mli + lmap[cgtoi.ang]][0], lx[mli + lmap[cgtoi.ang]][1], lx[mli + lmap[cgtoi.ang]][2]);
          printf("==============================================================\n");

          multipole_3d(
            rpj, rpi, cgtoj.alpha[jp], cgtoi.alpha[ip], 
            lx[mlj + lmap[cgtoj.ang]], lx[mli + lmap[cgtoi.ang]], 
            s1d, val, dip, quad);
          s3d(mli, mlj) += cc * val;
          for (size_t k = 0; k < 3; ++k)
            d3d(mli,mlj,k) += cc * dip[k];
          for (size_t k = 0; k < 6; ++k)
            q3d(mli, mlj, k) += cc * quad[k];
        }
      }
    }
  }

  /*
   call transform0(cgtoj%ang, cgtoi%ang, s3d, overlap)
   call transform1(cgtoj%ang, cgtoi%ang, d3d, dpint)
   call transform1(cgtoj%ang, cgtoi%ang, q3d, qpint)
  */
  transform0(cgtoj.ang, cgtoi.ang, s3d, overlap);
  // transform1(cgtoj.ang, cgtoi.ang, d3d.data, dpint.data);
  // transform1(cgtoj.ang, cgtoi.ang, q3d.data, qpint.data);

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

  void call_hello_kernel_()
  {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to finish.
  }
}

__global__ void get_hamiltonian(
    const structure_type mol,
    const tensor2d_t<double> trans,
    const adjacency_list alist,
    const basis_type bas,
    const tb_hamiltonian h0,
    const tensor1d_t<double> selfenergy,
    tensor2d_t<double> overlap,
    tensor3d_t<double> dpint,
    tensor3d_t<double> qpint,
    tensor2d_t<double> hamiltonian)
{
  printf("================= KERNEL =================\n");
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
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int msao[7] = {1, 3, 5, 7, 9, 11, 13};

  // locals
  int i = 0, j = 0, l = 0;
  int iat = 0, jat = 0, izp = 0, jzp = 0, itr = 0, k = 0, img = 0, inl = 0;
  int ish = 0, jsh = 0, is = 0, js = 0, ii = 0, jj = 0, iao = 0, jao = 0, nao = 0, ij = 0;
  double rr = 0.0, r2 = 0.0, vec[3] = {0.0}, cutoff2 = 0.0, hij = 0.0, shpoly = 0.0, dtmpj[3] = {0.0}, qtmpj[6] = {0.0};

  // clean overlap, dpint, qpint, hamiltonian
  if (thread_id == 0)
  {
    // printf("C: Cleaning overlap, dpint, qpint, hamiltonian\n");
    overlap.fill(0.0);
    dpint.fill(0.0);
    qpint.fill(0.0);
    hamiltonian.fill(0.0);
  }
  __syncthreads();

  // allocate stmp, dtmpi, qtmpi
  device_tensor2d_t<double> stmp(msao[bas.maxl], msao[bas.maxl]);
  stmp.fill(0.0);
  device_tensor3d_t<double> dtmpi(msao[bas.maxl], msao[bas.maxl], 3);
  dtmpi.fill(0.0);
  device_tensor3d_t<double> qtmpi(msao[bas.maxl], msao[bas.maxl], 6);
  qtmpi.fill(0.0);

  // stmp.fill(1.0);
  // dtmpi.fill(2.0);
  // qtmpi.fill(3.0);
  // // Assert device tensor works
  // printf("stmp = \n");
  // stmp.print();
  // printf("dtmpi = \n");
  // dtmpi.print();
  // printf("qtmpi = \n");
  // qtmpi.print();
  /*do iat = 1, mol%nat
      izp = mol%id(iat)
      is = bas%ish_at(iat)
      inl = alist%inl(iat)
      do img = 1, alist%nnl(iat)
        jat = alist%nlat(img+inl)
        itr = alist%nltr(img+inl)
        jzp = mol%id(jat)
        js = bas%ish_at(jat)
        vec(:) = mol%xyz(:, iat) - mol%xyz(:, jat) - trans(:, itr)
        r2 = vec(1)**2 + vec(2)**2 + vec(3)**2
        rr = sqrt(sqrt(r2) / (h0%rad(jzp) + h0%rad(izp)))
        do ish = 1, bas%nsh_id(izp)
            ii = bas%iao_sh(is+ish)
            do jsh = 1, bas%nsh_id(jzp)
              jj = bas%iao_sh(js+jsh)*/
  iat = thread_id;
  if (iat >= mol.nat)
    return;
  izp = mol.id[iat];
  is = bas.ish_at[iat];
  inl = alist.inl[iat];
  for (img = 0; img < alist.nnl[iat]; ++img)
  {
    jat = alist.nlat[img + inl];
    itr = alist.nltr[img + inl];
    jzp = mol.id[jat];
    js = bas.ish_at[jat];

    for (int k = 0; k < 3; ++k)
    {
      vec[k] = mol.xyz(iat, k) - mol.xyz(jat, k) - trans(itr, k);
    }

    r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    rr = sqrt(sqrt(r2) / (h0.rad[jzp] + h0.rad[izp]));
    for (ish = 0; ish < bas.nsh_id[izp]; ++ish)
    {
      ii = bas.iao_sh[is + ish];
      for (jsh = 0; jsh < bas.nsh_id[jzp]; ++jsh)
      {
        jj = bas.iao_sh[js + jsh];
        printf("ish = %d, jsh = %d, ii = %d, jj = %d\n", ish, jsh, ii, jj);
        /*call multipole_cgto(bas%cgto(jsh, jzp), bas%cgto(ish, izp), &
                  & r2, vec, bas%intcut, stmp, dtmpi, qtmpi)

                  shpoly = (1.0_wp + h0%shpoly(ish, izp)*rr) &
                     * (1.0_wp + h0%shpoly(jsh, jzp)*rr)

                  hij = 0.5_wp * (selfenergy(is+ish) + selfenergy(js+jsh)) &
                     * h0%hscale(jsh, ish, jzp, izp) * shpoly

                  nao = msao(bas%cgto(jsh, jzp)%ang)*/
        const auto &cgtoj = bas.cgto(jsh, jzp);
        const auto &cgtoi = bas.cgto(ish, izp);
        multipole_cgto(cgtoj, cgtoi, r2, vec, bas.intcut, stmp, dtmpi, qtmpi);
        // printf("ish = %d, jsh = %d, ii = %d, jj = %d\n", ish, jsh, ii, jj)
        // printf("stmp = \n");
        // stmp.print();
        // printf("dtmpi = \n");
        // dtmpi.print();
        // printf("qtmpi = \n");
        // qtmpi.print();
      }
    }
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
    const double *mol_lattice, int mol_lattice_dim1, int mol_lattice_dim2,
    const int *mol_periodic, int mol_periodic_dim1,
    const int *mol_bond, int mol_bond_dim1, int mol_bond_dim2,

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

    // Diagonal elememts of the Hamiltonian  (nel)
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
  printf("================= CUDA =================\n");
  // DEBUG print values of h0_hscale_dim1, int h0_hscale_dim2, int h0_hscale_dim3, int h0_hscale_dim4
  // printf("====================== DEBUG =================\n");
  // printf("h0_hscale_dim1 = %d\n", h0_hscale_dim1);
  // printf("h0_hscale_dim2 = %d\n", h0_hscale_dim2);
  // printf("h0_hscale_dim3 = %d\n", h0_hscale_dim3);
  // printf("h0_hscale_dim4 = %d\n", h0_hscale_dim4);
  // printf("======================= DEBUG =================\n");

  /* Pack args into structures */
  // debug print alist entires
  // printf("alist_inl = \n");
  // for (int i = 0; i < alist_inl_dim1; ++i)
  // {
  //   printf("%d, ", alist_inl[i]);
  // }
  // printf("\n");
  // printf("alist_nnl = \n");
  // for (int i = 0; i < alist_nnl_dim1; ++i)
  // {
  //   printf("%d, ", alist_nnl[i]);
  // }
  // printf("\n");
  // printf("alist_nlat = \n");
  // for (int i = 0; i < alist_nlat_dim1; ++i)
  // {
  //   printf("%d, ", alist_nlat[i]);
  // }
  // printf("\n");
  // printf("alist_nltr = \n");
  // for (int i = 0; i < alist_nltr_dim1; ++i)
  // {
  //   printf("%d, ", alist_nltr[i]);
  // }
  // printf("\n");

  const adjacency_list alist{
      tensor1d_t(alist_inl, alist_inl_dim1),
      tensor1d_t(alist_nnl, alist_nnl_dim1),
      tensor1d_t(alist_nlat, alist_nlat_dim1),
      tensor1d_t(alist_nltr, alist_nltr_dim1)};

  // printstruct(alist);

  const structure_type mol{
      mol_nat, mol_nid, mol_nbd,
      tensor1d_t(mol_id, mol_id_dim1),
      tensor1d_t(mol_num, mol_num_dim1),
      tensor2d_t(mol_xyz, mol_xyz_dim1, mol_xyz_dim2),
      mol_uhf,
      mol_charge,
      tensor2d_t(mol_lattice, mol_lattice_dim1, mol_lattice_dim2),
      tensor1d_t(mol_periodic, mol_periodic_dim1),
      tensor2d_t(mol_bond, mol_bond_dim1, mol_bond_dim2)};

  const tb_hamiltonian h0{
      // h0_selfenergy, h0_selfenergy_dim1, h0_selfenergy_dim2,
      tensor2d_t(h0_selfenergy, h0_selfenergy_dim1, h0_selfenergy_dim2),
      // h0_kcn, h0_kcn_dim1, h0_kcn_dim2,
      tensor2d_t(h0_kcn, h0_kcn_dim1, h0_kcn_dim2),
      // h0_kq1, h0_kq1_dim1, h0_kq1_dim2,
      tensor2d_t(h0_kq1, h0_kq1_dim1, h0_kq1_dim2),
      // h0_kq2, h0_kq2_dim1, h0_kq2_dim2,
      tensor2d_t(h0_kq2, h0_kq2_dim1, h0_kq2_dim2),
      // h0_hscale, h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4,
      tensor4d_t(h0_hscale, h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4),
      // h0_shpoly, h0_shpoly_dim1, h0_shpoly_dim2,
      tensor2d_t(h0_shpoly, h0_shpoly_dim1, h0_shpoly_dim2),
      // h0_rad, h0_rad_dim1,
      tensor1d_t(h0_rad, h0_rad_dim1),
      // h0_refocc, h0_refocc_dim1, h0_refocc_dim2};
      tensor2d_t(h0_refocc, h0_refocc_dim1, h0_refocc_dim2)};

  const basis_type bas{
      bas_maxl,
      bas_nsh,
      bas_nao,
      bas_intcut,
      bas_min_alpha,
      // bas_nsh_id, bas_nsh_id_dim1,
      tensor1d_t(bas_nsh_id, bas_nsh_id_dim1),
      // bas_nsh_at, bas_nsh_at_dim1,
      tensor1d_t(bas_nsh_at, bas_nsh_at_dim1),
      // bas_nao_sh, bas_nao_sh_dim1,
      tensor1d_t(bas_nao_sh, bas_nao_sh_dim1),
      // bas_iao_sh, bas_iao_sh_dim1,
      tensor1d_t(bas_iao_sh, bas_iao_sh_dim1),
      // bas_ish_at, bas_ish_at_dim1,
      tensor1d_t(bas_ish_at, bas_ish_at_dim1),
      // bas_ao2at, bas_ao2at_dim1,
      tensor1d_t(bas_ao2at, bas_ao2at_dim1),
      // bas_ao2sh, bas_ao2sh_dim1,
      tensor1d_t(bas_ao2sh, bas_ao2sh_dim1),
      // bas_sh2at, bas_sh2at_dim1,
      tensor1d_t(bas_sh2at, bas_sh2at_dim1),
      // cgto, cgto_dim1, cgto_dim2};
      tensor2d_t(cgto, cgto_dim1, cgto_dim2)};

  const structure_type d_mol{
      mol.nat,
      mol.nid,
      mol.nbd,
      mol.id.to_device(),
      mol.num.to_device(),
      mol.xyz.to_device(),
      mol.uhf,
      mol.charge,
      mol.lattice.to_device(),
      mol.periodic.to_device(),
      mol.bond.to_device()};

  const tensor2d_t<double> d_trans = tensor2d_t<double>(trans, trans_dim1, trans_dim2).to_device();

  const adjacency_list d_alist{
      alist.inl.to_device(),
      alist.nnl.to_device(),
      alist.nlat.to_device(),
      alist.nltr.to_device()};

  const basis_type d_basis{
      bas_maxl,
      bas_nsh,
      bas_nao,
      bas_intcut,
      bas_min_alpha,
      // tensor1d_t(bas.nsh_id, bas.nsh_id_dim1).to_device(),
      bas.nsh_id.to_device(),
      // tensor1d_t(bas.nsh_at, bas.nsh_at_dim1).to_device(),
      bas.nsh_at.to_device(),
      // tensor1d_t(bas.nao_sh, bas.nao_sh_dim1).to_device(),
      bas.nao_sh.to_device(),
      // tensor1d_t(bas.iao_sh, bas.iao_sh_dim1).to_device(),
      bas.iao_sh.to_device(),
      // tensor1d_t(bas.ish_at, bas.ish_at_dim1).to_device(),
      bas.ish_at.to_device(),
      // tensor1d_t(bas.ao2at, bas.ao2at_dim1).to_device(),
      bas.ao2at.to_device(),
      // tensor1d_t(bas.ao2sh, bas.ao2sh_dim1).to_device(),
      bas.ao2sh.to_device(),
      // tensor1d_t(bas.sh2at, bas.sh2at_dim1).to_device(),
      bas.sh2at.to_device(),
      // cgto, cgto_dim1, cgto_dim2};
      // tensor2d_t<cgto_type>(bas.cgto, bas.cgto_dim1, bas.cgto_dim2).to_device()};
      bas.cgto.to_device()};

  const tb_hamiltonian d_h0{
      // tensor2d_t(h0_selfenergy, h0_selfenergy_dim1, h0_selfenergy_dim2).to_device(),
      h0.selfenergy.to_device(),
      // tensor2d_t(h0_kcn, h0_kcn_dim1, h0_kcn_dim2).to_device(),
      h0.kcn.to_device(),
      // tensor2d_t(h0_kq1, h0_kq1_dim1, h0_kq1_dim2).to_device(),
      h0.kq1.to_device(),
      // tensor2d_t(h0_kq2, h0_kq2_dim1, h0_kq2_dim2).to_device(),
      h0.kq2.to_device(),
      // tensor4d_t(h0_hscale, h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4).to_device(),
      h0.hscale.to_device(),
      // tensor2d_t(h0_shpoly, h0_shpoly_dim1, h0_shpoly_dim2).to_device(),
      h0.shpoly.to_device(),
      // tensor1d_t(h0_rad, h0_rad_dim1).to_device(),
      h0.rad.to_device(),
      // tensor2d_t(h0_refocc, h0_refocc_dim1, h0_refocc_dim2).to_device()};
      h0.refocc.to_device()};

  const tensor1d_t<double> d_selfenergy = tensor1d_t<double>(selfenergy, nelem).to_device();

  tensor2d_t<double> d_overlap = tensor2d_t<double>(overlap, nao, nao).to_device();
  tensor3d_t<double> d_dpint = tensor3d_t<double>(dpint, nao, nao, 3).to_device();
  tensor3d_t<double> d_qpint = tensor3d_t<double>(qpint, nao, nao, 6).to_device();
  tensor2d_t<double> d_hamiltonian = tensor2d_t<double>(hamiltonian, nao, nao).to_device();

  // Start timing
  cudaEvent_t start, stop;

  // We need to set heap size for CUDA, to use malloc inside kernel
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 * 1024 * 1024););

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Launch kernel
  get_hamiltonian<<<1, mol.nat>>>(
      d_mol,
      d_trans,
      d_alist,
      d_basis,
      d_h0,
      d_selfenergy,
      d_overlap,
      d_dpint,
      d_qpint,
      d_hamiltonian);

  // Stop timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Clean up events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Exiting prematurely. Remove this once the kernel is working.\n");
  exit(EXIT_FAILURE);

  // printf("at %s:%i\n", __func__, __LINE__);
  // printf("nao = %i\n", nao);
  // printf("nelem = %i\n", nelem);
  // printstruct(bas);
  // printstruct(alist);
  // printstruct(mol);
  // printstruct(h0);

  // printf("bas_nsh_id = \n");
  // printr(bas_nsh_id_dim1, bas_nsh_id);
  // printf("bas_nsh_at = \n");
  // printr(bas_nsh_at_dim1, bas_nsh_at);
  // printf("bas_nao_sh = \n");
  // printr(bas_nao_sh_dim1, bas_nao_sh);
  // printf("bas_iao_sh = \n");
  // printr(bas_iao_sh_dim1, bas_iao_sh);
  // printf("bas_ish_at = \n");
  // printr(bas_ish_at_dim1, bas_ish_at);
  // printf("bas_ao2at = \n");
  // printr(bas_ao2at_dim1, bas_ao2at);
  // printf("bas_ao2sh = \n");
  // printr(bas_ao2sh_dim1, bas_ao2sh);
  // printf("bas_sh2at = \n");
  // printr(bas_sh2at_dim1, bas_sh2at);

  // for (int i = 0; i < cgto_dim1; ++i)
  // {
  //   for (int j = 0; j < cgto_dim2; ++j)
  //   {
  //     printf("cgto[%d][%d] = ", i, j);
  //     printstruct(cgto[i * cgto_dim2 + j]);
  //   }
  // }

  // printf("h0_selfenergy = \n");
  // printr(h0_selfenergy_dim1, h0_selfenergy_dim2, h0_selfenergy);
  // printf("h0_kcn = \n");
  // printr(h0_kcn_dim1, h0_kcn_dim2, h0_kcn);
  // printf("h0_kq1 = \n");
  // printr(h0_kq1_dim1, h0_kq1_dim2, h0_kq1);
  // printf("h0_kq2 = \n");
  // printr(h0_kq2_dim1, h0_kq2_dim2, h0_kq2);
  // printf("h0_hscale = \n");
  // printr(h0_hscale_dim1, h0_hscale_dim2, h0_hscale_dim3, h0_hscale_dim4, h0_hscale);

  // printf("selfenergy = \n"); printr(nelem, selfenergy);
  // printf("overlap = \n"); printr(nao, nao, overlap);
  // printf("dpint = \n"); printr(nao, nao, 3, dpint);
  // printf("qpint = \n"); printr(nao, nao, 6, qpint);
  // printf("hamiltonian = \n"); printr(nao, nao, hamiltonian);

  // get_hamiltonian<<<1,1>>>();
  // cudaDeviceSynchronize();
}
