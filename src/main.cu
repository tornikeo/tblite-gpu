#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "device_tensor.h"
#include "types.h"


template <typename T>
__device__ inline void transform0(int lj, int li, const device_tensor2d_t<T> &cart, device_tensor2d_t<T> &sphr)
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
      for(int i = 0; i < cart.dim2; ++i)
      {
        sphr(k, i, 0) = cart(k, i, 2) - 0.5 * (cart(k, i, 0) + cart(k, i, 1));
        sphr(k, i, 1) = s3 * cart(k, i, 4);
        sphr(k, i, 2) = s3 * cart(k, i, 5);
        sphr(k, i, 3) = s3_4 * (cart(k, i, 0) - cart(k, i, 1));
        sphr(k, i, 4) = s3 * cart(k, i, 3);
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



/*pure subroutine horizontal_shift(ae, l, cfs)
   integer, intent(in) :: l
   real(wp), intent(in) :: ae
   real(wp), intent(inout) :: cfs(*)
   select case(l)
   case(0) ! s
      continue
   case(1) ! p
      cfs(1)=cfs(1)+ae*cfs(2)
   case(2) ! d
      cfs(1)=cfs(1)+ae*ae*cfs(3)
      cfs(2)=cfs(2)+ 2*ae*cfs(3)
   case(3) ! f
      cfs(1)=cfs(1)+ae*ae*ae*cfs(4)
      cfs(2)=cfs(2)+ 3*ae*ae*cfs(4)
      cfs(3)=cfs(3)+ 3*ae*cfs(4)
   case(4) ! g
      cfs(1)=cfs(1)+ae*ae*ae*ae*cfs(5)
      cfs(2)=cfs(2)+ 4*ae*ae*ae*cfs(5)
      cfs(3)=cfs(3)+ 6*ae*ae*cfs(5)
      cfs(4)=cfs(4)+ 4*ae*cfs(5)
   end select
end subroutine horizontal_shift*/

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

/*
pure subroutine form_product(a, b, la, lb, d)
   real(wp), intent(in) :: a(*), b(*)
   integer, intent(in) :: la, lb
   real(wp), intent(inout) :: d(*)
   if(la.ge.4.or.lb.ge.4) goto 40
   if(la.ge.3.or.lb.ge.3) goto 30
   if(la.ge.2.or.lb.ge.2) goto 20
   ! <s|s> = <s>
   d(1)=a(1)*b(1)
   if(la.eq.0.and.lb.eq.0) return
   ! <s|p> = <s|*(|s>+|p>)
   !       = <s> + <p>
   d(2)=a(1)*b(2)+a(2)*b(1)
   if(la.eq.0.or.lb.eq.0) return
   ! <p|p> = (<s|+<p|)*(|s>+|p>)
   !       = <s> + <p> + <d>
   d(3)=a(2)*b(2)
   return
20 continue
   ! <s|d> = <s|*(|s>+|p>+|d>)
   !       = <s> + <p> + <d>
   d(1)=a(1)*b(1)
   d(2)=a(1)*b(2)+a(2)*b(1)
   d(3)=a(1)*b(3)+a(3)*b(1)
   if(la.eq.0.or.lb.eq.0) return
   ! <p|d> = (<s|+<p|)*(|s>+|p>+|d>)
   !       = <s> + <p> + <d> + <f>
   d(3)=d(3)+a(2)*b(2)
   d(4)=a(2)*b(3)+a(3)*b(2)
   if(la.le.1.or.lb.le.1) return
   ! <d|d> = (<s|+<p|+<d|)*(|s>+|p>+|d>)
   !       = <s> + <p> + <d> + <f> + <g>
   d(5)=a(3)*b(3)
   return
30 continue
   ! <s|f> = <s|*(|s>+|p>+|d>+|f>)
   !       = <s> + <p> + <d> + <f>
   d(1)=a(1)*b(1)
   d(2)=a(1)*b(2)+a(2)*b(1)
   d(3)=a(1)*b(3)+a(3)*b(1)
   d(4)=a(1)*b(4)+a(4)*b(1)
   if(la.eq.0.or.lb.eq.0) return
   ! <p|f> = (<s|+<p|)*(|s>+|p>+|d>+|f>)
   !       = <s> + <p> + <d> + <f> + <g>
   d(3)=d(3)+a(2)*b(2)
   d(4)=d(4)+a(2)*b(3)+a(3)*b(2)
   d(5)=a(2)*b(4)+a(4)*b(2)
   if(la.le.1.or.lb.le.1) return
   ! <d|f> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>)
   !       = <s> + <p> + <d> + <f> + <g> + <h>
   d(5)=d(5)+a(3)*b(3)
   d(6)=a(3)*b(4)+a(4)*b(3)
   if(la.le.2.or.lb.le.2) return
   ! <f|f> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>)
   !       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
   d(7)=a(4)*b(4)
   return
40 continue
   ! <s|g> = <s|*(|s>+|p>+|d>+|f>+|g>)
   !       = <s> + <p> + <d> + <f> + <g>
   d(1)=a(1)*b(1)
   d(2)=a(1)*b(2)+a(2)*b(1)
   d(3)=a(1)*b(3)+a(3)*b(1)
   d(4)=a(1)*b(4)+a(4)*b(1)
   d(5)=a(1)*b(5)+a(5)*b(1)
   if(la.eq.0.or.lb.eq.0) return
   ! <p|g> = (<s|+<p|)*(|s>+|p>+|d>+|f>+|g>)
   !       = <s> + <p> + <d> + <f> + <g> + <h>
   d(3)=d(3)+a(2)*b(2)
   d(4)=d(4)+a(2)*b(3)+a(3)*b(2)
   d(5)=d(5)+a(2)*b(4)+a(4)*b(2)
   d(6)=a(2)*b(5)+a(5)*b(2)
   if(la.le.1.or.lb.le.1) return
   ! <d|g> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>+|g>)
   !       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
   d(5)=d(5)+a(3)*b(3)
   d(6)=d(5)+a(3)*b(4)+a(4)*b(3)
   d(7)=a(3)*b(5)+a(5)*b(3)
   if(la.le.2.or.lb.le.2) return
   ! <f|g> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>+|g>)
   !       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k>
   d(7)=d(7)+a(4)*b(4)
   d(8)=a(4)*b(5)+a(5)*b(4)
   if(la.le.3.or.lb.le.3) return
   ! <g|g> = (<s|+<p|+<d|+<f|+<g|)*(|s>+|p>+|d>+|f>+|g>)
   !       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k> + <l>
   d(9)=a(5)*b(5)

end subroutine form_product
*/

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
  /*
   integer :: k, l
   real(wp) :: vi(0:maxl), vj(0:maxl), vv(0:maxl2), v1d(3, 3)

   v1d(:, :) = 0.0_wp
*/
  double vi[MAXL] = {0.0}, vj[MAXL] = {0.0}, vv[MAXL2] = {0.0}, v1d[3][3] = {0.0};
  /*do k = 1, 3
      vv(:) = 0.0_wp
      vi(:) = 0.0_wp
      vj(:) = 0.0_wp
      vi(li(k)) = 1.0_wp
      vj(lj(k)) = 1.0_wp
      call horizontal_shift(rpi(k), li(k), vi)
      call horizontal_shift(rpj(k), lj(k), vj)
      call form_product(vi, vj, li(k), lj(k), vv)
      do l = 0, li(k) + lj(k)
         v1d(k, 1) = v1d(k, 1) + s1d(l) * vv(l)
         v1d(k, 2) = v1d(k, 2) + (s1d(l+1) + rpi(k)*s1d(l)) * vv(l)
         v1d(k, 3) = v1d(k, 3) + (s1d(l+2) + 2*rpi(k)*s1d(l+1) + rpi(k)*rpi(k)*s1d(l)) * vv(l)
      end do
   end do*/
  for(int k = 0; k < 3; ++k)
  {
    for (int l = 0; l < MAXL2; ++l)
    {
      vv[l] = 0.0;
      vi[l] = 0.0;
      vj[l] = 0.0;
    }
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
  // s3d = v1d(1, 1) * v1d(2, 1) * v1d(3, 1)
  s3d = v1d[0][0] * v1d[1][0] * v1d[2][0];

  // d3d(1) = v1d(1, 2) * v1d(2, 1) * v1d(3, 1)
  // d3d(2) = v1d(1, 1) * v1d(2, 2) * v1d(3, 1)
  // d3d(3) = v1d(1, 1) * v1d(2, 1) * v1d(3, 2)
  d3d[0] = v1d[0][1] * v1d[1][0] * v1d[2][0];
  d3d[1] = v1d[0][0] * v1d[1][1] * v1d[2][0];
  d3d[2] = v1d[0][0] * v1d[1][0] * v1d[2][1];

  // q3d(1) = v1d(1, 3) * v1d(2, 1) * v1d(3, 1)
  // q3d(2) = v1d(1, 2) * v1d(2, 2) * v1d(3, 1)
  // q3d(3) = v1d(1, 1) * v1d(2, 3) * v1d(3, 1)
  // q3d(4) = v1d(1, 2) * v1d(2, 1) * v1d(3, 2)
  // q3d(5) = v1d(1, 1) * v1d(2, 2) * v1d(3, 2)
  // q3d(6) = v1d(1, 1) * v1d(2, 1) * v1d(3, 3)
  q3d[0] = v1d[0][2] * v1d[1][0] * v1d[2][0];
  q3d[1] = v1d[0][1] * v1d[1][1] * v1d[2][0];
  q3d[2] = v1d[0][0] * v1d[1][2] * v1d[2][0];
  q3d[3] = v1d[0][1] * v1d[1][0] * v1d[2][1];
  q3d[4] = v1d[0][0] * v1d[1][1] * v1d[2][1];
  q3d[5] = v1d[0][0] * v1d[1][0] * v1d[2][2];
}


/*pure subroutine transform1(lj, li, cart, sphr)
   integer, intent(in) :: li
   integer, intent(in) :: lj
   real(wp), intent(in) :: cart(:, :, :)
   real(wp), intent(out) :: sphr(:, :, :)
   integer :: k

   do k = 1, size(cart, 1)
      call transform0(lj, li, cart(k, :, :), sphr(k, :, :))
   end do
end subroutine transform1*/
template <typename T>
__device__ inline void transform1(int lj, int li, 
  const device_tensor3d_t<T> &cart, device_tensor3d_t<T> &sphr)
{
  for(int k = 0; k < cart.dim1; ++k)
  {
    /* cart is of shape tensor3d_t(mlao[cgtoi.ang], mlao[cgtoj.ang], 3); d3d.fill(0.0);
       shpr is of shape tensor3d_t(nao, nao, 3)*/
    transform0(lj, li, k, cart, sphr);
  }
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
  const double (&vec)[3],
  const double intcut,
  device_tensor2d_t<double> &overlap,
  device_tensor3d_t<double> &dpint,
  device_tensor3d_t<double> &qpint)
{
  // {
  //   printf("================== MULTIPOLE_CGTO =================\n");
  //   printf("bid %i tid %i: multipole_cgto\n", blockIdx.x, threadIdx.x);
  //   printf("%s:%d: %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);
  //   printf("Parameters\n");
  //   printf("cgtoj=\n");
  //   printstruct(cgtoj);
  //   printf("cgtoi=\n");
  //   printstruct(cgtoi);
  //   printf("r2=%f\n", r2);
  //   printf("vec=%f, %f, %f\n", vec[0], vec[1], vec[2]);
  //   printf("intcut=%f\n", intcut);
  //   printf("overlap=\n");
  //   overlap.print();
  //   printf("dpint=\n");
  //   dpint.print();
  //   printf("qpint=\n");
  //   qpint.print();
  //   printf("====================================================\n");
  // }
  /*integer, parameter :: msao(0:maxl) = [1, 3, 5, 7, 9, 11, 13]
   integer, parameter :: mlao(0:maxl) = [1, 3, 6, 10, 15, 21, 28]
   integer, parameter :: lmap(0:maxl) = [0, 1, 4, 10, 20, 35, 56]*/
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
        s1d[l] = overlap_1d(l, eab);
      double cc = cgtoi.coeff[ip] * cgtoj.coeff[jp] * pre;
      for (mli = 0; mli < mlao[cgtoi.ang]; ++mli)
      {
        for (mlj = 0; mlj < mlao[cgtoj.ang]; ++mlj)
        {
          // {
          //   printf("================== MULTIPOLE_CGTO INNER LOOP =================\n");
          //   printf("bid %i tid %i: multipole_cgto inner loop\n", blockIdx.x, threadIdx.x);
          //   printf("Parameters\n");
          //   printf("mli=%d, mlj=%d\n", mli, mlj);
          //   printf("rpi=%f, %f, %f\n", rpi[0], rpi[1], rpi[2]); printf("rpj=%f, %f, %f\n", rpj[0], rpj[1], rpj[2]);
          //   printf("cgtoj.alpha[%d]=%f, cgtoi.alpha[%d]=%f\n", jp, cgtoj.alpha[jp], ip, cgtoi.alpha[ip]);
          //   printf("lx[%i][:] = [%i, %i, %i]\n", mlj + lmap[cgtoj.ang], lx[mlj + lmap[cgtoj.ang]][0], lx[mlj + lmap[cgtoj.ang]][1], lx[mlj + lmap[cgtoj.ang]][2]);
          //   printf("lx[%i][:] = [%i, %i, %i]\n", mli + lmap[cgtoi.ang], lx[mli + lmap[cgtoi.ang]][0], lx[mli + lmap[cgtoi.ang]][1], lx[mli + lmap[cgtoi.ang]][2]);
          //   printf("==============================================================\n");
          // }
          multipole_3d(
            rpj, rpi, 
            cgtoj.alpha[jp], cgtoi.alpha[ip], 
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
  transform1(cgtoj.ang, cgtoi.ang, d3d, dpint);
  transform1(cgtoj.ang, cgtoi.ang, q3d, qpint);

  /*! remove trace from quadrupole integrals (transfrom to spherical harmonics and back)
   do mli = 1, msao(cgtoi%ang)
      do mlj = 1, msao(cgtoj%ang)
         tr = 0.5_wp * (qpint(1, mlj, mli) + qpint(3, mlj, mli) + qpint(6, mlj, mli))
         qpint(1, mlj, mli) = 1.5_wp * qpint(1, mlj, mli) - tr
         qpint(2, mlj, mli) = 1.5_wp * qpint(2, mlj, mli)
         qpint(3, mlj, mli) = 1.5_wp * qpint(3, mlj, mli) - tr
         qpint(4, mlj, mli) = 1.5_wp * qpint(4, mlj, mli)
         qpint(5, mlj, mli) = 1.5_wp * qpint(5, mlj, mli)
         qpint(6, mlj, mli) = 1.5_wp * qpint(6, mlj, mli) - tr
      end do
   end do*/

  for (mli = 0; mli < msao[cgtoi.ang]; ++mli)
  {
    for (mlj = 0; mlj < msao[cgtoj.ang]; ++mlj)
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

/*!> Shift multipole operator from Ket function (center i) to Bra function (center j),
!> the multipole operator on the Bra function can be assembled from the lower moments
!> on the Ket function and the displacement vector using horizontal shift rules.
   pure subroutine shift_operator(vec, s, di, qi, dj, qj)
      !> Displacement vector of center i and j
      real(wp),intent(in) :: vec(:)
      !> Overlap integral between basis functions
      real(wp),intent(in) :: s
      !> Dipole integral with operator on Ket function (center i)
      real(wp),intent(in) :: di(:)
      !> Quadrupole integral with operator on Ket function (center i)
      real(wp),intent(in) :: qi(:)
      !> Dipole integral with operator on Bra function (center j)
      real(wp),intent(out) :: dj(:)
      !> Quadrupole integral with operator on Bra function (center j)
      real(wp),intent(out) :: qj(:)

      real(wp) :: tr

      ! Create dipole operator on Bra function from Ket function and shift contribution
      ! due to monopol displacement
      dj(1) = di(1) + vec(1)*s
      dj(2) = di(2) + vec(2)*s
      dj(3) = di(3) + vec(3)*s

      ! For the quadrupole operator on the Bra function we first construct the shift
      ! contribution from the dipole and monopol displacement, since we have to remove
      ! the trace contribution from the shift and the moment integral on the Ket function
      ! is already traceless
      qj(1) = 2*vec(1)*di(1) + vec(1)**2*s
      qj(3) = 2*vec(2)*di(2) + vec(2)**2*s
      qj(6) = 2*vec(3)*di(3) + vec(3)**2*s
      qj(2) = vec(1)*di(2) + vec(2)*di(1) + vec(1)*vec(2)*s
      qj(4) = vec(1)*di(3) + vec(3)*di(1) + vec(1)*vec(3)*s
      qj(5) = vec(2)*di(3) + vec(3)*di(2) + vec(2)*vec(3)*s
      ! Now collect the trace of the shift contribution
      tr = 0.5_wp * (qj(1) + qj(3) + qj(6))

      ! Finally, assemble the quadrupole operator on the Bra function from the operator
      ! on the Ket function and the traceless shift contribution
      qj(1) = qi(1) + 1.5_wp * qj(1) - tr
      qj(2) = qi(2) + 1.5_wp * qj(2)
      qj(3) = qi(3) + 1.5_wp * qj(3) - tr
      qj(4) = qi(4) + 1.5_wp * qj(4)
      qj(5) = qi(5) + 1.5_wp * qj(5)
      qj(6) = qi(6) + 1.5_wp * qj(6) - tr
   end subroutine shift_operator*/
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

__global__ void get_hamiltonian_interatomic(
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
  printf("================= KERNEL I =================\n");

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
  constexpr int msao[7] = {1, 3, 5, 7, 9, 11, 13};
  double rr = 0.0, r2 = 0.0, vec[3] = {0.0}, cutoff2 = 0.0, hij = 0.0, shpoly = 0.0, dtmpj[3] = {0.0}, qtmpj[6] = {0.0};

  // allocate stmp, dtmpi, qtmpi
  device_tensor2d_t<double> stmp(msao[bas.maxl], msao[bas.maxl]);
  stmp.fill(0.0);
  device_tensor3d_t<double> dtmpi(msao[bas.maxl], msao[bas.maxl], 3);
  dtmpi.fill(0.0);
  device_tensor3d_t<double> qtmpi(msao[bas.maxl], msao[bas.maxl], 6);
  qtmpi.fill(0.0);

  {
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
  }
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
  // int printcounter = 0;
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
        // printf("ish = %d, jsh = %d, ii = %d, jj = %d\n", ish, jsh, ii, jj);
        /*call multipole_cgto(bas%cgto(jsh, jzp), bas%cgto(ish, izp), &
        & r2, vec, bas%intcut, stmp, dtmpi, qtmpi)

        shpoly = (1.0_wp + h0%shpoly(ish, izp)*rr) &
            * (1.0_wp + h0%shpoly(jsh, jzp)*rr)

        hij = 0.5_wp * (selfenergy(is+ish) + selfenergy(js+jsh)) &
            * h0%hscale(jsh, ish, jzp, izp) * shpoly

        nao = msao(bas%cgto(jsh, jzp)%ang)*/
        const auto &cgtoj = bas.cgto(jsh, jzp); /* TODO: aren't these swapped? */
        const auto &cgtoi = bas.cgto(ish, izp);
        multipole_cgto(cgtoj, cgtoi, r2, vec, bas.intcut, stmp, dtmpi, qtmpi);
        // Debug
        {
        //   printf("==================== DEBUG ====================\n");
        //   printf("cgtoj = bas.cgto(%i, %i)\n", jsh, jzp);
        //   printf("cgtoj = \n");
        //   printstruct(cgtoj);
        //   printf("cgtoi = bas.cgto(%i, %i)\n", ish, izp);
        //   printf("cgtoi = \n");
        //   printstruct(cgtoi);
        //   multipole_cgto(cgtoj, cgtoi, r2, vec, bas.intcut, stmp, dtmpi, qtmpi);
        //   printf("stmp = \n");
        //   stmp.print();
        //   printf("dtmpi = \n");
        //   dtmpi.print();
        //   printf("qtmpi = \n");
        //   qtmpi.print();
        //   printf("===================== DEBUG ====================\n");
        // if (printcounter++ > 5) return;
        // assert(false);
        }
        shpoly = (1.0 + h0.shpoly(izp, ish) * rr) *
                 (1.0 + h0.shpoly(jzp, jsh) * rr);
        hij = 0.5 * (selfenergy[is + ish] + selfenergy[js + jsh]) *
              h0.hscale(izp, jzp, ish, jsh) * shpoly;

        const int nao = msao[bas.cgto(jzp, jsh).ang];
        /*
        do iao = 1, msao(bas%cgto(ish, izp)%ang)
                     do jao = 1, nao
                        ij = jao + nao*(iao-1)
                        call shift_operator(vec, stmp(ij), dtmpi(:, ij), qtmpi(:, ij), &
                        & dtmpj, qtmpj)*/
        for(int iao = 0; iao < msao[bas.cgto(izp, ish).ang]; ++iao)
        {
          for(int jao = 0; jao < nao; ++jao)
          {
            // ij = jao + nao * iao;
            // printf("dtmpj = \n");
            // dtmpj.print();
            // printf("qtmpj = \n");
            // qtmpj.print();
            /* TODO: Implement slicing {}*/
            shift_operator(iao, jao, vec, stmp, dtmpi, qtmpi, dtmpj, qtmpj); 
            /*overlap(jj+jao, ii+iao) = overlap(jj+jao, ii+iao) &
                           + stmp(ij)*/

            if (threadIdx.x == 1)
            {
              // printf("stmp = \n");
              // stmp.print();
              // printf("dtmpi = \n");
              // dtmpi.print();
              // printf("qtmpi = \n");
              // qtmpi.print();
              // printf("dtmpj = \n");
              printf("before iat != jat; iat = %i, jat = %i; stmp = \n", iat, jat);
              stmp.print();
              printf("also, ii + iao = %d, jj + jao = %d, iao = %d, jao = %d\n", ii + iao, jj + jao, iao, jao);
            }
            atomicAdd(&overlap(ii + iao, jj + jao), stmp(iao, jao));
            /*do k = 1, 3
                           ! $omp atomic
                           dpint(k, jj+jao, ii+iao) = dpint(k, jj+jao, ii+iao) &
                              + dtmpi(k, ij)
                        end do*/
            for (int k = 0; k < 3; ++k)
              atomicAdd(&dpint(ii + iao, jj + jao, k), dtmpi(iao, jao, k));
            /*do k = 1, 6
                           ! $omp atomic
                           qpint(k, jj+jao, ii+iao) = qpint(k, jj+jao, ii+iao) &
                              + qtmpi(k, ij)
                        end do*/
            for (int k = 0; k < 6; ++k)
              atomicAdd(&qpint(ii + iao, jj + jao, k), qtmpi(iao, jao, k));
            /*                        ! $omp atomic
                        hamiltonian(jj+jao, ii+iao) = hamiltonian(jj+jao, ii+iao) &
                           + stmp(ij) * hij*/
            atomicAdd(&hamiltonian(ii + iao, jj + jao), stmp(iao, jao) * hij);
            
            /* TODO: This is a symmetrification of these matrices. Maybe this should be
            done in the outside this loop? */
            if (iat != jat) 
            {
              /*overlap(ii+iao, jj+jao) = overlap(ii+iao, jj+jao) + stmp(ij)*/
              if (threadIdx.x == 1)
              {
                printf("inside iat != jat (%i != %i); stmp = \n", iat, jat);
                stmp.print();
                printf("also, ii + iao = %d, jj + jao = %d, iao = %d, jao = %d\n", ii + iao, jj + jao, iao, jao);
              }
              atomicAdd(&overlap(jj + jao, ii + iao), stmp(iao, jao));
              /*do k = 1, 3
                              ! $omp atomic
                              dpint(k, ii+iao, jj+jao) = dpint(k, ii+iao, jj+jao) &
                                 + dtmpj(k)
                           end do*/
              for (int k = 0; k < 3; ++k)
                atomicAdd(&dpint(jj + jao, ii + iao,  k), dtmpj[k]);
              /*do k = 1, 6
                              ! $omp atomic
                              qpint(k, ii+iao, jj+jao) = qpint(k, ii+iao, jj+jao) &
                                 + qtmpj(k)
                           end do*/
              for (int k = 0; k < 6; ++k)
                atomicAdd(&qpint(jj + jao, ii + iao,  k), qtmpj[k]);
              /*! $omp atomic
                           hamiltonian(ii+iao, jj+jao) = hamiltonian(ii+iao, jj+jao) &
                              + stmp(ij) * hij*/
              atomicAdd(&hamiltonian(jj + jao, ii + iao), stmp(iao, jao) * hij);
            }
          }
        }
      }
    }
  }
  
  if(threadIdx.x == 1)
  {
    printf("================== DEBUG %i %i ==================\n", blockIdx.x, threadIdx.x);
    printf("overlap = \n");
    overlap.print();
    printf("dpint = \n");
    dpint.print();
    printf("qpint = \n");
    qpint.print();
    printf("hamiltonian = \n");
    hamiltonian.print();
  }

  printf("================= KERNEL I END =================\n");
}

__global__ void get_hamiltonian_intraatomic(
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
  printf("================= KERNEL II =================\n");
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


  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int msao[7] = {1, 3, 5, 7, 9, 11, 13};
  double rr = 0.0, r2 = 0.0, vec[3] = {0.0}, cutoff2 = 0.0, hij = 0.0, shpoly = 0.0, dtmpj[3] = {0.0}, qtmpj[6] = {0.0};

  // allocate stmp, dtmpi, qtmpi
  device_tensor2d_t<double> stmp(msao[bas.maxl], msao[bas.maxl]);
  stmp.fill(0.0);
  device_tensor3d_t<double> dtmpi(msao[bas.maxl], msao[bas.maxl], 3);
  dtmpi.fill(0.0);
  device_tensor3d_t<double> qtmpi(msao[bas.maxl], msao[bas.maxl], 6);
  qtmpi.fill(0.0);

  if (thread_id == 0)
  {
    printf("overlap = \n");
    overlap.print();
    printf("dpint = \n");
    dpint.print();
    printf("qpint = \n");
    qpint.print();
    printf("hamiltonian = \n");
    hamiltonian.print();
  }
  printf("================== KERNEL II END =================\n");
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
      tensor2d_t(mol_lattice, mol_lattice_dim1, mol_lattice_dim2),
      tensor1d_t(mol_periodic, mol_periodic_dim1),
      tensor2d_t(mol_bond, mol_bond_dim1, mol_bond_dim2)};

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
  
  // DEBUG
  // printf("at %s:%i\n", __func__, __LINE__);
  // printf("nao = %i\n", nao);
  // printf("nelem = %i\n", nelem);
  // printstruct(bas);
  // printstruct(alist);
  // printstruct(mol);
  // printstruct(h0);

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
      bas.nsh_id.to_device(),
      bas.nsh_at.to_device(),
      bas.nao_sh.to_device(),
      bas.iao_sh.to_device(),
      bas.ish_at.to_device(),
      bas.ao2at.to_device(),
      bas.ao2sh.to_device(),
      bas.sh2at.to_device(),
      bas.cgto.to_device()};

  const tb_hamiltonian d_h0{
      h0.selfenergy.to_device(),
      h0.kcn.to_device(),
      h0.kq1.to_device(),
      h0.kq2.to_device(),
      h0.hscale.to_device(),
      h0.shpoly.to_device(),
      h0.rad.to_device(),
      h0.refocc.to_device()};

  const tensor1d_t<double> d_selfenergy = tensor1d_t<double>(selfenergy, nelem).to_device();
  tensor2d_t<double> d_overlap = tensor2d_t<double>(overlap, nao, nao).to_device();
  tensor3d_t<double> d_dpint = tensor3d_t<double>(dpint, nao, nao, 3).to_device();
  tensor3d_t<double> d_qpint = tensor3d_t<double>(qpint, nao, nao, 6).to_device();
  tensor2d_t<double> d_hamiltonian = tensor2d_t<double>(hamiltonian, nao, nao).to_device();
  
  /* Zero out the arrays before the kernel starts */
  d_overlap.memset(static_cast<double>(0.0));
  d_dpint.memset(static_cast<double>(0.0));
  d_qpint.memset(static_cast<double>(0.0));
  d_hamiltonian.memset(static_cast<double>(0.0));
  
  ////////////////////////////////////////////
  // Launch kernel part I
  ////////////////////////////////////////////
  cudaEvent_t start, stop;
  float milliseconds = 0;

  {
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 * 1024 * 1024););
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    get_hamiltonian_interatomic<<<1, mol.nat>>>(
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel part I execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start); cudaEventDestroy(stop);
  }
  
  ////////////////////////////////////////////
  // Launch kernel part II
  ////////////////////////////////////////////
  {
    cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
    get_hamiltonian_intraatomic<<<1, mol.nat>>>(
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel part II execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start); cudaEventDestroy(stop);
  }
  printf("Exiting prematurely. Remove this once the kernel is working.\n");
  exit(EXIT_FAILURE);
}
