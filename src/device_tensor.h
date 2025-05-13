#ifndef DEVICE_TENSOR_H
#define DEVICE_TENSOR_H
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

template <typename T>
class device_tensor2d_t
{
public:
  int dim1, dim2;
  T *data;
  __device__ device_tensor2d_t() : dim1(0), dim2(0), data(nullptr) {}
  __device__ device_tensor2d_t(int dim1, int dim2)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->data = (T *)xmalloc(dim1 * dim2 * sizeof(T));
  }
  __device__ ~device_tensor2d_t()
  {
    free(data);
  }

  __device__ device_tensor2d_t(const device_tensor2d_t &other)
  {
    printf("Warning: device_tensor2d_t copy constructor called.\n");
    assert(false && "Copy constructor should not be used!");
  }
  __device__ device_tensor2d_t &operator=(const device_tensor2d_t &other)
  {
    printf("Warning: device_tensor2d_t assignment operator called.\n");
    assert(false && "Assignment operator should not be used!");
    return *this;
  }

  __device__ inline T &operator()(int i, int j)
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    return data[i * dim2 + j];
  }
  __device__ inline const T &operator()(int i, int j) const
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    return data[i * dim2 + j];
  }
  __device__ inline void fill(const T &value)
  {
    const int size = dim1 * dim2;
    for (int i = 0; i < size; ++i)
    {
      data[i] = value;
    }
  }
  __device__ inline void print() const
  {
    printf("(%i, %i)\n", dim1, dim2);
    printf("[");
    for (int i = 0; i < dim1; ++i)
    {
      printf("[");
      for (int j = 0; j < dim2; ++j)
      {
        printf("%f, ", static_cast<double>(data[i * dim2 + j]));
      }
      printf(i < dim1 - 1 ? "], \n " : "]");
    }
    printf("]\n");
  }
};


/* 3D tensor class */
template <typename T>
class device_tensor3d_t
{
public:
  int dim1, dim2, dim3;
  T *data;
  __device__ device_tensor3d_t() : dim1(0), dim2(0), dim3(0), data(nullptr) {}
  __device__ device_tensor3d_t(int dim1, int dim2, int dim3)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->data = (T *)xmalloc(dim1 * dim2 * dim3 * sizeof(T));
  }
  __device__ ~device_tensor3d_t()
  {
    free(data);
  }

  __device__ device_tensor3d_t(const device_tensor3d_t &other)
  {
    // Warn on use, we shouldn't need this!
    printf("Warning: device_tensor3d_t copy constructor called.\n");
    assert(false && "Copy constructor should not be used!");
  }
  __device__ device_tensor3d_t &operator=(const device_tensor3d_t &other)
  {
    /* Warn on use, we shouldn't need this! */
    printf("Warning: device_tensor3d_t assignment operator called.\n");
    assert(false && "Assignment operator should not be used!");
    return *this;
  }
  __device__ inline T &operator()(int i, int j, int k)
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    return data[i * dim2 * dim3 + j * dim3 + k];
  }
  __device__ inline const T &operator()(int i, int j, int k) const
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    return data[i * dim2 * dim3 + j * dim3 + k];
  }
  __device__ inline void fill(const T &value)
  {
    const int size = dim1 * dim2 * dim3;
    for (int i = 0; i < size; ++i)
    {
      data[i] = value;
    }
  }
  __device__ inline void print() const
  {
    printf("(%i, %i, %i)\n", dim1, dim2, dim3);
    printf("[");
    for (int i = 0; i < dim1; ++i)
    {
      printf("[");
      for (int j = 0; j < dim2; ++j)
      {
        printf("[");
        for (int k = 0; k < dim3; ++k)
        {
          printf("%f, ", static_cast<double>(data[i * dim2 * dim3 + j * dim3 + k]));
        }
        printf("], ");
      }
      printf("], ");
    }
    printf("]\n");
  }
};

#endif