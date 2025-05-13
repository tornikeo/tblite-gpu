#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include "utils.h"

/* 1D tensor class */ 
template <typename T>
class tensor1d_t
{
public:
  const int dim1;
  T *data;
  bool is_device; // Flag to indicate if the data is on the device
  tensor1d_t() : dim1(0), data(nullptr), is_device(false) {}
  tensor1d_t(T *data, int dim1) : data(data), dim1(dim1), is_device(false) {}
  __device__ __host__ inline const int size() const
  {
    return dim1;
  }

  __device__ __host__ inline const T &operator[](int i) const
  {
    assert(i >= 0 && i < dim1);
    return data[i];
  }

  __device__ __host__ inline T &operator[](int i)
  {
    assert(i >= 0 && i < dim1);
    return data[i];
  }

  __host__ __device__ inline void fill(const T &value)
  {
    for (int i = 0; i < dim1; ++i)
    {
      data[i] = value;
    }
  }

  __host__ inline void memset(const T &value)
  {
    CUDA_CHECK(cudaMemset(this->data, value, dim1 * sizeof(T)));
  }

  // to_device function
  tensor1d_t<T> to_device() const
  {
    assert(!is_device && "Cannot convert a device tensor to device");
    T* data = nullptr;
    const int dim1 = this->dim1;
    CUDA_CHECK(cudaMallocManaged(&data, dim1 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy((void *)data, this->data, dim1 * sizeof(T), cudaMemcpyHostToDevice));
    tensor1d_t<T> device_tensor(data, dim1);
    device_tensor.is_device = true;
    return device_tensor;
  }

  __host__ __device__ inline void print() const
  {
    printf("(%i)\n", dim1);
    printf("[");
    for (int i = 0; i < dim1; ++i)
    {
      printf("%f, ", static_cast<double>(data[i]));
    }
    printf("]\n");
  }
};

/* 2D tensor class */
template <typename T>
class tensor2d_t
{
public:
  int dim1, dim2;
  T *data;
  bool is_device; // Flag to indicate if the data is on the device

  tensor2d_t() : dim1(0), dim2(0), data(nullptr), is_device(false) {}

  tensor2d_t(T *data, int dim1, int dim2)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->is_device = false;
    this->data = data;
  }

  __device__ __host__ inline const int size() const
  {
    return dim1 * dim2;
  }

  __device__ __host__ inline T &operator()(int i, int j)
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    return data[i * dim2 + j];
  }

  __device__ __host__ inline const T &operator()(int i, int j) const
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    return data[i * dim2 + j];
  }

  __host__ __device__ inline void fill(const T &value)
  {
    const int size = dim1 * dim2;
    for (int i = 0; i < size; ++i)
    {
      data[i] = value;
    }
  }

  __host__ inline void memset(const T &value)
  {
    CUDA_CHECK(cudaMemset(data, value, dim1 * dim2 * sizeof(T)));
  }

  // to_device function
  tensor2d_t<T> to_device() const
  {
    assert(!is_device && "Cannot convert a device tensor to device");
    T* data = nullptr;
    const int dim1 = this->dim1;
    const int dim2 = this->dim2;
    CUDA_CHECK(cudaMallocManaged(&data, dim1 * dim2 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy((void *)data, this->data, dim1 * dim2 * sizeof(T), cudaMemcpyHostToDevice));
    tensor2d_t<T> device_tensor(data, dim1, dim2);
    device_tensor.is_device = true;
    return device_tensor;
  }


  __host__ __device__ inline void print() const
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
      printf(i == dim1 - 1 ? "]" : "],\n ");
    }
    printf("]\n");
  }
};

/* 3D tensor class */
template <typename T>
class tensor3d_t
{
public:
  int dim1, dim2, dim3;
  T *data;
  bool is_device; // Flag to indicate if the data is on the device

  tensor3d_t() : dim1(0), dim2(0), dim3(0), data(nullptr), is_device(false) {}

  tensor3d_t(T *data, int dim1, int dim2, int dim3)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->is_device = false;
    this->data = data;
  }

  __device__ __host__ inline const int size() const
  {
    return dim1 * dim2 * dim3;
  }

  __device__ __host__ inline T &operator()(int i, int j, int k)
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    return data[i * dim2 * dim3 + j * dim3 + k];
  }

  __device__ __host__ inline const T &operator()(int i, int j, int k) const
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    return data[i * dim2 * dim3 + j * dim3 + k];
  }

  __host__ __device__ inline void fill(const T &value)
  {
    const int size = dim1 * dim2 * dim3;
    for (int i = 0; i < size; ++i)
    {
      data[i] = value;
    }
  }

  __host__ inline void memset(const T &value)
  {
    CUDA_CHECK(cudaMemset(data, value, dim1 * dim2 * dim3 * sizeof(T)));
  }

  // to_device function
  tensor3d_t<T> to_device() const
  {
    assert(!is_device && "Cannot convert a device tensor to device");
    T* data = nullptr;
    const int dim1 = this->dim1;
    const int dim2 = this->dim2;
    const int dim3 = this->dim3;
    CUDA_CHECK(cudaMallocManaged(&data, dim1 * dim2 * dim3 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy((void *)data, this->data, dim1 * dim2 * dim3 * sizeof(T), cudaMemcpyHostToDevice));
    tensor3d_t<T> device_tensor(data, dim1, dim2, dim3);
    device_tensor.is_device = true;
    return device_tensor;
  }

  __host__ __device__ inline void print() const
  {
    printf("(%i, %i, %i)\n", dim1, dim2, dim3);
    printf("[");
    for (int i = 0; i < dim1; ++i)
    {
      printf("[");
      for (int j = 0; j < dim2; ++j)
      {
        printf(" [");
        for (int k = 0; k < dim3; ++k)
        {
          printf("%f, ", static_cast<double>(data[i * dim2 * dim3 + j * dim3 + k]));
        }
        printf(j == dim2 - 1 ? "]" : "],\n ");
      }
      printf(i == dim1 - 1 ? "]" : "],\n ");
    }
    printf("]\n");
  }
};

/* 4D tensor class */
template <typename T>
class tensor4d_t
{
public:
  int dim1, dim2, dim3, dim4;
  T *data;
  bool is_device; // Flag to indicate if the data is on the device

  tensor4d_t() : dim1(0), dim2(0), dim3(0), dim4(0), data(nullptr), is_device(false) {}

  tensor4d_t(T *data, int dim1, int dim2, int dim3, int dim4)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->dim4 = dim4;
    this->is_device = false;
    this->data = data;
  }
  
  __device__ __host__ inline const int size() const
  {
    return dim1 * dim2 * dim3 * dim4;
  }

  __device__ __host__ inline T &operator()(int i, int j, int k, int l)
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3 && l >= 0 && l < dim4);
    return data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l];
  }
  
  __device__ __host__ inline const T &operator()(int i, int j, int k, int l) const
  {
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3 && l >= 0 && l < dim4);
    return data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l];
  }

  __host__ __device__ inline void fill(const T &value)
  {
    const int size = dim1 * dim2 * dim3 * dim4;
    for (int i = 0; i < size; ++i)
    {
      data[i] = value;
    }
  }

  __host__ inline void memset(const T &value)
  {
    CUDA_CHECK(cudaMemset(data, value, dim1 * dim2 * dim3 * dim4 * sizeof(T)));
  }

  tensor4d_t<T> to_device() const
  {
    assert(!is_device && "Cannot convert a device tensor to device");
    T* data = nullptr;
    const int dim1 = this->dim1;
    const int dim2 = this->dim2;
    const int dim3 = this->dim3;
    const int dim4 = this->dim4;
    CUDA_CHECK(cudaMallocManaged(&data, dim1 * dim2 * dim3 * dim4 * sizeof(T)));
    CUDA_CHECK(cudaMemcpy((void *)data, this->data, dim1 * dim2 * dim3 * dim4 * sizeof(T), cudaMemcpyHostToDevice));
    tensor4d_t<T> device_tensor(data, dim1, dim2, dim3, dim4);
    device_tensor.is_device = true;
    return device_tensor;
  }

  __host__ __device__ inline void print() const
  {
    printf("(%i, %i, %i, %i) ", dim1, dim2, dim3, dim4);
    printf("[");
    for (int i = 0; i < dim1; ++i)
    {
      printf("[");
      for (int j = 0; j < dim2; ++j)
      {
        printf("[");
        for (int k = 0; k < dim3; ++k)
        {
          printf("[");
          for (int l = 0; l < dim4; ++l)
          {
            printf("%f, ", static_cast<double>(data[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l]));
          }
          printf(k == dim3 - 1 ? "]" : "],\n ");
        }
        printf("]\n");
      }
      printf("]\n");
    }
    printf("]\n");
  }
};
#endif