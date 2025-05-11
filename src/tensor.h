#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>


/* 1D tensor class */ template <typename T>
class tensor1d_t
{
public:
  int dim1;
  T *data;
  bool is_device; // Flag to indicate if the data is on the device
  tensor1d_t() : dim1(0), data(nullptr), is_device(false) {}
  tensor1d_t(const T *data, int dim1)
  {
    this->dim1 = dim1;
    this->is_device = false;
    this->data = (T *)malloc(dim1 * sizeof(T));
    for (int i = 0; i < dim1; ++i)
    {
      this->data[i] = data[i];
    }
  }

  ~tensor1d_t()
  {
    if (is_device)
    {
      cudaFree(data);
    }
    else
    {
      free(data);
    }
  }

  tensor1d_t(const tensor1d_t &other)
  {
    this->dim1 = other.dim1;
    this->is_device = other.is_device;
    if (is_device)
    {
      cudaMalloc(&this->data, dim1 * sizeof(T));
      cudaMemcpy(this->data, other.data, dim1 * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    else
    {
      this->data = (T *)malloc(dim1 * sizeof(T));
      for (int i = 0; i < dim1; ++i)
      {
        this->data[i] = other.data[i];
      }
    }
  }

  tensor1d_t &operator=(const tensor1d_t &other)
  {
    if (this != &other)
    {
      if (is_device)
      {
        cudaFree(this->data);
      }
      else
      {
        free(this->data);
      }

      this->dim1 = other.dim1;
      this->is_device = other.is_device;

      if (is_device)
      {
        cudaMalloc(&this->data, dim1 * sizeof(T));
        cudaMemcpy(this->data, other.data, dim1 * sizeof(T), cudaMemcpyDeviceToDevice);
      }
      else
      {
        this->data = (T *)malloc(dim1 * sizeof(T));
        for (int i = 0; i < dim1; ++i)
        {
          this->data[i] = other.data[i];
        }
      }
    }
    return *this;
  }

  __device__ __host__ inline const T &operator[](int i) const
  {
    assert (i >= 0 && i < dim1);
    return data[i];
  }

  __device__ __host__ inline T &operator[](int i)
  {
    assert (i >= 0 && i < dim1);
    return data[i];
  }

  // to_device function
  tensor1d_t<T> to_device() const
  {
    tensor1d_t<T> device_tensor;
    device_tensor.is_device = true;
    device_tensor.dim1 = dim1;
    cudaMalloc(&device_tensor.data, dim1 * sizeof(T));
    cudaMemcpy(device_tensor.data, this->data, dim1 * sizeof(T), cudaMemcpyHostToDevice);
    return device_tensor;
  }

  // to_host function
  tensor1d_t<T> to_host() const
  {
    tensor1d_t<T> host_tensor;
    host_tensor.is_device = false;
    host_tensor.dim1 = dim1;
    host_tensor.data = (T *)malloc(dim1 * sizeof(T));
    cudaMemcpy(host_tensor.data, this->data, dim1 * sizeof(T), cudaMemcpyDeviceToHost);
    return host_tensor;
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

  tensor2d_t(const T *data, int dim1, int dim2)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->is_device = false;
    this->data = (T *)malloc(dim1 * dim2 * sizeof(T));
    for (int i = 0; i < dim1 * dim2; ++i)
    {
      this->data[i] = data[i];
    }
  }

  ~tensor2d_t()
  {
    if (is_device)
    {
      cudaFree(data);
    }
    else
    {
      free(data);
    }
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

  tensor2d_t(const tensor2d_t &other)
  {
    this->dim1 = other.dim1;
    this->dim2 = other.dim2;
    this->is_device = other.is_device;
    if (is_device)
    {
      cudaMalloc(&this->data, dim1 * dim2 * sizeof(T));
      cudaMemcpy(this->data, other.data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    else
    {
      this->data = (T *)malloc(dim1 * dim2 * sizeof(T));
      for (int i = 0; i < dim1 * dim2; ++i)
      {
        this->data[i] = other.data[i];
      }
    }
  }

  tensor2d_t &operator=(const tensor2d_t &other)
  {
    if (this != &other)
    {
      if (is_device)
      {
        cudaFree(this->data);
      }
      else
      {
        free(this->data);
      }

      this->dim1 = other.dim1;
      this->dim2 = other.dim2;
      this->is_device = other.is_device;

      if (is_device)
      {
        cudaMalloc(&this->data, dim1 * dim2 * sizeof(T));
        cudaMemcpy(this->data, other.data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToDevice);
      }
      else
      {
        this->data = (T *)malloc(dim1 * dim2 * sizeof(T));
        for (int i = 0; i < dim1 * dim2; ++i)
        {
          this->data[i] = other.data[i];
        }
      }
    }
    return *this;
  }

  // to_device function
  tensor2d_t<T> to_device() const
  {
    tensor2d_t<T> device_tensor;
    device_tensor.is_device = true;
    device_tensor.dim1 = dim1;
    device_tensor.dim2 = dim2;
    cudaMalloc(&device_tensor.data, dim1 * dim2 * sizeof(T));
    cudaMemcpy(device_tensor.data, this->data, dim1 * dim2 * sizeof(T), cudaMemcpyHostToDevice);
    return device_tensor;
  }

  // to_host function
  tensor2d_t<T> to_host() const
  {
    tensor2d_t<T> host_tensor;
    host_tensor.is_device = false;
    host_tensor.dim1 = dim1;
    host_tensor.dim2 = dim2;
    host_tensor.data = (T *)malloc(dim1 * dim2 * sizeof(T));
    cudaMemcpy(host_tensor.data, this->data, dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
    return host_tensor;
  }
};

#endif