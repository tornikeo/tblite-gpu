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
    assert(i >= 0 && i < dim1);
    return data[i];
  }

  __device__ __host__ inline T &operator[](int i)
  {
    assert(i >= 0 && i < dim1);
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

  __host__ __device__ 
  inline void print() const
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

  __host__ __device__
  inline void print() const
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

  tensor3d_t(const T *data, int dim1, int dim2, int dim3)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->is_device = false;
    this->data = (T *)malloc(dim1 * dim2 * dim3 * sizeof(T));
    for (int i = 0; i < dim1 * dim2 * dim3; ++i)
    {
      this->data[i] = data[i];
    }
  }

  ~tensor3d_t()
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

  tensor3d_t(const tensor3d_t &other)
  {
    this->dim1 = other.dim1;
    this->dim2 = other.dim2;
    this->dim3 = other.dim3;
    this->is_device = other.is_device;
    if (is_device)
    {
      cudaMalloc(&this->data, dim1 * dim2 * dim3 * sizeof(T));
      cudaMemcpy(this->data, other.data, dim1 * dim2 * dim3 * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    else
    {
      this->data = (T *)malloc(dim1 * dim2 * dim3 * sizeof(T));
      for (int i = 0; i < dim1 * dim2 * dim3; ++i)
      {
        this->data[i] = other.data[i];
      }
    }
  }
  tensor3d_t &operator=(const tensor3d_t &other)
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
      this->dim3 = other.dim3;
      this->is_device = other.is_device;

      if (is_device)
      {
        cudaMalloc(&this->data, dim1 * dim2 * dim3 * sizeof(T));
        cudaMemcpy(this->data, other.data, dim1 * dim2 * dim3 * sizeof(T), cudaMemcpyDeviceToDevice);
      }
      else
      {
        this->data = (T *)malloc(dim1 * dim2 * dim3 * sizeof(T));
        for (int i = 0; i < dim1 * dim2 * dim3; ++i)
        {
          this->data[i] = other.data[i];
        }
      }
    }
    return *this;
  }
  // to_device function
  tensor3d_t<T> to_device() const
  {
    tensor3d_t<T> device_tensor;
    device_tensor.is_device = true;
    device_tensor.dim1 = dim1;
    device_tensor.dim2 = dim2;
    device_tensor.dim3 = dim3;
    cudaMalloc(&device_tensor.data, dim1 * dim2 * dim3 * sizeof(T));
    cudaMemcpy(device_tensor.data, this->data, dim1 * dim2 * dim3 * sizeof(T), cudaMemcpyHostToDevice);
    return device_tensor;
  }
  // to_host function
  tensor3d_t<T> to_host() const
  {
    tensor3d_t<T> host_tensor;
    host_tensor.is_device = false;
    host_tensor.dim1 = dim1;
    host_tensor.dim2 = dim2;
    host_tensor.dim3 = dim3;
    host_tensor.data = (T *)malloc(dim1 * dim2 * dim3 * sizeof(T));
    cudaMemcpy(host_tensor.data, this->data, dim1 * dim2 * dim3 * sizeof(T), cudaMemcpyDeviceToHost);
    return host_tensor;
  }

  __host__ __device__
  inline void print() const
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

  tensor4d_t(const T *data, int dim1, int dim2, int dim3, int dim4)
  {
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    this->dim4 = dim4;
    this->is_device = false;
    this->data = (T *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(T));
    for (int i = 0; i < dim1 * dim2 * dim3 * dim4; ++i)
    {
      this->data[i] = data[i];
    }
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
  
  tensor4d_t(const tensor4d_t &other)
  {
    this->dim1 = other.dim1;
    this->dim2 = other.dim2;
    this->dim3 = other.dim3;
    this->dim4 = other.dim4;
    this->is_device = other.is_device;
    if (is_device)
    {
      cudaMalloc(&this->data, dim1 * dim2 * dim3 * dim4 * sizeof(T));
      cudaMemcpy(this->data, other.data, dim1 * dim2 * dim3 * dim4 * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    else
    {
      this->data = (T *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(T));
      for (int i = 0; i < dim1 * dim2 * dim3 * dim4; ++i)
      {
        this->data[i] = other.data[i];
      }
    }
  }

  ~tensor4d_t()
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

  tensor4d_t &operator=(const tensor4d_t &other)
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
      this->dim3 = other.dim3;
      this->dim4 = other.dim4;
      this->is_device = other.is_device;

      if (is_device)
      {
        cudaMalloc(&this->data, dim1 * dim2 * dim3 * dim4 * sizeof(T));
        cudaMemcpy(this->data, other.data, dim1 * dim2 * dim3 * dim4 * sizeof(T), cudaMemcpyDeviceToDevice);
      }
      else
      {
        this->data = (T *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(T));
        for (int i = 0; i < dim1 * dim2 * dim3 * dim4; ++i)
        {
          this->data[i] = other.data[i];
        }
      }
    }
    return *this;
  }
  // to_device function
  tensor4d_t<T> to_device() const
  {
    tensor4d_t<T> device_tensor;
    device_tensor.is_device = true;
    device_tensor.dim1 = dim1;
    device_tensor.dim2 = dim2;
    device_tensor.dim3 = dim3;
    device_tensor.dim4 = dim4;
    cudaMalloc(&device_tensor.data, dim1 * dim2 * dim3 * dim4 * sizeof(T));
    cudaMemcpy(device_tensor.data, this->data, dim1 * dim2 * dim3 * dim4 * sizeof(T), cudaMemcpyHostToDevice);
    return device_tensor;
  }
  // to_host function
  tensor4d_t<T> to_host() const
  {
    tensor4d_t<T> host_tensor;
    host_tensor.is_device = false;
    host_tensor.dim1 = dim1;
    host_tensor.dim2 = dim2;
    host_tensor.dim3 = dim3;
    host_tensor.dim4 = dim4;
    host_tensor.data = (T *)malloc(dim1 * dim2 * dim3 * dim4 * sizeof(T));
    cudaMemcpy(host_tensor.data, this->data, dim1 * dim2 * dim3 * dim4 * sizeof(T), cudaMemcpyDeviceToHost);
    return host_tensor;
  }

  __host__ __device__
  inline void print() const
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