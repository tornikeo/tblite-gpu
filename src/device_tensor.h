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
  const int dim1, dim2;
  T * data;
  __device__ device_tensor2d_t() : dim1(0), dim2(0), data(nullptr) {}
  __device__ device_tensor2d_t(const int dim1, const int dim2, T *data)
  : dim1(dim1), dim2(dim2), data(data) {}

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

  __device__ inline T &operator()(const int j, const int i, char)
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2)
    {
      printf("FORTRAN indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    #endif

    return data[(i - 1) * dim2 + (j - 1)];
  }

  __device__ inline T &operator()(const int i, const int j)
  {
    #ifndef NDEBUG
    if(i < 0 || i >= dim1 || j < 0 || j >= dim2)
    {
      printf("Indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    #endif
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    return data[i * dim2 + j];
  }

  __device__ inline const T &operator()(const int j, const int i, char) const
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2)
    {
      printf("FORTRAN indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2);
    #endif
    return data[(i - 1) * dim2 + (j - 1)];
  }

  __device__ inline const T &operator()(const int i, const int j) const
  {
    #ifndef NDEBUG
    // need more verbose error message
    if(i < 0 || i >= dim1 || j < 0 || j >= dim2)
    {
      printf("Indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    #endif
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

template <typename T, int D1, int D2>
class device_tensor2d_fixed_t
{
public:
  static constexpr int dim1 = D1;
  static constexpr int dim2 = D2;
  T * data;
  __device__ device_tensor2d_fixed_t() : data(nullptr) {}
  __device__ device_tensor2d_fixed_t(T *data) : data(data) {}

  __device__ device_tensor2d_fixed_t(const device_tensor2d_fixed_t &other)
  {
    // Warn on use, we shouldn't need this!
    printf("Warning: device_tensor2d_fixed_t copy constructor called.\n");
    assert(false && "Copy constructor should not be used!");
  }
  __device__ device_tensor2d_fixed_t &operator=(const device_tensor2d_fixed_t &other)
  {
    /* Warn on use, we shouldn't need this! */
    printf("Warning: device_tensor2d_fixed_t assignment operator called.\n");
    assert(false && "Assignment operator should not be used!");
    return *this;
  }

  __device__ inline T &operator()(const int j, const int i, char)
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2)
    {
      printf("FORTRAN indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2);
    #endif
    return data[(i - 1) * dim2 + (j - 1)];
  }

  __device__ inline T &operator()(const int i, const int j)
  {
    #ifndef NDEBUG
    if(i < 0 || i >= dim1 || j < 0 || j >= dim2)
    {
      printf("Indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    #endif
    return data[i * dim2 + j];
  }

  __device__ inline const T &operator()(const int j, const int i, char) const
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2)
    {
      printf("FORTRAN indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2);
    #endif
    return data[(i - 1) * dim2 + (j - 1)];
  }
  __device__ inline const T &operator()(const int i, const int j) const
  {
    #ifndef NDEBUG
    // need more verbose error message
    if(i < 0 || i >= dim1 || j < 0 || j >= dim2)
    {
      printf("Indexing error: (%i, %i) out of bounds for tensor of size (%i, %i)\n", i, j, dim1, dim2);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2);
    #endif
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
  const int dim1, dim2, dim3;
  T * data;
  __device__ device_tensor3d_t() : dim1(0), dim2(0), dim3(0), data(nullptr) {}

  __device__ device_tensor3d_t(const int dim1, const int dim2, const int dim3, T *data) : dim1(dim1), dim2(dim2), dim3(dim3), data(data) {}

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

  __device__ inline T &operator()(const int k, const int j, const int i, char )
  {    
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2 || k < 1 || k > dim3)
    {
      printf("FORTRAN indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", k, j, i, dim3, dim2, dim1);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2 && k >= 1 && k <= dim3);
    #endif
    return data[(i - 1) * dim2 * dim3 + (j - 1) * dim3 + (k - 1)];
  }

  __device__ inline T &operator()(const int i, const int j, const int k)
  {
    #ifndef NDEBUG
    if(i < 0 || i >= dim1 || j < 0 || j >= dim2 || k < 0 || k >= dim3)
    {
      printf("Indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", i, j, k, dim1, dim2, dim3);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    #endif
    return data[i * dim2 * dim3 + j * dim3 + k];
  }

  __device__ inline const T &operator()(const int k, const  int j, const int i, char ) const
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2 || k < 1 || k > dim3)
    {
      printf("FORTRAN indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", k, j, i, dim3, dim2, dim1);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2 && k >= 1 && k <= dim3);
    #endif
    return data[(i - 1) * dim2 * dim3 + (j - 1) * dim3 + (k - 1)];
  }

  __device__ inline const T &operator()(const int i, const  int j, const  int k) const
  {
    #ifndef NDEBUG
    if (i < 0 || i >= dim1 || j < 0 || j >= dim2 || k < 0 || k >= dim3)
    {
      printf("Indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", i, j, k, dim1, dim2, dim3);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    #endif
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
      printf("[\n");
      for (int j = 0; j < dim2; ++j)
      {
        printf("[");
        for (int k = 0; k < dim3; ++k)
        {
          printf("%f, ", static_cast<double>(data[i * dim2 * dim3 + j * dim3 + k]));
        }
        printf("], \n");
      }
      printf("], ");
    }
    printf("]\n");
  }
};

template <typename T, int D1, int D2, int D3>
class device_tensor3d_fixed_t
{
  public:
  static constexpr int dim1 = D1;
  static constexpr int dim2 = D2;
  static constexpr int dim3 = D3;
  T * data;
  __device__ device_tensor3d_fixed_t() : data(nullptr) {}
  __device__ device_tensor3d_fixed_t(T *data) : data(data) {}

  __device__ device_tensor3d_fixed_t(const device_tensor3d_fixed_t &other)
  {
    // Warn on use, we shouldn't need this!
    printf("Warning: device_tensor3d_t copy constructor called.\n");
    assert(false && "Copy constructor should not be used!");
  }
  __device__ device_tensor3d_fixed_t &operator=(const device_tensor3d_fixed_t &other)
  {
    /* Warn on use, we shouldn't need this! */
    printf("Warning: device_tensor3d_fixed_t assignment operator called.\n");
    assert(false && "Assignment operator should not be used!");
    return *this;
  }

  __device__ inline T &operator()(const int k, const int j, const int i, char )
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2 || k < 1 || k > dim3)
    {
      printf("FORTRAN indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", k, j, i, dim3, dim2, dim1);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2 && k >= 1 && k <= dim3);
    #endif
    return data[(i - 1) * dim2 * dim3 + (j - 1) * dim3 + (k - 1)];
  }

  __device__ inline T &operator()(const int i, const int j, const int k)
  {
    #ifndef NDEBUG
    if(i < 0 || i >= dim1 || j < 0 || j >= dim2 || k < 0 || k >= dim3)
    {
      printf("Indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", i, j, k, dim1, dim2, dim3);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    #endif
    return data[i * dim2 * dim3 + j * dim3 + k];
  }

  __device__ inline const T &operator()(const int k, const int j, const int i, char ) const
  {
    #ifndef NDEBUG
    if(i < 1 || i > dim1 || j < 1 || j > dim2 || k < 1 || k > dim3)
    {
      printf("FORTRAN indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", k, j, i, dim3, dim2, dim1);
      assert(false);
    }
    assert(i >= 1 && i <= dim1 && j >= 1 && j <= dim2 && k >= 1 && k <= dim3);
    #endif
    return data[(i - 1) * dim2 * dim3 + (j - 1) * dim3 + (k - 1)];
  }

  __device__ inline const T &operator()(const int i, const int j, const int k) const
  {
    #ifndef NDEBUG
    if (i < 0 || i >= dim1 || j < 0 || j >= dim2 || k < 0 || k >= dim3)
    {
      printf("Indexing error: (%i, %i, %i) out of bounds for tensor of size (%i, %i, %i)\n", i, j, k, dim1, dim2, dim3);
      assert(false);
    }
    assert(i >= 0 && i < dim1 && j >= 0 && j < dim2 && k >= 0 && k < dim3);
    #endif
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
      printf("[\n");
      for (int j = 0; j < dim2; ++j)
      {
        printf("[");
        for (int k = 0; k < dim3; ++k)
        {
          printf("%f, ", static_cast<double>(data[i * dim2 * dim3 + j * dim3 + k]));
        }
        printf("], \n");
      }
      printf("], ");
    }
    printf("]\n");
  }
};

#endif