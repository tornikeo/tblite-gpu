# tblite-gpu

This repository contains kernels for fast calculation of the `get_hamiltonian` function using NVIDIA CUDA GPUs. 

Currently, only one function, `get_hamiltonian` is "kernelized". **NOTE** `get_hamiltonian` by itself is not all that useful. Other routines are needed to be ported to a GPU for this to become useful. Additionally, only the elements up to **Argon** are supported.

# Performance comparison
By itself, the current implementation of `get_hamiltonian` has the following performance profile:

## Scalability with Atom Count
A plot demonstrating how the runtime scales with an increasing number of atoms for each hardware type:

```
y_axis time
number of atoms (h2 to an entire protein)
lines: black 1 i9-14900HX CPU
         shades of green, dark to lime
- T4
- RTX 4060 Laptop
- A100 80GB
- H100 80GB
```

## Scalability depending on molecule class

```
y_axis: time
x_axis: adjacent bars, black - cpu, green gpus
bar type: chemical type:
  long alkanes, proteins, DNA, other polymers.
```

##  Memory Usage
A graph showing the memory usage for each hardware type during the computation. This can be useful for understanding resource requirements.

```
y_axis GBs.
x_axis number of atoms.
lines: black any CPU
       blue any GPU
denoted are limits for different GPUs, on y horizontal lines
```

## Kernel Execution Time Breakdown
A pie chart for showing what takes time inside the kernel.

```

```

# Future plans

Batched kernel implementation.