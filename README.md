# CGG 3D stencil - Kokkos implementation

## About
This repository aims at reimplementing the [3D stencil provided by CGG](https://github.com/arm-yourself-teratec-hackathon/stencil-cgg) during the [2022 HPC Hackaton organized by Teratec](https://teratec.eu/library/pdf/evenements/2022/HPC-Hackathon-TERATEC-Information-planning-en.pdf).
The reference implementation and its results are available in the `refs` directory (along with a Makefile if you wish to build/run it yourself).

## Requirements
To build this project, you need to have the following software installed on your machine:
- a C++ compiler supporting the C++23 standard (gcc 13.2.1 recommended)
- the [`{fmt}`](https://github.com/fmtlib/fmt) library 

## Building
Kokkos is provided as an inline build inside the project (c.f. `ext/kokkos`).

To build, use the provided CMake config file:
```
cmake -S . -B build -DKokkos_ENABLE_OPENMP=ON
```
Alternatively, you can configure the Kokkos backend to NVIDIA CUDA using: `-DCMAKE_CXX_COMPILER=nvcc -DKokkos_ENABLE_CUDA=ON`

Then, build the program and run it:
```
cmake --build build [-j <NJOBS>]
build/kokkos-scratch
```
