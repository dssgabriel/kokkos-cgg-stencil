#include <chrono>
#include <fmt/core.h>
#include <fmt/chrono.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <sys/time.h>
#include <vector>

namespace rv = std::ranges::views;
using namespace std::chrono;
using float64_t = double;
using Instant = high_resolution_clock;

/// Preset dimensions of the problem.
enum Preset: size_t {
    Mini = 4,
    Small = 100,
    Medium = 500,
    Big = 1000,
};

#if !defined(PRESET)
#    define PRESET Small
#endif
#if !defined(NB_ITERATIONS)
#    define NB_ITERATIONS 5
#endif

static constexpr float64_t ONE_THOUSAND = 1.0e+3;
static constexpr float64_t ONE_MILLION = 1.0e+6;

static constexpr float64_t EXPONENT = 17.0;
static constexpr size_t HALF_ORDER = 8;
static constexpr size_t ORDER = 16;

static constexpr size_t DIMX = PRESET;
static constexpr size_t DIMY = PRESET;
static constexpr size_t DIMZ = PRESET;
static constexpr size_t MAXX = DIMX + ORDER;
static constexpr size_t MAXY = DIMY + ORDER;
static constexpr size_t MAXZ = DIMZ + ORDER;

static constexpr size_t XY_PLANE = MAXX * MAXY;
static constexpr size_t TENSOR_SIZE = MAXX * MAXY * MAXZ;

/// Returns an offset in the center of the tensor of dimensions [0, DIM).
[[nodiscard]] static inline
auto dim_xyz(size_t x, size_t y, size_t z) -> size_t {
    size_t const z_offset = (z + HALF_ORDER) * XY_PLANE;
    size_t const y_offset = (y + HALF_ORDER) * MAXX;
    size_t const x_offset = x + HALF_ORDER;
    return z_offset + y_offset + x_offset;
}

/// Returns an offset in the center of the tensor of dimensions [-HALF_ORDER, DIM + HALF_ORDER) but
/// in indices of [0, DIM + ORDER).
[[nodiscard]] static inline
auto tensor_xyz(size_t x, size_t y, size_t z) -> size_t {
    size_t const z_offset = z * XY_PLANE;
    size_t const y_offset = y * MAXX;
    size_t const x_offset = x;
    return z_offset + y_offset + x_offset;
}

auto print_tensor(std::vector<double> const& T) -> void {
    for (auto x: rv::iota(0uz, MAXX)) {
        for (auto y: rv::iota(0uz, MAXY)) {
            for (auto z: rv::iota(0uz, MAXZ)) {
                fmt::print("{} ", T[tensor_xyz(x, y, z)]);
            }
            fmt::print("\n");
        }
        fmt::print("\n\n");
    }
}

/// Initializes the tensors for the problem to solve.
///
/// Optimizing this function is not part of the exercise. It can be optimized but is not part of the
/// measured code section and does not influence performance.
/// 
/// @param A The input data tensor, with it scenter cells initialized to 1.0 and its ghost cells
///          initialized to 0.0.
/// @param B The input constant tensor, with all its cells initialized as follows:
///          sin(z * cos(y + 0.817) * cos(x + 0.311) + 0.613)
auto initialize_tensors(
    std::vector<float64_t>& A,
    std::vector<float64_t>& B
) -> void {
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t z = 0; z < MAXZ; ++z) {
            for (size_t y = 0; y < MAXY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < MAXX; ++x) {
                    B[tensor_xyz(x, y, z)] = sin(z * cos(y + 0.817) * cos(x + 0.311) + 0.613);
                }
            }
        }

        #pragma omp for
        for (size_t z = 0; z < DIMZ; ++z) {
            for (size_t y = 0; y < DIMY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < DIMX; ++x) {
                    A[dim_xyz(x, y, z)] = 1.0;
                }
            }
        }
    } // pragma omp parallel
}

/// Performs the stencil operation on the tensors.
auto jacobi_iteration(
    std::vector<float64_t>& A,
    std::vector<float64_t> const& B,
    std::vector<float64_t>& C,
    std::array<float64_t, HALF_ORDER> const& exponents
) -> void {
    #pragma omp parallel schedule(dynamic)
    {
        #pragma omp for
        for (size_t z = 0; z < DIMZ; ++z) {
            for (size_t y = 0; y < DIMY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < DIMX; ++x) {
                    size_t const xyz = dim_xyz(x, y, z);
                    A[xyz] *= B[xyz];
                }
            }
        }

        #pragma omp for
        for (size_t z = 0; z < DIMZ; ++z) {
            for (size_t y = 0; y < DIMY; ++y) {
                size_t const yz = (z + HALF_ORDER) * XY_PLANE + (y + HALF_ORDER) * MAXX + HALF_ORDER;
                #pragma omp simd
                for (size_t x = 0; x < DIMX; ++x) {
                    size_t const xz = (z + HALF_ORDER) * XY_PLANE + HALF_ORDER * MAXX + (x + HALF_ORDER);
                    size_t const xy = HALF_ORDER * XY_PLANE + (y + HALF_ORDER) * MAXX + (x + HALF_ORDER);
                    size_t const xyz = dim_xyz(x, y, z);

                    float64_t acc = A[xyz];
                    acc += A[(x - 1) + yz] * exponents[0];
                    acc += A[(x - 2) + yz] * exponents[1];
                    acc += A[(x - 3) + yz] * exponents[2];
                    acc += A[(x - 4) + yz] * exponents[3];
                    acc += A[(x - 5) + yz] * exponents[4];
                    acc += A[(x - 6) + yz] * exponents[5];
                    acc += A[(x - 7) + yz] * exponents[6];
                    acc += A[(x - 8) + yz] * exponents[7];
                    acc += A[(x + 1) + yz] * exponents[0];
                    acc += A[(x + 2) + yz] * exponents[1];
                    acc += A[(x + 3) + yz] * exponents[2];
                    acc += A[(x + 4) + yz] * exponents[3];
                    acc += A[(x + 5) + yz] * exponents[4];
                    acc += A[(x + 6) + yz] * exponents[5];
                    acc += A[(x + 7) + yz] * exponents[6];
                    acc += A[(x + 8) + yz] * exponents[7];
                    acc += A[(y - 1) * MAXX + xz] * exponents[0];
                    acc += A[(y - 2) * MAXX + xz] * exponents[1];
                    acc += A[(y - 3) * MAXX + xz] * exponents[2];
                    acc += A[(y - 4) * MAXX + xz] * exponents[3];
                    acc += A[(y - 5) * MAXX + xz] * exponents[4];
                    acc += A[(y - 6) * MAXX + xz] * exponents[5];
                    acc += A[(y - 7) * MAXX + xz] * exponents[6];
                    acc += A[(y - 8) * MAXX + xz] * exponents[7];
                    acc += A[(y + 1) * MAXX + xz] * exponents[0];
                    acc += A[(y + 2) * MAXX + xz] * exponents[1];
                    acc += A[(y + 3) * MAXX + xz] * exponents[2];
                    acc += A[(y + 4) * MAXX + xz] * exponents[3];
                    acc += A[(y + 5) * MAXX + xz] * exponents[4];
                    acc += A[(y + 6) * MAXX + xz] * exponents[5];
                    acc += A[(y + 7) * MAXX + xz] * exponents[6];
                    acc += A[(y + 8) * MAXX + xz] * exponents[7];
                    acc += A[(z - 1) * XY_PLANE + xy] * exponents[0];
                    acc += A[(z - 2) * XY_PLANE + xy] * exponents[1];
                    acc += A[(z - 3) * XY_PLANE + xy] * exponents[2];
                    acc += A[(z - 4) * XY_PLANE + xy] * exponents[3];
                    acc += A[(z - 5) * XY_PLANE + xy] * exponents[4];
                    acc += A[(z - 6) * XY_PLANE + xy] * exponents[5];
                    acc += A[(z - 7) * XY_PLANE + xy] * exponents[6];
                    acc += A[(z - 8) * XY_PLANE + xy] * exponents[7];
                    acc += A[(z + 1) * XY_PLANE + xy] * exponents[0];
                    acc += A[(z + 2) * XY_PLANE + xy] * exponents[1];
                    acc += A[(z + 3) * XY_PLANE + xy] * exponents[2];
                    acc += A[(z + 4) * XY_PLANE + xy] * exponents[3];
                    acc += A[(z + 5) * XY_PLANE + xy] * exponents[4];
                    acc += A[(z + 6) * XY_PLANE + xy] * exponents[5];
                    acc += A[(z + 7) * XY_PLANE + xy] * exponents[6];
                    acc += A[(z + 8) * XY_PLANE + xy] * exponents[7];
                    C[xyz] = acc;
                }
            }
        }
    } // pragma omp parallel

    // Tensor copy: A <- C
    A.swap(C);
}

auto main() -> int32_t {
    std::vector<float64_t> A(TENSOR_SIZE, 0.0);
    std::vector<float64_t> B(TENSOR_SIZE, 0.0);
    std::vector<float64_t> C(TENSOR_SIZE, 0.0);
    constexpr auto exponents = [] {
        std::array<float64_t, HALF_ORDER> _{};
        for (size_t i = 0; i < HALF_ORDER; ++i) {
            _[i] = 1.0 / pow(EXPONENT, i + 1);
        }
        return _;
    }();

    initialize_tensors(A, B);

    for (size_t i = 0; i < NB_ITERATIONS; ++i) {
        fmt::print("#{} | ", i + 1);
        auto start = Instant::now();
        jacobi_iteration(A, B, C, exponents);
        auto stop = Instant::now();

        // Output iteration results
        for (auto idx: rv::iota(0, 5)) {
            fmt::print("{:<+018.15} ", A[dim_xyz(DIMX / 2 + idx, DIMY / 2 + idx, DIMZ / 2 + idx)]);
        }
        fmt::print("\t| {:>6}\n", duration_cast<microseconds>(stop - start));
    }

    return 0;
}
