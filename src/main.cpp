#include <fmt/core.h>
#include <fmt/chrono.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <ranges>

namespace rv = std::ranges::views;
using namespace std::chrono;

using Instant = high_resolution_clock;

/// Preset dimensions of the problem.
enum Preset: size_t {
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

static constexpr size_t ORDER = 16;
static constexpr size_t HALF_ORDER = ORDER / 2;

static constexpr size_t DIMX = PRESET;
static constexpr size_t DIMY = PRESET;
static constexpr size_t DIMZ = PRESET;
static constexpr size_t MAXX = DIMX + ORDER;
static constexpr size_t MAXY = DIMY + ORDER;
static constexpr size_t MAXZ = DIMZ + ORDER;

static constexpr double EXPONENT = 17.0;

// Elevate double-precision floating-point value to power `n`, similar to Rust's `f64::powi`
auto powi_f64 = [](double value, int n) {
    if (n == 0) {
        return 1.0;
    }
    double result = 1.0;
    for (auto _: rv::iota(0, n)) {
        result *= value;
    }
    return n >= 0 ? result : 1.0 / result;
};

auto main(int32_t argc, char* argv[]) -> int32_t {
    Kokkos::initialize(argc, argv);
    {
        // Exponents array initialization
        auto exponents = []() {
            std::array<double, HALF_ORDER> tmp {};
            for (auto [val, n]: rv::zip(tmp, rv::iota(0uz, HALF_ORDER))) {
                val = 1.0 / powi_f64(EXPONENT, n + 1);
            }
            return tmp;
        }();

        // Tensors allocation
        Kokkos::View<double[MAXX][MAXY][MAXZ]> A("A");
        Kokkos::View<double[MAXX][MAXY][MAXZ]> B("B");
        Kokkos::View<double[MAXX][MAXY][MAXZ]> C("C");

        // Tensors initialization
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {MAXX, MAXY, MAXZ}),
            KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                A(x, y, z) = 0.0;
                B(x, y, z) = sin(static_cast<double>(z) * cos(static_cast<double>(y) + 0.817)
                     * cos(static_cast<double>(x) + 0.311) + 0.613);
                C(x, y, z) = 0.0;
        });
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {HALF_ORDER, HALF_ORDER, HALF_ORDER},
                {DIMX + HALF_ORDER, DIMY + HALF_ORDER, DIMZ + HALF_ORDER}
            ),
            KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                A(x, y, z) = 1.0;
        });

        // Main loop
        for (auto _iter: rv::iota(1, NB_ITERATIONS + 1)) {
            fmt::print("#{} | ", _iter);

            // Benchmarked function: Jacobi iteration
            auto start = Instant::now();
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {HALF_ORDER, HALF_ORDER, HALF_ORDER},
                    {DIMX + HALF_ORDER, DIMY + HALF_ORDER, DIMZ + HALF_ORDER}
                ),
                KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                    A(x, y, z) *= B(x, y, z);
            });
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {HALF_ORDER, HALF_ORDER, HALF_ORDER},
                    {DIMX + HALF_ORDER, DIMY + HALF_ORDER, DIMZ + HALF_ORDER}
                ),
                KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                    double acc = A(x, y, z);
                    for (auto [n, exp]: rv::zip(rv::iota(1uz, HALF_ORDER + 1), exponents)) {
                        acc += A(x - n, y, z) * exp;
                        acc += A(x + n, y, z) * exp;
                        acc += A(x, y - n, z) * exp;
                        acc += A(x, y + n, z) * exp;
                        acc += A(x, y, z - n) * exp;
                        acc += A(x, y, z + n) * exp;
                    }
                    C(x, y, z) = acc;
            });
            std::swap(A, C);
            auto stop = Instant::now();

            // Output iteration results
            for (auto idx: rv::iota(0, 5)) {
                fmt::print("{:<+018.15} ", A(DIMX / 2 + idx, DIMY / 2 + idx, DIMZ / 2 + idx));
            }
            fmt::print("\t| {:>6}\n", duration_cast<microseconds>(stop - start));
        }
    }
    Kokkos::finalize();
    return 0;
}
