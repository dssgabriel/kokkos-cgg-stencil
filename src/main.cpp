#include <fmt/core.h>
#include <fmt/chrono.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <ranges>

using Instant = std::chrono::high_resolution_clock;

constexpr size_t ORDER = 16;
constexpr size_t HALF_ORDER = ORDER / 2;
constexpr size_t TENSOR_SIZE = 100;

constexpr size_t DIMX = TENSOR_SIZE;
constexpr size_t DIMY = TENSOR_SIZE;
constexpr size_t DIMZ = TENSOR_SIZE;
constexpr size_t MAXX = TENSOR_SIZE + ORDER;
constexpr size_t MAXY = TENSOR_SIZE + ORDER;
constexpr size_t MAXZ = TENSOR_SIZE + ORDER;

constexpr double EXPONENT = 17.0;

// Elevate double-precision floating-point value to power `n`, similar to Rust's `f64::powi`
auto powi_f64 = [](double value, int n) {
    if (n == 0) {
        return 1.0;
    }
    double result = 1.0;
    for (size_t _ = 0; _ < static_cast<size_t>(n); ++_) {
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
            for (auto [val, n]: std::ranges::zip_view(tmp, std::ranges::iota_view(0uz, HALF_ORDER))) {
                val = 1.0 / powi_f64(EXPONENT, n + 1);
            }
            return tmp;
        }();

        // Tensors allocation
        Kokkos::View<double[MAXX][MAXY][MAXZ]> A("A");
        Kokkos::View<double[MAXX][MAXY][MAXZ]> B("B");
        Kokkos::View<double[MAXX][MAXY][MAXZ]> C("C");
        // Get subviews on inner parts of the tensors
        auto sA = Kokkos::subview(
            A,
            Kokkos::make_pair(HALF_ORDER, MAXX - HALF_ORDER),
            Kokkos::make_pair(HALF_ORDER, MAXY - HALF_ORDER),
            Kokkos::make_pair(HALF_ORDER, MAXZ - HALF_ORDER)
        );
        auto sB = Kokkos::subview(
            B,
            Kokkos::make_pair(HALF_ORDER, MAXX - HALF_ORDER),
            Kokkos::make_pair(HALF_ORDER, MAXY - HALF_ORDER),
            Kokkos::make_pair(HALF_ORDER, MAXZ - HALF_ORDER)
        );
        auto sC = Kokkos::subview(
            C,
            Kokkos::make_pair(HALF_ORDER, MAXX - HALF_ORDER),
            Kokkos::make_pair(HALF_ORDER, MAXY - HALF_ORDER),
            Kokkos::make_pair(HALF_ORDER, MAXZ - HALF_ORDER)
        );

        // Tensors initialization
        Kokkos::parallel_for(
             Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {MAXX, MAXY, MAXZ}),
             KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                 B(x, y, z) = sin(static_cast<double>(z) * cos(static_cast<double>(y) + 0.817)
                     * cos(static_cast<double>(x) + 0.311) + 0.613);
        });
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {DIMX, DIMY, DIMZ}),
            KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                sA(x, y, z) = 1.0;
        });

        // Main loop
        for (auto _iter: std::ranges::iota_view(1, 6)) {
            fmt::print("#{} | ", _iter);

            // Benchmarked function: Jacobi iteration
            auto start = Instant::now();
            // Hadamar-Schur product A_tmp = (A * B)
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {DIMX, DIMY, DIMZ}),
                KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                    sA(x, y, z) *= sB(x, y, z);
            });
            // Multiply-Accumulate of C += A_tmp * exp
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {DIMX, DIMY, DIMZ}),
                KOKKOS_LAMBDA(size_t const x, size_t const y, size_t const z) {
                    double acc = sA(x, y, z);
                    for (auto n: std::ranges::iota_view(0uz, HALF_ORDER)) {
                        double exp = exponents[n];
                        acc += sA(x - n, y, z) * exp;
                        acc += sA(x + n, y, z) * exp;
                        acc += sA(x, y - n, z) * exp;
                        acc += sA(x, y + n, z) * exp;
                        acc += sA(x, y, z - n) * exp;
                        acc += sA(x, y, z + n) * exp;
                    }
                    sC(x, y, z) = acc;
            });
            // Swap pointers of A_tmp and C for next iteration (avoids tensor copy)
            std::swap(A, C);
            auto stop = Instant::now();

            // Output iteration results
            for (auto idx: std::ranges::iota_view(0, 5)) {
                fmt::print("{:<+018.15} ", A(DIMX / 2 + idx, DIMY / 2 + idx, DIMZ / 2 + idx));
            }
            fmt::print("\t| {:>6}\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
        }
    }
    Kokkos::finalize();
    return 0;
}
