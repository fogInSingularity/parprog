#include "algorithms.hpp"

#include <hwy/targets.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace matmul;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class F> void must_throw(F operation, const std::string& message) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_products() {
    std::size_t checks = 0;
    for (auto threads : {1U, 2U, 4U}) {
        ThreadPool pool(threads);
        for (auto n : {1U, 2U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 33U, 63U, 65U}) {
            Matrix a(n);
            Matrix b(n);
            Matrix c(n);
            for (int pattern = 0; pattern < 6; ++pattern) {
                const auto range = pattern == 5 ? InputRange::positive : InputRange::signed_unit;
                a.randomize(17, range);
                b.randomize(91, range);
                if (pattern == 0) {
                    a.fill(1);
                    b.fill(1);
                }
                if (pattern == 1) {
                    a.fill(0);
                }
                if (pattern == 2) {
                    b.fill(0);
                    for (std::size_t i = 0; i < n; ++i) {
                        b(i, i) = 1;
                    }
                }
                if (pattern == 4) {
                    for (std::size_t i = 0; i < n; ++i) {
                        for (std::size_t j = 0; j < n; ++j) {
                            a(i, j) = ((i + j) % 2 == 0 ? 1.0f : -1.0f) * 0.125f;
                            b(i, j) *= 16.0f;
                        }
                    }
                }
                const auto expected = reference_product(a, b);
                for (auto algorithm : algorithms) {
                    c.fill(std::numeric_limits<float>::quiet_NaN());
                    multiply(algorithm, a, b, c, {5, 8}, pool);
                    const auto error = compare(c, expected);
                    require(error.passed,
                            std::string(name(algorithm)) + " n=" + std::to_string(n) + " threads=" +
                                std::to_string(threads) + " pattern=" + std::to_string(pattern) +
                                " abs_error=" + std::to_string(error.max_absolute));
                    ++checks;
                }
            }
        }
    }
    std::cout << checks << " matrix comparisons passed\n";
}

void test_edge_configs() {
    ThreadPool pool(4);
    Matrix a(17);
    Matrix b(17);
    Matrix c(17);
    a.randomize(1, InputRange::signed_unit);
    b.randomize(2, InputRange::signed_unit);
    const auto expected = reference_product(a, b);
    for (auto algorithm : algorithms) {
        for (auto block : {1U, 7U, 32U}) {
            for (auto cutoff : {1U, 9U, 64U}) {
                multiply(algorithm, a, b, c, {block, cutoff}, pool);
                require(compare(c, expected).passed, "Edge configuration failed");
            }
        }
    }
    must_throw([] { Matrix invalid(0); }, "Zero matrix size accepted");
    must_throw([] { Matrix invalid(std::numeric_limits<std::size_t>::max()); },
               "Overflow accepted");
    must_throw([] { ThreadPool invalid(0); }, "Zero threads accepted");
    must_throw([&] { multiply(Algorithm::naive, a, b, c, {0, 1}, pool); }, "Zero block accepted");
    must_throw([&] { multiply(Algorithm::naive, a, b, c, {1, 0}, pool); }, "Zero cutoff accepted");
    must_throw([&] { multiply(Algorithm::naive, a, b, a, {}, pool); }, "Aliasing accepted");
    Matrix wrong(3);
    must_throw([&] { multiply(Algorithm::naive, a, wrong, c, {}, pool); },
               "Size mismatch accepted");
    c.fill(std::numeric_limits<float>::quiet_NaN());
    require(!compare(c, expected).passed, "NaN considered valid");
    c.fill(std::numeric_limits<float>::infinity());
    require(!compare(c, expected).passed, "Infinity considered valid");
}

void test_thread_pool() {
    for (auto threads : {1U, 2U, 4U}) {
        ThreadPool pool(threads);
        for (auto count : {0U, 1U, 3U, 31U}) {
            std::vector<std::atomic<int>> hits(count);
            for (auto& hit : hits) {
                hit.store(0);
            }
            pool.parallel_for(count, [&](std::size_t i) { ++hits[i]; });
            for (const auto& hit : hits) {
                require(hit.load() == 1, "Lost or repeated task");
            }
        }
        bool caught = false;
        try {
            pool.parallel_for(16, [](std::size_t i) {
                if (i == 3) {
                    throw std::runtime_error("intentional worker failure");
                }
            });
        } catch (const std::runtime_error&) {
            caught = true;
        }
        require(caught, "Worker exception not propagated");
        std::atomic<int> total{0};
        pool.parallel_for(19, [&](std::size_t) { ++total; });
        require(total == 19, "Pool unusable after exception");
    }
}

void test_validation_details() {
    Matrix actual(2);
    actual.fill(1);
    const std::vector<double> expected(4, 1);
    actual(0, 1) = 2;
    actual(1, 0) = 4;
    const auto error = compare(actual, expected);
    require(!error.passed && error.mismatches == 2, "Incorrect mismatch count");
    require(error.row == 1 && error.col == 0, "Incorrect worst-element coordinates");
    require(error.actual == 4 && error.expected == 1, "Incorrect diagnostic values");
    require(error.absolute == 3 && error.max_absolute == 3, "Incorrect absolute error");
    require(error.allowed == 4e-4 && error.max_scaled == 7500, "Incorrect allowed error");

    actual.fill(1);
    require(compare(actual, expected, 0, 0).passed, "Exact comparison rejects equal values");
    actual(1, 1) = 2;
    const auto exact = compare(actual, expected, 0, 0);
    require(!exact.passed && exact.mismatches == 1 && std::isinf(exact.max_scaled),
            "Exact comparison fails to diagnose mismatch");

    actual.fill(1);
    actual(0, 1) = std::numeric_limits<float>::quiet_NaN();
    const auto nan = compare(actual, expected);
    require(!nan.passed && nan.mismatches == 1 && nan.row == 0 && nan.col == 1 &&
                std::isnan(nan.actual) && std::isinf(nan.max_scaled),
            "NaN diagnostics failed");
    must_throw([&] { compare(actual, expected, -1, 0); }, "Negative tolerance accepted");
}

void test_simd_targets() {
    const auto targets = hwy::SupportedAndGeneratedTargets();
    ThreadPool pool(4);
    Matrix a(33);
    Matrix b(33);
    Matrix c(33);
    a.randomize(123, InputRange::signed_unit);
    b.randomize(321, InputRange::signed_unit);
    const auto expected = reference_product(a, b);
    for (auto target : targets) {
        hwy::SetSupportedTargetsForTest(target);
        for (auto algorithm :
             {Algorithm::simd, Algorithm::combined, Algorithm::combined_no_strassen}) {
            multiply(algorithm, a, b, c, {32, 16}, pool);
            require(compare(c, expected).passed,
                    std::string("SIMD target failed: ") + hwy::TargetName(target) + ", " +
                        std::string(name(algorithm)));
        }
        std::cout << "PASS SIMD target " << hwy::TargetName(target) << '\n';
    }
    hwy::SetSupportedTargetsForTest(0);
}

void test_input_ranges() {
    Matrix a(33);
    Matrix same(33);
    Matrix other(33);
    for (auto range : {InputRange::positive, InputRange::signed_unit}) {
        a.randomize(42, range);
        same.randomize(42, range);
        other.randomize(43, range);
        bool different_seed_changes_data = false;
        bool has_negative = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            for (std::size_t j = 0; j < a.size(); ++j) {
                const float value = a(i, j);
                const float lower = range == InputRange::positive ? 1.0f : -1.0f;
                const float upper = range == InputRange::positive ? 2.0f : 1.0f;
                require(value >= lower && value <= upper, "Generated value outside input range");
                require(value == same(i, j), "Same seed does not reproduce data");
                require(std::trunc(value * 1024) == value * 1024, "Input is not on binary grid");
                different_seed_changes_data |= value != other(i, j);
                has_negative |= value < 0;
            }
        }
        require(different_seed_changes_data, "Different seeds generate identical matrices");
        require(has_negative == (range == InputRange::signed_unit), "Signed input coverage lost");
    }
}
}

int main() {
    try {
        test_thread_pool();
        test_products();
        test_edge_configs();
        test_validation_details();
        test_input_ranges();
        test_simd_targets();
        std::cout << "All tests passed; maximum matrix dimension: 65\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
