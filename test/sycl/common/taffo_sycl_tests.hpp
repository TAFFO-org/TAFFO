#ifndef TAFFO_SYCL_TESTS_TAFFO_SYCL_TESTS_HPP
#define TAFFO_SYCL_TESTS_TAFFO_SYCL_TESTS_HPP

#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>

#define HOST_ACCESSOR_ANNOTATION(SCALAR_STR) __attribute__((annotate( \
"struct[struct[void, struct[void, void, void, " SCALAR_STR ", void], void]]")))

class RNG {
    std::mt19937 rng;
public:
    RNG(size_t seed) : rng(seed) {}

    float randomInRange(float min, float max) {
        std::uniform_real_distribution<> distribution(min, max);
        return distribution(rng);
    }
};

bool isCorrect(float x, float xExpected, float tolerancePercentage) {
    return std::abs(xExpected - x) / xExpected < tolerancePercentage;
}

/*void copy_sycl(sycl::queue queue, float *src, float *dst, size_t len) {
    sycl::buffer<float> srcBuffer(src, len);
    sycl::buffer<float> dstBuffer(dst, len);
    queue.submit([&](sycl::handler &cgh) {
        // Get access to buffers on device
        sycl::accessor srcAccessor{srcBuffer, cgh, sycl::read_only};
        sycl::accessor dstAccessor{dstBuffer, cgh, sycl::write_only};
        cgh.parallel_for(
                dstBuffer.get_range(), [=](sycl::id<1> itemID) {
                    // Kernel code
                    dstAccessor[itemID] = srcAccessor[itemID];
                });
    });
}*/

/**
 * Reads the bits in memory pointed by ptr and writes them into str
 * Make sure the length of str is at least bits+1
 * @param ptr memory address to read
 * @param bits number of bits to read and write
 * \n(max is sizeof(long long))
 * @param str output string
 */
SYCL_EXTERNAL void toBinary(const void *ptr, size_t bits, char *str) {
    bool bit;
    const auto *llPtr = reinterpret_cast<const long long*>(ptr);
    for (int i = 0; i < bits; i++) {
        bit = *llPtr >> i & 1;
        str[bits-i-1] = bit ? '1' : '0';
    }
    str[bits] = '\0';
}

#endif //TAFFO_SYCL_TESTS_TAFFO_SYCL_TESTS_HPP
