#include "../common/taffo_sycl_tests.hpp"

void gemm_naive(float alpha, float beta, const float *A, const float *B, float *C, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0;
            for (size_t l = 0; l < k; l++)
                sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

void gemm_sycl(sycl::queue queue,
               sycl::buffer<float> &alpha_buf,
               sycl::buffer<float> &beta_buf,
               sycl::buffer<float> &A_buf,
               sycl::buffer<float> &B_buf,
               sycl::buffer<float> &C_buf,
               size_t m, size_t k, size_t n) {
    queue.submit([&](sycl::handler &cgh) {
        // Get access to buffers on device
        sycl::accessor alpha_acc(alpha_buf, cgh, sycl::read_only);
        sycl::accessor beta_acc(beta_buf, cgh, sycl::read_only);
        sycl::accessor A_acc(A_buf, cgh, sycl::read_only);
        sycl::accessor B_acc(B_buf, cgh, sycl::read_only);
        sycl::accessor C_acc(C_buf, cgh, sycl::read_write);
        cgh.parallel_for(
                m, [=](sycl::id<1> itemID) {
                    // Kernel code
                    size_t i = itemID.get(0);
                    for (size_t j = 0; j < n; j++) {
                        __attribute__((annotate("scalar(range(0, 12001000) bufferid('C') final)"))) float sum = 0;
                        for (size_t l = 0; l < k; l++)
                            sum += A_acc[i * k + l] * B_acc[l * n + j];
                        C_acc[i * n + j] = alpha_acc[0] * sum + beta_acc[0] * C_acc[i * n + j];
                    }
                });
    });
}

#define ALPHA_MIN 0
#define ALPHA_MAX 10
#define BETA_MIN 0
#define BETA_MAX 20
#define A_MIN 0
#define A_MAX 30
#define B_MIN 0
#define B_MAX 40
#define C_INIT_MIN 0
#define C_INIT_MAX 50
#define ALPHA_ANNOTATION __attribute__((annotate("sycl_accessor('alpha_acc') scalar(range(0, 10) bufferid('alpha') final)")))
#define BETA_ANNOTATION __attribute__((annotate("sycl_accessor('beta_acc') scalar(range(0, 20) bufferid('beta') final)")))
#define A_ANNOTATION __attribute__((annotate("sycl_accessor('A_acc') scalar(range(0, 30) bufferid('A') final)")))
#define B_ANNOTATION __attribute__((annotate("sycl_accessor('B_acc') scalar(range(0, 40) bufferid('B') final)")))
#define C_ANNOTATION __attribute__((annotate("sycl_accessor('C_acc') scalar(range(0, 12001000) bufferid('C') final)")))

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <seed> <m> <k> <n> <warmupIterations> <outputFile>" << "\n";
        return 1;
    }
    size_t seed = std::atoi(argv[1]);
    size_t m = std::atoi(argv[2]);
    size_t k = std::atoi(argv[3]);
    size_t n = std::atoi(argv[4]);
    size_t warmupIterations = std::atoi(argv[5]);
    std::string outputFileName = argv[6];

    RNG rng(seed);

    std::ofstream outFile(outputFileName);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << outputFileName << "\n";
        return 1;
    }

    ALPHA_ANNOTATION float alpha;
    BETA_ANNOTATION float beta;
    A_ANNOTATION float *A = (float *) malloc(sizeof(float) * m * k);
    B_ANNOTATION float *B = (float *) malloc(sizeof(float) * k * n);
    C_ANNOTATION float *C = (float *) malloc(sizeof(float) * m * n);

    // Function called once just to let Taffo compute the right ranges on host
    // in order to have them on device with the bufferid system
    gemm_naive(alpha, beta, A, B, C, m, k, n);

    size_t executionTime = 0;

    sycl::queue queue;

    std::cout << "Warmup: " << warmupIterations << " iterations\n";
    for (size_t i = 0; i < warmupIterations + 1; i++) {
        if (i == warmupIterations)
            std::cout << "Warmup finished\n\n";

        // Initialize data
        alpha = rng.randomInRange(ALPHA_MIN, ALPHA_MAX);
        beta = rng.randomInRange(BETA_MIN, BETA_MAX);
        for (size_t j = 0; j < m * k; j++)
            A[j] = rng.randomInRange(A_MIN, A_MAX);
        for (size_t j = 0; j < k * n; j++)
            B[j] = rng.randomInRange(B_MIN, B_MAX);
        for (size_t j = 0; j < m * n; j++)
            C[j] = rng.randomInRange(C_INIT_MIN, C_INIT_MAX);

        std::cout << "Submitting work to device\n";
        auto start = std::chrono::high_resolution_clock::now();
        {
            sycl::buffer<float> alpha_buf(&alpha, 1);
            sycl::buffer<float> beta_buf(&beta, 1);
            sycl::buffer<float> A_buf(A, m * k);
            sycl::buffer<float> B_buf(B, k * n);
            sycl::buffer<float> C_buf(C, m * n);
            gemm_sycl(queue, alpha_buf, beta_buf, A_buf, B_buf, C_buf, m, k, n);
        }
        // End of scope destroys the buffers: a trigger to wait for the queue to complete the work and copy back data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        executionTime = duration.count();

        std::cout << "End of device work\n"
                  << "Execution time: " << executionTime << " ms\n";
    }
    std::cout << "Finished\n\n";

    outFile << "Time: " << executionTime << " ms\n";
    outFile << "Len: " << m * n << "\n";
    for (int i = 0; i < m * n; i++)
        outFile << i << " " << C[i] << "\n";
    outFile.close();

    free(A);
    free(B);
    free(C);
    return 0;
}
