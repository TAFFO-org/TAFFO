#include "../common/taffo_sycl_tests.hpp"

void gemv_naive(float alpha, float beta, const float *A, const float *x, float *y, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float sum = 0;
        for (size_t j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = alpha * sum + beta * y[i];
    }
}

void gemv_sycl(sycl::queue queue,
               sycl::buffer<float> &alpha_buf,
               sycl::buffer<float> &beta_buf,
               sycl::buffer<float> &A_buf,
               sycl::buffer<float> &x_buf,
               sycl::buffer<float> &y_buf,
               size_t m, size_t n) {
    queue.submit([&](sycl::handler &cgh) {
        // Get access to buffers on device
        sycl::accessor alpha_acc(alpha_buf, cgh, sycl::read_only);
        sycl::accessor beta_acc(beta_buf, cgh, sycl::read_only);
        sycl::accessor A_acc(A_buf, cgh, sycl::read_only);
        sycl::accessor x_acc(x_buf, cgh, sycl::read_only);
        sycl::accessor y_acc(y_buf, cgh, sycl::read_write);
        cgh.parallel_for(
                m, [=](sycl::id<1> itemID) {
                    // Kernel code
                    size_t i = itemID.get(0);
                    __attribute__((annotate("scalar(range(0, 120001000) bufferid('y') final)"))) float sum = 0;
                    for (size_t j = 0; j < n; j++)
                        sum += A_acc[i * n + j] * x_acc[j];
                    y_acc[i] = alpha_acc[0] * sum + beta_acc[0] * y_acc[i];
                });
    });
}

#define ALPHA_MIN 0
#define ALPHA_MAX 10
#define BETA_MIN 0
#define BETA_MAX 20
#define A_MIN 0
#define A_MAX 30
#define X_MIN 0
#define X_MAX 40
#define Y_INIT_MIN 0
#define Y_INIT_MAX 50
#define ALPHA_ANNOTATION __attribute__((annotate("sycl_accessor('alpha_acc') scalar(range(0, 10) bufferid('alpha') final)")))
#define BETA_ANNOTATION __attribute__((annotate("sycl_accessor('beta_acc') scalar(range(0, 20) bufferid('beta') final)")))
#define A_ANNOTATION __attribute__((annotate("sycl_accessor('A_acc') scalar(range(0, 30) bufferid('A') final)")))
#define X_ANNOTATION __attribute__((annotate("sycl_accessor('x_acc') scalar(range(0, 40) bufferid('x') final)")))
#define Y_ANNOTATION __attribute__((annotate("sycl_accessor('y_acc') scalar(range(0, 120001000) bufferid('y') final)")))

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <seed> <m> <n> <warmupIterations> <outputFile>" << "\n";
        return 1;
    }
    size_t seed = std::atoi(argv[1]);
    size_t m = std::atoi(argv[2]);
    size_t n = std::atoi(argv[3]);
    size_t warmupIterations = std::atoi(argv[4]);
    std::string outputFileName = argv[5];

    RNG rng(seed);

    std::ofstream outFile(outputFileName);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << outputFileName << "\n";
        return 1;
    }

    ALPHA_ANNOTATION float alpha;
    BETA_ANNOTATION float beta;
    A_ANNOTATION float *A = (float *) malloc(sizeof(float) * m * n);
    X_ANNOTATION float *x = (float *) malloc(sizeof(float) * n);
    Y_ANNOTATION float *y = (float *) malloc(sizeof(float) * m);

    // Function called once just to let Taffo compute the right ranges on host
    // in order to have them on device with the bufferid system
    gemv_naive(alpha, beta, A, x, y, m, n);

    size_t executionTime = 0;

    sycl::queue queue;

    std::cout << "Warmup: " << warmupIterations << " iterations\n";
    for (size_t i = 0; i < warmupIterations + 1; i++) {
        if (i == warmupIterations)
            std::cout << "Warmup finished\n\n";

        // Initialize data
        alpha = rng.randomInRange(ALPHA_MIN, ALPHA_MAX);
        beta = rng.randomInRange(BETA_MIN, BETA_MAX);
        for (size_t j = 0; j < m * n; j++)
            A[j] = rng.randomInRange(A_MIN, A_MAX);
        for (size_t j = 0; j < n; j++)
            x[j] = rng.randomInRange(X_MIN, X_MAX);
        for (size_t j = 0; j < m; j++)
            y[j] = rng.randomInRange(Y_INIT_MIN, Y_INIT_MAX);

        std::cout << "Submitting work to device\n";
        auto start = std::chrono::high_resolution_clock::now();
        {
            sycl::buffer<float> alpha_buf(&alpha, 1);
            sycl::buffer<float> beta_buf(&beta, 1);
            sycl::buffer<float> A_buf(A, m * n);
            sycl::buffer<float> x_buf(x, n);
            sycl::buffer<float> y_buf(y, m);
            gemv_sycl(queue, alpha_buf, beta_buf, A_buf, x_buf, y_buf, m, n);
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
    outFile << "Len: " << m << "\n";
    for (int i = 0; i < m; i++)
        outFile << i << " " << y[i] << "\n";
    outFile.close();

    free(x);
    free(y);
    return 0;
}
