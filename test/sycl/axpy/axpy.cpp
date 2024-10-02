#include "../common/taffo_sycl_tests.hpp"

void axpy_naive(float alpha, const float *x, float *y, size_t vecLen) {
    for (size_t i = 0; i < vecLen; i++)
        y[i] = alpha * x[i] + y[i];
}

void axpy_sycl(sycl::queue queue,
               sycl::buffer<float> &alpha_buf,
               sycl::buffer<float> &x_buf,
               sycl::buffer<float> &y_buf,
               size_t vecLen) {
    queue.submit([&](sycl::handler &cgh) {
        // Get access to buffers on device
        sycl::accessor alpha_acc(alpha_buf, cgh, sycl::read_only);
        sycl::accessor x_acc(x_buf, cgh, sycl::read_only);
        sycl::accessor y_acc(y_buf, cgh, sycl::read_write);

        cgh.parallel_for(
                vecLen, [=](sycl::id<1> itemID) {
                    // Kernel code
                    y_acc[itemID] = alpha_acc[0] * x_acc[itemID] + y_acc[itemID];

                    // Useless but at least 1 device-side annotation is needed at the moment for Taffo to work
                    __attribute__((annotate("scalar()"))) float annotatedVal = 0;
                });
    });
}

#define ALPHA_MIN 0
#define ALPHA_MAX 100
#define X_MIN 0
#define X_MAX 200
#define Y_INIT_MIN 0
#define Y_INIT_MAX 3000
#define ALPHA_ANNOTATION __attribute__((annotate("sycl_accessor('alpha_acc') scalar(range(0, 100) bufferid('alpha') final)")))
#define X_ANNOTATION __attribute__((annotate("sycl_accessor('x_acc') scalar(range(0, 200) bufferid('x') final)")))
#define Y_ANNOTATION __attribute__((annotate("sycl_accessor('y_acc') scalar(range(0, 23000) bufferid('y') final)")))

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <seed> <vecLen> <warmupIterations> <outputFile>" << "\n";
        return 1;
    }
    size_t seed = std::atoi(argv[1]);
    size_t vecLen = std::atoi(argv[2]);
    size_t warmupIterations = std::atoi(argv[3]);
    std::string outputFileName = argv[4];

    RNG rng(seed);

    std::ofstream outFile(outputFileName);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << outputFileName << "\n";
        return 1;
    }

    ALPHA_ANNOTATION float alpha;
    X_ANNOTATION float *x = (float *) malloc(sizeof(float) * vecLen);
    Y_ANNOTATION float *y = (float *) malloc(sizeof(float) * vecLen);

    // Function called once just to let Taffo compute the right ranges on host
    // in order to have them on device with the bufferid system
    axpy_naive(alpha, x, y, vecLen);

    size_t executionTime = 0;

    sycl::queue queue;

    std::cout << "Warmup: " << warmupIterations << " iterations\n";
    for (size_t i = 0; i < warmupIterations + 1; i++) {
        if (i == warmupIterations)
            std::cout << "Warmup finished\n\n";

        // Initialize data
        alpha = rng.randomInRange(ALPHA_MIN, ALPHA_MAX);
        for (size_t j = 0; j < vecLen; j++) {
            x[j] = rng.randomInRange(X_MIN, X_MAX);
            y[j] = rng.randomInRange(Y_INIT_MIN, Y_INIT_MAX);
        }

        std::cout << "Submitting work to device\n";
        auto start = std::chrono::high_resolution_clock::now();
        {
            sycl::buffer<float> alpha_buf(&alpha, 1);
            sycl::buffer<float> x_buf(x, vecLen);
            sycl::buffer<float> y_buf(y, vecLen);
            axpy_sycl(queue, alpha_buf, x_buf, y_buf, vecLen);
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
    outFile << "Len: " << vecLen << "\n";
    for (int i = 0; i < vecLen; i++)
        outFile << i << " " << y[i] << "\n";
    outFile.close();

    free(x);
    free(y);
    return 0;
}
