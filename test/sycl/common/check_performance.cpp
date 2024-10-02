#include "../common/taffo_sycl_tests.hpp"

void computeErrors_sycl(sycl::queue queue, float *resArray, float *expectedArray, size_t len, float &maxRelError, float &meanRelError) {
    maxRelError = 0;
    meanRelError = 0;
    sycl::buffer<float> res_buf(resArray, len);
    sycl::buffer<float> expected_buf(expectedArray, len);
    size_t workers = log2(len) + 1;
    size_t itemsPerWorker = len / workers;
    sycl::buffer<float> maxRelErr_buf(workers);
    sycl::buffer<float> meanRelErr_buf(workers);
    queue.submit([&](sycl::handler &cgh) {
        // Get access to buffers on device
        sycl::accessor res_acc(res_buf, cgh, sycl::read_only);
        sycl::accessor expected_acc(expected_buf, cgh, sycl::read_only);
        sycl::accessor maxRelErr_acc(maxRelErr_buf, cgh, sycl::write_only);
        sycl::accessor meanRelErr_acc(meanRelErr_buf, cgh, sycl::write_only);
        cgh.parallel_for(
                workers, [=](sycl::id<1> workerID) {
                    float localMaxRelErr = 0;
                    float localMeanRelErr = 0;
                    size_t offset = workerID * itemsPerWorker;
                    for (size_t i = offset; i < offset + itemsPerWorker && i < len; i++) {
                        float result = res_acc[i];
                        float expected = expected_acc[i];
                        float relErr = abs(expected - result) / abs(expected);
                        localMeanRelErr += relErr;
                        if (relErr > localMaxRelErr)
                            localMaxRelErr = relErr;
                    }
                    maxRelErr_acc[workerID] = localMaxRelErr;
                    meanRelErr_acc[workerID] = localMeanRelErr;
                });
    });
    sycl::host_accessor maxRelErr_hostAcc(maxRelErr_buf);
    sycl::host_accessor meanRelErr_hostAcc(meanRelErr_buf);
    for (size_t i = 0; i < workers; i++) {
        meanRelError += meanRelErr_hostAcc[i];
        float workerMaxRelErr = maxRelErr_hostAcc[i];
        if (workerMaxRelErr > maxRelError)
            maxRelError = workerMaxRelErr;
    }
    maxRelError *= 100;
    meanRelError = 100 * meanRelError / len;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <resultsFile> <expectedResultsFile> <outputFile>" << "\n";
        return 1;
    }
    std::string resFileName = argv[1];
    std::string expectedFileName = argv[2];
    std::string outputFileName = argv[3];

    std::ifstream resFile(resFileName);
    if (!resFile) {
        std::cerr << "Error: Could not open file " << resFileName << "\n";
        return 1;
    }
    std::ifstream expectedFile(expectedFileName);
    if (!expectedFile) {
        std::cerr << "Error: Could not open file " << expectedFileName << "\n";
        return 1;
    }
    std::ofstream outFile(outputFileName);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << outputFileName << "\n";
        return 1;
    }

    size_t taffo_time, standard_time;
    size_t len1, len2;

    std::string line;
    std::string junk;

    // Read Taffo results
    std::getline(resFile, line);
    std::istringstream(line) >> junk >> taffo_time;

    std::getline(resFile, line);
    std::istringstream(line) >> junk >> len1;

    float *res = (float*) malloc(sizeof(float) * len1);
    for (size_t i = 0; i < len1 && std::getline(resFile, line); i++)
        std::istringstream(line) >> junk >> res[i];
    resFile.close();

    // Read expected results
    std::getline(expectedFile, line);
    std::istringstream(line) >> junk >> standard_time;

    std::getline(expectedFile, line);
    std::istringstream(line) >> junk >> len2;
    assert(len1 == len2);

    float *expected = (float*) malloc(sizeof(float) * len1);
    for (size_t i = 0; i < len1 && std::getline(expectedFile, line); i++)
        std::istringstream(line) >> junk >> expected[i];
    expectedFile.close();

    // Compute speedup and errors
    float speedup = NAN;
    if (standard_time != 0)
        speedup = (float) standard_time / (float) taffo_time;
    float maxRelError, meanRelError;

    sycl::queue queue;
    computeErrors_sycl(queue, res, expected, len1, maxRelError, meanRelError);

    // Write results in output file
    outFile << "Speedup:             " << speedup << " x\n";
    outFile << std::scientific << std::setprecision(5);
    outFile << "Max relative error:  " << maxRelError << " %\n"
            << "Mean relative error: " << meanRelError << " %\n";
    outFile.close();

    std::cout << "Results written to " << outputFileName << "\n";
    return 0;
}
