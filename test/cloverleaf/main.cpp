
#include <cstdio>
#include "miosix.h"

using namespace std;
using namespace miosix;

extern "C" void clover_main();

int main()
{
    printf("Welcome to the CloverLeaf embedded benchmark\n");
    clover_main();
    MemoryProfiling::print();
    printf("CloverLeaf benchmark complete.");
    return 0;
}
