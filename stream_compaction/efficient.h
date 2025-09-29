#pragma once
#include "../src/sceneStructs.h";

void checkCUDAErrorFn(const char* msg, const char* file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)


inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

inline int divup(int n, int divisor) {
    return (n + divisor - 1) / divisor;
}

namespace StreamCompaction {
    namespace Efficient {

        void scan(int n, int *odata, const int *idata);

        void scanSharedMemory(int n, int* odata, const int* idata);

        int partitionOnBounces(int n, PathSegment* dev_idata);

        int partitionOnValidIntersect(int n, PathSegment* dev_idata, int* bools);

    }
}
