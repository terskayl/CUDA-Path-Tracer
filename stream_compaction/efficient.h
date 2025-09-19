#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanSharedMemory(int n, int* odata, const int* idata);

        int compact(int n, int *odata, const int *idata);

        int compactSharedMemory(int n, int* odata, const int* idata);

        void radixSort(int n, int* odata, const int* idata);
    }
}
