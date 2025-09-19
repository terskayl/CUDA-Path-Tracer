#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            bool timed = false;
            if (!timer().isStartedCpu()) {
                timer().startCpuTimer();
                timed = true;
            }
            
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                // We set out data first in order to do an exclusive scan. We
                // can reverse the next two lines in order to perform an 
                // inclusive scan.
                odata[i] = sum;
                sum += idata[i];
            }

            if (timed) {
                timer().endCpuTimer();
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int outIndex = 0;
            for (int i = 0; i < n; ++i) {
                int data = idata[i];
                if (data != 0) {
                    odata[outIndex] = data;
                    outIndex += 1;
                }
            }

            timer().endCpuTimer();
            return outIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // Is valid arr is an array the same size of our data that is 
            // 0 if the data is invalid and 1 if it is valid
            int* isValidArr = new int[n];
            for (int i = 0; i < n; ++i) {
                if (idata[i] == 0) {
                    isValidArr[i] = 0;
                } else {
                    isValidArr[i] = 1;
                }
            }

            // indicesArray will tell us, for all non-zero entries, which
            // index they will be in our output array.
            int* indicesArray = new int[n];
            scan(n, indicesArray, isValidArr);

            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[indicesArray[i]] = idata[i];
                }
            }

            int finalCount = indicesArray[n - 1] + (idata[n - 1] == 0 ? 0 : 1);
            delete[] isValidArr;
            delete[] indicesArray;

            timer().endCpuTimer();
            return finalCount;
        }
    }
}
