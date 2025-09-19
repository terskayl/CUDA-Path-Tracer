#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {

        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpsweepStep(int n, int exp, int* data) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            // Similar to array modifications when making a heap, we want 
            // our array to be 1-indexed instead of 0-indexed because 
            // 0 is divisible by all powers of two - but we want that position
            // on the right.
            idx += 1;
            unsigned lowerNeighbor = idx - (1 << (exp - 1));
                                                     // idx % powf(2, exp)
            if (idx <= n && lowerNeighbor >= 1 && (idx & (1 << exp) - 1) == 0) {
                //data[idx - 1] += data[idx - (1 << (exp - 1)) - 1];
                data[idx - 1] += data[lowerNeighbor - 1];
            }
        }

        __global__ void kernDownsweepStep(int n, int exp, int* data) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            // Similar to array modifications when making a heap, we want 
            // our array to be 1-indexed instead of 0-indexed because 
            // 0 is divisible by all powers of two - but we want that position
            // on the right.
            idx += 1;
            unsigned lowerNeighbor = idx - (1 << (exp - 1));
            if (idx <= n && lowerNeighbor >= 1 && idx % (1 << exp) == 0) {
                int temp = data[idx - 1];
                data[idx - 1] += data[lowerNeighbor - 1];
                data[lowerNeighbor - 1] = temp;
            }
        }

        __global__ void kernUpsweepBlock(int n, int* idata, int* odata, int padding) {
            // BLOCKSIZE must be power of two
            extern __shared__ int s[];

            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            //if (idx >= n) return;

            // virtual padding
            if (blockIdx.x == 0 && threadIdx.x < padding) {
                s[threadIdx.x] = 0;
            } else { 
                // load into shared memory
                s[threadIdx.x] = idata[idx - padding];
            }
            __syncthreads();

            for (int c = 2; c <= blockDim.x; c *= 2) {
                if (c * (threadIdx.x + 1) - 1 < blockDim.x && c * (threadIdx.x + 1) - (c / 2) - 1 >= 0) {
                    s[c * (threadIdx.x + 1) - 1] += s[c * (threadIdx.x + 1) - (c / 2) - 1];
                }
                __syncthreads();
            }
            if (idx >= padding) idata[idx - padding] = s[threadIdx.x];
            if (threadIdx.x == blockDim.x - 1) {
                odata[blockIdx.x] = s[threadIdx.x];
            }
        }

        __global__ void kernDownsweepBlock(int n, int* idata, int* odata, int padding) {
            // BLOCKSIZE must be power of two
            extern __shared__ int s[];

            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Recover the downsweep result for this block from the recursive
            // layer above. Also virtually pad the array to the left.
            if (idx < padding) {
                s[threadIdx.x] = 0;
            }
            else if (threadIdx.x == blockDim.x - 1) {
                s[blockDim.x - 1] = idata[blockIdx.x];
            }
            else {
                s[threadIdx.x] = odata[idx - padding];
            }

            __syncthreads();

             // The implementation is mirrored compared to downsweepStep
            for (int c = blockDim.x; c >= 2; c /= 2) {
                if (c * threadIdx.x < blockDim.x) {
                    int temp = s[c * (threadIdx.x + 1) - 1];
                    s[c * (threadIdx.x + 1) - 1] += s[c * (threadIdx.x + 1) - (c / 2) - 1];
                    s[c * (threadIdx.x + 1) - (c / 2) - 1] = temp;
                }
                __syncthreads();
            }

            if (idx >= padding) odata[idx - padding] = s[threadIdx.x];
        }

        __global__ void kernReverse(int n, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) odata[idx] = idata[n - idx - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            unsigned blocksize = 512;

            // n rounded up to the nearest power of two
            int roundUpN = ilog2ceil(n);
            int totalN = pow(2, roundUpN);
            
            int* d_data;
            cudaMalloc((void**)&d_data, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_data"); 

            cudaMemset(d_data, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to d_data");

            timer().startGpuTimer();
            // Up-Sweep
            for (int exp = 1; exp <= roundUpN; ++exp) {
                kernUpsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_data);
                checkCUDAError("kernUpsweepStep");
            }
            cudaDeviceSynchronize();
            // Down-Sweep
            cudaMemset(d_data + (totalN - 1), 0, 1 * sizeof(int));
            for (int exp = roundUpN; exp >= 1; --exp) {
                kernDownsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_data);
                checkCUDAError("kernDownsweepStep");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from d_data");
            cudaFree(d_data);

        }


        void scanSharedMemory(int n, int* odata, const int* idata) {
            unsigned blocksize = 512;
            // We will just use one buffer, each cycle the number of 
            // elements we process is divided by blocksize.

            int roundArraySize = n;
            int sum = n;
                                            // ceiling(ilog_blocksize(n)) via change of bases
            int* breakpoints = new int[2 + (ilog2(n - 1) / ilog2(blocksize)) + 1];
            breakpoints[0] = 0;
            breakpoints[1] = sum;
            int breakpointsSize = 2;

            // Output of the following is threefold:
            // breakpoints, an array containing all index transitions in d_data, 
            // breakpointsSize = breakpoints.size(),
            // sum = sum(breakpoints)
            while (roundArraySize > 1) {
                roundArraySize = divup(roundArraySize, blocksize);
                sum += roundArraySize;
                breakpoints[breakpointsSize] = sum;
                breakpointsSize++;
            }

            int* d_data;
            cudaMalloc((void**)&d_data, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_data");

            cudaMemset(d_data, 0, sum * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to d_data");

            timer().startGpuTimer();

            for (int i = 0; i < breakpointsSize - 2; ++i) {
                int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                kernUpsweepBlock<<<divup(breakpoints[i+1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_data + breakpoints[i], d_data + breakpoints[i + 1], padding);
                checkCUDAError("kernUpsweepBlock");
                cudaDeviceSynchronize();
            }

            cudaMemset(d_data + sum - 1, 0, 1 * sizeof(int));
            for (int i = breakpointsSize - 3; i >= 0; --i) {
                // interval is from breakpoints[i] to breakpoints[i+1]
                int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                kernDownsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_data + breakpoints[i + 1], d_data + breakpoints[i], padding);
                checkCUDAError("kernDownsweepBlock");
            }


            timer().endGpuTimer();

            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from d_data");
            cudaFree(d_data);

        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            unsigned blocksize = 128;

            // n rounded up to the nearest power of two
            int roundUpN = ilog2ceil(n);
            int totalN = pow(2, roundUpN);

            int *d_data, *d_bools, *d_indices, *d_output;
            cudaMalloc((void**)&d_data, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_data");
            cudaMalloc((void**)&d_bools, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_bools");
            cudaMalloc((void**)&d_indices, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_indices");
            cudaMalloc((void**)&d_output, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_output");

            cudaMemset(d_data, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into d_data");

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<divup(totalN, blocksize), blocksize>>>(totalN, d_bools, d_data);
            checkCUDAError("kernMapToBoolean");
            cudaDeviceSynchronize();
            // Up-Sweep
            for (int exp = 1; exp <= roundUpN; ++exp) {
                kernUpsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_bools);
            }
            cudaDeviceSynchronize();
            // Down-Sweep
            cudaMemset(d_bools + (totalN - 1), 0, 1 * sizeof(int));
            for (int exp = roundUpN; exp >= 1; --exp) {
                kernDownsweepStep<<<divup(totalN, blocksize), blocksize>>>(totalN, exp, d_bools);
            }
            checkCUDAError("upsweep and downsweep scan");
            int sizePlusMaybeOne;
            cudaMemcpy(&sizePlusMaybeOne, d_bools + (totalN - 1), 1 * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy sizePlusMaybeOne");

            cudaMemset(d_output, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset d_output");

            Common::kernScatter<<<divup(totalN, blocksize), blocksize>>>(totalN, d_output, d_data, d_data, d_bools);
            checkCUDAError("kernScatter");
            timer().endGpuTimer();

            cudaMemcpy(odata, d_output, std::min(sizePlusMaybeOne, n) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to output");

            if (odata[sizePlusMaybeOne - 1] == 0) {
                sizePlusMaybeOne--;
            }

            cudaFree(d_data);
            cudaFree(d_bools);
            cudaFree(d_indices);
            cudaFree(d_output);
            checkCUDAError("cudaFree");


            return sizePlusMaybeOne;
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compactSharedMemory(int n, int* odata, const int* idata) {
            unsigned blocksize = 128;

            int roundArraySize = n;
            int sum = n;
            // ceiling(ilog_blocksize(n)) via change of bases
            int* breakpoints = new int[2 + (ilog2(n - 1) / ilog2(blocksize)) + 1];
            breakpoints[0] = 0;
            breakpoints[1] = sum;
            int breakpointsSize = 2;
            while (roundArraySize > 1) {
                roundArraySize = divup(roundArraySize, blocksize);
                sum += roundArraySize;
                breakpoints[breakpointsSize] = sum;
                breakpointsSize++;
            }

            // n rounded up to the nearest power of two
            int roundUpN = ilog2ceil(n);
            int totalN = pow(2, roundUpN);

            int* d_data, * d_bools, * d_indices, * d_output;
            cudaMalloc((void**)&d_data, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_data");
            cudaMalloc((void**)&d_bools, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_bools");
            cudaMalloc((void**)&d_indices, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_indices");
            cudaMalloc((void**)&d_output, totalN * sizeof(int));
            checkCUDAError("cudaMalloc d_output");

            cudaMemset(d_bools, 0, sum * sizeof(int));
            checkCUDAError("cudaMemset d_bools");

            cudaMemset(d_data, 0, n * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into d_data");

            timer().startGpuTimer();

            Common::kernMapToBoolean<<<divup(totalN, blocksize), blocksize>>>(n, d_bools, d_data);
            checkCUDAError("kernMapToBoolean");
            cudaDeviceSynchronize();

            for (int i = 0; i < breakpointsSize - 2; ++i) {
                int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                kernUpsweepBlock<<<divup(breakpoints[i+1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_bools + breakpoints[i], d_bools + breakpoints[i + 1], padding);
                checkCUDAError("kernUpsweepBlock");
                cudaDeviceSynchronize();
            }

            int total;
            cudaMemcpy(&total, d_bools + sum - 1, 1 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemset(d_bools + sum - 1, 0, 1 * sizeof(int));
            for (int i = breakpointsSize - 3; i >= 0; --i) {
                // interval is from breakpoints[i] to breakpoints[i+1]
                int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                kernDownsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_bools + breakpoints[i + 1], d_bools + breakpoints[i], padding);
                checkCUDAError("kernDownsweepBlock");
                cudaDeviceSynchronize();
            }

            Common::kernScatter<<<divup(totalN, blocksize), blocksize>>>(n, d_output, d_data, d_data, d_bools);
            checkCUDAError("kernScatter");
            timer().endGpuTimer();

            cudaMemcpy(odata, d_output, total * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to output");

            cudaFree(d_data);
            cudaFree(d_bools);
            cudaFree(d_indices);
            cudaFree(d_output);
            checkCUDAError("cudaFree");

            return total;
        }


        __global__ void kernMapToBooleanRadix(int n, int* bools, const int* idata, int bit) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if ((idata[idx] & (1 << bit)) == 0) {
                bools[idx] = 1;
            }
            else {
                bools[idx] = 0;
            }
        }

        __global__ void kernScatterRadix(int n, int* odata,
            const int* idata, const int* falseIndices, int bit, int total) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if ((idata[idx] & (1 << bit)) == 0) {
                odata[falseIndices[idx]] = idata[idx];
            }
            else {
                odata[idx - falseIndices[idx] + total] = idata[idx];
            }
        }

        void radixSort(int n, int* odata, const int* idata) {
            int blocksize = 128;

            int roundArraySize = n;
            int sum = n;
                                            // ceiling(ilog_blocksize(n)) via change of bases
            int* breakpoints = new int[2 + (ilog2(n - 1) / ilog2(blocksize)) + 1];
            breakpoints[0] = 0;
            breakpoints[1] = sum;
            int breakpointsSize = 2;

            while (roundArraySize > 1) {
                roundArraySize = divup(roundArraySize, blocksize);
                sum += roundArraySize;
                breakpoints[breakpointsSize] = sum;
                breakpointsSize++;
            }

            int *d_ping, *d_pong, *d_bools;
            cudaMalloc((void**)&d_ping, n * sizeof(int));
            checkCUDAError("cudaMalloc d_ping");
            cudaMalloc((void**)&d_pong, n * sizeof(int));
            checkCUDAError("cudaMalloc d_pong");
            cudaMalloc((void**)&d_bools, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_bools");

            cudaMemcpy(d_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(d_bools, 0, sum * sizeof(int));

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; ++bit) {
                // Make boolean map
                kernMapToBooleanRadix<<<divup(n, blocksize), blocksize>>>(n, d_bools, d_ping, bit);

                // Scan
                for (int i = 0; i < breakpointsSize - 2; ++i) {
                    int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                    kernUpsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                        (breakpoints[i + 1] - breakpoints[i], d_bools + breakpoints[i], d_bools + breakpoints[i + 1], padding);
                    checkCUDAError("kernUpsweepBlock");
                    cudaDeviceSynchronize();
                }

                // Get total
                int total;
                cudaMemcpy(&total, d_bools + sum - 1, 1 * sizeof(int), cudaMemcpyDeviceToHost);

                // Continue Scan
                cudaMemset(d_bools + sum - 1, 0, 1 * sizeof(int));
                for (int i = breakpointsSize - 3; i >= 0; --i) {
                    // interval is from breakpoints[i] to breakpoints[i+1]
                    int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                    kernDownsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(int)>>>
                        (breakpoints[i + 1] - breakpoints[i], d_bools + breakpoints[i + 1], d_bools + breakpoints[i], padding);
                    checkCUDAError("kernDownsweepBlock");
                }

                // Scatter
                kernScatterRadix<<<divup(n, blocksize), blocksize>>>(n, d_pong, d_ping, d_bools, bit, total);
                checkCUDAError("scatter");

                // Swap ping pong
                int* temp = d_ping;
                d_ping = d_pong;
                d_pong = temp;
            }
            
            timer().endGpuTimer();

            cudaMemcpy(odata, d_ping, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy out to odata");

            cudaFree(d_ping);
            cudaFree(d_pong);
            cudaFree(d_bools);
        }
    }
}
