#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void addStride(int n, int stride, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx + stride < n) {
                odata[idx + stride] = idata[idx] + idata[idx + stride];
            }
        }

        __global__ void inclScanToExclScan(int n, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n - 1) {
                odata[idx + 1] = idata[idx];
            }
            if (idx == n - 1) {
                odata[0] = 0;
            }
        }

        __global__ void naiveScanSharedMem(int n, int* idata, int* odata) {
            // BLOCKSIZE must be power of 2
            // shared mem should have size threadsPerBlock * 2 * sizeof(int).
            // s[0:blockDim.x] represents the ping array
            // s[blockDim.x:2*blockDim.x] represents the pong array
            extern __shared__ int s[];
            
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            // Load into shared mem
            s[threadIdx.x] = idata[idx];
            s[blockDim.x + threadIdx.x] = s[threadIdx.x];
            __syncthreads();
            for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
                if (threadIdx.x + stride < blockDim.x) {
                    s[blockDim.x + threadIdx.x + stride] += s[threadIdx.x];
                }
            __syncthreads();
            s[threadIdx.x] = s[blockDim.x + threadIdx.x];
            __syncthreads();
            }
            idata[idx] = s[threadIdx.x];
            if (threadIdx.x == blockDim.x - 1) {
                odata[blockIdx.x] = s[threadIdx.x];
            }
        }

        __global__ void addPrefix(int n, int* idata, int* odata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx >= n) return;
            // manual incl to excl scan conversion
            int valueToAdd = 0;
            if (blockIdx.x > 0) {
                valueToAdd = idata[blockIdx.x - 1];
            }
            odata[idx] += valueToAdd;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            unsigned blocksize = 128;

            // n rounded up to the nearest power of two
            int roundUpN = ilog2ceil(n);
            int totalN = pow(2, roundUpN);

            int *d_ping, *d_pong;
            cudaMalloc((void**)&d_ping, totalN * sizeof(int));
            checkCUDAError("cudaMalloc ping");
            cudaMalloc((void**)&d_pong, totalN * sizeof(int));
            checkCUDAError("cudaMalloc pong");

            cudaMemset(d_ping, 0, totalN * sizeof(int));
            checkCUDAError("cudaMemset ping");
            cudaMemcpy(d_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to ping");

            cudaMemcpy(d_pong, d_ping, totalN * sizeof(int), cudaMemcpyDeviceToDevice);

            timer().startGpuTimer();
            for (int i = 0; i < roundUpN; ++i) {

                addStride<<<divup(totalN, blocksize), blocksize>>>(n, pow(2, i), d_ping, d_pong);
                checkCUDAError("addStride failed");
                cudaDeviceSynchronize();

                cudaMemcpy(d_ping, d_pong, totalN * sizeof(int), cudaMemcpyDeviceToDevice);

            }

            inclScanToExclScan<<<divup(totalN, blocksize), blocksize>>>(n, d_ping, d_pong);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_pong, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from pong");
            cudaFree(d_ping);
            cudaFree(d_pong);
        }

        void scanSharedMemory(int n, int* odata, const int* idata) {

            unsigned blocksize = 512;

            // We will just use one buffer, each cycle the number of elements we process
            // is divided by blocksize.
            int* d_data;

            int roundArraySize = n;
            int sum = n;
                                            // ceiling(log_blocksize(n)) via change of bases
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

            cudaMalloc((void**)&d_data, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_data");

            cudaMemset(d_data, 0, sum * sizeof(int));
            checkCUDAError("cudaMemset d_data");
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy initial data to d_data");

            timer().startGpuTimer();
            for (int i = 0; i < breakpointsSize - 1; ++i) {
                // interval is from breakpoints[i] to breakpoints[i+1]
                naiveScanSharedMem<<<divup(breakpoints[i+1] - breakpoints[i], blocksize), blocksize, blocksize * 2 * sizeof(int)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_data + breakpoints[i], d_data + breakpoints[i + 1]);
                checkCUDAError("naiveScanSharedMem");
            }
            for (int i = breakpointsSize - 2; i >= 0; --i) {
                // interval is from breakpoints[i] to breakpoints[i+1]

                // We will do manual incl scan to excl scan conversion in following kernel.
                addPrefix<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize>>>
                    (breakpoints[i + 1] - breakpoints[i], d_data + breakpoints[i + 1], d_data + breakpoints[i]);
                checkCUDAError("addPrefix");
            }
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, d_data, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy output data from d_data");
            cudaFree(d_data);



            delete[] breakpoints;
        }
    }
}
