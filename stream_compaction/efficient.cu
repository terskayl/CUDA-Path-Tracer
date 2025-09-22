#include <cuda.h>
#include <cuda_runtime.h>
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


        __global__ void kernMapToBooleanRadix(int n, int* bools, const PathSegment* idata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (idata[idx].remainingBounces <= 0) {
                bools[idx] = 1;
            }
            else {
                bools[idx] = 0;
            }
        }

        __global__ void kernMapToBooleanBounces(int n, int* bools, const PathSegment* idata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (idata[idx].remainingBounces > 0) {
                bools[idx] = 1;
            }
            else {
                bools[idx] = 0;
            }
        }

        __global__ void kernScatterRadixBounces(int n, PathSegment* odata,
            const PathSegment* idata, const int* falseIndices, int total) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (idata[idx].remainingBounces == 0) {
                odata[falseIndices[idx]].pixelIndex = idata[idx].pixelIndex;
                odata[falseIndices[idx]].ray.direction = idata[idx].ray.direction;
                odata[falseIndices[idx]].ray.origin = idata[idx].ray.origin;
                odata[falseIndices[idx]].remainingBounces = idata[idx].remainingBounces;
                odata[falseIndices[idx]].radiance = idata[idx].radiance;
                odata[falseIndices[idx]].throughput = idata[idx].throughput;
            }
            else {
                odata[idx - falseIndices[idx] + total].pixelIndex = idata[idx].pixelIndex;
                odata[idx - falseIndices[idx] + total].ray.direction = idata[idx].ray.direction;
                odata[idx - falseIndices[idx] + total].ray.origin = idata[idx].ray.origin;
                odata[idx - falseIndices[idx] + total].remainingBounces = idata[idx].remainingBounces;
                odata[idx - falseIndices[idx] + total].radiance = idata[idx].radiance;
                odata[idx - falseIndices[idx] + total].throughput = idata[idx].throughput;
            }
        }

        __global__ void kernScatterRadixIntersect(int n, PathSegment* odata,
            const PathSegment* idata, const int* falseIndices, int total, int* bools,
            ShadeableIntersection* iIntersects, ShadeableIntersection* oIntersects) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (bools[idx] == 1) {
                odata[falseIndices[idx]].pixelIndex = idata[idx].pixelIndex;
                odata[falseIndices[idx]].ray.direction = idata[idx].ray.direction;
                odata[falseIndices[idx]].ray.origin = idata[idx].ray.origin;
                odata[falseIndices[idx]].remainingBounces = idata[idx].remainingBounces;
                odata[falseIndices[idx]].radiance = idata[idx].radiance;
                odata[falseIndices[idx]].throughput = idata[idx].throughput;

                oIntersects[falseIndices[idx]].materialId = iIntersects[idx].materialId;
                oIntersects[falseIndices[idx]].t = iIntersects[idx].t;
                oIntersects[falseIndices[idx]].surfaceNormal = iIntersects[idx].surfaceNormal;
            }
            else {
                odata[idx - falseIndices[idx] + total].pixelIndex = idata[idx].pixelIndex;
                odata[idx - falseIndices[idx] + total].ray.direction = idata[idx].ray.direction;
                odata[idx - falseIndices[idx] + total].ray.origin = idata[idx].ray.origin;
                odata[idx - falseIndices[idx] + total].remainingBounces = idata[idx].remainingBounces;
                odata[idx - falseIndices[idx] + total].radiance = idata[idx].radiance;
                odata[idx - falseIndices[idx] + total].throughput = idata[idx].throughput;

                oIntersects[idx - falseIndices[idx] + total].materialId = iIntersects[idx].materialId;
                oIntersects[idx - falseIndices[idx] + total].t = iIntersects[idx].t;
                oIntersects[idx - falseIndices[idx] + total].surfaceNormal = iIntersects[idx].surfaceNormal;

            }
        }

        int partitionOnBounces(int n, PathSegment* dev_odata, const PathSegment* dev_idata) {
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

            PathSegment* d_ping, * d_pong;
            int *d_bools;
            cudaMalloc((void**)&d_ping, n * sizeof(PathSegment));
            checkCUDAError("cudaMalloc d_ping");
            cudaMalloc((void**)&d_pong, n * sizeof(PathSegment));
            checkCUDAError("cudaMalloc d_pong");
            cudaMalloc((void**)&d_bools, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_bools");

            cudaMemcpy(d_ping, dev_idata, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            cudaMemset(d_bools, 0, sum * sizeof(int));


            // Make boolean map
            kernMapToBooleanBounces<<<divup(n, blocksize), blocksize>>>(n, d_bools, d_ping);

            // Scan
            for (int i = 0; i < breakpointsSize - 2; ++i) {
                int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                kernUpsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(PathSegment)>>>
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
                kernDownsweepBlock<<<divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(PathSegment)>>>
                    (breakpoints[i + 1] - breakpoints[i], d_bools + breakpoints[i + 1], d_bools + breakpoints[i], padding);
                checkCUDAError("kernDownsweepBlock");
            }

            // Scatter
            kernScatterRadixBounces<<<divup(n, blocksize), blocksize>>>(n, d_pong, d_ping, d_bools, total);
            checkCUDAError("scatter");

            // Swap ping pong
            PathSegment* temp = d_ping;
            d_ping = d_pong;
            d_pong = temp;
            

            cudaMemcpy(dev_odata, d_ping, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy out to odata");

            cudaFree(d_ping);
            cudaFree(d_pong);
            cudaFree(d_bools);
            return total;
        }


        int partitionOnValidIntersect(int n, PathSegment* dev_odata, const PathSegment* dev_idata, int* dev_bools, ShadeableIntersection* dev_intersections) {
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

            PathSegment* d_ping, * d_pong;
            int *d_bools;
            ShadeableIntersection* d_intersectionsPing;
            cudaMalloc((void**)&d_ping, n * sizeof(PathSegment));
            checkCUDAError("cudaMalloc d_ping");
            cudaMalloc((void**)&d_pong, n * sizeof(PathSegment));
            checkCUDAError("cudaMalloc d_pong");
            cudaMalloc((void**)&d_bools, sum * sizeof(int));
            checkCUDAError("cudaMalloc d_bools");
            cudaMalloc((void**)&d_intersectionsPing, sum * sizeof(ShadeableIntersection));
            checkCUDAError("cudaMalloc d_bools");

            cudaMemcpy(d_ping, dev_idata, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            cudaMemset(d_bools, 0, sum * sizeof(int));
            cudaMemcpy(d_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_intersectionsPing, dev_intersections, n * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);

            // Scan
            for (int i = 0; i < breakpointsSize - 2; ++i) {
                int padding = divup(breakpoints[i + 1] - breakpoints[i], blocksize) * blocksize - (breakpoints[i + 1] - breakpoints[i]);
                kernUpsweepBlock << <divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(PathSegment) >> >
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
                kernDownsweepBlock << <divup(breakpoints[i + 1] - breakpoints[i], blocksize), blocksize, blocksize * 1 * sizeof(PathSegment) >> >
                    (breakpoints[i + 1] - breakpoints[i], d_bools + breakpoints[i + 1], d_bools + breakpoints[i], padding);
                checkCUDAError("kernDownsweepBlock");
            }

            //int* boolsHost = new int[n];
            //cudaMemcpy(boolsHost, d_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            //printf("Scanned Bools: \n");
            //for (int i = 0; i < n; ++i) {
            //    printf("%i , ", boolsHost[i]);
            //}
            //printf("\n");
            //printf("Total: %i \n", total);

            ////int* boolsHost = new int[n];
            //cudaMemcpy(boolsHost, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            //printf("dev Bools: \n");
            //for (int i = 0; i < n; ++i) {
            //    printf("%i , ", boolsHost[i]);
            //}
            //printf("\n");
            //printf("Total: %i \n", total);
            //delete[] boolsHost;

            // Scatter
            kernScatterRadixIntersect << <divup(n, blocksize), blocksize >> > (n, d_pong, d_ping, d_bools, total, dev_bools, d_intersectionsPing, dev_intersections);
            checkCUDAError("scatter");

            // Swap ping pong
            PathSegment* temp = d_ping;
            d_ping = d_pong;
            d_pong = temp;


            cudaMemcpy(dev_odata, d_ping, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy out to odata");

            //PathSegment* path = new PathSegment[n];
            //cudaMemcpy(path, d_ping, n * sizeof(PathSegment), cudaMemcpyDeviceToHost);
            //printf("Post Scatter: \n");
            //for (int i = 0; i < n; ++i) {
            //    printf("%i , ", path[i].pixelIndex);
            //}
            //printf("\n");
            //delete[] path;

            cudaFree(d_ping);
            cudaFree(d_pong);
            cudaFree(d_bools);
            cudaFree(d_intersectionsPing);
            return total;
        }
    }
}
