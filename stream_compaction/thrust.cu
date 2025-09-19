#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            int *d_idata, *d_odata;
            cudaMalloc((void**)&d_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_idata");
            cudaMalloc((void**)&d_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_odata");

            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into d_idata");
            
            thrust::device_ptr<int> d_thrust_idata = thrust::device_ptr<int>(d_idata);
            thrust::device_ptr<int> d_thrust_odata = thrust::device_ptr<int>(d_odata);
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            
            thrust::exclusive_scan(d_thrust_idata, d_thrust_idata + n, d_thrust_odata);

            timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy d_odata into odata");

            cudaFree(d_idata);
            cudaFree(d_odata);
        }
    }
}
