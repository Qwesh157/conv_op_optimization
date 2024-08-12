#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "./include/cute/tensor.hpp"
#include "./include/verify.h"
#include "./src/implgemm.cu"
using namespace std;
using namespace cute;
int main(int argc, char **argv)
{
    unsigned int n = atoi(argv[1]);
    unsigned int c = atoi(argv[2]);
    unsigned int h = atoi(argv[3]);
    unsigned int w = atoi(argv[4]);
    unsigned int k = atoi(argv[5]);
    unsigned int r = atoi(argv[6]);
    unsigned int s = atoi(argv[7]);

    unsigned int p = h - r + 1;
    unsigned int q = w - s + 1;

    using T = float;

    double M=k;
    double N=n*p*q;
    double K=c*r*s;
    double temp=n*p*q*1e-9f;
    double flopsPerConv = temp * M * K * 2.0;
    
    T *activations = (T *)malloc(n * c * h * w * sizeof(T));
    T *filter = (T *)malloc(k * c * r * s * sizeof(T));
    T *output = (T *)malloc(n * k * p * q * sizeof(T));
    T *output_host = (T *)malloc(n * k * p * q * sizeof(T));

    T *activations_device, *filter_device, *output_device;
    cudaMalloc((void **)&activations_device, n * c * h * w * sizeof(T));
    cudaMalloc((void **)&filter_device, k * c * r * s * sizeof(T));
    cudaMalloc((void **)&output_device, n * k * p * q * sizeof(T));

    for (int i = 0; i < n * c * h * w; i++)
    {
        activations[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        filter[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * k * p * q; i++)
    {
        output[i] = 0.0;
        output_host[i] = 0.0;
    }

    auto trans_act_layout = make_layout(
        make_shape(make_shape(q, p, n), make_shape(c, s, r)),
        make_stride(make_stride(c, c * w, c * w * h), make_stride(_1{}, c, c * w)));
    
    auto filter_layout = make_layout(
        make_shape(k, make_shape(c, s, r)),
        make_stride(c * s * r, make_stride(_1{}, c, c * s)));

    auto output_layout = make_layout(
        make_shape(k, make_shape(q, p, n)),
        make_stride(_1{}, make_stride(k, q * k, p * q * k)));

    Tensor act = make_tensor(make_gmem_ptr(activations_device), trans_act_layout);
    Tensor flt = make_tensor(make_gmem_ptr(filter_device), filter_layout);
    Tensor out = make_tensor(make_gmem_ptr(output_device), output_layout);
    
    Tensor act_host = make_tensor(make_gmem_ptr(activations), trans_act_layout);
    Tensor flt_host = make_tensor(make_gmem_ptr(filter), filter_layout);
    Tensor out_host = make_tensor(make_gmem_ptr(output), output_layout);

    cudaMemcpy(activations_device, activations, n * c * h * w * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_device, filter, k * c * r * s * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, n * k * p * q * sizeof(T), cudaMemcpyHostToDevice);

    int npq = static_cast<int>(size<1>(out));
    int rsc = static_cast<int>(size<1>(flt));

    // warm up
    launch_implgemm<T>(flt, act, out, k, npq, rsc);
    cudaMemcpy(output_host, output_device, n * k * p * q * sizeof(T), cudaMemcpyDeviceToHost);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        launch_implgemm<T>(flt, act, out, k, npq, rsc);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    float timePerConv=time_elapsed / iternum;
    double gflops =
        (flopsPerConv) / (timePerConv / 1000.0f);
    printf("time: %f ms\n", timePerConv);
    printf("Performance: %f Flops\n", gflops);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cpu direct conv
    direct_nhwc_conv2dcpu<T>(activations, filter, output, n, c, h, w, k, r, s, 1, 1, 0, 0);
    // cpu CuTe conv(make sure you got the right Tensor layout to use CuTe cpu function to verify result)
    // cute_conv2d_nhwc_cpu(flt_host, act_host, out_host);

    int error = 0;
    float threshold = 0.1;
    for (int i = 0; i < n * k * p * q; i++)
    {
        if (fabs(output_host[i] - output[i]) > threshold)
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
            error++;
        }
    }
    printf("Total error: %d\n", error);

    cudaFree(activations_device);
    cudaFree(filter_device);
    cudaFree(output_device);

    free(activations);
    free(filter);
    free(output);
    free(output_host);

    return 0;
}