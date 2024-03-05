#include <stdio.h>
#include <cuda_runtime.h>
#include "verify.h"
#include "cudnn.h"

int main(int argc, char **argv)
{
    unsigned int n = atoi(argv[1]);
    unsigned int c = atoi(argv[2]);
    unsigned int h = atoi(argv[3]);
    unsigned int w = atoi(argv[4]);
    unsigned int k = atoi(argv[5]);
    unsigned int r = atoi(argv[6]);
    unsigned int s = atoi(argv[7]);
    unsigned int u = atoi(argv[8]);
    unsigned int v = atoi(argv[9]);
    unsigned int p = atoi(argv[10]);
    unsigned int q = atoi(argv[11]);

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;
    double M = k;
    double N = n * outh * outw;
    double K = c * r * s;
    double temp = n * outh * outw * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0;
    float *input = (float *)malloc(n * c * h * w * sizeof(float));
    float *weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *output = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *output_host = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *input_device, *weight_device, *output_device;
    cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        weight[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        output[i] = 0.0;
        output_host[i] = 0.0;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_device, output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    
    cudnnStatus_t status;
    cudnnHandle_t handle;
    status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreate failed\n");
    cudnnTensorDescriptor_t input_descriptor;
    status = cudnnCreateTensorDescriptor(&input_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateTensorDescriptor input_descriptor failed\n");
    status = cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/n,
                                        /*channels=*/c,
                                        /*image_height=*/h,
                                        /*image_width=*/w);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor input_descriptor failed\n");
    cudnnFilterDescriptor_t kernel_descriptor;
    status = cudnnCreateFilterDescriptor(&kernel_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateTensorDescriptor kernel_descriptor failed\n");
    status = cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/k,
                                        /*in_channels=*/c,
                                        /*kernel_height=*/r,
                                        /*kernel_width=*/s);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor kernel_descriptor failed\n");
    cudnnConvolutionDescriptor_t convolution_descriptor;
    status = cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateConvolutionDescriptor convolution_descriptor failed\n");
    status = cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/p,
                                             /*pad_width=*/q,
                                             /*vertical_stride=*/u,
                                             /*horizontal_stride=*/v,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION, // how to compute
                                             /*computeType=*/CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor convolution_descriptor failed\n");

    cudnnMathType_t mathType=CUDNN_FMA_MATH;  //choose FMA or Tensor math

    // CUDNN_DEFAULT_MATH                    = 0,
    // CUDNN_TENSOR_OP_MATH                  = 1,
    // CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    // CUDNN_FMA_MATH                        = 3,

    status = cudnnSetConvolutionMathType(convolution_descriptor,mathType);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetConvolutionMathType failed\n");


    // create output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    status = cudnnCreateTensorDescriptor(&output_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateTensorDescriptor output_descriptor failed\n");
    status = cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/n,
                                        /*channels=*/k,
                                        /*image_height=*/outh,
                                        /*image_width=*/outw);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor output_descriptor failed\n");

    cudnnConvolutionFwdAlgoPerf_t perfResults[9];
    int returnedAlgoCount;
    status = cudnnFindConvolutionForwardAlgorithm(handle,
                                                  input_descriptor,
                                                  kernel_descriptor,
                                                  convolution_descriptor,
                                                  output_descriptor,
                                                  9,
                                                  &returnedAlgoCount,
                                                  perfResults);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnFindConvolutionForwardAlgorithm failed\n");
    
    // print all available convolution forward algorithm , ordered by time
    // for (int i = 0; i < 9; i++)
    // {
    //     printf("Algorithm %d: %d, time: %f\n", i, perfResults[i].algo, perfResults[i].time);
    // }

    // cuDNN all convolution forward algorithm

    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
    // CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2
    // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3
    // CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4
    // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5
    // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6
    // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7
    // CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8

    cudnnConvolutionFwdAlgo_t convolution_algorithm = (cudnnConvolutionFwdAlgo_t)0;  //choose implicit gemm

    int size;
    status = cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     (size_t *)&(size));
    // printf("Workspace size: %zu bytes\n", size);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnGetConvolutionForwardWorkspaceSize failed\n");
    float *extra;
    cudaMalloc((void **)&extra, size);

    float alpha = 1.0, beta = 0.0;

    status = cudnnConvolutionForward(handle, &alpha,
                                     input_descriptor, input_device, kernel_descriptor, weight_device,
                                     convolution_descriptor, convolution_algorithm,
                                     extra, size, &beta,
                                     output_descriptor, output_device);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("Not Successed!\n");
    cudaMemcpy(output_host, output_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        cudnnConvolutionForward(handle, &alpha,
                             input_descriptor, input_device, kernel_descriptor, weight_device,
                             convolution_descriptor, convolution_algorithm,
                             extra, size, &beta,
                             output_descriptor, output_device);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // printf("===================start verfiy===================\n");
    // direct_conv2dcpu(input, weight, output, n, c, h, w, k, r, s, u, v, p, q);

    // int error = 0;
    // for (int i = 0; i < n * k * outh * outw; i++)
    // {
    //     if (abs(output_host[i] - output[i]) > getPrecision(output[i]))
    //     {
    //         printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
    //         error++;
    //         break;
    //     }
    // }
    // printf("================finish,error:%d=========================\n", error);

    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n",  gflops);

    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(output_device);

    free(input);
    free(weight);
    free(output);
    free(output_host);

    return 0;
}