#include <stdio.h>
#include <cuda_runtime.h>
#include "verify.h"
#include "cudnn.h"

int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);
    int h = atoi(argv[3]);
    int w = atoi(argv[4]);
    int k = atoi(argv[5]);
    int r = atoi(argv[6]);
    int s = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int p = atoi(argv[10]);
    int q = atoi(argv[11]);

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;
    double M = c;
    double N = n * h * w;
    double K = k * r * s;
    double temp = N * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0; 

    float *grad_input = (float *)malloc(n * c * h * w * sizeof(float));
    float *grad_input_host = (float *)malloc(n * c * h * w * sizeof(float));
    float *weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *grad_output = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *weight_device, *grad_input_device, *grad_output_device;
    cudaMalloc((void **)&grad_input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&grad_output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        grad_input[i] = 0.0;
        grad_input_host[i] = 0.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        weight[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        grad_output[i] = (rand() % 255) / 255.0;
    }

    cudaMemcpy(grad_input_device, grad_input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_output_device, grad_output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    cudnnStatus_t status;
    cudnnHandle_t handle;
    status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreate failed\n");
    cudnnTensorDescriptor_t grad_input_descriptor;
    status = cudnnCreateTensorDescriptor(&grad_input_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateTensorDescriptor grad_input_descriptor failed\n");
    status = cudnnSetTensor4dDescriptor(grad_input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/n,
                                        /*channels=*/c,
                                        /*image_height=*/h,
                                        /*image_width=*/w);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor grad_input_descriptor failed\n");
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

    // create grad output descriptor
    cudnnTensorDescriptor_t grad_output_descriptor;
    status = cudnnCreateTensorDescriptor(&grad_output_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateTensorDescriptor grad_output_descriptor failed\n");
    status = cudnnSetTensor4dDescriptor(grad_output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/n,
                                        /*channels=*/k,
                                        /*image_height=*/outh,
                                        /*image_width=*/outw);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor grad_output_descriptor failed\n");

    cudnnConvolutionBwdDataAlgoPerf_t perfResults[7];
    int returnedAlgoCount;
    cudnnFindConvolutionBackwardDataAlgorithm(handle,
                                              kernel_descriptor,
                                              grad_output_descriptor,
                                              convolution_descriptor,
                                              grad_input_descriptor,
                                              7,
                                              &returnedAlgoCount,
                                              perfResults);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnFindConvolutionBackwardDataAlgorithm failed\n");
    
    // print all available convolution backward data algorithm , ordered by time
    // for (int i = 0; i < 7; i++)
    // {
    //     printf("Algorithm %d: %d, time: %f\n", i, perfResults[i].algo, perfResults[i].time);
    // }

    // cuDNN all convolution backward data algorithm

    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0, /* non-deterministic */
    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1,
    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2,
    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3,
    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4,
    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    // CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT             = 6
    
    cudnnConvolutionBwdDataAlgo_t convolution_algorithm = (cudnnConvolutionBwdDataAlgo_t)0;  //choose implicit gemm

    int size;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                     kernel_descriptor,
                                                     grad_output_descriptor,
                                                     convolution_descriptor,
                                                     grad_input_descriptor,
                                                     convolution_algorithm,
                                                     (size_t *)&(size));
    // printf("Workspace size: %zu bytes\n", size);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnGetConvolutionBackwardDataWorkspaceSize failed\n");
    float *extra;
    cudaMalloc((void **)&extra, size);

    float alpha = 1.0, beta = 0.0;

    status = cudnnConvolutionBackwardData(handle, &alpha,
                                          kernel_descriptor, weight_device,grad_output_descriptor, grad_output_device, 
                                          convolution_descriptor, convolution_algorithm,
                                          extra, size, &beta,
                                          grad_input_descriptor, grad_input_device);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("Not Successed!\n");

    cudaMemcpy(grad_input_host, grad_input_device, n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        cudnnConvolutionBackwardData(handle, &alpha,
                                     kernel_descriptor, weight_device,grad_output_descriptor, grad_output_device, 
                                     convolution_descriptor, convolution_algorithm,
                                     extra, size, &beta,
                                     grad_input_descriptor, grad_input_device);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // printf("===================start verfiy===================\n");
    // direct_conv2dbwddatacpu(grad_input, weight, grad_output, n, c, h, w, k, r, s, u, v, p, q, outh, outw);

    // int bwddataerror = 0;
    // for (int i = 0; i < n * c * h * w; i++)
    // {
    //     if (abs(grad_input_host[i] - grad_input[i]) > getPrecision(grad_input[i]))
    //     {
    //         printf("Backward data error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, grad_input_host[i], grad_input[i]);
    //         bwddataerror++;
    //         break;
    //     }
    // }
    // printf("========finish, Backward data error:%d=============\n", bwddataerror);

    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n",  gflops);

    cudaFree(grad_input_device);
    cudaFree(weight_device);
    cudaFree(grad_output_device);

    free(grad_input);
    free(grad_input_host);
    free(weight);
    free(grad_output);

    return 0;
}