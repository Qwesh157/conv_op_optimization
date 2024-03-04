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
    double N = c * r * s;
    double K = n * outh * outw;
    double temp = K * 1e-9f;
    double flopsPerConv = temp * M * N * 2.0;

    float *input = (float *)malloc(n * c * h * w * sizeof(float));
    float *grad_weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *grad_weight_host = (float *)malloc(k * c * r * s * sizeof(float));
    float *grad_output = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *input_device, *grad_weight_device, *grad_output_device;
    cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&grad_weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&grad_output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        grad_weight[i] = 0.0;
        grad_weight_host[i] = 0.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        grad_output[i] = (rand() % 255) / 255.0;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_weight_device, grad_weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_output_device, grad_output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

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
    cudnnFilterDescriptor_t grad_kernel_descriptor;
    status = cudnnCreateFilterDescriptor(&grad_kernel_descriptor);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnCreateTensorDescriptor grad_kernel_descriptor failed\n");
    status = cudnnSetFilter4dDescriptor(grad_kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/k,
                                        /*in_channels=*/c,
                                        /*kernel_height=*/r,
                                        /*kernel_width=*/s);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnSetTensor4dDescriptor grad_kernel_descriptor failed\n");
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

    // create grad_output descriptor
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

    cudnnConvolutionBwdFilterAlgoPerf_t perfResults[8];
    int returnedAlgoCount;
    status = cudnnFindConvolutionBackwardFilterAlgorithm(handle,
                                                input_descriptor,
                                                grad_output_descriptor,
                                                convolution_descriptor,
                                                grad_kernel_descriptor,
                                                8,
                                                &returnedAlgoCount,
                                                perfResults);
    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnFindConvolutionBackwardFilterAlgorithm failed\n");
    
    // print all available convolution backward weight algorithm , ordered by time
    // for (int i = 0; i < 8; i++)
    // {
    //     printf("Algorithm %d: %d, time: %f\n", i, perfResults[i].algo, perfResults[i].time);
    // }

    // cuDNN all convolution backward weight algorithm

    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                 = 0, /* non-deterministic */
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                 = 1,
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT               = 2,
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                 = 3, /* non-deterministic */
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD          = 4, /* not implemented */
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING        = 6,
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT             = 7
    
    cudnnConvolutionBwdFilterAlgo_t convolution_algorithm = (cudnnConvolutionBwdFilterAlgo_t)0;  //choose implicit gemm

    int size;
    status = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
                                                            input_descriptor,
                                                            grad_output_descriptor,
                                                            convolution_descriptor,
                                                            grad_kernel_descriptor,
                                                            convolution_algorithm,
                                                            (size_t *)&(size));
    // printf("Workspace size: %zu bytes\n", size);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("cudnnGetConvolutionBackwardFilterWorkspaceSize failed\n");
    float *extra;
    cudaMalloc((void **)&extra, size);

    float alpha = 1.0, beta = 0.0;

    status = cudnnConvolutionBackwardFilter(handle, &alpha,
                                            input_descriptor, input_device, grad_output_descriptor, grad_output_device,
                                            convolution_descriptor, convolution_algorithm,
                                            extra, size, &beta,
                                            grad_kernel_descriptor, grad_weight_device);

    if (status != CUDNN_STATUS_SUCCESS)
        printf("Not Successed!\n");
    cudaMemcpy(grad_weight_host, grad_weight_device, k * c * r * s * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        cudnnConvolutionBackwardFilter(handle, &alpha,
                                       input_descriptor, input_device, grad_output_descriptor, grad_output_device,
                                       convolution_descriptor, convolution_algorithm,
                                       extra, size, &beta,
                                       grad_kernel_descriptor, grad_weight_device);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // printf("===================start verfiy===================\n");
    // int bwdfiltererror = 0;
    // for (int i = 0; i < k * c * r * s; i++)
    // {
    //     if (abs(grad_weight_host[i] - grad_weight[i]) > getPrecision(grad_weight[i]))
    //     {
    //         printf("Backward filter error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, grad_weight_host[i], grad_weight[i]);
    //         bwdfiltererror++;
    //         break;
    //     }
    // }
    // printf("========finish, Backward filter error:%d===========\n", bwdfiltererror);

    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n",  gflops);
    
    cudaFree(input_device);
    cudaFree(grad_weight_device);
    cudaFree(grad_output_device);

    free(input);
    free(grad_weight);
    free(grad_weight_host);
    free(grad_output);

    return 0;
}