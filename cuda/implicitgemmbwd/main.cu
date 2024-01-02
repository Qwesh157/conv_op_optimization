#include <stdio.h>
#include <cuda_runtime.h>
// #include <cuda_ext.h>
#include "verify.h"
#include "conv2dbwd.h"

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
    M = c;
    N = n * h * w;
    K = k * r * s;
    temp = N * 1e-9f;
    flopsPerConv += temp * M * K * 2.0;
    float *input = (float *)malloc(n * c * h * w * sizeof(float));
    float *grad_input = (float *)malloc(n * c * h * w * sizeof(float));
    float *grad_input_host = (float *)malloc(n * c * h * w * sizeof(float));
    float *weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *grad_weight = (float *)malloc(k * c * r * s * sizeof(float));
    float *grad_weight_host = (float *)malloc(k * c * r * s * sizeof(float));
    float *grad_output = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *input_device, *weight_device, *grad_input_device, *grad_weight_device, *grad_output_device;
    cudaMalloc((void **)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&grad_input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&grad_weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&grad_output_device, n * k * outh * outw * sizeof(float));

    for (int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() % 255) / 255.0;
        grad_input[i] = 0.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        weight[i] = (rand() % 255) / 255.0;
        grad_weight[i] = 0.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        grad_output[i] = (rand() % 255) / 255.0;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_input_device, grad_input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_weight_device, grad_weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_output_device, grad_output, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    // Convolution parameter

    param_t param;

    param.input = input_device;
    param.grad_input = grad_input_device;
    param.weight = weight_device;
    param.grad_weight = grad_weight_device;
    param.grad_output = grad_output_device;
    param.n = n;
    param.c = c;
    param.h = h;
    param.w = w;
    param.k = k;
    param.r = r;
    param.s = s;
    param.u = u;
    param.v = v;
    param.p = p;
    param.q = q;
    param.Oh = outh;
    param.Ow = outw;
    /********************************** step 2****************************/

    /*******************************warm up and get result************************************/
    launch_implgemmbwd(param);

    cudaMemcpy(grad_input_host, grad_input_device, n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_weight_host, grad_weight_device, k * c * r * s * sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        launch_implgemmbwd(param);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    float timePerConv = time_elapsed / iternum;
    double gflops = flopsPerConv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n",  gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // verify
    // printf("===================start verfiy===================\n");
    // direct_conv2dbwddatacpu(grad_input, weight, grad_output, n, c, h, w, k, r, s, u, v, p, q, outh, outw);
    // direct_conv2dbwdfiltercpu(input, grad_weight, grad_output, n, c, h, w, k, r, s, u, v, p, q, outh, outw);

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
    cudaFree(input_device);
    cudaFree(grad_input_device);
    cudaFree(weight_device);
    cudaFree(grad_weight_device);
    cudaFree(grad_output_device);

    free(input);
    free(grad_input);
    free(grad_input_host);
    free(weight);
    free(grad_weight);
    free(grad_output);

    return 0;
}