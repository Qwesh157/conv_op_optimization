#include <stdio.h>
#include <cuda_runtime.h>
// #include <cuda_ext.h>
#include "verify.h"
#include "conv2d.h"

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

    //Convolution parameter

    param_t param;

    param.input = input_device;
    param.weight = weight_device;
    param.output = output_device;
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

    /********************************** step 2****************************/


    /*******************************warm up and get result************************************/
    launch_implgemm(param);

    cudaMemcpy(output_host, output_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for (int i = 0; i < iternum; i++)
    {
        launch_implgemm(param);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    printf("time: %f ms\n", time_elapsed / iternum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("===================start verfiy===================\n");
    direct_conv2dcpu(input, weight, output, n, c, h, w, k, r, s, u, v, p, q);

    int error = 0;
    for (int i = 0; i < n * k * outh * outw; i++)
    {
        if (abs(output_host[i] - output[i]) > getPrecision(output[i]))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
            error++;
            break;
        }
    }
    printf("================finish,error:%d=========================\n", error);

    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(output_device);

    free(input);
    free(weight);
    free(output);
    free(output_host);

    return 0;
}