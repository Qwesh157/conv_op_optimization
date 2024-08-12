# Convolution Operator Optimization
## Introduction

This project is about convolution operator optimization on GPU

## Content
 - [x] Cuda core Implicit GEMM forward
 - [x] Cuda core Implicit GEMM backward
 - [x] CuTe Implicit GEMM

This [blog](https://zhuanlan.zhihu.com/p/661879423) provides a detailed introduction to the optimization steps.

[/cuda](https://github.com/Qwesh157/conv_op_optimization/tree/main/cuda) Implementation on GPU  
&emsp;&emsp;[/implicitgemm](https://github.com/Qwesh157/conv_op_optimization/tree/main/cuda/implicitgemm) implicit gemm convolution implementation  
&emsp;&emsp;[/implicitgemmbwd](https://github.com/Qwesh157/conv_op_optimization/tree/main/cuda/implicitgemmbwd) implicit gemm convolution backward implementation  
[/cudnn](https://github.com/Qwesh157/conv_op_optimization/tree/main/cudnn) cuDNN test on GPU  
[/cute](https://github.com/Qwesh157/conv_op_optimization/tree/main/cute) Using CuTe implement convolution

## Build and run

```bash
$ cd cuda/implicitgemm
$ bash implgemm.sh
```

If you want to change the version of program, just change TARGET in Makefile

## Verification

There is verification code in main.cu, which was annotated due to slow running.
```cpp
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
```
If you need to verify the result, just unannotate the above code to verify the correctness of the results.

## TODO
 - [ ] Tensore core Implicit GEMM
 - [ ] Winograd-based convolution
