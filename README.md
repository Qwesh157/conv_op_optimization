# Convolution Operator Optimization
## Introduction

This project is about convolution operator optimization on GPU

## Content

[/cuda](https://github.com/Qwesh157/conv_op_optimization/tree/main/cuda) Implementation on GPU  
&emsp;&emsp;[/implicitgemm](https://github.com/Qwesh157/conv_op_optimization/tree/main/cuda/implicitgemm) implicit gemm convolution implementation  
&emsp;&emsp;[/implicitgemmbwd](https://github.com/Qwesh157/conv_op_optimization/tree/main/cuda/implicitgemmbwd) implicit gemm convolution backward implementation  

## Build and run

```bash
$ cd cuda/implicitgemm
$ bash implgemm.sh
```

If you want to change the version of program,just change TARGET in Makefile
