#include <cuda_runtime.h>
#include "conv2d.h"
/*
    合并KRS维度并使用shared memory减少global memory的访问次数
*/
__global__ void implgemm(param_t param)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * 16 + tx;
    int y = by * 16 + ty;
    int z = blockIdx.z;

    // if (y >= param.k || z >= param.n)
    // {
    //     return;
    // }
    __shared__ float smeminput[16][16];
    __shared__ float smemweight[16][16];
    // 当前线程处理的数据点在oh、ow上的坐标
    int posOh = x / param.Ow;
    int posOw = x % param.Ow;

    int posh_ori = posOh * param.u - param.p;
    int posw_ori = posOw * param.v - param.q;

    float sum = 0.0;

    int inOffset = z * param.c * param.h * param.w;
    int weiOffset = y * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;
    for (int i = 0; i < param.r * param.s * param.c; i += 16)
    {
        int weiOffsetTmp = i + tx;
        smemweight[ty][tx] = param.weight[weiOffset + weiOffsetTmp];

        int curC = (i + ty) / (param.r * param.s);             // channel offset
        int curR = ((i + ty) % (param.r * param.s)) / param.s; // kernel r offset
        int curS = ((i + ty) % (param.r * param.s)) % param.s; // kernel s offset
        int curH = posh_ori + curR;                            // input h
        int curW = posw_ori + curS;                            // input w
        int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
        smeminput[ty][tx] = param.input[inOffset + inOffsetTmp];

        __syncthreads();
#pragma unroll
        for (int subcrs = 0; subcrs < 16; ++subcrs)
        {
            sum += smemweight[ty][subcrs] * smeminput[subcrs][tx];
        }
        __syncthreads();
    }

    // 计算输出偏移
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    if (x < param.Oh * param.Ow && y < param.k)
    {
        param.output[outOffset] = sum;
    }
}
void launch_implgemm(param_t param)
{
    unsigned int n = param.n;
    unsigned int c = param.c;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;

    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;


    int blockx = ((outh * outw + 15) / 16); // blockx  number
    int blocky = (k + 15) / 16;             // blocky  number
    int blockz = n;                         // blockz  number
    int threadx = 16;                       // threadx number per block
    int thready = 16;                       // thready number per block
    int threadz = 1;                        // threadz number per block
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm<<<grid, block>>>(param);
}