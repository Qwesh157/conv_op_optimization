#include <cuda_runtime.h>
#include "conv2d.h"
/*
    Naive version
*/
__global__ void implgemm(param_t param)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= param.Oh * param.Ow || y >= param.k || z >= param.n)
    {
        return;
    }

    // 当前线程处理的数据点在oh、ow上的坐标
    int posOh = x / param.Ow;
    int posOw = x % param.Ow;

    int posh_ori = posOh * param.u - param.p;
    int posw_ori = posOw * param.v - param.q;

    float sum = 0.0;

    int inOffset = z * param.c * param.h * param.w + posh_ori * param.w + posw_ori;
    int weiOffset = y * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;

    for (int i = 0; i < param.r; i++)
    {
        for (int j = 0; j < param.s; j++)
        {
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;

            if (posh_real >= 0 && posw_real >= 0 && posw_real < param.w && posh_real < param.h)
            {
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for (int channel = 0; channel < param.c; channel++)
                {
                    sum += param.input[inOffsetTmp + i * param.w + j] * param.weight[weiOffsetTmp + i * param.s + j];
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }
            }
        }
    }

    // 计算输出偏移
    int outOffset = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[outOffset] = sum;
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

    param.Oh = outh;
    param.Ow = outw;

    int blockx =  ((outh * outw + 15) / 16); // blockx  number
    int blocky = (k + 15) / 16;           // blocky  number
    int blockz = n;                       // blockz  number
    int threadx = 16;                     // threadx number per block
    int thready = 16;                     // thready number per block
    int threadz = 1;                      // threadz number per block
    dim3 block(threadx, thready,threadz);
    dim3 grid(blockx, blocky,blockz);
    implgemm<<<grid,block>>>(param);
}