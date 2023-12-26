#include <cstdint>
#include <cuda_runtime.h>
#include "conv2d.h"
/*
    外积实现矩阵乘，重新安排线程、smem padding、ldg与sts分离避免bank conflict
*/
__global__ void implgemm(param_t param)
{
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Warp tile
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    uint32_t weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    uint32_t input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    __shared__ float smeminput[8 * 128];
    __shared__ float smemweight[8 * 132];

    float weight_ldg_reg[4];
    float input_ldg_reg[4];
    // 当前线程处理的数据点在oh、ow上的坐标
    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.u - param.p;
        posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.v - param.q;
    }

    int inOffset = z * param.c * param.h * param.w;
    int weiOffset = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // sts addr
    uint32_t weight_sts_addr = (tx % 8) * 132 +
                               (tx / 8) * 4;
    uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);

    float weight_frag[8];
    float input_frag[8];
    float output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }

    for (int crs = 0; crs < param.r * param.s * param.c; crs += 8)
    {
        //ldg
        int weiOffsetTmp = crs + tx % 8;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (weiOffsetTmp < weightKOffset)
            {
                weight_ldg_reg[i] = param.weight[weiOffset + weiOffsetTmp + i * weightKOffset];
            }
            else
            {
                weight_ldg_reg[i] = 0.0;
            }
        }
        int curC = (crs + tx / 32) / (param.r * param.s);             // channel offset
        int curR = ((crs + tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
        int curS = ((crs + tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori[i] + curR; // input h
            int curW = posw_ori[i] + curS; // input w
            int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h)
            {
                input_ldg_reg[i] = param.input[inOffset + inOffsetTmp];
            }
            else
            {
                input_ldg_reg[i] = 0.0;
            }
        }
        //sts
        for (int i = 0; i < 4; ++i)
        {
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];  
        }
        for (int i = 0; i < 4; ++i)
        {
            smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];  
        } 
        __syncthreads();
#pragma unroll
        for (int subcrs = 0; subcrs < 8; ++subcrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[i] = smemweight[weight_lds_addr + subcrs * 132 + i];
                weight_frag[i + 4] = smemweight[weight_lds_addr + subcrs * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[i] = smeminput[input_lds_addr + subcrs * 128 + i];
                input_frag[i + 4] = smeminput[input_lds_addr + subcrs * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[i] * input_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 计算输出偏移
    int outOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            outOffset = z * param.k * param.Oh * param.Ow + (y + i) * param.Oh * param.Ow + x + j;
            if (x + j < param.Oh * param.Ow && y + i < param.k)
            {
                param.output[outOffset] = output_frag[i][j];
            }
            outOffset = z * param.k * param.Oh * param.Ow + (y + i) * param.Oh * param.Ow + x + j + 32;
            if (x + j + 32 < param.Oh * param.Ow && y + i < param.k)
            {
                param.output[outOffset] = output_frag[i][j + 4];
            }
            outOffset = z * param.k * param.Oh * param.Ow + (y + i + 16) * param.Oh * param.Ow + x + j;
            if (x + j < param.Oh * param.Ow && y + i + 16 < param.k)
            {
                param.output[outOffset] = output_frag[i + 4][j];
            }
            outOffset = z * param.k * param.Oh * param.Ow + (y + i + 16) * param.Oh * param.Ow + x + j + 32;
            if (x + j + 32 < param.Oh * param.Ow && y + i + 16 < param.k)
            {
                param.output[outOffset] = output_frag[i + 4][j + 4];
            }
        }
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

    int blockx = ((outh * outw + 127) / 128); // blockx  number
    int blocky = (k + 127) / 128;             // blocky  number
    int blockz = n;                           // blockz  number
    // 合并threadx与thready
    int threadx = 256; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm<<<grid, block>>>(param);
}
