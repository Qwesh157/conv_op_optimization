#include <cstdint>
#include <cuda_runtime.h>
#include "../include/conv2dbwd.h"
/*
    Implicitgemm反向卷积计算实现
*/
__global__ void implgemmbwddata(param_t param)
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
    uint32_t gradoutput_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + gradoutput_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    __shared__ float smemgradoutput[8 * 128];
    __shared__ float smemweight[8 * 132];

    float weight_ldg_reg[4];
    float gradoutput_ldg_reg[4];
    // 当前线程处理的数据点在oh、ow上的坐标
    int posOh_ori[4];
    int posOw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posOh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.w) - param.r + 1;
        posOw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.w) - param.s + 1;
    }

    int outOffset = z * param.k * param.Oh * param.Ow;
    int weiC = (by * 128 + tx / 8 * 4);
    int outKOffset = param.Oh * param.Ow;
    int weiCOffset = param.r * param.s;
    int weiKOffset = param.c * param.r * param.s;

    // sts addr
    uint32_t weight_sts_addr = (tx % 8) * 132 +
                               (tx / 8) * 4;
    uint32_t gradoutput_sts_addr = (tx / 32) * 128 + (tx % 32);

    float weight_frag[8];
    float gradoutput_frag[8];
    float gradinput_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            gradinput_frag[i][j] = 0;
        }
    }

    for (int krs = 0; krs < param.r * param.s * param.k; krs += 8)
    {
        // ldg
        int curKRS = krs + tx % 8;
        int rs = param.r * param.s - 1 - curKRS % (param.r * param.s);
        int curK = curKRS / (param.r * param.s);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if ((curK * param.r * param.s + rs) < param.r * param.s * param.k)
            {
                weight_ldg_reg[i] = param.weight[curK * weiKOffset + (weiC + i) * weiCOffset + rs];
            }
            else
            {
                weight_ldg_reg[i] = 0.0;
            }
        }
        int curK2 = (krs + tx / 32) / (param.r * param.s);            // kernel k offset
        int curR = ((krs + tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
        int curS = ((krs + tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curOh = posOh_ori[i] + curR; // gradinput h
            int curOw = posOw_ori[i] + curS; // gradinput w
            int outOffsetTmp = curK2 * outKOffset + curOh * param.Ow + curOw;
            if (curOh >= 0 && curOw >= 0 && curOw < param.Ow && curOh < param.Oh)
            {
                gradoutput_ldg_reg[i] = param.grad_output[outOffset + outOffsetTmp];
            }
            else
            {
                gradoutput_ldg_reg[i] = 0.0;
            }
        }
        // sts
        for (int i = 0; i < 4; ++i)
        {
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            smemgradoutput[gradoutput_sts_addr + i * 32] = gradoutput_ldg_reg[i];
        }
        __syncthreads();
#pragma unroll
        for (int subkrs = 0; subkrs < 8; ++subkrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[i] = smemweight[weight_lds_addr + subkrs * 132 + i];
                weight_frag[i + 4] = smemweight[weight_lds_addr + subkrs * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                gradoutput_frag[i] = smemgradoutput[gradoutput_lds_addr + subkrs * 128 + i];
                gradoutput_frag[i + 4] = smemgradoutput[gradoutput_lds_addr + subkrs * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradinput_frag[i][j] += weight_frag[i] * gradoutput_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 计算输出偏移
    int gradinputOffset;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            gradinputOffset = z * param.c * param.h * param.w + (y + i) * param.h * param.w + x + j;
            if (x + j < param.h * param.w && y + i < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i][j];
            }
            gradinputOffset = z * param.c * param.h * param.w + (y + i) * param.h * param.w + x + j + 32;
            if (x + j + 32 < param.h * param.w && y + i < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i][j + 4];
            }
            gradinputOffset = z * param.c * param.h * param.w + (y + i + 16) * param.h * param.w + x + j;
            if (x + j < param.h * param.w && y + i + 16 < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i + 4][j];
            }
            gradinputOffset = z * param.c * param.h * param.w + (y + i + 16) * param.h * param.w + x + j + 32;
            if (x + j + 32 < param.h * param.w && y + i + 16 < param.c)
            {
                param.grad_input[gradinputOffset] = gradinput_frag[i + 4][j + 4];
            }
        }
    }
}
void launch_implgemmbwd(param_t param)
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
    unsigned int outh = param.Oh;
    unsigned int outw = param.Ow;

    int blockx = ((h * w + 127) / 128); // blockx  number
    int blocky = (c + 127) / 128;             // blocky  number
    int blockz = n;                           // blockz  number
    // 合并threadx与thready
    int threadx = 256; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 blockbwddata(threadx, thready, threadz);
    dim3 gridbwddata(blockx, blocky, blockz);
    implgemmbwddata<<<gridbwddata, blockbwddata>>>(param);
}
