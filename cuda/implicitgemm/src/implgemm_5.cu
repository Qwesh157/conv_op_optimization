#include <cstdint>
#include <cuda_runtime.h>
#include "conv2d.h"
/*
    线程通过共享内存交换数据，然后使用高效的条带访问模式协作访问全局内存，加入 bias Epilogue
*/
__global__ void implgemm(param_t param)
{
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 16 * 1024);

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

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
    int weight_sts_addr = (tx % 8) * 132 +
                          (tx / 8) * 4;
    int input_sts_addr = (tx / 32) * 128 + (tx % 32);

    int write_flag = 1;
    float weight_frag[2][8];
    float input_frag[2][8];
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
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (tx % 8 < weightKOffset)
        {
            weight_ldg_reg[i] = param.weight[weiOffset + tx % 8 + i * weightKOffset];
        }
        else
        {
            weight_ldg_reg[i] = 0.0;
        }
    }
    int curC = (tx / 32) / (param.r * param.s);             // channel offset
    int curR = ((tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
    int curS = ((tx / 32) % (param.r * param.s)) % param.s; // kernel s offset
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
    // sts
    for (int i = 0; i < 4; ++i)
    {
        smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
    }

    __syncthreads();
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        weight_frag[0][i] = smemweight[weight_lds_addr + i];
        weight_frag[0][i + 4] = smemweight[weight_lds_addr + i + 16];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        input_frag[0][i] = smeminput[input_lds_addr + i];
        input_frag[0][i + 4] = smeminput[input_lds_addr + i + 32];
    }
    for (int crs = 0; crs < param.r * param.s * param.c; crs += 8)
    {
        // ldg
        int weiOffsetTmp = crs + 8 + tx % 8;
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
        curC = (crs + 8 + tx / 32) / (param.r * param.s);             // channel offset
        curR = ((crs + 8 + tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
        curS = ((crs + 8 + tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

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
        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i];
                weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i];
                input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                }
            }
        }
        // sts
        for (int i = 0; i < 4; ++i)
        {
            smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        write_flag ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
            weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            input_frag[0][i] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i];
            input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i + 32];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
            }
        }
    }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);
    float *smembias = reinterpret_cast<float *>(smem + 16 * 1024);

    // bias ldg/sts
    if (tx < 128)
    {
        smembias[tx] = param.bias[by * 128 + tx];
    }

    uint32_t output_sts_addr = warp_id * 512 + mma_tid_y * 4 * 8 * 4 + mma_tid_x * 4;
    uint32_t output_lds_addr = warp_id * 512 + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            __syncthreads();

#pragma unroll
            for (int subi = 0; subi < 4; ++subi)
            {
#pragma unroll
                for (int subj = 0; subj < 4; ++subj)
                {
                    // output sts
                    smemoutput[output_sts_addr + subi * 8 * 4 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                }
            }
            __syncthreads();

#pragma unroll
            for (int subk = 0; subk < 16; ++subk)
            {
                int outOffset = z * param.k * param.Oh * param.Ow + (m_idx + i * 16 + subk) * param.Oh * param.Ow + n_idx + j * 32;
                if ((m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                    param.output[outOffset] = smemoutput[output_lds_addr + subk * 32] + smembias[m_idx + i * 16 + subk];
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
