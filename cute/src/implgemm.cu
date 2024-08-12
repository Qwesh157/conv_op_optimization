#include <cstdint>
#include <cuda_runtime.h>
/*
    外积实现矩阵乘
*/
template <typename T, class FilterTensor, class ActivationTensor, class OutputTensor>
__global__ void implgemm(FilterTensor flt, ActivationTensor act, OutputTensor out, int k, int npq, int rsc)
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
    uint32_t A_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    uint32_t B_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + B_lds_addr;
    int y = by * 128 + A_lds_addr;

    __shared__ T smemA[8 * 132];
    __shared__ T smemB[8 * 128];

    // sts addr
    uint32_t A_sts_addr = (tx % 8) * 132 +
                          (tx / 8) * 4;
    uint32_t B_sts_addr = (tx / 32) * 128 + (tx % 32);

    // ldg addr
    uint32_t A_ldg_addr = (by * 128 + tx / 8 * 4) * rsc + tx % 8;
    uint32_t B_ldg_addr = (tx / 32) * npq + bx * 128 + tx % 32;

    T A_frag[8];
    T B_frag[8];
    T output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }

    for (int subrsc = 0; subrsc < rsc; subrsc += 8)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemA[A_sts_addr + i] = flt(by * 128 + tx / 8 * 4 + i, subrsc + tx % 8);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            smemB[B_sts_addr + i * 32] = act(bx * 128 + tx % 32 + i * 32, tx / 32 + subrsc);
        }
        __syncthreads();
#pragma unroll
        for (int subk = 0; subk < 8; ++subk)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                A_frag[i] = smemA[A_lds_addr + subk * 132 + i];
                A_frag[i + 4] = smemA[A_lds_addr + subk * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                B_frag[i] = smemB[B_lds_addr + subk * 128 + i];
                B_frag[i + 4] = smemB[B_lds_addr + subk * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += A_frag[i] * B_frag[j];
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
            if (x + j < npq && y + i < k)
            {
                out(y + i, x + j) = output_frag[i][j];
            }
            if (x + j + 32 < npq && y + i < k)
            {
                out(y + i, x + j + 32) = output_frag[i][j + 4];
            }
            if (x + j < npq && y + i + 16 < k)
            {
                out(y + i + 16, x + j) = output_frag[i + 4][j];
            }
            if (x + j + 32 < npq && y + i + 16 < k)
            {
                out(y + i + 16, x + j + 32) = output_frag[i + 4][j + 4];
            }
        }
    }
}
template <typename T, class FilterTensor, class ActivationTensor, class OutputTensor>
void launch_implgemm(FilterTensor flt, ActivationTensor act, OutputTensor out, int k, int npq, int rsc)
{
    int blockx = (npq + 127) / 128; // blockx  number
    int blocky = (k + 127) / 128;   // blocky  number
    int blockz = 1;                 // blockz  number
    // 合并threadx与thready
    int threadx = 256; // threadx number per block
    int thready = 1;   // thready number per block
    int threadz = 1;   // threadz number per block
    dim3 block(threadx, thready, threadz);
    dim3 grid(blockx, blocky, blockz);
    implgemm<T><<<grid, block>>>(flt, act, out, k, npq, rsc);
}
