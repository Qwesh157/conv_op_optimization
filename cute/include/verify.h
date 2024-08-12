using namespace cute;
template <class FilterTensor, class ActivationTensor, class OutputTensor>
void cute_conv2d_nhwc_cpu(FilterTensor flt, ActivationTensor act, OutputTensor out)
{
    size_t k = static_cast<size_t>(size<0>(out));
    size_t npq = static_cast<size_t>(size<1>(out));
    size_t rsc = static_cast<size_t>(size<1>(flt));

    for (size_t m = 0; m < k; m++)
    {
        for (size_t n = 0; n < npq; n++)
        {
            auto accum = float(0);
            for (size_t i = 0; i < rsc; i++)
            {
                accum += flt(m, i) * act(n, i);
            }
            out(m, n) = accum;
        }
    }
}
template <typename T>
void direct_nhwc_conv2dcpu(T *input, T *filter, T *output, int N, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q)
{
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;

    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int oh = 0; oh < Oh; oh++)
            {
                for (int ow = 0; ow < Ow; ow++)
                {
                    float sum = 0.0;

                    for (int r = 0; r < R; r++)
                    {
                        for (int s = 0; s < S; s++)
                        {
                            for (int c = 0; c < C; c++)
                            {
                                int ih = oh * U - P + r;
                                int iw = ow * V - Q + s;
                                if (iw >= 0 && ih >= 0 && iw < W && ih < H)
                                {
                                    sum += input[n * C * H * W + ih * W * C + iw * C + c] * filter[k * R * S * C + r * S * C + s * C + c];
                                }
                            }
                        }
                    }
                    output[n * Oh * Ow * K + oh * Ow * K + ow * K + k] = (T)sum;
                }
            }
        }
    }
}