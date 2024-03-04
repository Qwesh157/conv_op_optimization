float getPrecision(float tmp)
{
    int tmpInt = (int)tmp;
    float eNum = 1.0e-6;
    if (abs(tmpInt) > 0)
    {
        while (tmpInt != 0)
        {
            tmpInt = (int)(tmpInt / 10);
            eNum *= 10;
        }
    }
    else
    {

        if (tmp == 0)
            return eNum;

        eNum = 1.0e-5;

        while (tmpInt == 0)
        {
            tmp *= 10;
            tmpInt = (int)(tmp);
            eNum /= 10;
        }
    }
    return eNum;
}
void direct_conv2dbwddatacpu(float *grad_input, float *filter, float *grad_output, int N, int C, int H, int W, int K, int R, int S, int U, int V, int P, int Q, int Oh, int Ow)
{
    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    float sum = 0.0;
                    for (int k = 0; k < K; k++)
                    {
                        for (int r = 0; r < R; r++)
                        {
                            for (int s = 0; s < S; s++)
                            {
                                // 不考虑padding与stride参数，进行full mode卷积
                                int oh = h + r - (R - 1 - P);
                                int ow = w + s - (S - 1 - Q);
                                if (ow >= 0 && oh >= 0 && ow < Ow && oh < Oh)
                                {
                                    sum += (grad_output[n * K * Oh * Ow + k * (Oh * Ow) + oh * Ow + ow] * filter[k * R * S * C + c * R * S + (R * S - 1 - (r * S + s))]);
                                }
                            }
                        }
                    }
                    grad_input[n * C * H * W + c * H * W + h * W + w] = sum;
                }
            }
        }
    }
}