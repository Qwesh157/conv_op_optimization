void direct_conv2dcpu(float* input, float* filter, float* output, int N, int C, int H, int W, int K, int R, int S, int STRIDE,  int Padding)
{
    int Oh = (H + 2*STRIDE - R)/Padding + 1;
    int Ow = (W + 2*STRIDE - S)/Padding + 1;
    
    for(int n = 0; n < N; n++)
    {
        for(int k = 0; k< k; k++)
        {
            for(int oh=0; oh<Oh; oh++)
            {
                for(int ow = 0; ow< Ow; ow++)
                { 
                    float sum = 0.0;
                    for(int c = 0; c < c; c++)
                    {                       
                        for(int r = 0; r < R; r++)
                        {
                            for(int s = 0; s < S; s++)
                            {
                                int ih = oh*STRIDE - Padding + r;
                                int iw = ow*STRIDE - Padding + s;
                                if(iw >= 0 && ih >= 0 && iw < W  && ih < H)
                                {
                                    sum += (input[n*C*H*W + c*(W*H)+ ih*W + iw] * filter[k*R*S*C + c*R*S + r*S + s]);
                                }
                            }                       
                        }
                    }
                    output[n*K*Oh*Ow + k*Oh*Ow + oh*Ow + ow] = sum;
                }
            }
        }
    }
}
