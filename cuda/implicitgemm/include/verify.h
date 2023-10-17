float getPrecision(float tmp)
{
    int tmpInt = (int)tmp;
    float eNum = 1.0e-6;
    if(abs(tmpInt) > 0)
    {
        while(tmpInt != 0)
        {
            tmpInt = (int)(tmpInt / 10);
            eNum *= 10;
        }
    }
    else
    {
        
        if(tmp == 0)
            return eNum;
            
        eNum = 1.0e-5;
        
        while(tmpInt == 0)
        {
            tmp *= 10;
            tmpInt = (int)(tmp);
            eNum /= 10;
        }
    }
    return eNum;
}
void direct_conv2dcpu(float* input, float* filter, float* output, int N, int C, int H, int W, int K, int R, int S, int U,int V,  int P, int Q)
{
    int Oh = (H + 2*P - R)/U + 1;
    int Ow = (W + 2*Q - S)/V + 1;
    
    for(int n = 0; n < N; n++)
    {
        for(int k = 0; k< K; k++)
        {
            for(int oh=0; oh<Oh; oh++)
            {
                for(int ow = 0; ow< Ow; ow++)
                { 
                    float sum = 0.0;
                    for(int c = 0; c < C; c++)
                    {                       
                        for(int r = 0; r < R; r++)
                        {
                            for(int s = 0; s < S; s++)
                            {
                                int ih = oh*U - P + r;
                                int iw = ow*V - Q + s;
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
