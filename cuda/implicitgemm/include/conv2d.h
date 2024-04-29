typedef struct
{
    float*   input;                                   //输入数据地址
    float*   weight;                                  //权值数据地址
    float*   bias;                                    //偏置值数据地址
    float*   output;                                  //输出数据地址
    unsigned int      n;                              //batch szie              
    unsigned int      c;                              //channel number          
    unsigned int      h;                              //数据高                  
    unsigned int      w;                              //数据宽                  
    unsigned int      k;                              //卷积核数量              
    unsigned int      r;                              //卷积核高                
    unsigned int      s;                              //卷积核宽                
    unsigned int      u;                              //卷积在高方向上的步长    
    unsigned int      v;                              //卷积在宽方向上的步长    
    unsigned int      p;                              //卷积在高方向上的补边    
    unsigned int      q;                              //卷积在宽方向上的补边    
    unsigned int      Oh;                             //卷积结果高             
    unsigned int      Ow;                             //卷积结果宽 
}param_t;
void launch_implgemm(param_t param);
