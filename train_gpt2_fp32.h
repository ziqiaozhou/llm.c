void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC);
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C);
void matmul_backward_bias_kernel4_host(float* dbias, const float* dout, int B, int T, int OC);
void matmul_forward_host(float* out, const float* inp, const float* weight, const float* bias,
                                  int B, int T, int C, int OC);
void gelu_backward(float* dinp, const float* inp, const float* dout, const int N);
void gelu_forward(float* out, const float* inp, int N);

void softmax_forward_host(float* out, const float* inp, int B, int T, int NH, float scale);
void softmax_autoregressive_backward_host(float* dpreatt, const float* datt, const float* att,
                                            int B, int T, int C, float scale);
