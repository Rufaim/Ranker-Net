#include "nalu.h"

//#include <cmath>
#include "math.h"

Nalu::Nalu(int input_dim, int out_dim, float *weight, float *gate) :
    weights_(weight),
    gate_(gate),
    Layer ( input_dim, out_dim)
{}

void Nalu::process(float* input,float* output) {
    float gate = 0;
    for(int i = 0; i < out_dim_; i++)
        gate += gate_[i] * input[i];
    gate = 0.5*tanh(gate)+1;
    for(int i = 0; i < out_dim_; i++) {
        float lin = 0;
        float log_ = 0;
        for(int j = 0; j < input_dim_; j++) {
            lin += weights_[i * input_dim_ + j] * input[j];
            log_ += weights_[i * input_dim_ + j] * log(fabs(input[j])+0.00001);
        }
        output[i] = gate * lin + (1-gate) * exp(log_);
    }
}

void Nalu::free() {
    delete [] weights_;
    delete [] gate_;
}
