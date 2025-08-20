// hls-fpga-machine-learning `TODO: make imports dynamic at writer level` imports
#include "dynamatic/Integration.h"
#include "stdlib.h"

// hls-fpga-machine-learning `TODO: make preicison dynamic at writer level` insert layer precision
#define NB_DEFAULT 16
#define INT_DEFAULT 6
#define FRAC_DEFAULT NB_DEFAULT - INT_DEFAULT

#define NB_ACC  32
#define INT_ACC 12
#define FRAC_ACC NB_ACC - INT_ACC

typedef long dense_accum_t;
typedef int default_t;

// hls-fpga-machine-learning insert dimensions


// hls-fpga-machine-learning load weights

// ****************************************
// NETWORK INSTANTIATION
// ****************************************
pub fn myproject_architecture(
    // hls-fpga-machine-learning architecture arguments
    ) ->
    // hls-fpga-machine-learning output 
    {

    // hls-fpga-machine-learning insert layers
}
void jet_tagging2(
        default_t input0, default_t input1, default_t input2, default_t input3, default_t input4, default_t input5, default_t input6, default_t input7, default_t input8, default_t input9, default_t input10, default_t input11, default_t input12, default_t input13, default_t input14, default_t input15, 
        default_t out0[OUT_L0], 
        default_t out1[OUT_L1], 
        default_t out2[OUT_L2], 
        default_t out3[OUT_L3]) {

    default_t tmp_input[IN_L0] = {
        input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15
    };

    /* ----------- Layer 0 ------------ */
    dense_accum_t acc0;
    default_t tmp_relu0 = 0;
    #pragma clang loop unroll_count(UNROLL_FACTOR)
    for (int j = 0; j < OUT_L0; j++) {                       
        acc0 = 0;       
        acc0 = (dense_accum_t)(b0[j] << (FRAC_DEFAULT));
        #pragma clang loop unroll_count(UNROLL_FACTOR)
        for (int i = 0; i < IN_L0; i++) {    
            acc0 += tmp_input[i] * w0[j*IN_L0 + i];           
        }
        /* TRUNCATE */                                       
        acc0 = acc0 >> (FRAC_DEFAULT);             
        tmp_relu0 = (default_t)acc0;                     
        /* RELU ACTIVATION */                                
        out0[j] = tmp_relu0 > 0 ? tmp_relu0 : 0;                   
    }

    /* ----------- Layer 1 ------------ */
    dense_accum_t acc1;
    default_t tmp_relu1 = 0;
    // #pragma clang loop unroll_count(UNROLL_FACTOR)
    for (int j = 0; j < OUT_L1; j++) {                            
        acc1 = (dense_accum_t)(b1[j] << (FRAC_DEFAULT));
        // #pragma clang loop unroll_count(UNROLL_FACTOR)
        for (int i = 0; i < IN_L1; i++) {    
            acc1 += out0[i] * w1[j*IN_L1 + i];           
        }                  
        /* TRUNCATE */                                       
        acc1 = acc1 >> (FRAC_DEFAULT);             
        tmp_relu1 = (default_t)acc1;                     
        /* RELU ACTIVATION */                                
        out1[j] = tmp_relu1 > 0 ? tmp_relu1 : 0;                   
    }

    /* ----------- Layer 2 ------------ */
    dense_accum_t acc2;
    default_t tmp_relu2 = 0;
    // #pragma clang loop unroll_count(UNROLL_FACTOR)
    for (int j = 0; j < OUT_L2; j++) {                            
        acc2 = (dense_accum_t)(b2[j] << (FRAC_DEFAULT));
        // #pragma clang loop unroll_count(UNROLL_FACTOR)
        for (int i = 0; i < IN_L2; i++) {    
            acc2 += out2[i] * w2[j*IN_L2 + i];           
        }                  
        /* TRUNCATE */                                       
        acc2 = acc2 >> (FRAC_DEFAULT);             
        tmp_relu2 = (default_t)acc2;                     
        /* RELU ACTIVATION */                                
        out2[j] = tmp_relu2 > 0 ? tmp_relu2 : 0;                   
    }

    /* ----------- Layer 3 ------------ */
    // #pragma clang loop unroll_count(UNROLL_FACTOR)
    dense_accum_t acc3;
    default_t tmp_relu3 = 0;
    for (int j = 0; j < OUT_L3; j++) {                        
        acc3 = (dense_accum_t)(b3[j] << (FRAC_DEFAULT));
        // #pragma clang loop unroll_count(UNROLL_FACTOR)
        for (int i = 0; i < IN_L3; i++) {    
            acc3 += out2[i] * w3[j*IN_L3 + i];           
        }                  
        /* TRUNCATE */
        acc3 = acc3 >> (FRAC_DEFAULT);
        tmp_relu3 = (default_t)acc3;
        /* RELU ACTIVATION */
        out3[j] = tmp_relu3 > 0 ? tmp_relu3 : 0;                   
    }
}



pub fn myproject(
    // hls-fpga-machine-learning top function input
    )-> 
    // hls-fpga-machine-learning top function output
    {

    
    myproject_architecture(
        // hls-fpga-machine-learning call inlined weights
    )
}