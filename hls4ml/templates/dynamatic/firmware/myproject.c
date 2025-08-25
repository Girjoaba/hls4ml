// hls-fpga-machine-learning `TODO: make imports dynamic at writer level` imports
#include "dynamatic/Integration.h"
#include "stdlib.h"
#include "stdio.h"
#include "nnet_utils/fc.h"

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
void myproject(
    // hls-fpga-machine-learning architecture arguments
    ) {

    // hls-fpga-machine-learning insert layers
}

int main(void) {
    srand(13);

    // hls-fpga-machine-learning input init

    CALL_KERNEL(
        myproject,
        // hls-fpga-machine-learning input kernel
    );
    return 0;
    
}