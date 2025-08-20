
#define DENSE_RELU_LAYER(input, output, input_sz, output_sz, w, b, acc, tmp)    \
    for (int j = 0; j < output_sz; j++) {                                       \
        acc = 0;                                                                \
        acc = (dense_accum_t)(b[j] << (FRAC_DEFAULT));                          \
        for (int i = 0; i < input_sz; i++) {                                    \
            acc += input[i] * w[j*input_sz + i];                                \
        }                                                                       \
        /* TRUNCATE */                                                          \
        acc = acc >> (FRAC_DEFAULT);                                            \
        tmp = (default_t)acc;                                                   \
        /* RELU ACTIVATION */                                                   \
        output[j] = tmp > 0 ? tmp : 0;                                          \
    }