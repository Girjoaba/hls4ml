

#define DENSE_RELU_LAYER(input, output, input_sz, output_sz, w, b, acc, tmp)    \
_Pragma("clang loop unroll(full)")                                              \
    for (int j = 0; j < output_sz; j++) {                                       \
        acc = 0;                                                                \
        acc = (dense_accum_t)(b[j] << (FRAC_DEFAULT));                          \
_Pragma("clang loop unroll(full)")                                              \
        for (int i = 0; i < input_sz; i++) {                                    \
            acc += input[i] * w[j][i];                                          \
        }                                                                       \
        /* TRUNCATE */                                                          \
        acc = acc >> (FRAC_DEFAULT);                                            \
        tmp = (default_t)acc;                                                   \
        /* RELU ACTIVATION */                                                   \
        output[j] = tmp > 0 ? tmp : 0;                                          \
    }

#define DENSE_LAYER(input, output, input_sz, output_sz, w, b, acc, tmp)    \
_Pragma("clang loop unroll(full)")                                              \
    for (int j = 0; j < output_sz; j++) {                                       \
        acc = 0;                                                                \
        acc = (dense_accum_t)(b[j] << (FRAC_DEFAULT));                          \
_Pragma("clang loop unroll(full)")                                              \
        for (int i = 0; i < input_sz; i++) {                                    \
            acc += input[i] * w[j][i];                                          \
        }                                                                       \
        /* TRUNCATE */                                                          \
        acc = acc >> (FRAC_DEFAULT);                                            \
        output[j] = (default_t)acc;                                             \
    }

#define ARGMAX(input, output, input_sz, tmp_argmax)                             \
    tmp_argmax = -(1 << (NB_DEFAULT-1));                                        \
    for (int i = 0; i < input_sz; i++) {                                        \
        tmp_argmax = (tmp_argmax < input[i]) ? input[i] : tmp_argmax;           \
    }                                                                           \
_Pragma("clang loop unroll(full)")                                              \
    for (int i = 0; i < input_sz; i++) {                                        \
        output[i] = (tmp_argmax == input[i]) ? (1 << (FRAC_DEFAULT)) : 0;       \
    }