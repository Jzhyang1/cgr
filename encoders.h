#include "tensors.h"


template<int N, int L>
ColVector<float, L * N> onehot_encode(ColVector<float, L> initial) {
    ColVector<float, L * N> ret;
    for (int i = 0; i < L; ++i) {
        ret[i * N + initial[i]] = 1.0f;
    }
    return ret;
}

template<int L>
int onehot_decode(ColVector<float, L> v) {
    float pmax = 0;
    int ret = -1;
    for (int i = 0; i < L; ++i) {
        if (v[i] > pmax) {
            pmax = v[i];
            ret = i;
        }
    }
    return ret;
}


template<int L>
ColVector<float, L> softmax_encode(ColVector<float, L> initial) {
    float max_val = initial[0];
    for (int i = 1; i < L; ++i) {
        if (initial[i] > max_val) max_val = initial[i];
    }
    float sum = 0.0f;
    ColVector<float, L> ret;
    for (int i = 0; i < L; ++i) {
        ret[i] = std::exp(initial[i] - max_val);
        sum += ret[i];
    }
    for (int i = 0; i < L; ++i) {
        ret[i] /= sum;
    }
    return ret;
}

template<int L>
ColVector<float, L> scaled_encode(ColVector<float, L> initial, int lo, int hi) {
    // scales [lo, hi] -> [0, 1]
    ColVector<float, L> ret;
    for (int i = 0; i < L; ++i) {
        ret[i] = (initial[i] - lo) / (hi - lo);
    }
    return ret;
}