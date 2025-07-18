#pragma once
#include <concepts>
#include <iostream>
#include "tensors.h"
#include "random.h"

template<int SZ>
struct Activation {
    static constexpr int size = SZ;

    template<int IN_SZ>
    static void init_weights_glorot(Matrix<float, SZ, IN_SZ>& weights) {
        float r = sqrtf(6.0/(IN_SZ + SZ)); // used for Glorot initialization
        for (int i = 0; i < SZ; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                float temp = weights[i, j] = sample<UNIFORM>(-r, r);
            }
        }
    }

    template<int IN_SZ>
    static void init_weights_he(Matrix<float, SZ, IN_SZ>& weights) {
        float r = sqrtf(2.0/IN_SZ); // used for He initialization
        for (int i = 0; i < SZ; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                float temp = weights[i, j] = sample<NORMAL>(-r, r);
            }
        }
    }
};

// Base case: does not exist
template<typename, typename = void>
struct has_init_weights : std::false_type {};

// Specialization: exists if calling it with some int works
template<typename T>
struct has_init_weights<T, std::void_t<
    decltype(std::declval<T>().template init_weights<1>(std::declval<Matrix<float, T::size, 1>&>()))
>> : std::true_type {};


template<typename A>
concept Activation_ = 
    std::is_base_of_v<Activation<A::size>, A> &&
    has_init_weights<A>::value &&
    requires(ColVector<float, A::size> const& x) {
        { A::activate(x) } -> std::same_as<ColVector<float, A::size>>;
        { A::derivative(x) } -> std::same_as<ColVector<float, A::size>>;
    };


// -------------------------------------------------------------------
// |                                                                 |
// |                                                                 |
// -------------------------------------------------------------------


template<int SZ>
struct Sigmoid : Activation<SZ> {
    static ColVector<float, SZ> activate(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            ret[i] = 1 / (1 + expf(-x[i]));
        }
        return ret;
    }

    static ColVector<float, SZ> derivative(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            float s = 1 / (1 + expf(-x[i]));
            ret[i] = s * (1 - s);
        }
        return ret;
    }

    template<int IN_SZ>
    static void init_weights(Matrix<float, SZ, IN_SZ>& weights) {
        return Activation<SZ>::template init_weights_glorot<IN_SZ>(weights);
    }
};

template<int SZ>
struct Tanh : Activation<SZ> {
    static ColVector<float, SZ> activate(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            ret[i] = tanh(x[i]);
        }
        return ret;
    }

    static ColVector<float, SZ> derivative(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            float temp = tanh(x[i]);
            ret[i] = 1 - temp * temp;
        }
        return ret;
    }

    template<int IN_SZ>
    static void init_weights(Matrix<float, SZ, IN_SZ>& weights) {
        return Activation<SZ>::template init_weights_glorot<IN_SZ>(weights);
    }
};


template<int SZ>
struct ReLU : Activation<SZ> {
    static ColVector<float, SZ> activate(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            ret[i] = x[i] > 0 ? x[i] : 0.0f;
        }
        return ret;
    }

    static ColVector<float, SZ> derivative(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            ret[i] = x[i] > 0 ? 1.0f : 0.0f;
        }
        return ret;
    }

    template<int IN_SZ>
    static void init_weights(Matrix<float, SZ, IN_SZ>& weights) {
        return Activation<SZ>::template init_weights_he<IN_SZ>(weights);
    }
};


template<int SZ>
struct Softmax : Activation<SZ> {
    static ColVector<float, SZ> activate(ColVector<float, SZ> const& x) {
        float max_val = x[0];
        for (int i = 1; i < SZ; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }
        float sum = 0.0f;
        ColVector<float, SZ> ret;
        for (int i = 0; i < SZ; ++i) {
            ret[i] = std::exp(x[i] - max_val);
            sum += ret[i];
        }
        for (int i = 0; i < SZ; ++i) {
            ret[i] /= sum;
        }
        return ret;
    }

    static ColVector<float, SZ> derivative(ColVector<float, SZ> const& x) {
        ColVector<float, SZ> ret;
        ColVector<float, SZ> y = activate(x);
        for (int i = 0; i < SZ; ++i) {
            ret[i] = y[i] * (1 - y[i]);
        }
        return ret;
    }

    template<int IN_SZ>
    static void init_weights(Matrix<float, SZ, IN_SZ>& weights) {
        return Activation<SZ>::template init_weights_glorot<IN_SZ>(weights);
    }
};
