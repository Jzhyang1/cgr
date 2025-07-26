#pragma once
#include <concepts>
#include <iostream>
#include "tensors.h"
#include "random.h"

template<int SZ, int IN_SZ> 
struct ActivateLayer {
    static constexpr int size = SZ;
    static constexpr int int_size = IN_SZ;

    virtual ColVector<float, SZ> fwd(ColVector<float, IN_SZ> input) = 0;
    virtual ColVector<float, IN_SZ> bwd(ColVector<float, SZ> loss, float learning_rate) = 0;
    virtual void save_weights(std::ostream& os) = 0;
    virtual void load_weights(std::istream& is) = 0;
};

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


template<typename A>
concept Activation_ = 
    std::is_base_of_v<Activation<A::size>, A> &&
    requires {
        typename A::template storage<1>;
        requires std::is_base_of_v<ActivateLayer<A::size, 1>, typename A::template storage<1>>;
    };


// -------------------------------------------------------------------
// |                                                                 |
// |                                                                 |
// -------------------------------------------------------------------
// helper functions

float safe_expf(float x) {
    if (x > 30.0) return expf(30.0);
    if (x < -30.0) return expf(-30.0);
    return expf(x);
}



// -------------------------------------------------------------------
// |                                                                 |
// |                                                                 |
// -------------------------------------------------------------------


template<int SZ>
struct Sigmoid : Activation<SZ> {
    template<int IN_SZ>
    struct storage : ActivateLayer<SZ, IN_SZ> {
        Matrix<float, SZ, IN_SZ> weights;
        ColVector<float, IN_SZ> prev_input;
        ColVector<float, SZ> prev_output;
        storage() {
            Activation<SZ>::template init_weights_glorot<IN_SZ>(weights);
        }

        ColVector<float, SZ> activate(ColVector<float, SZ> input_evaluated) {
            return input_evaluated.template applied<float>([](float f){return 1 / (1 + safe_expf(-f));});
        }
        ColVector<float, SZ> derivative() {
            return prev_output.template applied<float>([](float f){return f * (1 - f);});
        }

        ColVector<float, SZ> fwd(ColVector<float, IN_SZ> input) override {
            prev_input = input;
            prev_output = activate(weights * input);
            return prev_output;
        }
        ColVector<float, IN_SZ> bwd(ColVector<float, SZ> loss, float learning_rate) override {
            auto intermediate = dot_had(loss, derivative());
            weights -= fmatmul(learning_rate, intermediate * prev_input.transposed());
            return weights.transposed() * intermediate;
        }

        void save_weights(std::ostream& os) override {
            os << weights;
        }
        void load_weights(std::istream& is) override {
            is >> weights;
        }
    };
};

template<int SZ>
struct Tanh : Activation<SZ> {
    template<int IN_SZ>
    struct storage : ActivateLayer<SZ, IN_SZ> {
        Matrix<float, SZ, IN_SZ> weights;
        ColVector<float, IN_SZ> prev_input;
        ColVector<float, SZ> prev_output;
        storage() {
            Activation<SZ>::template init_weights_glorot<IN_SZ>(weights);
        }
        ColVector<float, SZ> activate(ColVector<float, SZ> input_evaluated) {
            return input_evaluated.template applied<float>([](float f){return tanh(f);});
        }
        ColVector<float, SZ> derivative() {
            return prev_output.template applied<float>([](float f){return 1 - f * f;});
        }
        ColVector<float, SZ> fwd(ColVector<float, IN_SZ> input) override {
            prev_input = input;
            prev_output = activate(weights * input);
            return prev_output;
        }
        ColVector<float, IN_SZ> bwd(ColVector<float, SZ> loss, float learning_rate) override {
            auto intermediate = dot_had(loss, derivative());
            weights -= fmatmul(learning_rate, intermediate * prev_input.transposed());
            return weights.transposed() * intermediate;
        }

        void save_weights(std::ostream& os) override {
            os << weights;
        }
        void load_weights(std::istream& is) override {
            is >> weights;
        }
    };
};


template<int SZ>
struct ReLU : Activation<SZ> {
    template<int IN_SZ>
    struct storage : ActivateLayer<SZ, IN_SZ> {
        Matrix<float, SZ, IN_SZ> weights;
        ColVector<float, IN_SZ> prev_input;
        ColVector<float, SZ> input_evaluated;
        storage() {
            Activation<SZ>::template init_weights_he<IN_SZ>(weights);
        }
        ColVector<float, SZ> activate() {
            return input_evaluated.template applied<float>([](float f){return f > 0 ? f : 0;});
        }
        ColVector<float, SZ> derivative() {
            return input_evaluated.template applied<float>([](float f){return f > 0 ? 1 : 0;});
        }
        ColVector<float, SZ> fwd(ColVector<float, IN_SZ> input) override {
            prev_input = input;
            input_evaluated = weights * input;
            return activate();
        }
        ColVector<float, IN_SZ> bwd(ColVector<float, SZ> loss, float learning_rate) override {
            auto intermediate = dot_had(loss, derivative());
            weights -= fmatmul(learning_rate, intermediate * prev_input.transposed());
            return weights.transposed() * intermediate;
        }

        void save_weights(std::ostream& os) override {
            os << weights;
        }
        void load_weights(std::istream& is) override {
            is >> weights;
        }
    };
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
            ret[i] = safe_expf(x[i] - max_val);
            sum += ret[i];
        }
        for (int i = 0; i < SZ; ++i) {
            ret[i] /= sum;
        }
        return ret;
    }

    static ColVector<float, SZ> derivative(ColVector<float, SZ> const& x) {
        // assuming we're using cross-entropy loss immediately after
        return x;
    }

    template<int IN_SZ>
    struct storage : ActivateLayer<SZ, IN_SZ> {
        Matrix<float, SZ, IN_SZ> weights;
        ColVector<float, IN_SZ> prev_input;
        ColVector<float, SZ> input_evaluated;
        storage() {
            Activation<SZ>::template init_weights_glorot<IN_SZ>(weights);
        }
        ColVector<float, SZ> fwd(ColVector<float, IN_SZ> input) override {
            prev_input = input;
            input_evaluated = weights * input;
            return activate(input_evaluated);
        }
        ColVector<float, IN_SZ> bwd(ColVector<float, SZ> loss, float learning_rate) override {
            auto intermediate = dot_had(loss, derivative(input_evaluated));
            weights -= fmatmul(learning_rate, intermediate * prev_input.transposed());
            return weights.transposed() * intermediate;
        }

        void save_weights(std::ostream& os) override {
            os << weights;
        }
        void load_weights(std::istream& is) override {
            is >> weights;
        }
    };
};


template<int SZ>
struct Quadratic : Activation<SZ> {
    template<int IN_SZ>
    struct storage : ActivateLayer<SZ, IN_SZ> {
        ColVector<Matrix<float, IN_SZ, IN_SZ>, SZ> weights;
        ColVector<float, IN_SZ> prev_input;
        ColVector<float, SZ> input_evaluated;
        storage() {
            for (int i = 0; i < SZ; ++i) {
                Activation<IN_SZ>::template init_weights_glorot<IN_SZ>(weights[i]);
            }
        }
        ColVector<float, SZ> fwd(ColVector<float, IN_SZ> input) override {
            RowVector<float, IN_SZ> transposed = input.transposed();
            prev_input = input;
            return weights.template applied<float>([transposed, input](Matrix<float, IN_SZ, IN_SZ> mat)->float{return 0.5 * (transposed * mat * input);});
        }
        ColVector<float, IN_SZ> bwd(ColVector<float, SZ> loss, float learning_rate) override {
            // loss: dL/dy (delta), size SZ
            ColVector<float, IN_SZ> grad_input;

            // dL/dW update and dL/dx computation
            for (int i = 0; i < SZ; ++i) {
                // Weight update
                weights[i] -= fmatmul(learning_rate * loss[i], (prev_input * prev_input.transposed()));
                // Input gradient
                grad_input += fmatmul(loss[i], (weights[i] + weights[i].transposed()) * prev_input);
            }
            return grad_input;
        }

        void save_weights(std::ostream& os) override {
            os << weights;
        }
        void load_weights(std::istream& is) override {
            is >> weights;
        }
    };
};