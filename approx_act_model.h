#pragma once
#include "model.h"


// ======= second degree activation approximator models =======

template<int FINAL_SZ, int IN_SZ, int OUT_SZ, int ...NEURONS>
struct ApproxLayers;
template<int FINAL_SZ, int IN_SZ, int OUT_SZ> requires (FINAL_SZ == OUT_SZ)
struct ApproxLayers<FINAL_SZ, IN_SZ, OUT_SZ>;


template<int FINAL_SZ, int IN_SZ, int OUT_SZ, int ...NEURONS>
struct ApproxLayers : public Model<FINAL_SZ, IN_SZ> {
    RowVector<Matrix<float, OUT_SZ, IN_SZ>, IN_SZ> weights;
    ApproxLayers<FINAL_SZ, OUT_SZ, NEURONS...> next_weights;
    ColVector<float, IN_SZ> prev_input;
    ColVector<float, OUT_SZ> prev_output;
    float learning_rate;
    bool tanh_activation;
public:
    ApproxLayers(float learning_rate): learning_rate(learning_rate), next_weights(learning_rate) {
        init_weights();
    }

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        prev_input = input;
        Matrix<float, OUT_SZ, IN_SZ> first_degree_weights;
        for (int i = 0; i < IN_SZ; ++i) {
            first_degree_weights += fmatmul(input[i], weights[i]);
        }    // the jacobian
        auto res = first_degree_weights * input;

        prev_output = res;
        return next_weights.fwd(prev_output);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> last_loss) override {
        ColVector<float, OUT_SZ> loss = next_weights.bwd(last_loss);

        // loss: dL/dy (delta), size OUT_SZ
        ColVector<float, IN_SZ> grad_input;

        // dL/dW update and dL/dx computation
        for (int i = 0; i < OUT_SZ; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                for (int k = 0; k < IN_SZ; ++k) {
                    // Weight gradient
                    float grad_w = loss[i] * prev_input[j] * prev_input[k];
                    weights[j][i, k] -= learning_rate * grad_w;

                    // Input gradient: (W + W^T) x
                    grad_input[j] += loss[i] * (weights[j][i, k] + weights[k][i, j]) * prev_input[k];
                }
            }
        }
        return grad_input;
    }

    void init_weights() {
        for (int i = 0; i < IN_SZ; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                for (int k = 0; k < IN_SZ; ++k) {
                    weights[i][j, k] = sample<UNIFORM>(0.01, 0.1);
                }
            }
        }
    }
};

// base case
template<int FINAL_SZ, int IN_SZ, int OUT_SZ> requires (FINAL_SZ == OUT_SZ)
struct ApproxLayers<FINAL_SZ, IN_SZ, OUT_SZ> : public Model<FINAL_SZ, IN_SZ> {
    RowVector<Matrix<float, OUT_SZ, IN_SZ>, IN_SZ> weights; // the hessian
    ColVector<float, IN_SZ> prev_input;
    ColVector<float, OUT_SZ> prev_output;
    float learning_rate;
    bool tanh_activation;
public:
    ApproxLayers(float learning_rate): learning_rate(learning_rate) {
        init_weights();
    }

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        prev_input = input;
        Matrix<float, OUT_SZ, IN_SZ> first_degree_weights;
        for (int i = 0; i < IN_SZ; ++i) {
            first_degree_weights += fmatmul(input[i], weights[i]);
        }    // the jacobian
        auto res = first_degree_weights * input;
        
        prev_output = res;
        return prev_output;
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        // loss: dL/dy (delta), size OUT_SZ
        ColVector<float, IN_SZ> grad_input;

        // dL/dW update and dL/dx computation
        for (int i = 0; i < OUT_SZ; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                for (int k = 0; k < IN_SZ; ++k) {
                    // Weight gradient
                    float grad_w = loss[i] * prev_input[j] * prev_input[k];
                    weights[j][i, k] -= learning_rate * grad_w;

                    // Input gradient: (W + W^T) x
                    grad_input[j] += loss[i] * (weights[j][i, k] + weights[k][i, j]) * prev_input[k];
                }
            }
        }
        return grad_input;
    }

    void init_weights() {
        for (int i = 0; i < IN_SZ; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                for (int k = 0; k < IN_SZ; ++k) {
                    weights[i][j, k] = sample<UNIFORM>(0.01, 0.1);
                }
            }
        }
    }
};