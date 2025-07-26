#include "approx_act_model.h"
#include <vector>
#include <cstdio>
#include <memory>
#include <chrono>
#include <math.h>
#include <cctype>
#include <iostream> // Required for input/output operations (e.g., cout)
#include <fstream>  // Required for file stream operations (e.g., ifstream)



float target_function(float x1, float x2) {
    return std::sin(3 * x1) * std::cos(2 * x2) + x1 * x1 * x1 - 0.5f * x2 * x2;
}

int main() {
    SequentialLayers<1, 4, Tanh<4>, Tanh<4>, Tanh<1>> traditional_net(0.01);
    SequentialLayers<1, 4, Quadratic<4>, Tanh<4>, Tanh<1>> partial_qnet(0.01);
    SequentialLayers<1, 4, Quadratic<4>, Quadratic<4>, Quadratic<1>> full_qnet(0.01);
    // ApproxLayers<1, 4, 4, 4, 1> full_qnet(0.01);

    // training data
    const int num_samples = 500;
    std::vector<std::array<float, 2>> X(num_samples);
    std::vector<float> Y(num_samples);

    for (int i = 0; i < num_samples; i++) {
        float x1 = -1.0f + 2.0f * (rand() / float(RAND_MAX));
        float x2 = -1.0f + 2.0f * (rand() / float(RAND_MAX));
        X[i] = {x1, x2};
        Y[i] = target_function(x1, x2);
    }

    int epochs = 200;
    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss_trad = 0.0f, loss_fullq = 0.0f, loss_partq = 0.0f;

        for (int i = 0; i < num_samples; i++) {
            ColVector<float, 4> input = DataSlice<float>({X[i][0], X[i][1], 1.0, -1.0});
            ColVector<float, 1> target = DataSlice<float>({Y[i]});

            // traditional net
            auto pred_t = traditional_net.fwd(input);
            ColVector<float, 1> loss_grad_t = pred_t - target; // we drop the 2x factor since that's embedded in the learning rate
            traditional_net.bwd(loss_grad_t);
            loss_trad += dot(loss_grad_t, loss_grad_t);

            // partial quadratic net
            auto pred_p = partial_qnet.fwd(input);
            ColVector<float, 1> loss_grad_p = pred_p - target;
            float sq_loss_grad_p = dot(loss_grad_p, loss_grad_p);
            partial_qnet.bwd(loss_grad_p);
            // partial_qnet.bwd(fmatmul(1/sqrt(sq_loss_grad_p), loss_grad_p)); // take unitized to prevent exploding gradient
            loss_partq += sq_loss_grad_p;

            // full quadratic net
            auto pred_q = full_qnet.fwd(input);
            ColVector<float, 1> loss_grad_q = pred_q - target;
            float sq_loss_grad_q = dot(loss_grad_q, loss_grad_q);
            full_qnet.bwd(loss_grad_q);
            // full_qnet.bwd(fmatmul(1/sqrt(sq_loss_grad_q), loss_grad_q)); // take unitized to prevent exploding gradient
            loss_fullq += dot(loss_grad_q, loss_grad_q);
        }

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Traditional: " << loss_trad / num_samples
                      << " | Partial QNet: " << loss_partq / num_samples
                      << " | Full QNet: " << loss_fullq / num_samples
                      << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Training finished in " << duration.count() << "s\n";

    return 0;
}
