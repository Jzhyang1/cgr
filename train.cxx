#include <vector>
#include <cstdio>
#include <memory>
#include <chrono>
#include <math.h>
#include <cctype>
#include "train.h"
#include <iostream> // Required for input/output operations (e.g., cout)
#include <fstream>  // Required for file stream operations (e.g., ifstream)



using namespace std;

// supported characters are:
// a-z (lower), comma, semicolon, period, dash (-), space, newline
map<pair<char, char>, map<char, int>> probabilities; // used for loss calculations


int main() {
    init_alphabet();

    //begin training
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream initial_weights("weights.txt");
    base_model.load_weights(initial_weights);
    initial_weights.close();

    //our training corpus
    vector<float> one_hot_vec;
    vector<char> queue;
    int n;
    {
        std::ifstream inputFile("train_data/crimeandpunishment.txt");
        char c, cp = ' ', cpp = ' ';
        
        while (inputFile.get(c)) {
            c = tolower(c);
            if (!alphabet.contains(c)) {
                continue;
            }
            queue.push_back(c);

            // update probabilities
            map<char, int>& dist = probabilities[{cpp, cp}];
            ++dist['\0'];
            ++dist[c];

            cpp = cp;
            cp = c;
        }
        n = queue.size();
        one_hot_vec.resize(n * alphsize);
        for (int i = 0; i < n; ++i) {
            one_hot_vec[i * alphsize + alphabet[queue[i]]] = 1.0f;
        }
        inputFile.close();
    }

    n = 1000; // we don't want to see too much text at once
    int first_index = 100; //first_index at minimum is equal to lookback
    vector<float>::iterator one_hot_begin = one_hot_vec.begin() + first_index * alphsize;

    for (int _ = 0; _ < epochs; ++_) {
        ColVector<float, alphsize> error_sum;
        for (int k = 0; k < n; ++k) {
            span<float> one_hot_it{one_hot_begin + (k - lookback) * alphsize, one_hot_begin + k * alphsize};
            span<float> one_hot_ot{one_hot_begin + k * alphsize, one_hot_begin + (k + 1) * alphsize};

            one_hot_it[0] = 1;  // trick for biases
            ColVector<float, lookback * alphsize> input{DataSlice<float>(one_hot_it)};
            ColVector<float, alphsize> correct{DataSlice<float>(one_hot_ot)};

            ColVector<float, alphsize> pred = base_model.fwd(input);
            ColVector<float, alphsize> error = pred - correct; // cross entropy
            error_sum += error;

            // logging
            float loss = -logf(dot(pred, correct));
            cout << k << " loss: " << loss << endl;

            // gradient accumulation
            if ((k + 1) % batchsize == 0) {
                constexpr float frac = 1 / batchsize;
                base_model.bwd(fmatmul(frac, error_sum));
                error_sum -= error_sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    //print some output.
    vector<char> oqueue;
    string init = "For in that sleep what dreams may come must give us pause. Aye, there's the rub";
    for (char c : init) {
        if (alphabet.contains(tolower(c)))
        oqueue.push_back(tolower(c));
    }

    cout << init;
    for (int i = 0; i < 100; ++i) {
        //encode with one_hot
        shared_ptr<vector<float>> one_hot = make_shared<vector<float>>(lookback * alphsize);
        for (int i = 0; i < lookback; ++i) {
            (*one_hot)[i * alphsize + alphabet[oqueue[oqueue.size() - lookback + i]]] = 1.0f;
        }
        (*one_hot)[0] = 1.0f; // trick for biases

        ColVector<float, lookback * alphsize> input(one_hot);
        ColVector<float, alphsize> pred = base_model.fwd(input);

        float p = 0;
        char c = '#';
        for (int j = 0; j < alphsize; ++j) {
            if (pred[j] > p) {
                c = inverse_alphabet[j];
                p = pred[j];
            }
            j++;
        }
        oqueue.push_back(c);
        cout << c;
    }

    // print probabilities
    // cout << probabilities.size() << endl;
    // for (auto& entry : probabilities) {
    //     auto context = entry.first;
    //     auto& dist = entry.second;
    //     int ct = dist['\0'];
    //     std::cout << "Context: [" << context.first << ", " << context.second << "]\n";
    //     for (const auto& prob : dist) {
    //         if (prob.first == '\0') continue; // skip count
    //         std::cout << "  '" << prob.first << "': " << ((float)prob.second / ct) << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    std::ofstream weights_file("weights.txt");
    base_model.save_weights(weights_file);
    weights_file.close();
}