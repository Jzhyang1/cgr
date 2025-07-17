#include <vector>
#include <cstdio>
#include <memory>
#include <chrono>
#include <map>
#include <math.h>
#include <cctype>
#include "model.h"
#include <iostream> // Required for input/output operations (e.g., cout)
#include <fstream>  // Required for file stream operations (e.g., ifstream)


using namespace std;

// supported characters are:
// a-z (lower), comma, semicolon, period, dash (-), space, newline
map<char, int> alphabet;
map<pair<char, char>, map<char, int>> probabilities; // used for loss calculations

constexpr int alphsize = 32;
constexpr int lookback = 64;
constexpr int epochs = 1;

SequentialLayers<alphsize, lookback * alphsize + 1, 
    Tanh<128>, Tanh<64>, Tanh<32>, Sigmoid<32>
> base_model(0.001f); // predicts the next letter

int main() {
    //define our alphabet
    for (char c = 'a'; c <= 'z'; ++c) {
        alphabet[c] = alphabet.size();
    }
    alphabet['-'] = alphabet.size();
    alphabet[' '] = alphabet.size();
    alphabet[','] = alphabet.size();
    alphabet[';'] = alphabet.size();
    alphabet['.'] = alphabet.size();
    alphabet[':'] = alphabet.size();
    assert(alphabet.size() == alphsize);

    //begin training
    auto start = std::chrono::high_resolution_clock::now();
    base_model.init_weights();

    //our training corpus
    vector<float> one_hot_vec;
    vector<char> queue;
    int n;
    {
        std::ifstream inputFile("train_data/crimeandpunishment.txt");
        char c;
        while (inputFile.get(c)) {
            c = tolower(c);
            if (!alphabet.contains(c)) continue;
            queue.push_back(c);
        }
        n = queue.size();
        one_hot_vec.resize(n * alphsize);
        for (int i = 0; i < n; ++i) {
            one_hot_vec[i * alphsize + alphabet[queue[i]]] = 1.0f;
        }
        inputFile.close();
    }

    for (int _ = 0; _ < epochs; ++_) {
        for (int k = lookback; k < n; ++k) {
            vector<float>::iterator one_hot_it = one_hot_vec.begin() + (k - lookback) * alphsize;
            ColVector<float, lookback * alphsize + 1> input(one_hot_it);
            ColVector<float, alphsize> correct;
            
            // define correct weights
            correct[alphabet[queue[k]]] = 0.9f; // don't give too high of confidence
            //also give everything else reasonable a reasonable score
            map<char, int>& dist = probabilities[{queue[k - 2], queue[k - 1]}];
            int count = ++dist['\0']; ++dist[queue[k]];
            for (auto e : dist) {
                if (e.first == '\0') continue;
                correct[alphabet[e.first]] += 0.1 * e.second / count;
            }

            ColVector<float, alphsize> pred = base_model.fwd(input);
            ColVector<float, alphsize> error = pred - correct;

            base_model.bwd(error);
            cout << "loss: " << (error * error) << endl;
        }
    }

    //print some output.
    queue.clear();
    string init = "For in that sleep what dreams may come must give us pause. Aye, there's the rub";
    for (char c : init) {
        if (alphabet.contains(tolower(c)))
        queue.push_back(tolower(c));
    }

    cout << init;
    for (int i = 0; i < 1000; ++i) {
        //encode with one_hot
        shared_ptr<vector<float>> one_hot = make_shared<vector<float>>(lookback * alphsize + 1);
        for (int i = 0; i < lookback; ++i) {
            (*one_hot)[i * alphsize + alphabet[queue[queue.size() - lookback + i]]] = 1.0f;
        }
        (*one_hot)[lookback * alphsize] = 1.0f; // trick for biases

        //occasionally flush the queue a bit
        if (queue.size() > 4 * lookback) {
            queue = vector<char>(queue.begin() + 3 * lookback, queue.end());
        }

        ColVector<float, lookback * alphsize + 1> input(one_hot);
        ColVector<float, alphsize> pred = base_model.fwd(input);

        int j = 0, p = 0;
        char c = '#';
        auto iter = alphabet.begin();
        while (j < alphsize) {
            if (pred[j] > p) {
                c = iter->first;
                p = iter->second;
            }
            j++;
            iter++;
        }
        queue.push_back(c);
        cout << c;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    // print probabilities
    cout << probabilities.size() << endl;
    for (auto& entry : probabilities) {
        auto context = entry.first;
        auto& dist = entry.second;
        int ct = dist['\0'];
        std::cout << "Context: [" << context.first << ", " << context.second << "]\n";
        for (const auto& prob : dist) {
            if (prob.first == '\0') continue; // skip count
            std::cout << "  '" << prob.first << "': " << ((float)prob.second / ct) << std::endl;
        }
        std::cout << std::endl;
    }

    base_model.save_weights();
}