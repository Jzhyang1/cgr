#pragma once
#include "math.h"
#include <random>

enum SampleType {
    UNIFORM, BINOMIAL, BIMODAL,  //bimodal returns either MIN or MAX
    NORMAL
};

template<SampleType M = UNIFORM>
float sample(float minimum_inclusive, float maximum_exclusive) {
    return minimum_inclusive + (maximum_exclusive - minimum_inclusive) * (rand() / (float) RAND_MAX);
}

template<>
float sample<BIMODAL>(float minimum_inclusive, float maximum_inclusive) {
    return rand()&1 ? maximum_inclusive : minimum_inclusive;
}

template<>
float sample<BINOMIAL>(float minimum_inclusive, float maximum_inclusive) {
    // takes 4 samples randomly-ish. This is stupid but whatever
    return 0.4 * sample<BIMODAL>(minimum_inclusive, maximum_inclusive) 
        + 0.3 * sample<BIMODAL>(minimum_inclusive, maximum_inclusive)
        + 0.2 * sample<BIMODAL>(minimum_inclusive, maximum_inclusive)
        + 0.1 * sample<BIMODAL>(minimum_inclusive, maximum_inclusive);
}

std::random_device rd{}; // Provides non-deterministic seed
std::mt19937 gen{rd()};  // Mersenne Twister engine0
std::normal_distribution<> distr{5.0, 2.0};

template<>
float sample<NORMAL>(float minimum_inclusive, float maximum_inclusive) {
    return distr(gen);
}