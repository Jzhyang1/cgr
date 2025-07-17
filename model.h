#pragma once
#include <iostream>
#include "activations.h"
#include "math.h"
#include "random.h"

template<int OUT_SZ, int IN_SZ>
struct Model {
    static constexpr int in_size = IN_SZ;
    static constexpr int out_size = OUT_SZ;

    virtual ColVector<float, OUT_SZ> fwd(ColVector<float, IN_SZ> input) = 0;
    virtual ColVector<float, IN_SZ> bwd(ColVector<float, OUT_SZ> loss) = 0;
    virtual void init_weights() = 0;
};

template<typename A>
concept Model_ = std::is_base_of_v<Model<A::in_size, A::out_size>, A>;


// ======= wrapper model =======

template<int FINAL_SZ, int IN_SZ>
struct ReplacableModel : public Model<FINAL_SZ, IN_SZ> {
    Model<FINAL_SZ, IN_SZ>* assigned_model;

    ReplacableModel() = default;
    ReplacableModel(Model<FINAL_SZ, IN_SZ>* initial_assignment) : assigned_model(initial_assignment) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) {
        return (*assigned_model).fwd(input);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) {
        return (*assigned_model).bwd(loss);
    }
    void init_weights() {
        if (assigned_model != nullptr)
            (*assigned_model).init_weights();
    }
};


// ======= dense models =======

template<int FINAL_SZ, int IN_SZ, Activation_ OUT, Activation_ ...NEURONS>
struct SequentialLayers;
template<int FINAL_SZ, int IN_SZ, Activation_ OUT> requires (FINAL_SZ == OUT::size)
struct SequentialLayers<FINAL_SZ, IN_SZ, OUT>;


template<int FINAL_SZ, int IN_SZ, Activation_ OUT, Activation_ ...NEURONS>
struct SequentialLayers : public Model<FINAL_SZ, IN_SZ> {
    Matrix<float, OUT::size, IN_SZ> weights;
    ColVector<float, IN_SZ> prev_input;
    ColVector<float, OUT::size> input_evaluated;
    SequentialLayers<FINAL_SZ, OUT::size, NEURONS...> next_weights;
    float learning_rate;
public:
    SequentialLayers(float learning_rate): learning_rate(learning_rate), next_weights(learning_rate) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        prev_input = input;
        input_evaluated = weights * input;
        return next_weights.fwd(OUT::activate(input_evaluated));
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        ColVector<float, OUT::size> intermediate = dot_had(next_weights.bwd(loss), OUT::derivative(input_evaluated));
        weights -= fmatmul(learning_rate, intermediate * prev_input.transposed());
        return weights.transposed() * intermediate;
    }
    void init_weights() override {
        OUT::template init_weights<IN_SZ>(weights);
        next_weights.init_weights();
    }
    void save_weights() {
        std::cout << "Layer weights (" << OUT::size << " x " << IN_SZ << "):\n" << weights << "\n\n"
        next_weights.save_weights();
    }
};

// base case
template<int FINAL_SZ, int IN_SZ, Activation_ OUT> requires (FINAL_SZ == OUT::size)
struct SequentialLayers<FINAL_SZ, IN_SZ, OUT> : public Model<FINAL_SZ, IN_SZ> {
    Matrix<float, OUT::size, IN_SZ> weights;
    ColVector<float, IN_SZ> prev_input;
    ColVector<float, OUT::size> input_evaluated;
    float learning_rate;
public:
    SequentialLayers(float learning_rate): learning_rate(learning_rate) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        prev_input = input;
        input_evaluated = weights * input;
        return OUT::activate(input_evaluated);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        auto intermediate = dot_had(loss, OUT::derivative(input_evaluated));
        weights -= fmatmul(learning_rate, intermediate * prev_input.transposed());
        return weights.transposed() * intermediate;
    }
    void init_weights() override {
        OUT::template init_weights<IN_SZ>(weights);
    }
    void save_weights() {
        std::cout << "Layer weights (" << OUT::size << " x " << IN_SZ << "):\n";
        for (int i = 0; i < OUT::size; ++i) {
            for (int j = 0; j < IN_SZ; ++j) {
                std::cout << weights[i, j] << " ";
            }
            std::cout << "\n";
        }
    }
};


// ======= sequential models =======

template<int FINAL_SZ, int IN_SZ, Model_ FIRST, Model_ ...COMPONENTS> requires (IN_SZ == FIRST::in_size)
struct SequentialModels;
template<int FINAL_SZ, int IN_SZ, Model_ FIRST> requires (IN_SZ == FIRST::in_size && FINAL_SZ == FIRST::out_size)
struct SequentialModels<FINAL_SZ, IN_SZ, FIRST>;



template<int FINAL_SZ, int IN_SZ, Model_ FIRST, Model_ ...COMPONENTS> requires (IN_SZ == FIRST::in_size)
struct SequentialModels : public Model<FINAL_SZ, IN_SZ> {
    FIRST model;
    SequentialModels<FINAL_SZ, FIRST::out_size, COMPONENTS...> next_models;

    SequentialModels() = default;
    SequentialModels(FIRST model, COMPONENTS...components): model(model), next_models(components...) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        return next_models.fwd(model.fwd(input));
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        return model.bwd(next_models.bwd(loss));
    }
    void init_weights() override {
        model.init_weights();
        next_models.init_weights();
    }
};

// base case
template<int FINAL_SZ, int IN_SZ, Model_ FIRST> requires (IN_SZ == FIRST::in_size && FINAL_SZ == FIRST::out_size)
struct SequentialModels<FINAL_SZ, IN_SZ, FIRST> : public Model<FINAL_SZ, IN_SZ> {
    FIRST model;

    SequentialModels() = default;
    SequentialModels(FIRST model): model(model) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        return model.fwd(input);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        return model.bwd(loss);
    }
    void init_weights() override {
        model.init_weights();
    }
};

template<int FINAL_SZ, int IN_SZ, Model_ ...COMPONENTS> 
SequentialModels<FINAL_SZ, IN_SZ, COMPONENTS...> make_sequential(COMPONENTS... components) {
    return SequentialModels<FINAL_SZ, IN_SZ, COMPONENTS...>(components...);
}


// ======= stacked models =======

template<int FINAL_SZ, int IN_SZ, Model_ FIRST, Model_ ...COMPONENTS>
struct StackedModels;
template<int FINAL_SZ, int IN_SZ, Model_ FIRST> requires (IN_SZ == FIRST::in_size && FINAL_SZ == FIRST::out_size)
struct StackedModels<FINAL_SZ, IN_SZ, FIRST>;



template<int FINAL_SZ, int IN_SZ, Model_ FIRST, Model_ ...COMPONENTS>
struct StackedModels : public Model<FINAL_SZ, IN_SZ> {
    FIRST model;
    StackedModels<FINAL_SZ - FIRST::out_size, IN_SZ - FIRST::in_size, COMPONENTS...> next_models;

    StackedModels() = default;
    StackedModels(FIRST model, COMPONENTS...components): model(model), next_models(components...) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        ColVector<float, FIRST::out_size> top = model.fwd(input.template slice<0, FIRST::in_size>());
        ColVector<float, FINAL_SZ - FIRST::out_size> bot = next_models.fwd(input.template slice<0, IN_SZ - FIRST::in_size>());
        return ColVector<float, FINAL_SZ>(top, bot);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        return model.bwd(next_models.bwd(loss));
    }
    void init_weights() override {
        model.init_weights();
        next_models.init_weights();
    }
};

// base case
template<int FINAL_SZ, int IN_SZ, Model_ FIRST> requires (IN_SZ == FIRST::in_size && FINAL_SZ == FIRST::out_size)
struct StackedModels<FINAL_SZ, IN_SZ, FIRST> : public Model<FINAL_SZ, IN_SZ> {
    FIRST model;

    StackedModels() = default;
    StackedModels(FIRST model): model(model) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        return model.fwd(input);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        return model.bwd(loss);
    }
    void init_weights() override {
        model.init_weights();
    }
};

template<int FINAL_SZ, int IN_SZ, Model_ ...COMPONENTS> 
StackedModels<FINAL_SZ, IN_SZ, COMPONENTS...> make_stacked(COMPONENTS... components) {
    return StackedModels<FINAL_SZ, IN_SZ, COMPONENTS...>(components...);
}
