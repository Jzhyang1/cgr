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
};


// ======= dense models =======

template<int FINAL_SZ, int IN_SZ, Activation_ OUT, Activation_ ...NEURONS>
struct SequentialLayers;
template<int FINAL_SZ, int IN_SZ, Activation_ OUT> requires (FINAL_SZ == OUT::size)
struct SequentialLayers<FINAL_SZ, IN_SZ, OUT>;


template<int FINAL_SZ, int IN_SZ, Activation_ OUT, Activation_ ...NEURONS>
struct SequentialLayers : public Model<FINAL_SZ, IN_SZ> {
    OUT::template storage<IN_SZ> weights;
    SequentialLayers<FINAL_SZ, OUT::size, NEURONS...> next_weights;
    float learning_rate;
public:
    SequentialLayers(float learning_rate): learning_rate(learning_rate), next_weights(learning_rate) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        return next_weights.fwd(weights.fwd(input));
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        ColVector<float, OUT::size> intermediate = next_weights.bwd(loss);
        return weights.bwd(intermediate, learning_rate);
    }

    void save_weights(std::ostream& os) {
        weights.save_weights(os);
        next_weights.save_weights(os);
    }
    void load_weights(std::istream& is) {
        weights.load_weights(is);
        next_weights.load_weights(is);
    }
};

// base case
template<int FINAL_SZ, int IN_SZ, Activation_ OUT> requires (FINAL_SZ == OUT::size)
struct SequentialLayers<FINAL_SZ, IN_SZ, OUT> : public Model<FINAL_SZ, IN_SZ> {
    OUT::template storage<IN_SZ> weights;
    float learning_rate;
public:
    SequentialLayers(float learning_rate): learning_rate(learning_rate) {}

    ColVector<float, FINAL_SZ> fwd(ColVector<float, IN_SZ> input) override {
        return weights.fwd(input);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        return weights.bwd(loss, learning_rate);
    }
    
    // Overload the stream insertion operator
    void save_weights(std::ostream& os) {
        weights.save_weights(os);
    }
    void load_weights(std::istream& is) {
        weights.load_weights(is);
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
        ColVector<float, FINAL_SZ - FIRST::out_size> bot = next_models.fwd(input.template slice<FIRST::in_size, IN_SZ>());
        return ColVector<float, FINAL_SZ>(top, bot);
    }
    ColVector<float, IN_SZ> bwd(ColVector<float, FINAL_SZ> loss) override {
        ColVector<float, FIRST::in_size> top = model.bwd(loss.template slice<0, FIRST::out_size>());
        ColVector<float, IN_SZ - FIRST::in_size> bot = next_models.bwd(loss.template slice<FIRST::out_size, FINAL_SZ>());
        return ColVector<float, IN_SZ>(top, bot);
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
};

template<int FINAL_SZ, int IN_SZ, Model_ ...COMPONENTS> 
StackedModels<FINAL_SZ, IN_SZ, COMPONENTS...> make_stacked(COMPONENTS... components) {
    return StackedModels<FINAL_SZ, IN_SZ, COMPONENTS...>(components...);
}
