#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <functional>
#include <regex>
#include "tensors_cuda.h"


constexpr int BIG_SIZE = 16384; 



template<typename T, int L>
struct RowVector;
template<typename T, int L>
struct ColVector;


template<typename T>
struct DataSlice {
    // A lightweight wrapper for full-on vector data

    std::shared_ptr<std::vector<T>> raw_data;   // only kept if data is owned by a tensor
    std::span<T> data_view;


    DataSlice(int size): raw_data(std::make_shared<std::vector<T>>(size)), data_view(std::span(*raw_data)) {}
    DataSlice(std::shared_ptr<std::vector<T>> idata): raw_data(idata), data_view(std::span(*raw_data)) {}
    DataSlice(std::initializer_list<T> idata): raw_data(std::make_shared<std::vector<T>>(idata)), data_view(std::span(*raw_data)) {}
    DataSlice(std::span<T> iview): data_view(iview) {}

    
    T& operator[](int i) {
        return data_view[i];
    }
    T operator[](int i) const {
        return data_view[i];
    }
};




template<typename T, int V = 1, int H = 1>
struct Matrix {
    DataSlice<T> data;
    int v_multiplier = H, h_multiplier = 1;
    int v_start = 0, h_start = 0; // for submatrix view
    
    Matrix(): data(V * H) {};
    Matrix(RowVector<T, H> mat): Matrix(mat.data, mat.v_multiplier, mat.h_multiplier, mat.v_start, mat.h_start) {}
    Matrix(ColVector<T, V> mat): Matrix(mat.data, mat.v_multiplier, mat.h_multiplier, mat.v_start, mat.h_start) {}

    Matrix(DataSlice<T> data): data(data) {}
    Matrix(DataSlice<T> data, int v_multiplier, int h_multiplier): data(data), v_multiplier(v_multiplier), h_multiplier(h_multiplier) {}
    Matrix(DataSlice<T> data, int v_multiplier, int h_multiplier, int v_start, int h_start): 
        data(data), v_multiplier(v_multiplier), h_multiplier(h_multiplier), v_start(v_start), h_start(h_start) {}


    T& operator[](int const v, int const h) {
        // gets the v'th row as a row std::vector
        return data[(v_start + v) * v_multiplier + (h_start + h) * h_multiplier];
    }

    T operator[](int const v, int const h) const {
        // gets the v'th row as a row std::vector
        return data[(v_start + v) * v_multiplier + (h_start + h) * h_multiplier];
    }

    template<int N, typename std::enable_if<(V * H + H * N < BIG_SIZE), int>::type = 0>
    Matrix<float, V, N> operator*(const Matrix<float, H, N>& B) {
        Matrix<float, V, N> result;
        matmul_cuda(
              data.data_view.data(),   data.data_view.size(),   v_start,   h_start,   v_multiplier,   h_multiplier,
            B.data.data_view.data(), B.data.data_view.size(), B.v_start, B.h_start, B.v_multiplier, B.h_multiplier,
            result.data.data_view.data(), V, N, H
        );
        return result;
    }
    
    template<int N, typename std::enable_if<(V * H + H * N >= BIG_SIZE), int>::type = 0>
    Matrix<float, V, N> operator*(const Matrix<float, H, N>& m2) {
        constexpr int AV1 = V/2;
        constexpr int AH1 = H/2;
        constexpr int BV1 = H/2;
        constexpr int BH1 = N/2;
        auto A00 = view<0, 0, AV1, AH1>();
        auto A01 = view<0, AH1, AV1, H>();
        auto A10 = view<AV1, 0, V, AH1>();
        auto A11 = view<AV1, AH1, V, H>();
        auto B00 = m2.template view<0, 0, AV1, AH1>();
        auto B01 = m2.template view<0, AH1, AV1, H>();
        auto B10 = m2.template view<AV1, 0, V, AH1>();
        auto B11 = m2.template view<AV1, AH1, V, H>();

        A11 = A11 + A01 - A10;
        B11 = B11 + B01 - B10;

        auto t0 = A10 + A11;
        auto t1 = A11 - A01;
        auto t2 = A11 - A00;
        auto t3 = B11 - B00;
        auto t4 = B10 + B11;
        auto t5 = B11 - B01;

        auto M0 = A00 * B00;
        auto M1 = A01 * B10;
        auto M2 = A10 * t3;
        auto M3 = A11 * B11;
        auto M4 = t0 * t4;
        auto M5 = t1 * t5;
        auto M6 = t2 * B01;

        auto D = M4 + M5 - M1 - M3;

        Matrix<float, V, N> ret;
        ret.template view<0, 0, AV1, AH1>() += M0 + M1;
        ret.template view<AV1, AH1, V, N>() += D;
        ret.template view<0, AH1, AV1, H>() += M4 - M6 - D;
        ret.template view<AV1, 0, V, AH1>() += D - (M2 + M5);
        return ret;
    }

    ColVector<float, V> operator*(const ColVector<float, H> m2) {
        ColVector<float, V> ret;
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < H; ++j) {
                ret[i] += operator[](i, j) * m2[j];
            }
        }
        return ret;
    }

    Matrix<float, V, H>& operator+=(Matrix<float, V, H> m2) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < H; ++j) {
                (*this)[i, j] += m2[i, j];
            }
        }
        return *this;
    }

    Matrix<float, V, H>& operator-=(Matrix<float, V, H> m2) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < H; ++j) {
                (*this)[i, j] -= m2[i, j];
            }
        }
        return *this;
    }

    Matrix<T, H, V> transposed() {
        return Matrix<T, H, V>(data, h_multiplier, v_multiplier, h_start, v_start);
    }

    template<typename U>
    Matrix<U, V, H> applied(std::function<U(T)> func) {
        Matrix<U, V, H> ret;
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < H; ++j) {
                ret[i, j] = func(operator[](i, j));
            }
        }
        return ret;
    }

    template <int V0, int H0, int V1, int H1>
    Matrix<T, V1 - V0, H1 - H0> const view() const {
        // returns a matrix view starting from (v0, h0) inclusive to (v1, h1) exclusive
        return Matrix<T, V1 - V0, H1 - H0>(data, v_multiplier, h_multiplier, v_start + V0, h_start + H0);
    }
    template <int V0, int H0, int V1, int H1>
    Matrix<T, V1 - V0, H1 - H0> view() {
        // returns a matrix view starting from (v0, h0) inclusive to (v1, h1) exclusive
        return Matrix<T, V1 - V0, H1 - H0>(data, v_multiplier, h_multiplier, v_start + V0, h_start + H0);
    }

    RowVector<T, H*V> flattened() {
        if (h_start == 0)
            return RowVector<T, H*V>(data, v_multiplier, h_multiplier, v_start, 0);
        else
            throw "flattening a matrix view is inefficient";
    }


    // Overload the stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, const Matrix& obj) {
        os << "Matrix(" << V << ", " << H << ")[\n";
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < H; ++j) {
                os << obj[i, j] << ",\t";
            }
            os << std::endl;
        }
        os << "]";
        return os;
    }
    // Overload the stream read operator
    friend std::istream& operator>>(std::istream& is, Matrix& obj) {
        // read the first line
        std::string line;
        std::getline(is, line);

        std::regex first_line_pattern(R"(Matrix\((\d+),\s*(\d+)\)\[)");
        std::smatch match;
        if (std::regex_search(line, match, first_line_pattern)) {
            int _V = std::stoi(match[1]);
            int _H = std::stoi(match[2]);
            
            if (_V != V || _H != H) {
                std::cerr << "trying to read Matrix of " << _V << 'x' << _H << " into " << V << 'x' << H;
                throw "Matrix read from stream failed";
            }
        } else {
            std::cerr << "trying to read Matrix header but encountered " << line;
            throw "Matrix read from stream failed";
        }
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < H; ++j) {
                char comma;
                is >> obj[i, j] >> comma;
            }
            std::getline(is, line); // clear the remainder of the line
        }
        char endbrace;
        is >> endbrace;
        return is;
    }
};

template<typename T, int L>
struct RowVector: public Matrix<T, 1, L> {
    using Matrix<T, 1, L>::v_multiplier;
    using Matrix<T, 1, L>::h_multiplier;
    using Matrix<T, 1, L>::v_start;
    using Matrix<T, 1, L>::h_start;
    using Matrix<T, 1, L>::data;
    
    RowVector(RowVector<T, L> const&) = default;
    RowVector(Matrix<T, 1, L> const& mat) : Matrix<T, 1, L>(mat) {}
    RowVector(DataSlice<T> data, int h_multiplier, int h_start): Matrix<T, 1, L>(data, 1, h_multiplier, 0, h_start) {}
    using Matrix<T, 1, L>::Matrix;

    T& operator[](int h) {
        return Matrix<T, 1, L>::operator[](0, h);
    }
    T operator[](int h) const {
        return Matrix<T, 1, L>::operator[](0, h);
    }

    ColVector<T, L> transposed() {
        return ColVector(data, h_multiplier, v_multiplier, h_start, v_start);
    }

    T operator*(ColVector<T, L> const& m2) const {
        T ret;
        for (int i = 0; i < L; ++i) {
            ret += operator[](i) * m2[i];
        }
        return ret;
    }
    template<int H>
    RowVector<T, H> operator*(Matrix<T, L, H> const& m2) const {
        RowVector<T, H> result;
        for (int h = 0; h < H; ++h) {
            for (int l = 0; l < L; ++l) {
                result[h] += operator[](l) * m2[l, h];
            }
        }
        return result;
    }

    // // Overload the stream insertion operator
    // friend std::ostream& operator<<(std::ostream& os, const RowVector& obj) {
    //     os << "RowVector(" << L << ")[\n";
    //     for (int i = 0; i < L; ++i) {
    //         os << obj[i] << ",\t";
    //     }
    //     os << "\n]";
    //     return os;
    // }
};

template<typename T, int L>
struct ColVector: public Matrix<T, L, 1> {
    using Matrix<T, L, 1>::v_multiplier;
    using Matrix<T, L, 1>::h_multiplier;
    using Matrix<T, L, 1>::v_start;
    using Matrix<T, L, 1>::h_start;
    using Matrix<T, L, 1>::data;
    
    template<int L1, int L2>
    ColVector(ColVector<T, L1> first, ColVector<T, L2> second): Matrix<T, L1 + L2, 1>() {
        for (int i = 0; i < L1; ++i) {
            operator[](i) = first[i];
        }
        for (int i = 0; i < L2; ++i) {
            operator[](L1 + i) = second[i];
        }
    }
    ColVector(ColVector<T, L> const&) = default;
    ColVector(Matrix<T, L, 1> const& mat) : Matrix<T, L, 1>(mat) {}
    ColVector(DataSlice<T> data, int v_multiplier, int v_start): Matrix<T, L, 1>(data, v_multiplier, 1, v_start, 0) {}
    using Matrix<T, L, 1>::Matrix;
    using Matrix<T, L, 1>::operator*;

    T& operator[](int v) {
        return Matrix<T, L, 1>::operator[](v, 0);
    }
    T operator[](int v) const {
        return Matrix<T, L, 1>::operator[](v, 0);
    }


    RowVector<T, L> transposed() {
        return RowVector<T, L>(data, h_multiplier, v_multiplier, h_start, v_start);
    }

    template<int I, int J>
    ColVector<T, J - I> slice() {
        return ColVector<T, J - I>(data, v_multiplier, h_multiplier, I * v_multiplier + v_start, h_start);
    }

    // // Overload the stream insertion operator
    // friend std::ostream& operator<<(std::ostream& os, const ColVector& obj) {
    //     os << "ColVector(" << L << ")[\n";
    //     for (int i = 0; i < L; ++i) {
    //         os << obj[i] << ",\t";
    //         os << std::endl;
    //     }
    //     os << ']';
    //     return os;
    // }
};



template<typename T, int V, int H>
Matrix<T, V, H> operator+(Matrix<T, V, H> const m1, Matrix<T, V, H> const m2) {
    Matrix<T, V, H> ret;
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < H; ++j) {
            ret[i, j] = m1[i, j] + m2[i, j];
        }
    }
    return ret;
}

template<typename T, int V1, int V2, int H1, int H2>
Matrix<T, std::max(V1, V2), std::max(H1, H2)> operator+(Matrix<T, V1, H1> const m1, Matrix<T, V2, H2> const m2) {
    Matrix<T, std::max(V1, V2), std::max(H1, H2)> ret;
    for (int i = 0; i < std::max(V1, V2); ++i) {
        for (int j = 0; j < std::max(H1, H2); ++j) {
            if (i < V1 && j < H1) ret[i, j] = m1[i, j];
            if (i < V2 && j < H2) ret[i, j] += m2[i, j];
        }
    }
    return ret;
}


template<typename T, int V, int H>
Matrix<T, V, H> operator-(Matrix<T, V, H> const m1, Matrix<T, V, H> const m2) {
    Matrix<T, V, H> ret;
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < H; ++j) {
            ret[i, j] = m1[i, j] - m2[i, j];
        }
    }
    return ret;
}

template<typename T, int V1, int V2, int H1, int H2>
Matrix<T, std::max(V1, V2), std::max(H1, H2)> operator-(Matrix<T, V1, H1> const m1, Matrix<T, V2, H2> const m2) {
    Matrix<T, std::max(V1, V2), std::max(H1, H2)> ret;
    for (int i = 0; i < std::max(V1, V2); ++i) {
        for (int j = 0; j < std::max(H1, H2); ++j) {
            if (i < V1 && j < H1) ret[i, j] = m1[i, j];
            if (i < V2 && j < H2) ret[i, j] -= m2[i, j];
        }
    }
    return ret;
}


template<typename T, int L>
ColVector<T, L> operator-(ColVector<T, L> const& m1, ColVector<T, L> const& m2) {
    ColVector<T, L> ret;
    for (int i = 0; i < L; ++i) {
        ret[i] = m1[i] - m2[i];
    }
    return ret;
}



template<typename T, int L>
T dot(ColVector<T, L> const& m1, ColVector<T, L> const& m2) {
    T ret;
    for (int i = 0; i < L; ++i) {
        ret += m1[i] * m2[i];
    }
    return ret;
}



template<int N>
ColVector<float, N> dot_had(ColVector<float, N> m1, ColVector<float, N> m2) {
    // Hadamard product
    ColVector<float, N> ret;
    for (int i = 0; i < N; ++i) {
        ret[i] = m1[i] * m2[i];
    }
    return ret;
}

template<int N, int M>
Matrix<float, N, M> fmatmul(float n, Matrix<float, N, M> m) {
    Matrix<float, N, M> ret;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            ret[i, j] = m[i, j] * n;
        }
    }
    return ret;
}