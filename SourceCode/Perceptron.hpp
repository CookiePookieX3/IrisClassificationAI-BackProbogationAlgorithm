#pragma once
#include <string>



struct Gradient;



struct Perceptron{
    int layers = 0;
    int* structure = nullptr;
    double*** weights = nullptr;
    double** biases = nullptr;

    Perceptron();
    explicit Perceptron(int _layers);

    void makeStructure(const int _layers, const int* _structure);

    bool fillIn(const std::string& fileName);

    bool printOut(const std::string& fileName);

    void randomizeWeights(unsigned seed);

    void randomizeBioses(unsigned seed);


    void calculateNeuronActivation(double** neuronActiv,const double* in) const;

    void calculate(double* in, double* out);

    void addGradient(const Gradient& gradient, double k);


    ~Perceptron();

    void free();
};