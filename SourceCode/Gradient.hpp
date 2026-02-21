#pragma once
#include <string>



struct Perceptron;



struct Gradient{

    int layers = 0;
    int* structure = nullptr;
    double*** weightGr = nullptr;
    double** biasGr = nullptr;

    Gradient();
    explicit Gradient(int _layers);
    
    void makeEmptyGradient(const Perceptron& example);

    bool printOut(const std::string& fileName);


    double addUpGradient(const Perceptron& net, const double* in, const double* goal, int batchSize);

    void clear();
    

    ~Gradient();
    
    void free();
};