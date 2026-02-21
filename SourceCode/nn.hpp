#pragma once
#include <fstream>
#include <chrono>
#include <random>
#include "Perceptron.hpp"
#include "Gradient.hpp"



double sigmoid(double num){
    return (1.0/(1.0+std::exp(-num)));
}

double cost(const double* out, const double* goal, int n){
    double sum = 0.0;

    for(int i = 0; i<n; i++){
        sum+=(out[i]-goal[i])*(out[i] - goal[i]);
    }

    return sum;
}




    Perceptron::Perceptron() = default;
    Perceptron::Perceptron(int _layers): layers(_layers), 
    structure(_layers? new int[_layers] : nullptr),
    weights(_layers? new double** [_layers-1]{} : nullptr),
    biases(_layers? new double*[_layers-1]{} : nullptr) {}

    void Perceptron::makeStructure(const int _layers, const int* _structure){
        layers = _layers;
        structure =  new int[layers];
        for(int l = 0; l<layers; l++){
            structure[l] = _structure[l];
        }
    }

    bool Perceptron::fillIn(const std::string& fileName){
        
        std::ifstream fin(fileName);
        if(!fin) return 0;

        fin >> layers;
        weights = new double**[layers-1];

        structure = new int[layers];
        for(int i = 0; i<layers; i++) fin >> structure[i];
        biases = new double*[layers-1];
        
        for(int l = 0; l<layers-1; l++){
            weights[l] = new double*[structure[l+1]];
            for(int i = 0; i<structure[l+1]; i++){
                weights[l][i] = new double[structure[l]];
                for(int j = 0; j<structure[l]; j++) fin >> weights[l][i][j];
            }
        }
        
        for(int l = 0; l<layers-1; l++){
            biases[l] = new double[structure[l+1]];
            for(int i = 0; i<structure[l+1]; i++) fin >> biases[l][i];
        }
        
        return 1;
    }

    bool Perceptron::printOut(const std::string& fileName){
        
        std::ofstream fout(fileName);
        if(!fout) return 0;

        fout << layers << '\n';
        for(int l = 0; l<layers; l++) fout << structure[l] << ' ';
        fout << '\n';

        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l+1]; i++){
                for(int j = 0; j<structure[l]; j++) fout << weights[l][i][j] << ' ';
                fout << '\n';
            }
        }
        
        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l+1]; i++) fout << biases[l][i] << ' ';
            fout << '\n';
        }
        
        return 1;
    }

    void Perceptron::randomizeWeights(unsigned seed = 0){

        std::mt19937 rng(seed);

        if (!weights || !biases || !structure || layers < 2) return;

        for(int l = 0; l<layers-1; l++){
            int fan_in  = structure[l];
            int fan_out = structure[l + 1];

            double limit = std::sqrt(6.0 / (fan_in + fan_out));
            std::uniform_real_distribution<double> dist(-limit, limit);

            for(int i = 0; i<structure[l]; i++){
                for(int j = 0; j<structure[l+1]; j++){
                    weights[l][j][i] = dist(rng);
                }
            }
        }
    }

    void Perceptron::randomizeBioses(unsigned seed = 0){

        std::mt19937 rng(seed);

        if (!weights || !biases || !structure || layers < 2) return;

        for(int l = 1; l<layers; l++){
            int fan_in  = structure[l];
            int fan_out = structure[l + 1];

            double limit = std::sqrt(6.0 / (fan_in + fan_out));
            std::uniform_real_distribution<double> dist(-limit, limit);

            for(int i = 0; i<structure[l]; i++){
                biases[l-1][i] = dist(rng)/10;
            }
        }
    }


    void Perceptron::calculateNeuronActivation(double** neuronActiv,const double* in) const {
        
        for(int i = 0; i<structure[0]; i++) neuronActiv[0][i] = in[i];

        for(int l = 1; l<layers; l++){
            for(int i = 0; i<structure[l]; i++){
                double sum = 0;
                for(int j = 0; j<structure[l-1]; j++){
                    sum+=neuronActiv[l-1][j] * weights[l-1][i][j];
                }
                neuronActiv[l][i] = sigmoid(sum + biases[l-1][i]);
            }
        }
    }

    void Perceptron::calculate(double* in, double* out){
        double** neuronActiv = new double*[layers];
        for(int l = 0; l<layers; l++){
            neuronActiv[l] = new double[structure[l]];
        }
        
        calculateNeuronActivation(neuronActiv, in);

        for(int i = 0; i<structure[layers-1]; i++){
            out[i] = neuronActiv[layers-1][i];
        }

        for(int l = 0; l<layers; l++){
            delete[] neuronActiv[l];
        }
        delete[] neuronActiv;
    }

    void Perceptron::addGradient(const Gradient& gradient, double k){
        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l]; i++){
                for(int j = 0; j<structure[l+1]; j++){
                    weights[l][j][i]+=k*gradient.weightGr[l][j][i];
                }
            }
        }

        for(int l = 1; l<layers; l++){
            for(int i = 0; i<structure[l]; i++){
                biases[l-1][i]+=k*gradient.biasGr[l-1][i];
            }
        }
    }


    Perceptron::~Perceptron(){
        free();
    }

    void Perceptron::free(){
        if(weights){
            for(int l = 0; l<layers-1; l++){
                if(weights[l]){
                    for(int i = 0; i<structure[l+1]; i++){
                        delete[] weights[l][i];
                        weights[l][i] = nullptr;
                    }
                    delete[] weights[l];
                    weights[l] = nullptr;
                }
            }
            delete[] weights;
            weights = nullptr;
        }
        if(biases){
            for(int l = 0; l<layers-1; l++){
                if(biases[l]) delete[] biases[l];
                biases[l] = nullptr;
            }
            delete[] biases;
        }
        biases = nullptr;
        if(structure) delete[] structure;
        structure = nullptr;
    }



    Gradient::Gradient() = default;
    Gradient::Gradient(int _layers): layers(_layers), 
    structure(_layers? new int[_layers] : nullptr),
    weightGr(_layers? new double** [_layers-1]{} : nullptr),
    biasGr(_layers? new double*[_layers-1]{} : nullptr) {}
    
    void Gradient::makeEmptyGradient(const Perceptron& example){
        
        layers = example.layers;
        structure = new int[layers];
        weightGr = new double**[layers-1];
        biasGr = new double*[layers-1];
        
        for(int l = 0; l<layers; l++){
            structure[l] = example.structure[l];
        }
        
        for(int l = 0; l<layers-1; l++){
            weightGr[l] = new double*[structure[l+1]];
            biasGr[l] = new double[structure[l+1]];
            for(int i = 0; i<structure[l+1]; i++){
                biasGr[l][i] = 0;
                weightGr[l][i] = new double[structure[l]];
                for(int j = 0; j<structure[l]; j++){
                    weightGr[l][i][j] = 0;
                }
            }
        }
    }

    bool Gradient::printOut(const std::string& fileName){
        
        std::ofstream fout(fileName);
        if(!fout) return 0;

        fout << layers << '\n';
        for(int l = 0; l<layers; l++) fout << structure[l] << ' ';
        fout << '\n';

        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l+1]; i++){
                for(int j = 0; j<structure[l]; j++) fout << weightGr[l][i][j] << ' ';
                fout << '\n';
            }
        }
        
        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l+1]; i++) fout << biasGr[l][i] << ' ';
            fout << '\n';
        }
        
        return 1;
    }


    double Gradient::addUpGradient(const Perceptron& net, const double* in, const double* goal, int batchSize){
        
        double** neuronActiv = new double*[layers];
        for(int l = 0; l<layers; l++){
            neuronActiv[l] = new double[structure[l]];
        }
        
        net.calculateNeuronActivation(neuronActiv, in);

        double** dCdNA = new double*[layers-1];
        for(int l = 0; l<layers-1; l++){
            dCdNA[l] = new double[structure[l+1]];
        }
        for(int i = 0; i<structure[layers-1]; i++){
            dCdNA[layers-2][i] = 2*(neuronActiv[layers-1][i] - goal[i]);
        }

        for(int l = layers - 2; l>0; l--){
            for(int i = 0; i<structure[l]; i++){
                double sum = 0.0;
                for(int j = 0; j<structure[l+1]; j++){
                    sum +=dCdNA[l][j]*neuronActiv[l+1][j]*(1-neuronActiv[l+1][j])*net.weights[l][j][i];
                }
                dCdNA[l-1][i]=sum;
            }
        }
        

        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l]; i++){
                for(int j = 0; j<structure[l+1]; j++){
                    weightGr[l][j][i]+=(dCdNA[l][j]*neuronActiv[l+1][j]*(1-neuronActiv[l+1][j])*neuronActiv[l][i])/batchSize;
                }
            }
        }

        for(int l = 1; l<layers; l++){
            for(int i = 0; i<structure[l]; i++){
                biasGr[l-1][i]+=(dCdNA[l-1][i]*neuronActiv[l][i]*(1-neuronActiv[l][i]))/batchSize;
            }
        }

        double* out = new double[structure[layers-1]];
        for(int i = 0; i<structure[layers-1]; i++){
            out[i] = neuronActiv[layers-1][i];
        }

        double d_cost = cost(out, goal, structure[layers-1]);

        for(int l = 0; l<layers-1; l++){
            delete[] dCdNA[l];
            delete[] neuronActiv[l];
        }
        delete[] neuronActiv[layers-1];
        delete[] dCdNA;
        delete[] neuronActiv;

        return d_cost;
    }

    void Gradient::clear(){
        for(int l = 0; l<layers-1; l++){
            for(int i = 0; i<structure[l]; i++){
                for(int j = 0; j<structure[l+1]; j++){
                    weightGr[l][j][i] = 0;
                }
            }
        }

        for(int l = 1; l<layers; l++){
            for(int i = 0; i<structure[l]; i++){
                biasGr[l-1][i] = 0;
            }
        }
    }


    Gradient::~Gradient(){
        free();
    }
    
    void Gradient::free(){
        if(weightGr){
            for(int l = 0; l<layers-1; l++){
                if(weightGr[l]){
                    for(int i = 0; i<structure[l+1]; i++){
                        delete[] weightGr[l][i];
                        weightGr[l][i] = nullptr;
                    }
                    delete[] weightGr[l];
                    weightGr[l] = nullptr;
                }
            }
            delete[] weightGr;
            weightGr = nullptr;
        }
        if(biasGr){
            for(int l = 0; l<layers-1; l++){
                if(biasGr[l]) delete[] biasGr[l];
                biasGr[l] = nullptr;
            }
            delete[] biasGr;
        }
        biasGr = nullptr;
        if(structure) delete[] structure;
        structure = nullptr;
    }