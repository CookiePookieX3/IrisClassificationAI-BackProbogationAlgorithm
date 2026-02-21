#include <chrono>
#include <iostream>
#include <fstream>
#include "../SourceCode/nn.hpp"

int main(){

    Perceptron net = Perceptron();
    net.fillIn("blankNet.txt");
    net.randomizeWeights(std::chrono::steady_clock::now().time_since_epoch().count());

    Gradient GR = Gradient();
    GR.makeEmptyGradient(net);

    int N;
    std::cout << "Enter number of learning cycles: ";
    std::cin >> N;

    double* in = new double[4];
    double* goal = new double[3];
    double* out = new double[3];

    for(int i = 0; i<N; i++){
        std::ifstream fin("TrainingData.txt");
        std::ifstream fin2("TestData.txt");

        double accCost = 0;
        
        for(int j = 0; j<120; j++){
            for(int k = 0; k<4; k++) fin >> in[k];
            for(int k = 0; k<3; k++) fin >> goal[k];

            accCost+=GR.addUpGradient(net, in, goal, 120);
        }

        int right = 0;
        for(int j = 0; j<30; j++){
            for(int k = 0; k<4; k++) fin2 >> in[k];
            for(int k = 0; k<3; k++) fin2 >> goal[k];

            net.calculate(in, out);
            for(int k = 0; k<3; k++){
                if(goal[k]&&(out[k] > out[(k+1)%3])&&(out[k] > out[(k+2)%3])) right++;
            }
        }
        
        std::cout << "Run " << i+1 << ", cost=" << accCost/120 << ",\n";
        std::cout << right << " tests passed.\n";
        net.addGradient(GR, -1);
        GR.clear();
    }

    return 0;
}