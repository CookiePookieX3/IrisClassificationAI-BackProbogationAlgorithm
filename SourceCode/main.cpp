#include <iostream>
#include "nn.hpp"

int main(){

    Perceptron net = Perceptron();
    net.fillIn("BlankNet.txt");
    net.randomizeWeights();
    net.randomizeBioses();
    net.printOut("randomNet.txt");

    double* in = new double[4];
    in[0] = 1; in[1] = 1; in[2] = 1; in[3] = 1;

    double* goal = new double[3];
    goal[0] = 1; goal[1] = 0; goal[2] = 0; goal[3] = 0;

    double* out = new double[3];

    Gradient GR = Gradient();
    GR.makeEmptyGradient(net);

    net.calculate(in ,out);

    std::cout << cost(out, goal, net.structure[net.layers-1]) << '\n';
    std::cout << out[0] << ' ' << out[1] << ' ' << out[2] << '\n';


    for(int i = 0; i<10; i++){
        GR.clear();
        GR.addUpGradient(net, in, goal, 1);
        net.addGradient(GR, -10);

        net.calculate(in ,out);
        std::cout << cost(out, goal, net.structure[net.layers-1]) << '\n';
        std::cout << out[0] << ' ' << out[1] << ' ' << out[2] << '\n';
    }
    return 0;
}