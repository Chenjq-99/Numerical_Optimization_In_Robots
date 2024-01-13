#pragma once

#include<iostream>
#include<vector>

class Logger {
public:
    void LOG(int iter_count, const std::vector<double> &x, double delta) {
        std::cout << "【In " << iter_count << "th Iteration】: " << "x = [";
        int x_size = x.size();
        for (int i = 0; i < x_size; i++) {
            std::cout << x[i];
            if (i != x_size - 1) {
                std::cout << ",";
            }
        }
        std::cout << "]" << ", " << "delta = " << delta << std::endl;
    }

    void LOG(const std::vector<double> &x) {
        std::cout << "Result: ";
        int x_size = x.size();
        for (int i = 0; i < x_size; i++) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }
};
