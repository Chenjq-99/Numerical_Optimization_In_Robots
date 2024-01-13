#include "../include/RosenbrockCost.h"
#include "../include/GradientDescentSolver.h"
#include "../include/Logger.h"

int main(int argc, char const *argv[])
{

    std::vector<double> x0;

    int dimensions = 3;
    for (int i = 0; i < dimensions; ++i) {
        x0.push_back(0.0);
    }

    GradientDescentSolver solver(x0, 100000, 1e-4);

    auto result = solver.Solve();

    Logger().LOG(result);

    return 0;
}
