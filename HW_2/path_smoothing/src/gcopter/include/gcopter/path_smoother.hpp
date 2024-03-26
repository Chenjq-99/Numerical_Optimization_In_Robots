#ifndef PATH_SMOOTHER_HPP
#define PATH_SMOOTHER_HPP

#include "cubic_spline.hpp"
#include "lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <vector>
#include <iomanip>

namespace path_smoother
{

    class PathSmoother
    {
    private:
        cubic_spline::CubicSpline cubSpline;

        int pieceN;
        Eigen::Matrix3Xd diskObstacles;
        double penaltyWeight;
        Eigen::Vector2d headP;
        Eigen::Vector2d tailP;
        Eigen::Matrix2Xd points;
        Eigen::Matrix2Xd gradByPoints;

        lbfgs::lbfgs_parameter_t lbfgs_params;

    private:
        static inline double costFunction(void *ptr,
                                          const Eigen::VectorXd &x,
                                          Eigen::VectorXd &g)
        {

            // TODO
            auto smoother_ptr = reinterpret_cast<path_smoother::PathSmoother *>(ptr);
            const int points_nums = smoother_ptr->pieceN - 1;
            Eigen::Matrix2Xd grad;
            grad.resize(2, points_nums);
            grad.setZero();
            double cost = 0.0;

            Eigen::Matrix2Xd inner_pts;
            inner_pts.resize(2, points_nums);
            inner_pts.row(0) = x.head(points_nums);
            inner_pts.row(1) = x.tail(points_nums);
            smoother_ptr->cubSpline.setInnerPoints(inner_pts);

            double energy = 0.0;
            Eigen::Matrix2Xd energy_grad;
            energy_grad.resize(2, points_nums);
            energy_grad.setZero();
            smoother_ptr->cubSpline.getStretchEnergy(energy);
            smoother_ptr->cubSpline.getGrad(energy_grad);

            double obstacles = 0.0;
            Eigen::Matrix2Xd potential_grad;
            potential_grad.resize(2, points_nums);
            potential_grad.setZero();
            for (int i = 0; i < points_nums; ++i)
            {
                for (int j = 0; j < smoother_ptr->diskObstacles.cols(); ++j)
                {
                    Eigen::Vector2d diff = inner_pts.col(i) - smoother_ptr->diskObstacles.col(j).head(2);
                    double distance = diff.norm();
                    double delta = smoother_ptr->diskObstacles(2, j) - distance;

                    if (delta > 0.0)
                    {
                        obstacles += smoother_ptr->penaltyWeight * delta;
                        potential_grad.col(i) += smoother_ptr->penaltyWeight * (-diff / distance);
                    }
                }
            }
            cost = energy + obstacles;
            grad = energy_grad + potential_grad;

            g.setZero();
            g.head(points_nums) = grad.row(0).transpose();
            g.tail(points_nums) = grad.row(1).transpose();

            return cost;
        }

    public:
        inline bool setup(const Eigen::Vector2d &initialP,
                          const Eigen::Vector2d &terminalP,
                          const int &pieceNum,
                          const Eigen::Matrix3Xd &diskObs,
                          const double penaWeight)
        {
            pieceN = pieceNum;
            diskObstacles = diskObs;
            penaltyWeight = penaWeight;
            headP = initialP;
            tailP = terminalP;

            cubSpline.setConditions(headP, tailP, pieceN);

            points.resize(2, pieceN - 1);
            gradByPoints.resize(2, pieceN - 1);

            return true;
        }

        inline double optimize(CubicCurve &curve,
                               const Eigen::Matrix2Xd &iniinner_pts,
                               const double &relCostTol)
        {
            // TODO
            Eigen::VectorXd x(pieceN * 2 - 2);
            Eigen::Map<Eigen::Matrix2Xd> innerP(x.data(), 2, pieceN - 1);
            innerP = iniinner_pts;

            double minCost = 0.0;

            int status = lbfgs::lbfgs_optimize(x,
                                               minCost,
                                               &PathSmoother::costFunction,
                                               nullptr,
                                               this,
                                               lbfgs_params);

            if (status >= 0)
            {
                cubSpline.getCurve(curve);
            }
            else
            {
                std::cout << "Generate cubic spline failed!" << std::endl;
            }
            return minCost;
        }
    };

}

#endif