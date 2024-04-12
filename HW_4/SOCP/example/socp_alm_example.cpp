#include "lbfgs.hpp"
#include <iostream>
#include <Eigen/Eigen>
#include <chrono>

class SOCP_ALM
{
    /*
        minimize f^T * x
        s.t. ||A * x + b|| <= c^T * x + d;

        [c^T; A] * x + [d; b] belong to Qn
    */
public:
    enum SolverType
    {
        LBFGS,
        Semi_Smooth_Newton
    };

    SOCP_ALM(const int N, const int m)
        : N_(N), m_(m) {}

    inline void Init(const Eigen::MatrixXd A, const Eigen::VectorXd b,
                     const Eigen::VectorXd c, const double d,
                     const Eigen::VectorXd f)
    {
        A_hat_ = Eigen::MatrixXd::Zero(m_ + 1, N_);
        A_hat_.block(0, 0, 1, N_) = c.transpose();
        A_hat_.block(1, 0, m_, N_) = A;
        b_hat_ = Eigen::VectorXd::Zero(m_ + 1);
        b_hat_.segment(0, 1) << d;
        b_hat_.segment(1, m_) = b;
        f_ = f;
        mu_ = Eigen::VectorXd::Zero(m_ + 1);
    }

    void Solve(const SolverType &solver_type, Eigen::VectorXd &x, double &cost)
    {
        double res_cons = 1e5;
        double res_prec = 1e5;
        int iteration = 0;

        Eigen::VectorXd x_opt = Eigen::VectorXd::Zero(N_);

        if (solver_type == SolverType::LBFGS)
        {
            std::cout << "==================L-BFGS==================" << std::endl;
            // l-bfgs parameters
            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.mem_size = 16;
            lbfgs_params.past = 0;
            lbfgs_params.g_epsilon = 0.0;
            lbfgs_params.min_step = 1e-32;
            lbfgs_params.delta = 1e-4;

            while ((res_cons > eps_cons || res_prec > eps_prec) && iteration <= max_iterations)
            {
                ++iteration;

                double min_cost = 0.0;

                // l-bfgs optimization
                lbfgs::lbfgs_optimize(x_opt, min_cost, &CostFunction, nullptr, this, lbfgs_params);
                // calculate residual
                Eigen::VectorXd v = mu_ / rho_ - A_hat_ * x_opt - b_hat_;
                Eigen::VectorXd proj = Project2Soc(v);
                Eigen::VectorXd g = f_ - rho_ * A_hat_.transpose() * proj;
                Eigen::VectorXd p = mu_ / rho_ - proj;

                res_cons = p.lpNorm<Eigen::Infinity>();
                res_prec = g.lpNorm<Eigen::Infinity>();
                // update dual variables
                mu_ = Project2Soc(mu_ - rho_ * (A_hat_ * x_opt + b_hat_));
                rho_ = std::min((1 + gamma_) * rho_, beta_);

                // print iteration info
                std::cout << "============================================" << std::endl;
                std::cout << "iteration: " << iteration << std::endl;
                std::cout << "x: " << x_opt.transpose() << std::endl;
                std::cout << "min_cost: " << min_cost << std::endl;
                std::cout << "residual cons: " << res_cons << std::endl;
                std::cout << "residual prec: " << res_prec << std::endl;
            }
        }
        else if (solver_type == SolverType::Semi_Smooth_Newton)
        {
            std::cout << "==================Semi-Smooth-Newton==================" << std::endl;
            while ((res_cons > eps_cons || res_prec > eps_prec) && iteration <= max_iterations)
            {
                ++iteration;

                double min_cost = 0.0;

                // semi-smooth-newton optimization
                SemiSmoothNewtonMethod(x_opt, min_cost);
                // calculate residual
                Eigen::VectorXd v = mu_ / rho_ - A_hat_ * x_opt - b_hat_;
                Eigen::VectorXd proj = Project2Soc(v);
                Eigen::VectorXd g = f_ - rho_ * A_hat_.transpose() * proj;
                Eigen::VectorXd p = mu_ / rho_ - proj;

                res_cons = p.lpNorm<Eigen::Infinity>();
                res_prec = g.lpNorm<Eigen::Infinity>();
                // update dual variables
                mu_ = Project2Soc(mu_ - rho_ * (A_hat_ * x_opt + b_hat_));
                rho_ = std::min((1 + gamma_) * rho_, beta_);

                // print iteration info
                std::cout << "============================================" << std::endl;
                std::cout << "iteration: " << iteration << std::endl;
                std::cout << "x: " << x_opt.transpose() << std::endl;
                std::cout << "min_cost: " << min_cost << std::endl;
                std::cout << "residual cons: " << res_cons << std::endl;
                std::cout << "residual prec: " << res_prec << std::endl;
            }
        }
        x = x_opt;
        cost = f_.dot(x);
    }

private:
    inline static double CostFunction(void *instance, const Eigen::VectorXd &x,
                                      Eigen::VectorXd &g)
    {
        SOCP_ALM &obj = *(SOCP_ALM *)instance;
        g = obj.CalculateGradient(x);
        double cost = obj.CalculateCost(x);
        return cost;
    }

    inline void SemiSmoothNewtonMethod(Eigen::VectorXd &x, double &cost)
    {
        const double eps = 1e-5;
        const double c = 1e-4;        //Armijo condition c
        auto g = CalculateGradient(x);
        const Eigen::MatrixXd identity_N = Eigen::MatrixXd::Identity(N_, N_);
        while (g.norm() >= eps)
        {
            auto prox_hessian = CalculateProximateHessian(x);
            auto direction = -prox_hessian.inverse() * g;
            double step = 1.0;
            double f = CalculateCost(x);
            while (CalculateCost(x + step * direction) > f + c * step * direction.dot(g))
            {
                step = 0.5 * step;
            }
            x = x + step * direction;
            g = CalculateGradient(x);
        }
        cost = CalculateCost(x);
    }

    inline Eigen::VectorXd CalculateGradient(const Eigen::VectorXd &x)
    {
        Eigen::VectorXd v = mu_ / rho_ - A_hat_ * x - b_hat_;
        Eigen::VectorXd proj = Project2Soc(v);
        Eigen::VectorXd g = f_ - rho_ * A_hat_.transpose() * proj;
        return g;
    }

    inline double CalculateCost(const Eigen::VectorXd &x)
    {
        Eigen::VectorXd v = mu_ / rho_ - A_hat_ * x - b_hat_;
        Eigen::VectorXd proj = Project2Soc(v);
        double cost = f_.dot(x) + rho_ * proj.squaredNorm() / 2;
        return cost;
    }

    inline Eigen::MatrixXd CalculateProximateHessian(const Eigen::VectorXd &x)
    {
        auto grad_project2soc = CalculateBDifferentialOfProject(mu_ - rho_ * (A_hat_ * x + b_hat_));
        Eigen::MatrixXd prox_hessian = rho_ * A_hat_.transpose() * grad_project2soc * A_hat_;
        return prox_hessian;
    }

    inline Eigen::VectorXd Project2Soc(const Eigen::VectorXd &v)
    {
        const double v0 = v(0);
        const Eigen::VectorXd v1 = v.segment(1, m_);
        const double v1_norm = v1.norm();
        Eigen::VectorXd proj_v = Eigen::VectorXd::Zero(m_ + 1);
        if (v0 <= -v1_norm)
        {
            proj_v = Eigen::VectorXd::Zero(m_ + 1);
        }
        else if (v0 >= v1_norm)
        {
            proj_v = v;
        }
        else
        {
            double coeff = (v0 + v1_norm) / (2 * v1_norm);
            proj_v << v1_norm, v1;
            proj_v *= coeff;
        }
        return proj_v;
    }

    inline Eigen::MatrixXd CalculateBDifferentialOfProject(const Eigen::VectorXd &v)
    {
            const double v0 = v(0);
            const Eigen::VectorXd v1 = v.segment(1, m_);
            const double v1_norm = v1.norm();
            Eigen::MatrixXd b_differential(m_ + 1, m_ + 1);
            if (v0 <= -v1_norm)
            {
                b_differential = Eigen::MatrixXd::Zero(m_ + 1, m_ + 1);
            }
            else if (v0 >= v1_norm)
            {
                b_differential = Eigen::MatrixXd::Identity(m_ + 1, m_ + 1);
            }
            else
            {
                auto top_left = b_differential.block(0, 0, 1, 1);
                auto top_right = b_differential.block(0, 1, 1, m_);
                auto bottom_left = b_differential.block(1, 0, m_, 1);
                auto bottom_right = b_differential.block(1, 1, m_, m_);

                top_left << 0.5;
                top_right << v1.transpose() / v1_norm / 2;
                bottom_left << v1 / v1_norm / 2;
                bottom_right << -v0 * v1 * v1.transpose() / std::pow(v1_norm, 3) / 2 
                                + (v0 + v1_norm) / v1_norm / 2 * Eigen::MatrixXd::Identity(m_, m_);
            }
            
            return b_differential;
    }

    const double eps_cons = 1e-5;
    const double eps_prec = 1e-5;
    const int max_iterations = 50;
    int N_;
    int m_;
    double rho_ = 1.0, gamma_ = 1.0, beta_ = 1e5;
    Eigen::MatrixXd A_hat_;
    Eigen::VectorXd f_, b_hat_, mu_;
};

int main()
{
    int N = 7, m = 7;
    Eigen::MatrixXd A(m, N);
    Eigen::VectorXd b(m), c(N), f(N);
    double d = 1.0;

    Eigen::DiagonalMatrix<double, 7> diagonalMatrix;
    diagonalMatrix.diagonal() << 7, 6, 5, 4, 3, 2, 1;
    A = diagonalMatrix;

    b << 1, 3, 5, 7, 9, 11, 13;
    c << 1, 0, 0, 0, 0, 0, 0;
    f << 1, 2, 3, 4, 5, 6, 7;

    SOCP_ALM socp_alm(N, m);
    socp_alm.Init(A, b, c, d, f);

    Eigen::VectorXd x1;
    double cost1;
    std::chrono::high_resolution_clock::time_point start_time_1 = std::chrono::high_resolution_clock::now();
    socp_alm.Solve(SOCP_ALM::SolverType::LBFGS, x1, cost1);
    std::chrono::high_resolution_clock::time_point end_time_1 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration time_cost_1 = end_time_1 - start_time_1;

    std::cout << "****************************************" << std::endl;
    std::cout << "optimal x with lbfgs: " << x1.transpose() << std::endl;
    std::cout << "primal problem minimal cost: " << cost1 << std::endl;
    std::cout << "time cost with lbfgs: " << time_cost_1.count() / 1000000.0 << "ms" << std::endl; 

    Eigen::VectorXd x2;
    double cost2;
    std::chrono::high_resolution_clock::time_point start_time_2 = std::chrono::high_resolution_clock::now();
    socp_alm.Solve(SOCP_ALM::SolverType::Semi_Smooth_Newton, x2, cost2);
    std::chrono::high_resolution_clock::time_point end_time_2 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration time_cost_2 = end_time_2 - start_time_2;

    std::cout << "****************************************" << std::endl;
    std::cout << "optimal x with semi-smooth-newton: " << x2.transpose() << std::endl;
    std::cout << "primal problem minimal cost: " << cost2 << std::endl;
    std::cout << "time cost with semi-smooth-newton: " << time_cost_2.count() / 1000000.0 << "ms" << std::endl; 


    return 0;
}
