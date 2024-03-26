#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP

#include "cubic_curve.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <vector>

namespace cubic_spline
{

    // The banded system class is used for solving
    // banded linear system Ax=b efficiently.
    // A is an N*N band matrix with lower band width lowerBw
    // and upper band width upperBw.
    // Banded LU factorization has O(N) time complexity.
    class BandedSystem
    {
    public:
        // The size of A, as well as the lower/upper
        // banded width p/q are needed
        inline void create(const int &n, const int &p, const int &q)
        {
            // In case of re-creating before destroying
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
            return;
        }

        inline void destroy()
        {
            if (ptrData != nullptr)
            {
                delete[] ptrData;
                ptrData = nullptr;
            }
            return;
        }

    private:
        int N;
        int lowerBw;
        int upperBw;
        // Compulsory nullptr initialization here
        double *ptrData = nullptr;

    public:
        // Reset the matrix to zero
        inline void reset(void)
        {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
            return;
        }

        // The band matrix is stored as suggested in "Matrix Computation"
        inline const double &operator()(const int &i, const int &j) const
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline double &operator()(const int &i, const int &j)
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        // This function conducts banded LU factorization in place
        // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
        inline void factorizeLU()
        {
            int iM, jM;
            double cVl;
            for (int k = 0; k <= N - 2; ++k)
            {
                iM = std::min(k + lowerBw, N - 1);
                cVl = operator()(k, k);
                for (int i = k + 1; i <= iM; ++i)
                {
                    if (operator()(i, k) != 0.0)
                    {
                        operator()(i, k) /= cVl;
                    }
                }
                jM = std::min(k + upperBw, N - 1);
                for (int j = k + 1; j <= jM; ++j)
                {
                    cVl = operator()(k, j);
                    if (cVl != 0.0)
                    {
                        for (int i = k + 1; i <= iM; ++i)
                        {
                            if (operator()(i, k) != 0.0)
                            {
                                operator()(i, j) -= operator()(i, k) * cVl;
                            }
                        }
                    }
                }
            }
            return;
        }

        // This function solves Ax=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solve(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; ++j)
            {
                iM = std::min(j + lowerBw, N - 1);
                for (int i = j + 1; i <= iM; ++i)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; --j)
            {
                b.row(j) /= operator()(j, j);
                iM = std::max(0, j - upperBw);
                for (int i = iM; i <= j - 1; ++i)
                {
                    if (operator()(i, j) != 0.0)
                    {
                        b.row(i) -= operator()(i, j) * b.row(j);
                    }
                }
            }
            return;
        }

        // This function solves ATx=b, then stores x in b
        // The input b is required to be N*m, i.e.,
        // m vectors to be solved.
        template <typename EIGENMAT>
        inline void solveAdj(EIGENMAT &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; ++j)
            {
                b.row(j) /= operator()(j, j);
                iM = std::min(j + upperBw, N - 1);
                for (int i = j + 1; i <= iM; ++i)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
            for (int j = N - 1; j >= 0; --j)
            {
                iM = std::max(0, j - lowerBw);
                for (int i = iM; i <= j - 1; ++i)
                {
                    if (operator()(j, i) != 0.0)
                    {
                        b.row(i) -= operator()(j, i) * b.row(j);
                    }
                }
            }
        }
    };

    class CubicSpline
    {
    public:
        CubicSpline() = default;
        ~CubicSpline() { A.destroy(); }

    private:
        int N;
        Eigen::Vector2d headP;
        Eigen::Vector2d tailP;
        BandedSystem A;
        Eigen::MatrixX2d b;
        Eigen::Matrix2Xd coeff;
        Eigen::MatrixXd partial_c;
        Eigen::MatrixXd partial_d;

    public:
        inline void setConditions(const Eigen::Vector2d &headPos,
                                  const Eigen::Vector2d &tailPos,
                                  const int &pieceNum)
        {
            N = pieceNum;
            headP = headPos;
            tailP = tailPos;

            A.create(N - 1, 3, 3);
            b.resize(N - 1, 2);

            calculatePartial();
            return;
        }

        inline void setInnerPoints(const Eigen::Ref<const Eigen::Matrix2Xd> &inPs)
        {
            // TODO
            A.reset();
            b.setZero();
            for (int i = 0; i < N - 1; ++i)
            {
                if (i == 0)
                {
                    A(0, 0) = 4;
                    A(0, 1) = 1;
                }
                else if (i == N - 2)
                {
                    A(N - 2, N - 3) = 1;
                    A(N - 2, N - 2) = 4;
                }
                else
                {
                    A(i, i - 1) = 1;
                    A(i, i) = 4;
                    A(i, i + 1) = 1;
                }
            }

            b.row(0) = 3 * (inPs.col(1).transpose() - headP.transpose());
            for (int i = 1; i < N - 2; ++i)
            {
                b.row(i) = 3 * (inPs.col(i + 1).transpose() - inPs.col(i - 1).transpose());
            }
            b.row(N - 2) = 3 * (tailP.transpose() - inPs.col(N - 3).transpose());

            A.factorizeLU();
            A.solve(b);

            std::vector<Eigen::Vector2d> D(N + 1, Eigen::Vector2d::Zero());
            for (int i = 0; i < N - 1; ++i)
            {
                D[i + 1] = b.row(i).transpose();
            }
            coeff.resize(2, 4 * N);
            coeff.setZero();
            for (int i = 0; i < N; ++i)
            {
                if (i == 0)
                {
                    coeff.col(0) = headP;
                    coeff.col(1) = Eigen::Vector2d::Zero();
                    coeff.col(2) = 3 * (inPs.col(0) - headP) - D[1];
                    coeff.col(3) = 2 * (headP - inPs.col(0)) + D[1];
                }
                else if (i == N - 1)
                {
                    coeff.col(4 * (N - 1) + 0) = inPs.col(N - 2);
                    coeff.col(4 * (N - 1) + 1) = D[N - 1];
                    coeff.col(4 * (N - 1) + 2) = 3 * (tailP - inPs.col(N - 2)) - 2 * D[N - 1];
                    coeff.col(4 * (N - 1) + 3) = 2 * (inPs.col(N - 2) - tailP) + 1 * D[N - 1];
                }
                else
                {
                    coeff.col(4 * i + 0) = inPs.col(i - 1);
                    coeff.col(4 * i + 1) = D[i];
                    coeff.col(4 * i + 2) = 3 * (inPs.col(i) - inPs.col(i - 1)) - 2 * D[i] - D[i + 1];
                    coeff.col(4 * i + 3) = 2 * (inPs.col(i - 1) - inPs.col(i)) + 1 * D[i] + D[i + 1];
                }
            }

            b.resize(4 * N, 2);
            b = coeff.transpose();

            return;
        }

        inline void getCurve(CubicCurve &curve) const
        {
            // TODO
            curve.clear();
            curve.reserve(N);
            for (int i = 0; i < N; ++i)
            {
                auto cMat = b.block<4, 2>(4 * i, 0).transpose().rowwise().reverse();
                curve.emplace_back(1.0, cMat);
            }
            return;
        }

        inline void getStretchEnergy(double &energy) const
        {
            energy = 0.0;
            for (int i = 0; i < N; ++i)
            {
                const auto c_i = b.row(4 * i + 2), d_i = b.row(4 * i + 3);
                energy += 4.0 * c_i.squaredNorm() + 12.0 * c_i.dot(d_i) + 12.0 * d_i.squaredNorm();
            }
            return;
        }

        inline const Eigen::MatrixX2d &getCoeffs(void) const
        {
            return b;
        }

        inline void calculatePartial()
        {
            Eigen::MatrixXd A_tmp;
            A_tmp.resize(N - 1, N - 1);
            A_tmp.setZero();
            Eigen::MatrixXd Q;
            Q.resize(N - 1, N - 1);
            Q.setZero();
            for (int i = 0; i < N - 1; ++i)
            {
                if (i == 0)
                {
                    A_tmp(0, 0) = 4;
                    A_tmp(0, 1) = 1;
                    Q(0, 1) = 3;
                }
                else if (i == N - 2)
                {
                    A_tmp(N - 2, N - 3) = 1;
                    A_tmp(N - 2, N - 2) = 4;
                    Q(N - 2, N - 3) = -3;
                }
                else
                {
                    A_tmp(i, i - 1) = 1;
                    A_tmp(i, i) = 4;
                    A_tmp(i, i + 1) = 1;
                    Q(i, i - 1) = -3;
                    Q(i, i + 1) = 3;
                }
            }
            Eigen::MatrixXd partial_D = A_tmp.inverse() * Q;

            Eigen::MatrixXd P;
            P.resize(N, N - 1);
            P.setZero();
            for (int i = 0; i < N; ++i)
            {
                if (i == 0)
                {
                    P(0, 0) = 1;
                }
                else if (i == N - 1)
                {
                    P(N - 1, N - 2) = -1;
                }
                else
                {
                    P(i, i) = 1;
                    P(i, i - 1) = -1;
                }
            }

            partial_c.resize(N, N - 1);
            partial_c.setZero();
            partial_d.resize(N, N - 1);
            partial_d.setZero();

            Eigen::MatrixXd Qc;
            Qc.resize(N, N - 1);
            Qc.setZero();
            Eigen::MatrixXd Qd;
            Qd.resize(N, N - 1);
            Qd.setZero();

            for (int i = 0; i < N; ++i)
            {
                if (i == 0)
                {
                    Qc(0, 0) = 1;
                    Qd(0, 0) = 1;
                }
                else if (i == N - 1)
                {
                    Qc(N - 1, N - 2) = 2;
                    Qd(N - 1, N - 2) = 1;
                }
                else
                {
                    Qc(i, i) = 1;
                    Qc(i, i - 1) = 2;
                    Qd(i, i) = 1;
                    Qd(i, i - 1) = 1;
                }
            }
            partial_c = 3 * P - Qc * partial_D;
            partial_d = -2 * P + Qd * partial_D;
        }
        inline void getGrad(Eigen::Ref<Eigen::Matrix2Xd> gradByPoints) const
        {
            // TODO
            gradByPoints.setZero();

            for (int i = 0; i < N; ++i)
            {
                Eigen::Vector2d c_i = coeff.col(4 * i + 2);
                Eigen::Vector2d d_i = coeff.col(4 * i + 3);
                gradByPoints += (24 * d_i + 12 * c_i) * partial_d.row(i) +
                                (12 * d_i + 8 * c_i) * partial_c.row(i);
            }
        }
    };
}

#endif