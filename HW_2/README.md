## HW2  LBFGS 求解三次样条曲线

### How to run

```
cd HW_2/path_smoothing
rm -rf devel/ build/
catkin_make
source devel/setup.bash 
roslaunch gcopter curve_gen.launch
```

### Workflow

1. cubic spline

   给定N-1个中间点，将起点，中间点，终点这N+1拟合成N段三次样条曲线，并解出系数矩阵

   ```cpp
   // path_smoothing/src/gcopter/include/gcopter/cubic_spline.hpp
   inline void setInnerPoints(const Eigen::Ref<const Eigen::Matrix2Xd> &inPs)
   {
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
   ```

   计算曲线的能量以及梯度

   ```cpp
   // path_smoothing/src/gcopter/include/gcopter/cubic_spline.hpp
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
   
   ```

   ```cpp
   // path_smoothing/src/gcopter/include/gcopter/cubic_spline.hpp
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
       gradByPoints.setZero();
       for (int i = 0; i < N; ++i)
       {
           Eigen::Vector2d c_i = coeff.col(4 * i + 2);
           Eigen::Vector2d d_i = coeff.col(4 * i + 3);
           gradByPoints += (24 * d_i + 12 * c_i) * partial_d.row(i) + (12 * d_i + 8 * c_i) * partial_c.row(i);
       }
   }
   ```

2. path_smoother

   调用lbfgs_optimize进行优化，evaluate function 选择 PathSmoother::costFunction

   ```cpp
   // path_smoothing/src/gcopter/include/gcopter/path_smoother.hpp
   inline double optimize(CubicCurve &curve,
                          const Eigen::Matrix2Xd &iniinner_pts,
                          const double &relCostTol)
   {
       Eigen::VectorXd x(pieceN * 2 - 2);
       Eigen::Map<Eigen::Matrix2Xd> innerP(x.data(), 2, pieceN - 1);
       innerP = iniinner_pts;
   
       double minCost = 0.0;
   
       int status = lbfgs::lbfgs_optimize(x, minCost, &PathSmoother::costFunction,
                                          nullptr, this, lbfgs_params);
   
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
   ```

   costFunction由两部分组成包括三次样条曲线本身的energy，还有障碍物势场的惩罚

   ```cpp
   // path_smoothing/src/gcopter/include/gcopter/path_smoother.hpp
   static inline double costFunction(void *ptr,
                                     const Eigen::VectorXd &x,
                                     Eigen::VectorXd &g)
   {
   
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
   ```

3. lbfgs中的线搜索采用lewisoverton形式

   ```cpp
   // path_smoothing/src/gcopter/include/gcopter/lbfgs.hpp
   inline int line_search_lewisoverton(Eigen::VectorXd &x,
                                       double &f,
                                       Eigen::VectorXd &g,
                                       double &stp,
                                       const Eigen::VectorXd &s,
                                       const Eigen::VectorXd &xp,
                                       const Eigen::VectorXd &gp,
                                       const double stpmin,
                                       const double stpmax,
                                       const callback_data_t &cd,
                                       const lbfgs_parameter_t &param)
   {
       // x is the decision variable vector
       // f is function value at x
       // g is the gradient value at x
       // stp is the initial stepsize for line search
       // s is the search direction vector
       // xp is the decision variable vector at the current iteration
       // gp is the gradient vector at the current iteration
       // stpmin is the minimum allowable stepsize
       // stpmax is the maximum allowable stepsize
       // the struct param contains all necessary parameters
       // the cd contains all necessary callback function
   
       // eg.             x = xp; f = cd.proc_evaluate(cd.instance, x, g);
       // the above line assigns x with xp and computes the function and grad at x
   
       // note the output x, f and g which satisfy the weak wolfe condition when the function returns
       int iter_num = 0;
       double fxp = f, fx = f;
       double l = stpmin, u = stpmax;
       while (true)
       {
           x = xp + stp * s;
           fx = cd.proc_evaluate(cd.instance, x, g);
           iter_num++;
           // If Armijo condition fails
           if (fxp - fx < -stp * param.f_dec_coeff * gp.dot(s))
           {
               u = stp;
           }
           // If Weak Wolfe condition fails
           else if (g.dot(s) < param.s_curv_coeff * gp.dot(s))
           {
               l = stp;
           }
           else
           {
               f = fx;
               return iter_num;
           }
   
           if (u < stpmax)
           {
               stp = 0.5 * (l + u);
           }
           else
           {
               stp = 2 * l;
           }
           if (iter_num >= param.max_linesearch)
           {
               break;
           }
       }
       return LBFGSERR_MAXIMUMLINESEARCH;
   }
   ```

## Result

由于环境问题，障碍物可视化有些问题，等后期补上![](https://raw.githubusercontent.com/Chenjq-99/Blog-pic/main/image-20240207142014650.png)