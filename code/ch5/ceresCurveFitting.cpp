#include <ceres/ceres.h>

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

// residual
struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // implement operator () to compute error
    template <typename T>
    bool operator()(const T *const abc, T *residual) const {
        // y - exp(ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);

        return true;
    }

    const double _x, _y;  // x, y data
};

int main() {
    // ground-thruth values
    double ar = 1.0;
    double br = 2.0;
    double cr = 1.0;

    // initial estimation values
    double ae = 2.0;
    double be = -1.0;
    double ce = 5.0;

    int n = 100;           // num of data points
    double w_sigma = 1.0;  // sigma of the noise
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;  // random numer

    vector<double> x_data;
    vector<double> y_data;  // the data

    for (int i = 0; i < n; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};

    // construct the problem in ceres
    ceres::Problem problem;
    for (int i = 0; i < n; i++) {
        problem.AddResidualBlock(  // add i-th residual into the problem
                                   // use auto-diff, template params: residual type, output dimension, input dimension
                                   // should be same as the struct written before
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr,  // kernel function, don't use here
            abc       // estimated variables
        );
    }

    // set the solver options
    ceres::Solver::Options options;                             // actually there're lots of params can be adjusted
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // use cholesky to solve the normal equation
    options.minimizer_progress_to_stdout = true;                // print to cout

    ceres::Solver::Summary summary;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // do optimization!
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // get the outputs
    cout << summary.BriefReport() << endl;
    cout << "estimated a, b, c = ";
    for (auto a : abc) cout << a << " ";
    cout << endl;

    return 0;
}