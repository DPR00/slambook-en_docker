#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

using Eigen::Matrix3d;
using Eigen::Vector3d;

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
    vector<double> y_data;  // Data

    for (int i = 0; i < n; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // start Gauss-Newton iterations
    int iterations = 100;
    double cost = 0;
    double last_cost = 0;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        Matrix3d H = Matrix3d::Zero();  // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero();  // bias
        cost = 0;

        for (int i = 0; i < n; i++) {
            double xi = x_data[i];
            double yi = y_data[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J;                                          // Jacobian
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);       // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);            // de/dc

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }

        // solve Hx = b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= last_cost) {
            cout << "cost: " << cost << ">= last cost: " << last_cost << ", break." << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        last_cost = cost;

        cout << "total cost: " << cost << ", \t\t update: " << dx.transpose() << "\t\t estimated params: " << ae << ", "
             << be << ", " << ce << endl;
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = = " << ae << ", " << be << ", " << ce << endl;

    return 0;
}