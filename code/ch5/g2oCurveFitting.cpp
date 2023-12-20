#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>

using Eigen::Vector3d;
using std::cout;
using std::endl;
using std::istream;
using std::ostream;
using std::vector;

// Vertex: 3D vector
class CurveFittingVertex : public g2o::BaseVertex<3, Vector3d> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // override the reset function
    void setToOriginImpl() override { _estimate << 0, 0, 0; }

    // override the plus operator, just plain vector addition
    void oplusImpl(const double *update) override { _estimate += Vector3d(update); }

    // the dummy read/write function
    bool read(istream &in) override {}
    bool write(ostream &out) const override {}
};

// edge: 1D error term, connected to exactly one vertex
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // define the error term computation
    void computeError() override {
        auto *v = static_cast<const CurveFittingVertex *>(_vertices[0]);  // const CurveFittingVertex
        const Vector3d &abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }

    // the jacobian
    void linearizeOplus() override {
        auto *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Vector3d &abc = v->estimate();
        double y = std::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    bool read(istream &in) override {}
    bool write(ostream &out) const override {}

  public:
    double _x;  // x data, note y is given in the _measurement
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
    vector<double> y_data;  // Data

    for (int i = 0; i < n; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // choose the optimization method from GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // graph optimizer
    optimizer.setAlgorithm(solver);  // set the algorithm
    optimizer.setVerbose(true);      // print the results

    // add vertex
    auto *v = new CurveFittingVertex();
    v->setEstimate(Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    // add edges
    for (int i = 0; i < n; i++) {
        auto *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }

    // carry out the optimization
    cout << "start optimization" << endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // print the results
    Vector3d abc_estimate = v->estimate();
    cout << "estimated model = " << abc_estimate.transpose() << endl;

    return 0;
}