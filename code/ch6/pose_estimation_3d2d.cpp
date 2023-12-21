#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>
#include <utility>

using cv::DMatch;
using cv::KeyPoint;
using cv::Mat;

using std::cout;
using std::endl;
using std::vector;

using VecVector2d = vector<Eigen::Vector2d, Eigen::aligned_allocator_indirection<Eigen::Vector2d>>;
using VecVector3d = vector<Eigen::Vector3d, Eigen::aligned_allocator_indirection<Eigen::Vector3d>>;

void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const Mat &K);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,
                                 const VecVector2d &points_2d,
                                 const Mat &K,
                                 Sophus::SE3d &pose);

// BA by G2O
void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }

    // read image
    Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Total found" << matches.size() << " group matching points" << endl;

    // Create 3d points
    Mat d1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;

    for (DMatch m : matches) {
        auto d = d1.ptr<uint16_t>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) continue;
        float dd = d / 5000.0;  // scale
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, k_intrinsics);
        pts_2d.emplace_back(keypoints_2[m.trainIdx].pt);
        pts_3d.emplace_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    Mat rot;  // r
    Mat t;
    cv::solvePnP(pts_3d, pts_2d, k_intrinsics, Mat(), rot, t, false);  // OpenCV's PNP solution
    Mat rod;                                                           // R
    cv::Rodrigues(rot, rod);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

    cout << "R = " << endl << rod << endl;
    cout << "t = " << endl << t << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;

    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;
    t1 = std::chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, k_intrinsics, pose_gn);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    t1 = std::chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, k_intrinsics, pose_g2o);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;

    return 0;
}

void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches) {
    // Initialization
    Mat descriptors_1;
    Mat descriptors_2;

    // OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //-- Step 1: Detecting Oriented FAST corner positions
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- Step 2: Calculate the BRIEF descriptor based on the location of the corner points.
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- Step 3:Match the BRIEF descriptors in the two images, using the Hamming distance
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //-- Step 4: Match Point Pair Filtering
    double min_dist = 10000;
    double max_dist = 0;

    // Find the minimum and maximum distance between all matches, i.e., the distance between the most and least similar
    // sets of points.
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::printf("-- Max dist: %f \n", max_dist);
    std::printf("-- Min dist: %f \n", min_dist);

    // If the distance between descriptors is more than twice the minimum distance, the match is considered wrong. But
    // sometimes the minimum distance can be very small, set an empirical value of 30 as a lower limit.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const Mat &K) {
    return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

// BA by gauss-newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,
                                 const VecVector2d &points_2d,
                                 const Mat &K,
                                 Sophus::SE3d &pose) {
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    const int iterations = 10;
    double cost = 0;
    double last_cost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        cost = 0;
        // compute cost
        for (int i = 0; i < points_3d.size(); i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);  // u, v

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> jacobian;
            jacobian << -fx * inv_z, 0, fx * pc[0] * inv_z2, fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z, 0, -fy * inv_z, fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2, -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;

            hessian += jacobian.transpose() * jacobian;
            b += -jacobian.transpose() * e;
        }

        Vector6d dx;
        dx = hessian.ldlt().solve(b);

        if (std::isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= last_cost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << last_cost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        last_cost = cost;

        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

    cout << "pose by g-n: \n" << pose.matrix() << endl;
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

    // left multiplication on SE3
    void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    bool read(std::istream &in) override{};

    bool write(std::ostream &out) const override{};
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(Eigen::Vector3d &pos, Eigen::Matrix3d &K) : pos3d_(pos), K_(K) {}

    void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d trans = v->estimate();
        Eigen::Vector3d pos_pixel = K_ * (trans * pos3d_);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d trans = v->estimate();
        Eigen::Vector3d pos_cam = trans * pos3d_;
        double fx = K_(0, 0);
        double fy = K_(1, 1);
        double cx = K_(0, 2);
        double cy = K_(1, 2);
        double x_c = pos_cam[0];
        double y_c = pos_cam[1];
        double z_c = pos_cam[2];
        double z2_c = z_c * z_c;
        _jacobianOplusXi << -fx / z_c, 0, fx * x_c / z2_c, fx * x_c * y_c / z2_c, -fx - fx * x_c * x_c / z2_c,
            fx * y_c / z_c, 0, -fy / z_c, fy * y_c / (z_c * z_c), fy + fy * y_c * y_c / z2_c, -fy * x_c * y_c / z2_c,
            -fy * x_c / z_c;
    }

    bool read(std::istream &in) override{};

    bool write(std::ostream &out) const override{};

  private:
    Eigen::Vector3d pos3d_;
    Eigen::Matrix3d K_;
};

// BA by G2O
void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose) {
    // Build graph optimization, first set up to go
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // Gradiente descent method, you can choose from GN, LM and DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    auto *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // k_eigen
    Eigen::Matrix3d k_eigen;
    k_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2), K.at<double>(1, 0), K.at<double>(1, 1),
        K.at<double>(1, 2), K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        auto *edge = new EdgeProjection(p3d, k_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optimization cost time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}