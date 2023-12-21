#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
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

void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const Mat &K);

void pose_estimation_3d3d(const vector<cv::Point3f> &pts1, const vector<cv::Point3f> &pts2, Mat &R, Mat &t);

void bundleAdjustment(const vector<cv::Point3f> &pts1, const vector<cv::Point3f> &pts2, Mat &R, Mat &t);

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

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    explicit EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d point) : point_(std::move(point)) {}

    void computeError() override {
        const auto *pose = static_cast<const VertexPose *>(_vertices[0]);
        _error = _measurement - pose->estimate() * point_;
    }

    void linearizeOplus() override {
        auto *pose = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d trans = pose->estimate();
        Eigen::Vector3d xyz_trans = trans * point_;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }

    bool read(std::istream &in) override{};

    bool write(std::ostream &out) const override{};

  protected:
    Eigen::Vector3d point_;
};

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }

    // read image
    Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Total found" << matches.size() << " group matching points" << endl;

    // Create 3d points
    Mat depth1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    Mat depth2 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point3f> pts1;
    vector<cv::Point3f> pts2;

    for (DMatch m : matches) {
        auto d1 = depth1.ptr<uint16_t>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        auto d2 = depth2.ptr<uint16_t>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d1 == 0 || d2 == 0) continue;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, k_intrinsics);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.queryIdx].pt, k_intrinsics);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts1.emplace_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.emplace_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3e pairs: " << pts1.size() << endl;
    Mat rot;
    Mat t;
    pose_estimation_3d3d(pts1, pts2, rot, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << rot << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << rot.t() << endl;
    cout << "t_inv = " << -rot.t() << endl;

    cout << "calling vundle adjustment" << endl;

    bundleAdjustment(pts1, pts2, rot, t);

    // verify p1 = R*p2 + t
    for (int i = 0; i < 5; i++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "(R*p2 + t) = " << rot * (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << endl;
        cout << endl;
    }

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

void pose_estimation_3d3d(const vector<cv::Point3f> &pts1, const vector<cv::Point3f> &pts2, Mat &R, Mat &t) {
    cv::Point3f p1;
    cv::Point3f p2;
    int n_pts1 = pts1.size();
    for (int i = 0; i < n_pts1; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 = cv::Point3f(cv::Vec3f(p1) / n_pts1);
    p2 = cv::Point3f(cv::Vec3f(p2) / n_pts1);
    vector<cv::Point3f> q1(n_pts1);
    vector<cv::Point3f> q2(n_pts1);
    for (int i = 0; i < n_pts1; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
    for (int i = 0; i < n_pts1; i++) {
        w += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    cout << "W= " << w << endl;

    // SVD on w
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d &u_svd = svd.matrixU();
    Eigen::Matrix3d v_svd = svd.matrixV();

    cout << "U = " << u_svd << endl;
    cout << "V = " << v_svd << endl;

    Eigen::Matrix3d r_svd = u_svd * (v_svd.transpose());
    if (r_svd.determinant() < 0) {
        r_svd = -r_svd;
    }
    Eigen::Vector3d t_svd = Eigen::Vector3d(p1.x, p1.y, p1.z) - r_svd * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (cv::Mat_<double>(3, 3) << r_svd(0, 0), r_svd(0, 1), r_svd(0, 2), r_svd(1, 0), r_svd(1, 1), r_svd(1, 2),
         r_svd(2, 0), r_svd(2, 1), r_svd(2, 2));
    t = (cv::Mat_<double>(3, 1) << t_svd(0, 0), t_svd(1, 0), t_svd(2, 0));
}

void bundleAdjustment(const vector<cv::Point3f> &pts1, const vector<cv::Point3f> &pts2, Mat &R, Mat &t) {
    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // graphical model
    optimizer.setAlgorithm(solver);  // set up solver
    optimizer.setVerbose(true);      // turn on debug output

    // vertex
    auto *pose = new VertexPose();
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    // edges
    for (size_t i = 0; i < pts1.size(); i++) {
        auto *edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    cout << endl << "after optimization: " << endl;
    cout << "T=\n " << pose->estimate().matrix() << endl;

    // convert to cv::Mat
    Eigen::Matrix3d rot = pose->estimate().rotationMatrix();
    Eigen::Vector3d tras = pose->estimate().translation();

    R = (cv::Mat_<double>(3, 3) << rot(0, 0), rot(0, 1), rot(0, 2), rot(1, 0), rot(1, 1), rot(1, 2), rot(2, 0),
         rot(2, 1), rot(2, 2));

    t = (cv::Mat_<double>(3, 1) << tras(0, 0), tras(1, 0), tras(2, 0));
}