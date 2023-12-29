#include <pangolin/pangolin.h>

#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using VecVector2d = vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

// Camera intrinsics
double fx = 718.856;
double fy = 718.856;
double cx = 607.1928;
double cy = 185.2157;

// baseline
double baseline = 0.573;

// paths
string left_file = "./../images/left.png";
string disparity_file = "./../images/disparity.png";
boost::format fmt_others("./../images/%06d.png");  // other files

// useful typedefs
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix26d = Eigen::Matrix<double, 2, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// class for accumulator jacobians in parallel
class JacobianAccumulator {
  public:
    JacobianAccumulator(const cv::Mat &img1,
                        const cv::Mat &img2,
                        const VecVector2d &px_ref,
                        const vector<double> depth_ref,
                        Sophus::SE3d &T21)
        : img1_(img1), img2_(img2), px_ref_(px_ref), depth_ref_(depth_ref), T21_(T21) {
        projection_ = VecVector2d(px_ref_.size(), Eigen::Vector2d(0, 0));
    }

    // accumulate_jacobian in a range
    void accumulate_jacobian(const cv::Range &range);

    // get hassian matrix
    Matrix6d hessian() const { return hessian_; }

    // get bias
    Vector6d bias() const { return bias_; }

    // get total cost
    double cost_func() const { return cost_; }

    // get projected points
    VecVector2d projected_points() const { return projection_; }

    // reset h, b, cost to zero
    void reset() {
        hessian_ = Matrix6d::Zero();
        bias_ = Vector6d::Zero();
        cost_ = 0;
    }

  private:
    const cv::Mat &img1_;
    const cv::Mat &img2_;
    const VecVector2d &px_ref_;
    const vector<double> depth_ref_;
    Sophus::SE3d &T21_;
    VecVector2d projection_;  // projected points

    std::mutex hessian_mutex_;
    Matrix6d hessian_ = Matrix6d::Zero();
    Vector6d bias_ = Vector6d::Zero();
    double cost_ = 0;
};

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(const cv::Mat &img1,
                                    const cv::Mat &img2,
                                    const VecVector2d &px_ref,
                                    const vector<double> depth_ref,
                                    Sophus::SE3d &T21);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(const cv::Mat &img1,
                                     const cv::Mat &img2,
                                     const VecVector2d &px_ref,
                                     const vector<double> depth_ref,
                                     Sophus::SE3d &T21);

// bilinear interpolation
float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 1;
    if (y >= img.rows - 1) y = img.rows - 1;

    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);

    return (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step] +
           xx * yy * data[img.step + 1];
}

int main() {
    Mat left_img = cv::imread(left_file, 0);
    Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int n_points = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < n_points; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        double disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity;  // disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01-05.png's pose using this information
    Sophus::SE3d t_curf_ref;

    for (int i = 1; i < 6; i++) {
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, t_curf_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, t_curf_ref);
    }

    return 0;
}

void DirectPoseEstimationSingleLayer(const cv::Mat &img1,
                                     const cv::Mat &img2,
                                     const VecVector2d &px_ref,
                                     const vector<double> depth_ref,
                                     Sophus::SE3d &T21) {
    const int iterations = 10;
    double cost = 0;
    double last_cost = 0;
    auto t1 = std::chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d hessian = jaco_accu.hessian();
        Vector6d bias = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = hessian.ldlt().solve(bias);
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occured when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > last_cost) {
            cout << "cost increased: " << cost << ", " << last_cost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }
        last_cost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {
        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref_[i] * Eigen::Vector3d((px_ref_[i][0] - cx) / fx, (px_ref_[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21_ * point_ref;
        if (point_cur[2] < 0) continue;  // depth invalid

        float u = fx * point_cur[0] / point_cur[2] + cx;
        float v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2_.cols - half_patch_size || v < half_patch_size ||
            v > img2_.rows - half_patch_size)
            continue;

        projection_[i] = Eigen::Vector2d(u, v);
        double x_pc = point_cur[0];
        double y_pc = point_cur[1];
        double z_pc = point_cur[2];
        double z2_pc = z_pc * z_pc;
        double z_pc_inv = 1.0 / z_pc;
        double z2_pc_inv = z_pc_inv * z_pc_inv;

        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {
                double error =
                    GetPixelValue(img1_, px_ref_[i][0] + x, px_ref_[i][1] + y) - GetPixelValue(img2_, u + x, v + y);
                Matrix26d jacobian_pixel_xi;
                Eigen::Vector2d jacobian_img_pixel;

                jacobian_pixel_xi(0, 0) = fx * z_pc_inv;
                jacobian_pixel_xi(0, 1) = 0;
                jacobian_pixel_xi(0, 2) = -fx * x_pc * z2_pc_inv;
                jacobian_pixel_xi(0, 3) = -fx * x_pc * y_pc * z2_pc_inv;
                jacobian_pixel_xi(0, 4) = fx + fx * x_pc * x_pc * z2_pc_inv;
                jacobian_pixel_xi(0, 5) = -fx * y_pc * z_pc_inv;

                jacobian_pixel_xi(1, 0) = 0;
                jacobian_pixel_xi(1, 1) = fy * z_pc_inv;
                jacobian_pixel_xi(1, 2) = -fy * y_pc * z2_pc_inv;
                jacobian_pixel_xi(1, 3) = -fy - fy * y_pc * y_pc * z2_pc_inv;
                jacobian_pixel_xi(1, 4) = fy * x_pc * y_pc * z2_pc_inv;
                jacobian_pixel_xi(1, 5) = fy * x_pc * z_pc_inv;

                jacobian_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2_, u + 1 + x, v + y) - GetPixelValue(img2_, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2_, u + x, v + 1 + y) - GetPixelValue(img2_, u + x, v - 1 + y)));

                // total jacobian
                Vector6d jacobian = -1.0 * (jacobian_img_pixel.transpose() * jacobian_pixel_xi).transpose();

                hessian += jacobian * jacobian.transpose();
                bias += -error * jacobian;
                cost_tmp += error * error;
            }
    }
    if (cnt_good) {
        // set hessian, bias and cost
        std::unique_lock<std::mutex> lck(hessian_mutex_);
        hessian_ += hessian;
        bias_ += bias;
        cost_ += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(const cv::Mat &img1,
                                    const cv::Mat &img2,
                                    const VecVector2d &px_ref,
                                    const vector<double> depth_ref,
                                    Sophus::SE3d &T21) {
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    std::array<double, 4> scales{1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1;
    vector<Mat> pyr2;
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);

        } else {
            Mat img1_pyr;
            Mat img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].rows * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // backup the old values
    double fx_g = fx;
    double fy_g = fy;
    double cx_g = cx;
    double cy_g = cy;

    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr;  // set the keypoints in this pyramid ;eve;
        for (auto &px : px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx,x fy, cx, cy in different pyramid levels
        fx = fx_g * scales[level];
        fy = fy_g * scales[level];
        cx = cx_g * scales[level];
        cy = cy_g * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
}