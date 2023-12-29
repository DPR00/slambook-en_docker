#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

using cv::KeyPoint;
using cv::Mat;

string file_1 = "./../images/LK1.png";  // first image
string file_2 = "./../images/LK2.png";  // second image

// Optical flow tracker and interface
class OpticalFlowTracker {
  public:
    OpticalFlowTracker(const Mat &img1,
                       const Mat &img2,
                       const vector<KeyPoint> &kp1,
                       vector<KeyPoint> &kp2,
                       vector<bool> &success,
                       bool inverse = true,
                       bool has_initial = false)
        : img1_(img1),
          img2_(img2),
          kp1_(kp1),
          kp2_(kp2),
          success_(success),
          inverse_(inverse),
          has_initial_(has_initial) {}

    void calculateOpticalFlow(const cv::Range &range);

  private:
    const Mat &img1_;
    const Mat &img2_;
    const vector<KeyPoint> &kp1_;
    vector<KeyPoint> &kp2_;
    vector<bool> &success_;
    bool inverse_ = true;
    bool has_initial_ = false;
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp2
 * @param [out] success true if a keypoint is tracked succeessfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(const Mat &img1,
                            const Mat &img2,
                            const vector<KeyPoint> &kp1,
                            vector<KeyPoint> &kp2,
                            vector<bool> &success,
                            bool inverse = false,
                            bool has_initial = false);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp2
 * @param [out] success true if a keypoint is tracked succeessfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(const Mat &img1,
                           const Mat &img2,
                           const vector<KeyPoint> &kp1,
                           vector<KeyPoint> &kp2,
                           vector<bool> &success,
                           bool inverse = false);

/**
 * Get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, x_a1) +
           (1 - xx) * yy * img.at<uchar>(y_a1, x) + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main() {
    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = cv::imread(file_1, 0);
    Mat img2 = cv::imread(file_2, 0);

    // key points, using GFTT here
    vector<KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);  // max 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optical flow by OF single level: " << time_used.count() << endl;

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    t1 = std::chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // use opencv's flow for validation
    vector<cv::Point2f> pt1;
    vector<cv::Point2f> pt2;
    pt1.reserve(kp1.size());
    for (auto &kp : kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);  // LK flow
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // plot the difference of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_cv;
    cv::cvtColor(img2, img2_cv, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_cv, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_cv, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_cv);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {
    // parameters
    int half_patch_size = 4;  // w : window size
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1_[i];
        double dx = 0;
        double dy = 0;
        if (has_initial_) {
            dx = kp2_[i].pt.x - kp.pt.x;
            dy = kp2_[i].pt.y - kp.pt.y;
        }

        double cost = 0;
        double last_cost = 0;
        bool succ = true;  // indicate if this point succeeded

        // Gauss-Neeton iterations
        Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d jacobian;
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse_ == false) {
                hessian = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++) {
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2_, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    if (inverse_ == false) {
                        jacobian =
                            -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img2_, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -  // dI/dx
                                              GetPixelValue(img2_, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                       0.5 * (GetPixelValue(img2_, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -  // dI/dy
                                              GetPixelValue(img2_, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this jacobian does not change when dx, dy is updated, so we can store it and
                        // only compute error
                        jacobian = -1.0 * Eigen::Vector2d(0.5 * (GetPixelValue(img1_, kp.pt.x + x + 1, kp.pt.y + y) -
                                                                 GetPixelValue(img1_, kp.pt.x + x - 1, kp.pt.y + y)),
                                                          0.5 * (GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y + 1) -
                                                                 GetPixelValue(img1_, kp.pt.x + x, kp.pt.y + y - 1)));
                    };
                    // compute Hessian, b and set cost
                    b += -error * jacobian;  //
                    cost += error * error;
                    if (inverse_ == false || iter == 0) {
                        // also update Hessian
                        hessian += jacobian * jacobian.transpose();
                    }
                }
            }
            // compute update
            Eigen::Vector2d update = hessian.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and Hessian is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > last_cost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            last_cost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }
        success_[i] = succ;
        // set kp2
        kp2_[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

void OpticalFlowSingleLevel(const Mat &img1,
                            const Mat &img2,
                            const vector<KeyPoint> &kp1,
                            vector<KeyPoint> &kp2,
                            vector<bool> &success,
                            bool inverse,
                            bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    cv::parallel_for_(cv::Range(0, kp1.size()),
                      std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowMultiLevel(const Mat &img1,
                           const Mat &img2,
                           const vector<KeyPoint> &kp1,
                           vector<KeyPoint> &kp2,
                           vector<bool> &success,
                           bool inverse) {
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    std::array<double, 4> scales{1.0, 0.5, 0.25, 0.125};

    // create pyramids
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
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
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr;
    vector<KeyPoint> kp2_pyr;
    for (auto &kp : kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = std::chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        cout << "track pyr" << level << " cost time" << time_used.count() << endl;

        if (level > 0) {
            for (auto &kp : kp1_pyr) kp.pt /= pyramid_scale;
            for (auto &kp : kp2_pyr) kp.pt /= pyramid_scale;
        }
    }
    for (auto &kp : kp2_pyr) kp2.push_back(kp);
}
