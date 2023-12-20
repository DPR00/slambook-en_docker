#include <iostream>
#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;
using std::vector;

using cv::DMatch;
using cv::KeyPoint;
using cv::Mat;

void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

void pose_estimation_2d2d(const vector<KeyPoint> &keypoints_1,
                          const vector<KeyPoint> &keypoints_2,
                          const vector<DMatch> &matches,
                          Mat &R,
                          Mat &t);

void triangulation(const vector<KeyPoint> &keypoint_1,
                   const vector<KeyPoint> &keypoint_2,
                   const vector<DMatch> &matches,
                   const Mat &R,
                   const Mat &t,
                   vector<cv::Point3d> &points);
// Plot
inline cv::Scalar get_color(float depth) {
    float up_th = 50;
    float low_th = 10;
    float th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// Convert pixel coordinates to camera normalized coordinates
cv::Point2f pixel2cam(const cv::Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: triangulation img1 img2" << endl;
        return 1;
    }

    // Read image
    Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Total found " << matches.size() << " Group matching points" << endl;

    // Estimating motion between two images
    Mat rot;
    Mat t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, rot, t);

    // triangulation
    vector<cv::Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, rot, t, points);

    // Verify the reprojection relationship between triangulated points and feature points
    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();

    for (int i = 0; i < matches.size(); i++) {
        // first picture
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        cv::Point2f pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, k_intrinsics);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // The second picture
        Mat pt2_trans = rot * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }

    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

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

cv::Point2f pixel2cam(const cv::Point2d &p, const Mat &K) {
    return cv::Point2f((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(const vector<KeyPoint> &keypoints_1,
                          const vector<KeyPoint> &keypoints_2,
                          const vector<DMatch> &matches,
                          Mat &R,
                          Mat &t) {
    // Cámara Insider, TUM Friburgo2
    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- Convert matches into vector<Point2f> form
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for (auto &match : matches) {
        points1.push_back(keypoints_1[match.queryIdx].pt);
        points2.push_back(keypoints_2[match.trainIdx].pt);
    }

    //-- Calculate the essential matrix
    cv::Point2d principal_point(325.1, 249.7);  // Camera optical center, TUM dataset calibration value
    int focal_length = 521;                     // Camera focal length, TUM dataset calibration value
    Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    // cout << "essential matrix is " << endl << essential_matrix << endl;

    //-- Recover rotation and translation information from the essential matrix.
    // This function is only available in Opencv3>.
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(const vector<KeyPoint> &keypoint_1,
                   const vector<KeyPoint> &keypoint_2,
                   const vector<DMatch> &matches,
                   const Mat &R,
                   const Mat &t,
                   vector<cv::Point3d> &points) {
    Mat t1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    Mat t2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0), R.at<double>(2, 0),
              R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point2f> pts_1;
    vector<cv::Point2f> pts_2;

    for (DMatch m : matches) {
        // Convert pixel coordinate to camera coordinates
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, k_intrinsics));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, k_intrinsics));
    }

    Mat pts_4d;
    cv::triangulatePoints(t1, t2, pts_1, pts_2, pts_4d);

    // Convert to non-homogeneous coordinates
    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);  // Normalized
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}