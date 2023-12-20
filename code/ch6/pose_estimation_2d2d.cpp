#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::cout;
using std::endl;
using std::vector;

using cv::DMatch;
using cv::imread;
using cv::KeyPoint;
using cv::Mat;
using cv::Point2d;

void find_feature_matches(const Mat &img_1,
                          const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

void pose_estimation_2d2d(
    vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat &R, Mat &t);

Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "In total, we get " << matches.size() << " set of future points" << endl;

    // -- Estimate the motion between two frames
    Mat rot;
    Mat t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, rot, t);

    // -- Check E = t^R*scale
    Mat t_x = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0), t.at<double>(2, 0), 0,
               -t.at<double>(0, 0), -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R = " << endl << t_x * rot << endl;

    // -- Check epipolar constraints
    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m : matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, k_intrinsics);
        Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, k_intrinsics);
        Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * rot * y1;
        cout << "epipolar constraint = " << d << endl;
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

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(
    vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat &R, Mat &t) {
    // Cámara Insider, TUM Friburgo2
    Mat k_intrinsics = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- Convert matches into vector<Point2f> form
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for (auto &match : matches) {
        points1.push_back(keypoints_1[match.queryIdx].pt);
        points2.push_back(keypoints_2[match.trainIdx].pt);
    }

    //-- Calculation basis matrix
    Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    cout << "fundamental_matrix is" << endl << fundamental_matrix << endl;

    //-- Calculate the essential matrix
    Point2d principal_point(325.1, 249.7);  // Camera optical center, TUM dataset calibration value
    double focal_length = 521;              // Camera focal length, TUM dataset calibration value
    Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential matrix is " << endl << essential_matrix << endl;

    //-- Calculate the single response matrix
    //- But in this case the scene is not planar, and the single response matrix is not very meaningful.
    Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- Recover rotation and translation information from the essential matrix.
    // This function is only available in Opencv3>.
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}