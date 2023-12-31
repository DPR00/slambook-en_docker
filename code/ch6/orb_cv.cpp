#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::cout;
using std::endl;

using cv::Mat;

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: feature_extraction img1 img2" << endl;
    }

    // -- read images
    Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // -- initialization
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    Mat descriptors_1;
    Mat descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // -- detect oriented FAST
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    //-- compute BRIEF descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    Mat outimg1;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB Features", outimg1);

    //-- use Hamming distance to match the features
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "Match ORB cost = " << time_used.count() << " seconds." << endl;

    //-- sort and remove the outliers
    // min and max distance
    auto min_max = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) {
        return m1.distance < m2.distance;
    });

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::printf("-- Max dist: %f \n", max_dist);
    std::printf("-- Min dist: %f \n", min_dist);

    // remove the bad matching
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // draw the results
    Mat img_match;
    Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("All matches", img_match);
    cv::imshow("Good matches", img_goodmatch);
    cv::waitKey(0);

    return 0;
}