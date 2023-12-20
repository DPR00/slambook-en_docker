#include <Eigen/Core>
#include <Eigen/Geometry>

using Eigen::Isometry3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

#include <algorithm>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

int main() {
    Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize();
    q2.normalize();

    Vector3d t1(0.3, 0.1, 0.1);
    Vector3d t2(-0.1, 0.5, 0.3);
    Vector3d p1(0.5, 0, 0.2);

    Isometry3d tse3_1w(q1);
    Isometry3d tse3_2w(q2);
    tse3_1w.pretranslate(t1);
    tse3_2w.pretranslate(t2);

    Vector3d p2 = tse3_2w * tse3_1w.inverse() * p1;
    cout << p2.transpose() << endl;

    return 0;
}