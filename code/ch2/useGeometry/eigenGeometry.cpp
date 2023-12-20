#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

using std::cout;
using std::endl;

int main() {
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();

    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));

    cout.precision(3);
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;

    cout << "(1, 0, 0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;

    v_rotated = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);  // Z,Y,X
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    Eigen::Isometry3d t_se3 = Eigen::Isometry3d::Identity();  // 4x4 matrix
    t_se3.rotate(rotation_vector);
    t_se3.pretranslate(Eigen::Vector3d(1, 3, 4));
    cout << "Transform matrix = \n" << t_se3.matrix() << endl;

    Eigen::Vector3d v_transformed = t_se3 * v;
    cout << "v transformed = " << v_transformed.transpose() << endl;

    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;

    q = Eigen::Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;

    v_rotated = q * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    cout << "should be equal to " << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

    return 0;
}