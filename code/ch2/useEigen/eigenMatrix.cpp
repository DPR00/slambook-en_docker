#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

// eigen core
#include <Eigen/Core>  // NOLINT(bugprone-macro-parentheses)
// Algebraic operations of dense matrices (inverse, eigenvalues, etc)
#include <Eigen/Dense>  // NOLINT(bugprone-macro-parentheses)

#define MATRIX_SIZE 50

int main() {
    Eigen::Matrix<float, 2, 3> matrix_23;

    Eigen::Vector3d v_3d;
    Eigen::Matrix<float, 3, 1> vd_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;

    Eigen::MatrixXd matrix_x;

    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

    cout << "Print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix_23(i, j) << "\t ";
            cout << endl;
        }
    }

    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1]= " << result.transpose() << endl;

    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << "[1,2,3;4,5,6]*[4,5,6]= " << result2.transpose() << endl;

    matrix_33 = Eigen::Matrix3d::Random();
    cout << "random matrix: \n" << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl;
    cout << "sum: " << matrix_33.sum() << endl;
    cout << "trace: " << matrix_33.trace() << endl;
    cout << "times 10: \n" << 10 * matrix_33 << endl;
    cout << "inverse: \n" << matrix_33.inverse() << endl;
    cout << "det: " << matrix_33.determinant() << endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_nn = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_nn = matrix_nn * matrix_nn.transpose();
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_nn.inverse() * v_nd;
    cout << "time of normal inverse is " << 1000 * (clock() - time_stt) / static_cast<double>(CLOCKS_PER_SEC) << "ms"
         << endl;
    cout << "x = " << x.transpose() << endl;

    time_stt = clock();
    x = matrix_nn.colPivHouseholderQr().solve(v_nd);
    cout << "time of Qr decomposition is " << 1000 * (clock() - time_stt) / static_cast<double>(CLOCKS_PER_SEC) << "ms"
         << endl;

    time_stt = clock();
    x = matrix_nn.ldlt().solve(v_nd);
    cout << "time of ldlt decomposition is " << 1000 * (clock() - time_stt) / static_cast<double>(CLOCKS_PER_SEC)
         << "ms" << endl;

    return 0;
}
