#include <pangolin/pangolin.h>
#include <unistd.h>

#include <Eigen/Core>

using std::cout;
using std::endl;
using std::vector;

using Eigen::Isometry3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

std::string trajectory_file = "./../trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main() {
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
    std::ifstream fin(trajectory_file);
    if (!fin) {
        cout << "cannot find trajectory file at " << trajectory_file << endl;
    }
    while (!fin.eof()) {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
        Twr.pretranslate(Vector3d(tx, ty, tz));
        poses.push_back(Twr);
    }

    cout << "read total " << poses.size() << " poses entries" << endl;

    // Draw trajectory pangolin
    DrawTrajectory(poses);

    return 0;
}

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);  // image dimensions: 1024 x768
    glEnable(GL_DEPTH_TEST);                                        // enables the depth testing functionality in OpenGL
    glEnable(GL_BLEND);                                             // enables blending
    glBlendFunc(GL_SRC_ALPHA,
                GL_ONE_MINUS_SRC_ALPHA);  // sets the blending function for the alpha (transparency) channel

    // configure the OpenGL projection matrix and model-view matrix for rendering the 3D scene
    // Projection Matrix:
    // Image dimensions: 1024 x 768
    // Focal Length: (500, 500)
    // Principal point: (512, 389)
    // near/far clipping planes (0.1, 1000)
    // Model View Look At:
    // Position of the camera: (0.0, -1.0, 1.8)
    // The point it is looking at: (0, 0, 0)
    // the up vector: (0.0, -1.0, 0.0)
    pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                                      pangolin::ModelViewLookAt(-1.5, -0.1, -2.5, -1, 0, 0, 0.0, -1.0, 0.0));

    // Creates a Pangolin display and sets its properties
    // .SetBunds(): sets the bounds of the display in normalized coordinates (from 0.0 to 1.0 in both x and y
    // directions). The last argument is the aspect ratio, which is set to match the specified image dimensions
    // (1024x768).
    // .SetHandler(): sets the input handler for the display. In this case, it uses pangolin::Handler3D to handle 3D
    // navigation interactions, and it is configured with the previously defined s_cam (render state) to synchronize the
    // navigation with the camera parameters
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {                // Until the user quit
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // prepares the framebuffer for rendering a new frame
        d_cam.Activate(s_cam);  // activates the Pangolin camera (d_cam) with the specified rendering state (s_cam).
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // clear the color buffer
        glLineWidth(2);                        // sets the width of the lines to be drawn
        for (auto posei : poses) {
            // draw three axes of each pose
            Vector3d Ow = posei.translation();
            Vector3d Xw = posei * (0.1 * Vector3d(1, 0, 0));
            Vector3d Yw = posei * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = posei * (0.1 * Vector3d(0, 1, 0));
            glBegin(GL_LINES);
            // axis x
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            // axis y
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            // axis z
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }
        for (size_t i = 0; i < poses.size() - 1; i++) {
            // Draws black line connecting all the xyz frame
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i];
            auto p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);
    }
}