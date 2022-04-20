#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    auto image = cv::Mat(480, 640, CV_8U);
    putText(image, "Hello World!", cv::Point(200, 400), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC,
            1.0, cv::Scalar(255, 255, 0));
    imshow("My Window", image);
    cv::waitKey();
}
