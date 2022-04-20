#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <spdlog/spdlog.h>

namespace {
bool divided_by_8(unsigned int n) { return ((n >> 3U) << 3U) == n; }
} // namespace

int main() {
    using namespace std::string_literals;

    auto capture = cv::VideoCapture(0);
    if (!capture.isOpened()) {
        spdlog::critical("cam 0 failed");
        return 1;
    }
    auto const width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    auto const height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    spdlog::info(FMT_STRING("native width {}, height {}"), width, height);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    auto const org_name = "original"s;
    auto const blurred_name = "blurred"s;

    namedWindow(org_name, cv::WINDOW_AUTOSIZE);
    namedWindow(blurred_name, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(org_name, 0, 0);
    cv::moveWindow(blurred_name, 640, 0);

    auto frame = cv::Mat();
    auto blurred = cv::Mat();
    auto count = 0;
    auto const start = cv::getTickCount();
    for (;;) {
        capture.read(frame);
        if (frame.empty()) {
            spdlog::info("end of stream");
            break;
        }
        cv::GaussianBlur(frame, blurred, cv::Size(11, 11), 1.5);
        ++count;
        cv::imshow(org_name, frame);
        cv::imshow(blurred_name, blurred);
        if (divided_by_8(static_cast<unsigned>(count)) && static_cast<char>(cv::waitKey(1)) == 27) {
            break;
        }
    }
    auto const elapsed_time =
        static_cast<double>(cv::getTickCount() - start) / cv::getTickFrequency();
    spdlog::info("{} frames in {:.2} seconds, {:.2} fps", count, elapsed_time,
                 count / elapsed_time);
}
