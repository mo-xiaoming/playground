#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

constexpr int alpha_slider_max = 100;

struct Trackbar_data {
    cv::Mat src1;
    cv::Mat src2;
    std::string title;
};

static void on_trackbar(int pos, void* data) {
    auto const alpha = static_cast<double>(pos) / alpha_slider_max;
    auto const beta = (1.0 - alpha);
    auto* user_data = static_cast<Trackbar_data*>(data);
    cv::Mat dst;
    addWeighted(user_data->src1, alpha, user_data->src2, beta, 0.0, dst);
    imshow(user_data->title, dst);
}

int main() {
    auto const src1 = cv::imread("/Users/mx/Desktop/WechatIMG8.jpeg");
    if (src1.empty()) {
        spdlog::critical("error loading src1");
        return 1;
    }
    auto const src2 = cv::Mat::zeros(src1.size(), src1.type());
    //    auto const src2 = cv::imread("/Users/mx/Desktop/WechatIMG8.jpeg");
    //    if (src2.empty()) {
    //        spdlog::critical("error loading src2");
    //        return 1;
    //    }

    auto user_data = Trackbar_data{.src1 = src1, .src2 = src2, .title = "Linear Blend"};
    namedWindow(user_data.title, cv::WINDOW_AUTOSIZE);
    auto alpha_slider = 0;
    cv::createTrackbar(fmt::format("Alpha x {}", alpha_slider_max), "Linear Blend", &alpha_slider,
                       alpha_slider_max, on_trackbar, &user_data);
    on_trackbar(alpha_slider, &user_data);
    cv::waitKey(0);
}
