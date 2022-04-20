#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>
#include <utility>

namespace {
struct Cmd_options {
    std::string file_to_show;
};

struct Cmd_options_exception : std::exception {
    Cmd_options_exception(std::string msg, std::string usage)
        : msg_(std::move(msg)), usage_(std::move(usage)) {}
    [[nodiscard]] char const* what() const noexcept override { return msg_.c_str(); }
    [[nodiscard]] std::string const& usage() const noexcept { return usage_; }

private:
    std::string const msg_;
    std::string const usage_;
};

Cmd_options parse_options(int argc, char** argv) {
    using namespace std::literals;

    auto const file_to_show_flag = "f"s;

    auto options = cxxopts::Options("DisplayImage", "Display an image file");
    options.add_options()(file_to_show_flag, "a image path", cxxopts::value<std::string>());
    try {
        auto const result = options.parse(argc, argv);
        if (result.count(file_to_show_flag) != 1) {
            throw cxxopts::OptionParseException(
                fmt::format("Option '{}' is missing", file_to_show_flag));
        }
        return Cmd_options{.file_to_show = result[file_to_show_flag].as<std::string>()};
    } catch (cxxopts::OptionException const& e) {
        throw Cmd_options_exception(e.what(), options.help());
    }
}

std::optional<cv::Mat> read_image(std::string const& file_to_show) {
    auto img = cv::imread(file_to_show);
    if (img.data == nullptr) {
        return {};
    }
    return {img};
}
} // namespace

int main(int argc, char** argv) {
    auto options = Cmd_options();
    try {
        options = parse_options(argc, argv);
    } catch (Cmd_options_exception const& e) {
        spdlog::error(e.what());
        spdlog::info(e.usage());
        return 1;
    }

    auto const image = read_image(options.file_to_show);
    if (!image || image->empty()) {
        spdlog::error("no valid image");
        return 1;
    }

    auto const win_title = std::string("Display Image");
    namedWindow(win_title, cv::WINDOW_AUTOSIZE);
    imshow(win_title, *image);
    cv::waitKey(0);
}
