#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

auto file_reader(const std::filesystem::path &path) {
  return [file = std::ifstream(path), line = std::string(),
          state = 0]() mutable -> std::optional<std::string> {
    switch (state) {
    case 0:
      while (file.good()) {
        std::getline(file, line);
        state = 1;
        return std::move(line);
      case 1:;
      }
    }

    return {};
  };
}

int main() {
  auto file = file_reader("some_path.txt");

  auto line = std::optional<std::string>{};

  while ((line = file())) {
    std::puts(line->c_str());
  }

  std::puts("done reading lines");
}
