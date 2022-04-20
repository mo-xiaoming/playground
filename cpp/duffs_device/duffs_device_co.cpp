#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#define co_begin() switch (state) { case 0:
#define co_end(X) } return X
#define co_yield(X) state = __LINE__; return X; case __LINE__:

auto file_reader(const std::filesystem::path &path) {
  return [file = std::ifstream(path), line = std::string(),
          state = 0]() mutable -> std::optional<std::string> {
    co_begin();

    while (std::getline(file, line)) {
      co_yield(std::move(line));
      co_yield("Another line");
    }

    co_end({});
  };
}

int main() {
  auto file = file_reader("some_path.txt");

  std::optional<std::string> line;

  while ((line = file())) {
    puts(line->c_str());
  }

  puts("done reading lines");
}
