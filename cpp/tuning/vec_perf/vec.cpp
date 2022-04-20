#include <boost/program_options/errors.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

using namespace std::string_literals;

template <auto Size> struct S {
  int a[Size] = {};
};

struct Options {
  int container_size = 0;
  int struct_size = 0;
};

auto getOptions(int argc, char** argv) -> std::optional<Options> {
  namespace po = boost::program_options;

  Options options{};
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "container-size,c", po::value<int>(&options.struct_size)->required(), "struct size in sizeof(int)")(
      "struct-size,s", po::value<int>(&options.container_size)->required(), "container size in sizeof struct");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (po::error& e) {
    fmt::print(stderr, "{}\n{}\n", e.what(), desc);
    return {};
  }

  if (vm.count("help") != 0U) {
    fmt::print("{}\n", desc);
    return {};
  }

  return options;
}

auto main(int argc, char** argv) -> int {
  const auto options = getOptions(argc, argv);
  if (!options) {
    return -1;
  }
  if (options->struct_size == 1) {
    auto v = std::vector<S<1>>{};
    v.reserve(options->container_size);
    v.insert(v.cbegin(), S<1>{});
    v.erase(v.cbegin());
  }
}
