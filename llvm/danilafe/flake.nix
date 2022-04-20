{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        buildTools = with pkgs; [ cmake ninja gcc9 ];
        libDeps = with pkgs; [ catch2 spdlog zlib ncurses readline llvmPackages_12.llvm ];
      in rec {
        packages = {
          dl = pkgs.stdenv.mkDerivation {
            pname = "dl";
            version = "0.1.0";
            src = self;

            nativeBuildInputs = buildTools;
            buildInputs = libDeps;

            cmakeBuildType = "ASanAndUBSan";
            ninjaFlags = [ "-v" ];

            doCheck = true;

            UBSAN_OPTIONS="print_stacktrace=1";
            ASAN_OPTIONS="strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:use_odr_indicator=1";
          };
        };
        defaultPackage = packages.dl;

        devShell = pkgs.mkShell {
          cmakeBuildType = "ASanAndUBSan";
          ninjaFlags = [ "-v" ];

          shellHook = ''
            export UBSAN_OPTIONS=print_stacktrace=1
            export ASAN_OPTIONS=strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:use_odr_indicator=1
          '';

          nativeBuildInputs = buildTools;
          buildInputs = libDeps;
        };
      });
}
