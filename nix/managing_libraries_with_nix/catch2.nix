{ stdenv, fetchFromGitHub, cmake }:

stdenv.mkDerivation rec {
  pname = "catch2";
  version = "2.13.7";

  src = fetchFromGitHub {
    owner = "catchorg";
    repo = "Catch2";
    rev = "v${version}";
    sha256 = "0cizwi5lj666xn9y0qckxf35m1kchjf2j2h1hb5ax4fx3qg7q5in";
  };

  nativeBuildInputs = [ cmake ];
}
