pkgs: attrs:
  with pkgs;
  let defaultAttrs = {
    builder = "${bash}/bin/bash";
    args = [ ./builder_generic.sh ];
    baseInputs = [ gnutar gzip gnumake gcc binutils-unwrapped coreutils gawk gnused gnugrep patchelf findutils ];
    buildInputs = [];
    system = builtins.currentSystem;
  };
  in
  derivation (defaultAttrs // attrs)
