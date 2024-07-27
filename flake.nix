{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    naersk.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = args:
    args.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = (import args.nixpkgs) {
          inherit system;
        };

        runtimeDeps = [
          pkgs.xorg.libxcb
          pkgs.xorg.libX11
          pkgs.xorg.libXcursor
          pkgs.xorg.libXrandr
          pkgs.xorg.libXi
          pkgs.fontconfig
          pkgs.gtk3
          pkgs.python3
          pkgs.libGL
          pkgs.libGLU
          pkgs.wayland
          pkgs.libxkbcommon
          pkgs.pkg-config
          pkgs.openssl
          pkgs.openssl.dev
          pkgs.kdialog
          pkgs.yad
        ];

        LD_LIBRARY_PATH = "/run/opengl-driver/lib/:${pkgs.lib.makeLibraryPath runtimeDeps}";

        devShellPkgs = [
          pkgs.cargo-deny
          pkgs.cargo-bloat
          pkgs.cargo-flamegraph
          pkgs.cargo-udeps
          pkgs.rustfmt
          pkgs.pkg-config
          pkgs.just
          pkgs.cmake
        ] ++ runtimeDeps;

        fenix = args.fenix.packages.${system};

        toolchain = with fenix;
          combine [
            complete.rustc
            complete.cargo
          ];

        naersk = args.naersk.lib.${system}.override {
          cargo = toolchain;
          rustc = toolchain;
        };

        glibc = if pkgs.stdenv.isx86_64 then pkgs.glibc_multi else pkgs.glibc;
        self = {
          devShells.default = self.devShells.rustup-dev;

          devShells.rustup-dev = pkgs.stdenv.mkDerivation {
            inherit LD_LIBRARY_PATH;
            name = "rustup-dev-shell";

            shellHook = ''
              export CC=
              export NIX_CFLAGS_COMPILE=
              export NIX_CFLAGS_COMPILE_FOR_TARGET=
            '';

            depsBuildBuild = with pkgs; [
              pkg-config
            ];

            nativeBuildInputs = with pkgs; [
              lld
            ];

            GLIBC_PATH = "${glibc}/lib";

            buildInputs = [
              glibc
              pkgs.rustup
              pkgs.libunwind
            ] ++ devShellPkgs;
          };
        };
      in
      self
    );
}
