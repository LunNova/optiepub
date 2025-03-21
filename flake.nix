{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
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
					pkgs.bzip2
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
