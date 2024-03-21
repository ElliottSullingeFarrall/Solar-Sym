{
  description = "A Nix flake for the ScriptCheck application.";

  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs/nixos-23.11";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, ...}@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = inputs.nixpkgs.legacyPackages.${system};
        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs;};
      in
      {
        devShells = {
          default = pkgs.mkShell rec {
            packages = with pkgs; [ poetry python3Full zlib ];

            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.lib.makeLibraryPath packages}:";

            POETRY_VIRTUALENVS_IN_PROJECT = true;
            shellHook = ''
              poetry env use $(which python3)
              poetry install --no-root
            '';
          };
        };
      });
}