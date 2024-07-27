all: clippy test

build:
    cargo build --workspace --all

clippy:
    cargo clippy --workspace --all

test:
    cargo test --workspace --all
