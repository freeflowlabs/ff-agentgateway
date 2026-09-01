# syntax=docker/dockerfile:1.11
ARG BUILDER=base

FROM docker.io/library/node:23.11.0-bookworm AS node

WORKDIR /app

COPY ui .

RUN --mount=type=cache,target=/app/npm/cache npm install

RUN --mount=type=cache,target=/app/npm/cache npm run build

FROM docker.io/library/rust:1.93.0-trixie AS musl-builder

ARG TARGETARCH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && apt-get install -y --no-install-recommends \
    make musl-tools

RUN <<EOF
mkdir /build
if [ "$TARGETARCH" = "arm64" ]; then
  rustup target add aarch64-unknown-linux-musl;
  echo aarch64-unknown-linux-musl > /build/target
else
  rustup target add x86_64-unknown-linux-musl;
  echo x86_64-unknown-linux-musl > /build/target
fi
EOF

FROM docker.io/library/rust:1.93.0-trixie AS base-builder

ARG TARGETARCH

RUN <<EOF
mkdir /build
if [ "$TARGETARCH" = "arm64" ]; then
  echo aarch64-unknown-linux-gnu > /build/target
else
  echo x86_64-unknown-linux-gnu > /build/target
fi
echo "Building $(cat /build/target)"
EOF

FROM ${BUILDER}-builder AS builder
ARG TARGETARCH
ARG PROFILE=release
ARG VERSION
ARG GIT_REVISION

WORKDIR /app

COPY Makefile Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY crates ./crates
COPY tools ./tools
COPY --from=node /app/out ./ui/out

RUN \
    --mount=type=cache,id=cargo,target=/usr/local/cargo/registry \
    --mount=type=cache,id=cargo-git,target=/usr/local/cargo/git \
    cargo fetch --locked
RUN --mount=type=cache,target=/app/target \
    --mount=type=cache,id=cargo,target=/usr/local/cargo/registry  \
    --mount=type=cache,id=cargo-git,target=/usr/local/cargo/git \
    <<EOF
export VERSION="${VERSION}"
export GIT_REVISION="${GIT_REVISION}"
cargo build --features ui  --target "$(cat /build/target)" --profile ${PROFILE} || exit 1
mkdir /out
mv /app/target/$(cat /build/target)/${PROFILE}/agentgateway /out
/out/agentgateway --version
# Fail if version is not set
if /out/agentgateway --version | grep -q '"unknown"'; then
  exit 1
fi
EOF

# Test gate: the image cannot be produced unless the Vertex provider tests pass.
# The runner stage copies a marker from here, which forces buildx to run this stage;
# a failing test fails the build before anything is pushed or rolled out.
FROM builder AS tester
RUN --mount=type=cache,target=/app/target \
    --mount=type=cache,id=cargo,target=/usr/local/cargo/registry \
    --mount=type=cache,id=cargo-git,target=/usr/local/cargo/git \
    <<EOF
cargo test -p agentgateway llm::vertex -- --nocapture || exit 1
mkdir -p /out && touch /out/tests-passed
EOF

FROM cgr.dev/chainguard/glibc-dynamic AS runner

ARG TARGETARCH

WORKDIR /

COPY --from=builder /out/agentgateway /app/agentgateway
COPY --from=tester /out/tests-passed /app/tests-passed

LABEL org.opencontainers.image.source=https://github.com/agentgateway/agentgateway
LABEL org.opencontainers.image.description="Agentgateway is an open source project that is built on AI-native protocols to connect, secure, and observe agent-to-agent and agent-to-tool communication across any agent framework and environment."

ENTRYPOINT ["/app/agentgateway"]
