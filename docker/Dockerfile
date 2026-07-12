# syntax=docker/dockerfile:1
ARG ISAACLAB_BASE_IMAGE_ARG

# Layer the Qmini_lab extension on top of the prebuilt Isaac Lab image
# (nvcr.io/nvidia/isaac-lab:3.0.0-beta1 — set via ISAACLAB_BASE_IMAGE in docker/.env.base).
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Qmini_lab project into the image. The compose build context is the project root
# (`context: ../`), so the source is `.` here. (Upstream template ships `COPY ../`, which
# escapes the build context and fails — corrected to `.`.) Exclusions live in .dockerignore.
COPY . ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Editable-install the extension so the live bind-mount picks up edits without a rebuild.
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/qmini_tasks && \
    pip install -e ."

# Stop NvStreamer (WebRTC livestream) from dumping event-trace *.etli files into the CWD — i.e.
# the mounted project — on every stream session. We never use these QoS diagnostics; there is no
# output-path knob, only this on/off toggle (omni.kit.livestream.app primaryStream.enableEventTracing,
# default true). Patch the image default so it's off for every script/launch. Version-agnostic glob;
# best-effort (won't fail the build if a future base renames the extension). ponytail: image-wide off.
RUN find /isaac-sim -path '*omni.kit.livestream*/config/extension.toml' \
        -exec sed -i 's/^enableEventTracing = true/enableEventTracing = false/' {} + \
    && echo "NvStreamer event tracing (.etli) disabled in:" \
    && grep -rl 'enableEventTracing = false' /isaac-sim/extscache/omni.kit.livestream*/config/ \
    || echo "WARN: enableEventTracing line not found — check extension path if .etli still appear"

# Default working directory inside the container.
WORKDIR /workspace
