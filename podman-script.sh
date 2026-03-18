podman run -d \
  --name torch29-ssh-mount-1 \
  --hostname torch29 \
  -v $(pwd):/app \
  -v torch-venv:/app/.venv \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /app \
  -p 2222:22 \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --cap-add MKNOD \
  --security-opt apparmor=unconfined \
  --security-opt seccomp=unconfined \
  --privileged \
  torch29-ssh-image:v1