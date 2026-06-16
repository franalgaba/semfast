#!/usr/bin/env bash
set -euo pipefail

onnx_version="${SEMFAST_ONNX_VERSION:-1.24.2}"
runtime_root="${SEMFAST_RUNTIME_DIR:-/private/tmp/semfast-runtime}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/.." && pwd)"
fastembed_cache_dir="${FASTEMBED_CACHE_DIR:-${repository_root}/.fastembed_cache}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    onnx_archive="onnxruntime-osx-arm64-${onnx_version}.tgz"
    onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/${onnx_archive}"
    onnx_dir="${runtime_root}/onnxruntime-osx-arm64-${onnx_version}"
    onnx_dylib="${onnx_dir}/lib/libonnxruntime.${onnx_version}.dylib"
    ;;
  Darwin-x86_64)
    onnx_archive="onnxruntime-osx-x86_64-${onnx_version}.tgz"
    onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/${onnx_archive}"
    onnx_dir="${runtime_root}/onnxruntime-osx-x86_64-${onnx_version}"
    onnx_dylib="${onnx_dir}/lib/libonnxruntime.${onnx_version}.dylib"
    ;;
  Linux-aarch64)
    onnx_archive="onnxruntime-linux-aarch64-${onnx_version}.tgz"
    onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/${onnx_archive}"
    onnx_dir="${runtime_root}/onnxruntime-linux-aarch64-${onnx_version}"
    onnx_dylib="${onnx_dir}/lib/libonnxruntime.so.${onnx_version}"
    ;;
  Linux-x86_64)
    onnx_archive="onnxruntime-linux-x64-${onnx_version}.tgz"
    onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/${onnx_archive}"
    onnx_dir="${runtime_root}/onnxruntime-linux-x64-${onnx_version}"
    onnx_dylib="${onnx_dir}/lib/libonnxruntime.so.${onnx_version}"
    ;;
  *)
    echo "unsupported platform: $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

mkdir -p "${runtime_root}" "${fastembed_cache_dir}"

if [[ ! -f "${onnx_dylib}" ]]; then
  archive_path="${runtime_root}/${onnx_archive}"
  echo "downloading ONNX Runtime ${onnx_version} to ${archive_path}" >&2
  curl -L -o "${archive_path}" "${onnx_url}"
  tar -xzf "${archive_path}" -C "${runtime_root}"
fi

repo_dir="${fastembed_cache_dir}/models--Qdrant--all-MiniLM-L6-v2-onnx"
snapshot_root="${repo_dir}/snapshots"
snapshot_commit=""

if [[ -f "${repo_dir}/refs/main" ]]; then
  snapshot_commit="$(tr -d '[:space:]' < "${repo_dir}/refs/main")"
fi

if [[ -z "${snapshot_commit}" ]]; then
  snapshot_commit="5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
  mkdir -p "${repo_dir}/refs"
  printf '%s\n' "${snapshot_commit}" > "${repo_dir}/refs/main"
fi

snapshot_dir="${snapshot_root}/${snapshot_commit}"
mkdir -p "${snapshot_dir}"

for lock_file in $(find "${repo_dir}" -name '*.lock' -type f 2>/dev/null); do
  echo "removing stale FastEmbed lock ${lock_file}" >&2
  rm -f "${lock_file}"
done

download_if_missing() {
  local file_name="$1"
  local output_path="${snapshot_dir}/${file_name}"
  local url="https://huggingface.co/Qdrant/all-MiniLM-L6-v2-onnx/resolve/main/${file_name}"

  if [[ -e "${output_path}" ]]; then
    return
  fi

  mkdir -p "$(dirname "${output_path}")"
  echo "downloading ${file_name}" >&2
  curl -L -o "${output_path}" "${url}"
}

download_if_missing "model.onnx"
download_if_missing "tokenizer.json"
download_if_missing "config.json"
download_if_missing "special_tokens_map.json"
download_if_missing "tokenizer_config.json"

env_file="${runtime_root}/semfast-production.env"
cat > "${env_file}" <<EOF
export ORT_DYLIB_PATH="${onnx_dylib}"
export FASTEMBED_CACHE_DIR="${fastembed_cache_dir}"
export SEMFAST_ONNX_INTRA_THREADS="${SEMFAST_ONNX_INTRA_THREADS:-4}"
EOF

echo "runtime ready" >&2
echo "source ${env_file}" >&2
