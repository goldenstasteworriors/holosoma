#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nvme04/holodiffusion/holosoma}"
AMASS_ROOT="${AMASS_ROOT:-/home/nvme04/holodiffusion/AMASS_SMPLX_N}"
PROCESSED_AMASS_DIR="${PROCESSED_AMASS_DIR:-${REPO_ROOT}/src/holosoma_retargeting/holosoma_retargeting/amass_result}"
RETARGET_SAVE_DIR="${RETARGET_SAVE_DIR:-${REPO_ROOT}/src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/robot_only/amass_smplx}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/amass_pipeline}"
MAX_WORKERS="${MAX_WORKERS:-}"
SUBDATASET_FOLDER="${SUBDATASET_FOLDER:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_ROOT="${MODEL_ROOT:-}"
HUMAN_BODY_PRIOR_DIR="${REPO_ROOT}/src/holosoma_retargeting/holosoma_retargeting/data_utils/human_body_prior"
HUMAN_BODY_PRIOR_BODYMODEL="${HUMAN_BODY_PRIOR_DIR}/src/human_body_prior/body_model/body_model.py"

detect_model_root() {
    local candidates=(
        "/home/nvme04/holodiffusion/smpl_models/models"
        "/home/nvme04/holodiffusion/thirdparties/smpl_models/models"
        "/home/nvme04/thirdparties/smpl_models/models"
        "/home/nvme04/Downloads/thirdparties/smpl_models/models"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/smplx/SMPLX_NEUTRAL.npz" ]]; then
            MODEL_ROOT="${candidate}"
            return 0
        fi
    done
    return 1
}

ensure_human_body_prior() {
    if [[ -f "${HUMAN_BODY_PRIOR_BODYMODEL}" ]]; then
        return 0
    fi

    echo "未检测到 human_body_prior 源码，准备自动拉取到:"
    echo "  ${HUMAN_BODY_PRIOR_DIR}"

    if [[ -d "${HUMAN_BODY_PRIOR_DIR}" ]]; then
        if find "${HUMAN_BODY_PRIOR_DIR}" -mindepth 1 -print -quit | grep -q .; then
            echo "目录已存在但缺少关键文件:"
            echo "  ${HUMAN_BODY_PRIOR_BODYMODEL}"
            echo "请手动检查该目录内容后重试。"
            exit 1
        fi
    fi

    git clone --depth 1 https://github.com/nghorbani/human_body_prior.git "${HUMAN_BODY_PRIOR_DIR}"

    if [[ ! -f "${HUMAN_BODY_PRIOR_BODYMODEL}" ]]; then
        echo "human_body_prior 拉取后仍缺少关键文件:"
        echo "  ${HUMAN_BODY_PRIOR_BODYMODEL}"
        exit 1
    fi
}

if [[ -z "${MODEL_ROOT}" ]]; then
    detect_model_root || {
        echo "未找到 SMPL-X model 路径。"
        echo "请设置环境变量 MODEL_ROOT，例如："
        echo "  MODEL_ROOT=/path/to/smpl_models/models bash scripts/run_server_amass_pipeline.sh"
        exit 1
    }
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "REPO_ROOT 不存在: ${REPO_ROOT}"
    exit 1
fi

if [[ ! -d "${AMASS_ROOT}" ]]; then
    echo "AMASS_ROOT 不存在: ${AMASS_ROOT}"
    exit 1
fi

if [[ ! -f "${MODEL_ROOT}/smplx/SMPLX_NEUTRAL.npz" ]]; then
    echo "SMPL-X model 不完整，缺少: ${MODEL_ROOT}/smplx/SMPLX_NEUTRAL.npz"
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "当前环境里找不到 Python: ${PYTHON_BIN}"
    exit 1
fi

mkdir -p "${PROCESSED_AMASS_DIR}" "${RETARGET_SAVE_DIR}" "${LOG_DIR}"

PIPELINE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
RETARGET_LOG="${LOG_DIR}/parallel_robot_retarget.log"

exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "REPO_ROOT=${REPO_ROOT}"
echo "AMASS_ROOT=${AMASS_ROOT}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "PROCESSED_AMASS_DIR=${PROCESSED_AMASS_DIR}"
echo "RETARGET_SAVE_DIR=${RETARGET_SAVE_DIR}"
echo "PIPELINE_LOG=${PIPELINE_LOG}"
echo "RETARGET_LOG=${RETARGET_LOG}"
echo "PYTHON_BIN=$(command -v "${PYTHON_BIN}")"

cd "${REPO_ROOT}"
ensure_human_body_prior

PREP_CMD=(
    "${PYTHON_BIN}"
    src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py
    --amass-root-folder "${AMASS_ROOT}"
    --output-folder "${PROCESSED_AMASS_DIR}"
    --model-root-folder "${MODEL_ROOT}"
)

if [[ -n "${SUBDATASET_FOLDER}" ]]; then
    PREP_CMD+=(--subdataset-folder "${SUBDATASET_FOLDER}")
fi

echo
echo "[1/2] 预处理 AMASS SMPL-X"
"${PREP_CMD[@]}"

RETARGET_CMD=(
    "${PYTHON_BIN}"
    src/holosoma_retargeting/holosoma_retargeting/examples/parallel_robot_retarget.py
    --data-dir "${PROCESSED_AMASS_DIR}"
    --task-type robot_only
    --data_format smplx
    --save_dir "${RETARGET_SAVE_DIR}"
    --task-config.object-name ground
    --task-config.ground-range -10 10
)

if [[ -n "${MAX_WORKERS}" ]]; then
    RETARGET_CMD+=(--max-workers "${MAX_WORKERS}")
fi

echo
echo "[2/2] 后台启动并行 retarget"
echo "命令: ${RETARGET_CMD[*]}"
nohup "${RETARGET_CMD[@]}" > "${RETARGET_LOG}" 2>&1 &
RETARGET_PID=$!

echo "parallel_robot_retarget 已启动"
echo "PID=${RETARGET_PID}"
echo "日志: ${RETARGET_LOG}"
echo "查看日志: tail -f ${RETARGET_LOG}"
