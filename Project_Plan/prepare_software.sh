#!/usr/bin/env bash
# User-invoked software preparation only; never loads model weights or uses a GPU.
set -euo pipefail
PROJECT=/data/chenjiayu/changzihe/DistillPP_Validation
PLAN="$PROJECT/Project_Plan"
ENV_ROOT=/data/chenjiayu/miniconda3/envs
GEN_PY="$ENV_ROOT/changzihe/bin/python"
EVAL_ENV="$ENV_ROOT/changzihe_eval"
CONDA=/data/chenjiayu/miniconda3/bin/conda
MODE="${1:-}"
if [[ "$MODE" != generation && "$MODE" != evaluation ]]; then
    printf 'Usage: bash %s generation|evaluation\n' "$0" >&2
    exit 2
fi
[[ "$(realpath -e "$PLAN")" == "$PLAN" ]] || exit 2
[[ "$(realpath -e "$ENV_ROOT")" == "$ENV_ROOT" ]] || exit 2
[[ -x "$GEN_PY" ]] || exit 2
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=""
export MAX_JOBS=4
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL
export PIP_DEFAULT_TIMEOUT=60
export PIP_RETRIES=5
export PIP_CACHE_DIR="$PROJECT/resources/software_cache/pip"
export CONDA_PKGS_DIRS="$PROJECT/resources/software_cache/conda"
export XDG_CACHE_HOME="$PROJECT/resources/software_cache/xdg"
export HF_HOME="$PROJECT/resources/evaluation/.cache/hf"
export TORCH_HOME="$PROJECT/resources/evaluation/.cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PROJECT/Project_Progress/Stage_Progress" "$PIP_CACHE_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$PROJECT/Project_Progress/Stage_Progress/software_${MODE}_${STAMP}.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; printf "SOFTWARE_EXIT_CODE=%s\n" "$rc"' EXIT

if [[ "$MODE" == generation ]]; then
    "$GEN_PY" "$PLAN/prepare_sources.py" --group generation
    "$GEN_PY" -m pip install --timeout 60 --retries 5 \
        --no-build-isolation --constraint "$PLAN/constraints_generation.txt" \
        --report "$PROJECT/Project_Progress/Stage_Progress/pip_generation_${STAMP}.json" \
        --editable "$PROJECT/src/LightX2V" 'langdetect==1.0.9' 'pyzmq>=25,<28' 'jsonschema>=4,<5'
    PY="$GEN_PY"
else
    # Separate prefix avoids downgrading Transformers in the general changzihe env.
    if [[ -L "$EVAL_ENV" ]]; then
        printf 'Refusing redirected evaluation environment: %s\n' "$EVAL_ENV" >&2
        exit 2
    fi
    if [[ ! -e "$EVAL_ENV" ]]; then
        "$CONDA" create --yes --prefix "$EVAL_ENV" --no-default-packages \
            --override-channels \
            --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
            python=3.10 pip
    fi
    [[ -x "$EVAL_ENV/bin/python" ]] || { printf 'Incomplete environment; stop for review\n'; exit 2; }
    PY="$EVAL_ENV/bin/python"
    "$PY" -m pip install --timeout 60 --retries 5 \
        --constraint "$PLAN/constraints_evaluation.txt" \
        --requirement "$PLAN/requirements_evaluation.txt" \
        --report "$PROJECT/Project_Progress/Stage_Progress/pip_evaluation_${STAMP}.json"
    "$PY" "$PLAN/prepare_sources.py" --group evaluation
    # Frozen CLIP and VBench source take precedence over generic pip distributions.
    # The later scoring launcher must set the same PYTHONPATH explicitly.
    PYTHONPATH="$PROJECT/src/CLIP:$PROJECT/src/VBench:$PROJECT/src/DINO" \
        "$PY" -c 'import clip, pyiqa, decord; import vbench; print("CLIP_SOURCE", clip.__file__); print("PYIQA_VERSION", pyiqa.__version__); print("DECODER", decord.__version__); print("CPU_IMPORTS_OK")'
fi

"$PY" -m pip freeze > "$PROJECT/Project_Progress/Stage_Progress/pip_freeze_${MODE}_${STAMP}.txt"
"$PY" -c 'import sys, torch; print("PYTHON",sys.executable); print("TORCH",torch.__version__); print("TORCH_CUDA_BUILD",torch.version.cuda); print("CUDA_INITIALIZED",torch.cuda.is_initialized())'
set +e
CHECK="$("$PY" -m pip check 2>&1)"
RC=$?
set -e
printf '%s\n' "$CHECK"
if [[ "$RC" != 0 && "$CHECK" != 'decord 0.6.0 is not supported on this platform' ]]; then
    printf 'Dependency check has an additional/unexpected error; stop for review.\n' >&2
    exit "$RC"
fi
if [[ "$RC" != 0 ]]; then
    printf 'KNOWN_WARNING: decord internal wheel tag disagrees with its filename; this is recorded, not repaired or hidden. Decoder acceptance still required.\n'
fi
printf 'Software preparation finished. Model inference and six-metric runtime acceptance have NOT run.\n'
