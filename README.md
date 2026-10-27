# GenAI C1 Project: Teach an LLM to Reason

This project guides you through teaching an instruction-tuned LLM to reason step‑by‑step using GRPO (Group Relative Policy Optimization). The model learns to break a word into letters and count a specific letter.

Key notebooks and files:
- `starter/gen_ai_fundamentals_project_starter.ipynb`
- `requirements.txt`

## Hardware

- Runs on a single "T4 Tesla" GPU with 16GB VRAM (e.g., AWS `g4dn.xlarge`).

## NVIDIA Driver and CUDA (tested)

- OS: Ubuntu 24.04 (x86_64)
- NVIDIA driver: 575.57.08
- CUDA: 12.9.1

Install driver + CUDA toolkit only if you are sure drivers/CUDA are not already installed on your system (if you can run nvidia-smi, you have them installed definitely).
```
sudo apt update && sudo apt install gcc make -y
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit --driver --no-drm
```

## Other Dependencies

Install uv and Python headers:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart your terminal to ensure `uv` is available.

## Environment Setup

From the repository root:
```
uv python pin 3.12.3
uv init
uv sync
uv add ipykernel pip
uv pip install -r requirements.txt  --no-deps
```

## Run the Project

- Open `starter/gen_ai_fundamentals_project_starter.ipynb` in Jupyter and select the `.venv` kernel.
- Execute cells in order. The notebook installs any remaining Python dependencies needed for training and evaluation.


## Results

The notebook was run end to end on a Tesla T4 (16 GB) with `Qwen 2.5 3B Instruct`, LoRA rank 64 and 100 GRPO steps (about 23 minutes). All outputs and plots are saved in the notebook.

| Metric (mean of first 20 vs. last 20 steps) | Start | End |
|---|---|---|
| Total reward | 1.62 | 3.96 |
| Correct-answer reward (-1 wrong, +2 right) | 0.875 | 1.438 |
| Approximate answer accuracy | ~62% | ~81% |

- Before training, the model answered that "idea" contains **3** a's. After training it spells the word, keeps a correct running count and answers **1**.
- The model still answers general questions correctly (capital of the Philippines: Manila), so there is no sign of catastrophic forgetting.

See the **Training analysis** section of the notebook for the full discussion.

### Reward revision (after review)

After the first review, `format_reward_func` was revised to give a **negative** reward for malformed output (no tags, missing reasoning or answer block, non-numeric answer, or extra text, checked with `re.fullmatch`) and +1.0 only for a fully valid response. `spelling_reward_func` now also penalizes letters in the wrong order (anagrams). Both were re-validated with assertions in the notebook. The training results above were produced with the **original** reward configuration; the model was not retrained after the revision.

## Built With

- PyTorch, Transformers, Unsloth, vLLM, TRL

