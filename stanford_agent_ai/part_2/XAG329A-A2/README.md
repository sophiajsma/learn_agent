# XAG329A2 – Self-Improving AI Agents – Part 2 Final Assignment

Enhanced assignment covering test-time compute scaling techniques for self-improving AI agents.

## Overview

This assignment explores various techniques to improve model performance during inference, including:

1. Zero-shot Evaluation (10 points)
2. Majority Voting (10 points)
3. Best-of-N with a Generative Reward Model (10 points)
4. Self-Improvement with Feedback (16 points)
5. Evaluation summaries (4 points)

**Total: 50 points.** An optional Analysis and Discussion section in the notebook is ungraded.

## Submission

Submit your completed notebook to Gradescope:

- Upload **`assignment2.ipynb`** (from `src/submission/assignment2.ipynb`).
- **Do not rename the file** — it must be named exactly `assignment2.ipynb`.
- Run all required evaluation cells before submitting so the notebook contains the saved outputs the autograder checks.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.
Think of `uv` as a fast replacement for `pip` + `venv`: one tool creates a virtual environment, installs packages from `pyproject.toml` / `uv.lock`, and runs commands inside that environment.

We only support `uv` for this assignment (not Conda).

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS, you can also use Homebrew:

```bash
brew install uv
```

After installing, restart your terminal (or open a new one), then confirm it works:

```bash
uv --version
```

### 2. Install the project dependencies

From the root of this repository (the folder that contains `pyproject.toml`), run:

```bash
uv sync
```

This creates a local `.venv/` folder and installs everything listed in `pyproject.toml` (locked by `uv.lock`).
You usually do **not** need to activate the environment yourself — use `uv run ...` (or select the `.venv` kernel in your editor).

### 3. Set up API keys

[OpenRouter](https://openrouter.ai/) is a unified API gateway that routes requests to many model providers through one interface and API key. We use it in this assignment so students can access both open-source and closed (proprietary) models without configuring each provider separately. For this assignment, however, we will only use specific open-source models (see the notebook and [Useful links](#useful-links)).

1. Copy `.env.example` to `.env`
2. Edit `.env` and add your API keys (see [Additional information](#additional-information)):
   - `OPENROUTER_API_KEY`: Your [OpenRouter](https://openrouter.ai/) API key (create one at [openrouter.ai/keys](https://openrouter.ai/keys))
   - `HUGGINGFACE_HUB_TOKEN`: Your [Hugging Face](https://huggingface.co/) token (create one at [settings/tokens](https://huggingface.co/settings/tokens))
3. The notebook loads these with:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 4. Open the notebook (recommended: VS Code)

The easiest way to run the assignment locally is in **[VS Code](https://code.visualstudio.com/)**:

1. Install the **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)** extension (search “Jupyter” by Microsoft in Extensions)
2. Open this folder in VS Code (the repo root that contains `pyproject.toml`)
3. Open `src/submission/assignment2.ipynb`
4. Select the kernel / interpreter created by `uv`:
   - Choose **Python Environments** → the `.venv` in this project
   - Or run the kernel registration command below and pick `xag329a-a2`
5. Run cells as usual

See also: [VS Code Jupyter notebooks docs](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

### Alternative: Jupyter in the browser

If you prefer classic [Jupyter Notebook](https://jupyter.org/), from the repo root:

```bash
uv run jupyter notebook src/submission/assignment2.ipynb
```

`uv run` runs the command inside the project environment, so imports like `from xag329a_a2.methods import get_sampler` should work.

See also: [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/).

### Troubleshooting

**Kernel / wrong environment**: If cells fail with `ModuleNotFoundError`, make sure the notebook kernel is the project `.venv` (or `xag329a-a2`), then re-run:

```bash
uv sync
uv run python -m ipykernel install --user --name xag329a-a2 --display-name "xag329a-a2"
```

**Package import errors**: From the repo root:

```bash
uv sync
```

**Missing API tokens**: If you see an error about `HUGGINGFACE_HUB_TOKEN` or `OPENROUTER_API_KEY`, confirm `.env` exists (copied from `.env.example`), the variable names match exactly, and you ran `load_dotenv()` in the notebook.

## Project structure

```
.
├── pyproject.toml              # dependencies / packaging (install from repo root)
├── uv.lock                     # locked dependency versions (used by `uv sync`)
├── .python-version             # default Python for `uv` (3.11)
├── .env.example                # copy to `.env` and add your API keys
└── src/submission/
    ├── assignment2.ipynb # main assignment notebook
    └── xag329a_a2/            # Python package (tasks, methods, inference, tests)
```

After `uv sync`, import the package as `xag329a_a2` (not via a relative path into `src/submission`).

## Usage

```python
from xag329a_a2.tasks import AIME25
from xag329a_a2.methods import get_sampler, get_verifier

aime25 = AIME25()
problems = aime25.get_problems(debug_mode=True)
system_prompt = aime25.get_system_prompt()
verifier = get_verifier("aime25")

# Example: Sample with OpenRouter using Qwen model
# Model page: https://openrouter.ai/qwen/qwen3-next-80b-a3b-instruct
method = get_sampler("sample_multiple", "openrouter/qwen/qwen3-next-80b-a3b-instruct", temperature=0.7, n_samples=16, system_prompt=system_prompt)
```

## Features

- **AIME25 Dataset**: Challenging mathematical problems from the American Invitational Mathematics Examination
- **Multiple Sampling Methods**: Greedy, multiple sampling, majority voting, LLM voting
- **Self-Improvement System**: RLEF (Reinforcement Learning from Execution Feedback) with critique and regeneration
- **Comprehensive Analysis**: Detailed comparison of all methods with cost-effectiveness analysis
- **Concurrent Processing**: ThreadPoolExecutor for efficient API calls

## Requirements

- Python 3.10–3.12 (default: 3.11 via `.python-version`; 3.13 is unsupported)
- OpenRouter API key
- Hugging Face token
- Required packages listed in `pyproject.toml` (installed via `uv sync`)

## Additional information

### Steps to create an OpenRouter API key

1. **Create an account / sign in**
   - Go to [openrouter.ai](https://openrouter.ai/) and sign up or log in. ([docs](https://openrouter.ai/docs))

2. **Open API Keys**
   - Go to [openrouter.ai/keys](https://openrouter.ai/keys)

3. **Create a new key**
   - Click **Create Key**, give it a name (e.g., `xag329a`), and copy the key immediately

4. **Add credits**
   - OpenRouter is a paid API gateway; add credits in your account billing settings so requests can succeed

5. **Use the key in your app**
   - Put it in `.env` as `OPENROUTER_API_KEY=...`, or export it in your shell:
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```
   ([OpenRouter docs](https://openrouter.ai/docs/quickstart))

### Steps to get a free Hugging Face token

1. **Create a free account**
   - Go to [huggingface.co/join](https://huggingface.co/join) and sign up (or log in).

2. **Open token settings**
   - Go to your [token settings](https://huggingface.co/settings/tokens)

3. **Create a new token**
   - Click **“New token”**
   - Give it a name (e.g., `xag329a`)

4. **Choose the token role (permissions)**
   - **Read**: download models/datasets (most common)
   - **Write**: upload/update repos
   - **Admin**: full control (usually not needed)

5. **Generate and copy the token**
   - Copy it immediately and store it somewhere safe (password manager is best).
   - Put it in `.env` as `HUGGINGFACE_HUB_TOKEN=...`

See also: [Hugging Face Hub documentation](https://huggingface.co/docs/hub/en/index).

### Useful links

- [Jupyter Notebook](https://jupyter.org/) / [docs](https://docs.jupyter.org/en/latest/)
- [VS Code](https://code.visualstudio.com/) / [Jupyter in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Hugging Face](https://huggingface.co/) / [Access tokens](https://huggingface.co/settings/tokens) / [Hub docs](https://huggingface.co/docs/hub/en/index)
- [OpenRouter](https://openrouter.ai/) / [API keys](https://openrouter.ai/keys) / [Docs](https://openrouter.ai/docs) / [LiteLLM OpenRouter provider](https://docs.litellm.ai/docs/providers/openrouter)
- Assignment model: [qwen/qwen3-next-80b-a3b-instruct](https://openrouter.ai/qwen/qwen3-next-80b-a3b-instruct)
