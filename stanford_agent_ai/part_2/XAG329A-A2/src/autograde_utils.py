# Defines helper functions for autograder

import ast
import json
import os
import sys
import types

import numpy as np

# Cells with these tags run real LLM evals — skip them when loading student code.
EVAL_TAGS = frozenset(
    {
        "zero_shot_eval",
        "majority_voting_eval",
        "llm_voting_eval",
        "self_improvement_eval",
    }
)

# Top-level defs/classes we need from assignment2.ipynb
_STUDENT_SYMBOLS = frozenset(
    {
        "evaluate_zero_shot",
        "MajorityVoting",
        "LLMVoting",
        "SelfImprovementSystem",
    }
)


def assert_allclose(a, b, err_msg="", rtol=1e-5, atol=1e-7, squeeze=True):
    """
    Inputs:
    - a: np.ndarray. First array to compare.
    - b: np.ndarray. Second array to compare.
    - err_msg: optional, str. Error message to print on failure.
    - rtol: optional, float. Relative tolerance; see documentation for np.allclose.
        Defaults to 1e-5.
    - atol: optional, float. Absolute tolerance; see documentation for np.allclose.
        Defaults to 1e-7.
    - squeeze: optional, bool. Squeeze inputs before comparing. Defaults to True.
    """

    # By default, let's squeeze to ignore errors related to row/column vector
    # convention, etc ¯\_(ツ)_/¯
    a_orig, b_orig = a, b
    if squeeze and type(a) == np.ndarray:
        a = a.squeeze()
    if squeeze and type(b) == np.ndarray:
        b = b.squeeze()

    exception = None
    try:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=err_msg)
    except Exception as e:
        exception = e

    if exception is not None:
        # If the test fails, try to give students some useful error messages
        assert (a is None) == (b is None), "Comparison failed! Unexpected 'None' value."
        assert type(a) == type(b), "Comparison type error! {} doesn't match {}.".format(
            type(a).__name__, type(b).__name__
        )
        assert (
            a.shape == b.shape
        ), "Comparison shape error! {} doesn't match {}.".format(
            a_orig.shape, b_orig.shape
        )
        assert (
            a.dtype == b.dtype
        ), "Comparison datatype error! {} doesn't match {}.".format(a.dtype, b.dtype)

        # Resort to original error message
        raise exception


def text_in_cell(ipynb_path, metadata):
    """
    Get the output texts from the CODE cell with metadata == metadata in the provided
    .ipynb file.

    Inputs:
    - ipynb_path: str. Path to the .ipynb file.
    - metadata: str. Contains cell identifier.

    Returns:
    - str. All output texts in the cell with the specified metadata.
    """

    with open(ipynb_path) as f:
        ipynb = json.load(f)

    # Pick out the target cell
    code_cells = [cell for cell in ipynb["cells"] if cell["cell_type"] == "code"]
    success = False
    for code_cell in code_cells:
        if "test" in code_cell["metadata"].keys():
            if code_cell["metadata"]["test"] == metadata:
                tg_cell = code_cell
                success = True
                break

    # Couldn't find cell
    error_file = ipynb_path.split("/")[-1]
    if not success:
        raise ValueError(
            "Corrupted notebook metadata: you may have accidentally deleted or"
            " modified a critical code cell. \n"
            f"Please try copying your {error_file} changes back into the "
            "skeleton code, rerunning, and resubmitting."
        )

    if tg_cell["outputs"]:
        # When there are multiple output blocks, combine all stdout and return
        stdout = []
        for output in tg_cell["outputs"]:
            if "name" in output and output["name"] == "stdout":
                if type(output["text"]) is list:
                    stdout += output["text"]
                elif type(output["text"]) is str:
                    # Hack to fix one student's output...
                    # For some reason, their output comes as a single string instead of
                    # a list of strings
                    stdout.extend([s for s in output["text"].split("\n")])
                else:
                    raise ValueError(
                        "Error processing code cell output in {error_file}!"
                    )

        # Strip out empty lines -- these can appear inconsistently
        stdout = [x.strip() for x in stdout if len(x.strip()) > 0]
        return stdout

    # Error message if nothing was returned
    raise ValueError(
        f"Missing code cell output in {error_file}. Make sure all code cells "
        "in your submission have been run."
    )


def if_text_in_py(py_path, string):
    """
    Check if the provided .py file contains the specified string.

    Inputs:
    - py_path: str. Path to the .py file.
    - string: str. The text to be checked.

    Returns:
    - bool. True if the .py file contains string, False otherwise.
    """

    with open(py_path) as f:
        py = f.readlines()

    for line in py:
        line = line.lstrip()  # Remove unrelevant leading characters
        if string in line and line[0] != "#":  # string is not in a comment
            return True

    return False


def _cell_source(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def _strip_ipython_magics(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        if "get_ipython(" in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def _top_level_names(source: str):
    """Return names of top-level function/class defs in source (best-effort)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _ensure_package_path(ipynb_path: str):
    """Put the notebook's parent dir on sys.path so `xag329a_a2` imports work."""
    parent = os.path.abspath(os.path.dirname(ipynb_path))
    if parent not in sys.path:
        sys.path.insert(0, parent)


def notebook_to_py(ipynb_path, py_path=None, skip_eval_tags=True):
    """
    Convert notebook code cells to a .py file (for debugging).

    Skips eval-tagged cells and IPython magics by default.
    """
    with open(ipynb_path) as f:
        ipynb = json.load(f)

    chunks = []
    for cell in ipynb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []) or [])
        if skip_eval_tags and tags & EVAL_TAGS:
            continue
        source = _strip_ipython_magics(_cell_source(cell)).strip()
        if not source:
            continue
        chunks.append(source)

    text = "\n\n".join(chunks) + "\n"
    if py_path is not None:
        with open(py_path, "w") as f:
            f.write(text)
    return text


def load_notebook_module(ipynb_path, module_name="assignment2"):
    """
    Load student TODO symbols from assignment2.ipynb into a module.

    Only executes cells that define evaluate_zero_shot / MajorityVoting /
    LLMVoting / SelfImprovementSystem. Skips eval-tagged cells and magics so
    no LLM API calls run during import.
    """
    error_file = os.path.basename(ipynb_path)
    if not os.path.isfile(ipynb_path):
        raise FileNotFoundError(
            f"Could not find {error_file}. Upload assignment2.ipynb and try again."
        )

    _ensure_package_path(ipynb_path)

    with open(ipynb_path) as f:
        ipynb = json.load(f)

    module = types.ModuleType(module_name)
    module.__file__ = ipynb_path
    # Seed imports used by class cells (also available if a cell re-imports).
    exec(
        "from typing import List, Union\n"
        "from xag329a_a2.methods import get_verifier\n"
        "from xag329a_a2.methods.simple_samplers import SampleMultiple\n"
        "from xag329a_a2.tasks.math_utils import strip_string, extract_answer\n",
        module.__dict__,
    )

    found = set()
    for cell in ipynb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []) or [])
        if tags & EVAL_TAGS:
            continue
        source = _strip_ipython_magics(_cell_source(cell)).strip()
        if not source:
            continue
        names = _top_level_names(source)
        if not (names & _STUDENT_SYMBOLS):
            continue
        try:
            exec(source, module.__dict__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load student code from {error_file}: {e}"
            ) from e
        found |= names & _STUDENT_SYMBOLS

    missing = _STUDENT_SYMBOLS - found
    if missing:
        raise RuntimeError(
            f"Could not find required definitions in {error_file}: "
            f"{', '.join(sorted(missing))}. "
            "Do not delete the TODO cells that define these names."
        )

    sys.modules[module_name] = module
    return module


def _find_tagged_cell(ipynb_path, tag):
    with open(ipynb_path) as f:
        ipynb = json.load(f)

    for cell in ipynb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", []) or []
        if tag in tags:
            return cell
    return None


def stdout_from_tagged_cell(ipynb_path, tag):
    """
    Return non-empty stdout lines from the code cell tagged `tag`.
    """
    error_file = os.path.basename(ipynb_path)
    tg_cell = _find_tagged_cell(ipynb_path, tag)
    if tg_cell is None:
        raise ValueError(
            f"Missing tagged cell '{tag}' in {error_file}. "
            "Please restore the skeleton notebook and re-run that cell."
        )

    if not tg_cell.get("outputs"):
        raise ValueError(
            f"Missing output for tagged cell '{tag}' in {error_file}. "
            "Run all evaluation cells before submitting."
        )

    stdout = []
    for output in tg_cell["outputs"]:
        # Prefer stdout streams. Some notebook editors strip the `name` field
        # when saving; treat unnamed streams as stdout, but skip stderr.
        if output.get("output_type") != "stream":
            continue
        name = output.get("name")
        if name == "stderr":
            continue
        if name not in (None, "stdout"):
            continue
        text = output.get("text", "")
        if isinstance(text, list):
            stdout.extend(text)
        elif isinstance(text, str):
            stdout.extend(text.split("\n"))
        else:
            raise ValueError(f"Error processing stdout in {error_file} (tag={tag}).")

    return [x.strip() for x in stdout if len(x.strip()) > 0]


def json_from_tagged_cell(ipynb_path, tag):
    """
    Parse the JSON object printed by the code cell tagged `tag`.

    Scans stdout for valid JSON objects via JSONDecoder.raw_decode so braces
    in earlier LLM/LaTeX output do not poison extraction. Prefers a dict whose
    "section" matches `tag`; otherwise uses the last valid dict. If the chosen
    object has a "section" field, it must match `tag`.
    """
    error_file = os.path.basename(ipynb_path)
    lines = stdout_from_tagged_cell(ipynb_path, tag)
    text = "\n".join(lines)

    # Eval cells print noisy prediction traces before the summary JSON; a greedy
    # \{...\} match can start on LaTeX braces. Decode from each '{' instead.
    decoder = json.JSONDecoder()
    candidates = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            candidates.append(data)

    if not candidates:
        raise ValueError(
            f"No JSON object found in stdout for tagged cell '{tag}' in {error_file}. "
            "Make sure the cell prints a JSON summary via json.dumps(...)."
        )

    matching = [d for d in candidates if d.get("section") == tag]
    data = matching[-1] if matching else candidates[-1]
    if "section" in data and data["section"] != tag:
        raise ValueError(
            f"JSON section mismatch for tag '{tag}': got {data['section']!r}."
        )
    return data
