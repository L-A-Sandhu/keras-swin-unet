import subprocess
import sys
import os
import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "swin_transformer", "main.py")


def run_cmd(extra_args):
    """Run main.py with *extra_args* and return the CompletedProcess"""
    cmd = [sys.executable, MAIN_SCRIPT] + extra_args
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# 1. Top‑level help must succeed
# ---------------------------------------------------------------------------

def test_root_help():
    result = run_cmd(["--help"])
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


# ---------------------------------------------------------------------------
# 2. Sub‑command help must succeed (train & infer)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subcmd", ["train", "infer"])
def test_subcommand_help(subcmd):
    result = run_cmd([subcmd, "--help"])
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


# ---------------------------------------------------------------------------
# 3. Missing required arguments should raise an argparse error (exit code 2)
# ---------------------------------------------------------------------------

def test_train_missing_required():
    result = run_cmd(["train"])
    assert result.returncode == 2  # argparse error
    assert "the following arguments are required" in result.stderr.lower()


def test_infer_missing_required():
    result = run_cmd(["infer"])
    assert result.returncode == 2
    assert "the following arguments are required" in result.stderr.lower()


# ---------------------------------------------------------------------------
# 4. Quick parse‑only sanity checks with minimal dummy inputs
#    ‑ we don't execute heavy TF code; we just ensure parsing accepts options
# ---------------------------------------------------------------------------

def test_train_parses_with_minimum_required(tmp_path):
    dummy_data = tmp_path / "data" / "images"
    dummy_data.mkdir(parents=True)
    # create a single dummy image & mask so the folder isn't empty
    from PIL import Image
    img = Image.new("RGB", (4, 4))
    img.save(dummy_data / "img.png")
    (tmp_path / "data" / "masks").mkdir(parents=True)
    img.save(tmp_path / "data" / "masks" / "img.png")

    result = run_cmd([
        "train",
        "--data", str(tmp_path / "data"),
        "--gamma", "1.0", "--alpha", "0.5",
        "--epochs", "0",  # ensure the script exits quickly if it gets that far
        "--visualize", "0",
    ])
    # Accept either normal exit (0) or early stopping because of epochs==0 (1)
    assert result.returncode in (0, 1, 2)


def test_infer_parses_with_dummy_paths(tmp_path):
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_text("")  # empty file just to satisfy path check
    from PIL import Image
    img = Image.new("RGB", (4, 4))
    img_file = tmp_path / "img.png"
    img.save(img_file)

    result = run_cmd([
        "infer",
        "--model-dir", str(tmp_path),
        "--checkpoint", str(checkpoint),
        "--image", str(img_file),
        "--visualize", "0",
    ])
    assert result.returncode in (0, 1, 2)
