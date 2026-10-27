from pathlib import Path
import subprocess
import sys


def test_release_root_has_expected_scaffold() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        (root / "README.md", Path.is_file),
        (root / "pyproject.toml", Path.is_file),
        (root / "requirements.txt", Path.is_file),
        (root / "src" / "group_auth_dnn" / "__init__.py", Path.is_file),
        (root / "scripts", Path.is_dir),
        (root / "experiments", Path.is_dir),
        (root / "tests", Path.is_dir),
        (root / "results", Path.is_dir),
    ]
    missing = [path.as_posix() for path, predicate in expected if not predicate(path)]
    assert not missing, f"missing release paths: {missing}"


def test_public_scripts_expose_help() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts" / "prepare_data.py",
        root / "scripts" / "prepare_checkpoints.py",
        root / "scripts" / "run_main_experiments.py",
        root / "scripts" / "run_ablations.py",
        root / "scripts" / "generate_figures.py",
    ]
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage" in completed.stdout.lower()


def test_readme_mentions_release_workflow() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    required_phrases = [
        "prepare_data.py",
        "prepare_checkpoints.py",
        "run_main_experiments.py",
        "run_ablations.py",
        "generate_figures.py",
        "main results",
        "key ablations",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    assert not missing, f"README missing phrases: {missing}"
