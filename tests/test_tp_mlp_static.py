from pathlib import Path


def test_tp_mlp_lab_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "labs" / "01_minimal_tp_mlp.py").exists()
    assert (root / "labs" / "02_tp_mlp.py").exists()
    assert (root / "labs" / "03_tp_regions.py").exists()
    assert (root / "parallel" / "mlp.py").exists()
    assert (root / "parallel" / "linear.py").exists()
    assert (root / "parallel" / "mappings.py").exists()
    assert (root / "tests" / "distributed_tp_correctness.py").exists()


def test_tp_mlp_uses_nanogpt_reference():
    root = Path(__file__).resolve().parents[1]
    minimal_lab = (root / "labs" / "01_minimal_tp_mlp.py").read_text()
    assert "copy_mlp_to_tensor_parallel(reference, tp_mlp)" in minimal_lab

    lab = (root / "labs" / "02_tp_mlp.py").read_text()
    assert "from model import GPTConfig, MLP" in lab
    assert "copy_mlp_to_tensor_parallel(reference, tp_mlp)" in lab
    assert "torch.testing.assert_close(tp_y, reference_y" in lab
