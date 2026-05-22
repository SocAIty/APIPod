from apipod.deploy.profile import PROFILE_ML_GPU, PROFILE_SERVERLESS_MINIMAL, infer_profile


def test_infer_serverless_minimal_for_apipod_only():
    profile = infer_profile(
        pytorch=False,
        tensorflow=False,
        onnx=False,
        transformers=False,
        diffusers=False,
        cuda=False,
        compute="serverless",
        provider="runpod",
        python_deps={"apipod", "runpod"},
    )
    assert profile == PROFILE_SERVERLESS_MINIMAL


def test_infer_ml_gpu_when_torch_present():
    profile = infer_profile(
        pytorch=True,
        tensorflow=False,
        onnx=False,
        transformers=False,
        diffusers=False,
        cuda=True,
        compute="serverless",
        provider="runpod",
        python_deps={"torch", "runpod"},
    )
    assert profile == PROFILE_ML_GPU


def test_tensorboard_does_not_imply_tensorflow():
    from apipod.deploy.detectors.framework import FrameworkDetector
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            '[project]\nname="x"\nrequires-python=">=3.12"\n'
            'dependencies = ["apipod", "runpod", "tensorboard"]\n'
        )
        (root / "main.py").write_text(
            'from apipod import APIPod\napp = APIPod(compute="serverless", provider="runpod")\n'
        )
        info = FrameworkDetector(str(root)).detect()
        assert info["pytorch"] is False
        assert info["tensorflow"] is False
        assert info["onnx"] is False


if __name__ == "__main__":
    test_infer_serverless_minimal_for_apipod_only()
    test_infer_ml_gpu_when_torch_present()
    test_tensorboard_does_not_imply_tensorflow()
    print("ok")
