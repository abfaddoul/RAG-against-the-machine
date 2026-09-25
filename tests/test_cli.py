import pytest

import src.__main__ as application


def test_main_handles_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test hiding unexpected CLI tracebacks."""

    def raise_runtime_error() -> None:
        raise RuntimeError("model failed")

    monkeypatch.setattr(
        application,
        "run_cli",
        raise_runtime_error,
    )

    with pytest.raises(SystemExit) as error:
        application.main()

    assert error.value.code == 1
    assert (
        capsys.readouterr().err
        == "Error: model failed\n"
    )
