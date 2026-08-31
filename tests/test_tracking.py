from types import SimpleNamespace

from progressive_tokenizer.tracking import generate_wandb_run_id


def test_generate_wandb_run_id_uses_legacy_public_location() -> None:
    module = SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "legacy-id"))
    assert generate_wandb_run_id(module) == "legacy-id"


def test_generate_wandb_run_id_supports_current_wandb() -> None:
    module = SimpleNamespace(util=SimpleNamespace())
    run_id = generate_wandb_run_id(module)
    assert isinstance(run_id, str)
    assert run_id
