# Contributing

Bug reports and focused pull requests are welcome.

Before opening a pull request:

1. Keep existing node IDs and Registry IDs stable unless a breaking change is intentional.
2. Add or update tests for behavior changes.
3. Run the command used by `.github/workflows/tests.yml` locally.
4. Do not include models, generated checkpoints, credentials, or private workflow data.

Please describe the problem, the expected behavior, and the ComfyUI version used.

Run the regression checks with:

```sh
python -m pip install -r requirements.txt numpy
python -m unittest discover -s tests -v
node --test tests/test_lora_folder_filter.mjs
```

The Python tests use real PyTorch and safetensors with small fixtures; ComfyUI's
environment and model application API are mocked. The JavaScript tests exercise
the extension's workflow and widget callbacks with a minimal frontend harness.
