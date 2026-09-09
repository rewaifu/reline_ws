"""Unit tests for rebind.preprocess (no server needed).

Run: uv run python tests/test_preprocess.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebind.preprocess import (  # noqa: E402
    _dearchive,
    _ensure_model,
    _rewrite_upscale_models,
    run_preprocessors,
)

checks: list[str] = []


def ok(name: str) -> None:
    checks.append(name)
    print(f"PASS {name}", flush=True)


def make_zip(path: str, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="reline_preprocess_")
    try:
        models = os.path.join(tmp, "models")
        os.makedirs(models)

        # 1. already installed (stem match) -> no download
        with open(os.path.join(models, "4x_a.pth"), "wb") as f:
            f.write(b"weights")
        got = _ensure_model(models, "4x_a", None)
        assert got == os.path.join(models, "4x_a.pth"), got
        ok("installed model: stem match, no download")

        # 2. archive url: model extracted from a nested folder and moved
        archive = os.path.join(tmp, "4x_b.tar.xz")
        make_zip(archive, {"sub/dir/4x_b.pth": b"weights-b"})
        got = _ensure_model(models, "4x_b", "file://" + archive)
        assert got == os.path.join(models, "4x_b.pth"), got
        assert open(got, "rb").read() == b"weights-b"
        ok("archive url: model found inside and moved to models dir")

        # 3. direct model file url
        raw = os.path.join(tmp, "4x_c.safetensors")
        with open(raw, "wb") as f:
            f.write(b"weights-c")
        got = _ensure_model(models, "4x_c", "file://" + raw)
        assert got == os.path.join(models, "4x_c.safetensors"), got
        ok("model file url: moved as-is")

        # 4. missing model without url -> error
        try:
            _ensure_model(models, "4x_missing", None)
            raise AssertionError("expected ValueError")
        except ValueError:
            ok("missing model without url raises")

        # 5. upscale rewrite matches by model name
        nodes = [
            {"type": "upscale", "options": {"model": "4x_b", "dtype": "F32"}},
            {"type": "upscale", "options": {"model": "4x_a"}},
            {"type": "folder_reader", "options": {"path": "/x"}},
        ]
        count = _rewrite_upscale_models(nodes, "4x_b", "/models/4x_b.pth")
        assert count == 1, count
        assert nodes[0]["options"]["model"] == "/models/4x_b.pth"
        assert nodes[1]["options"]["model"] == "4x_a"
        ok("upscale nodes repointed by name")

        # 6. recursive dearchive: zip in a folder + nested archive, sources removed
        room = os.path.join(tmp, "room")
        os.makedirs(room)
        inner = os.path.join(tmp, "inner.tar.gz")
        import tarfile

        with tarfile.open(inner, "w:gz") as t:
            t.add(os.path.join(tmp, "models", "4x_a.pth"), arcname="img.png")
        make_zip(os.path.join(room, "pack.zip"), {"data/inner.tar.gz": open(inner, "rb").read()})
        _dearchive(room)
        assert not os.path.exists(os.path.join(room, "pack.zip"))
        assert os.path.isfile(os.path.join(room, "pack", "data", "inner", "img.png")), os.listdir(room)
        ok("recursive dearchive: zip -> nested tar.gz, archives removed")
        # 7. run_preprocessors: skip + rewrite + unarchive, then cancel
        base = os.path.join(tmp, "base")
        os.makedirs(base)
        make_zip(os.path.join(base, "raws.zip"), {"a.png": b"a"})
        preprocess = [
            {"type": "download", "options": {"name": "4x_b", "url": "file://" + archive}},
            {"type": "unarchive", "options": {"path": "raws.zip"}},
        ]
        nodes2 = [{"type": "upscale", "options": {"model": "4x_b"}}]
        percents: list[int] = []

        async def progress(p: int) -> None:
            percents.append(p)

        async def run() -> None:
            import asyncio as aio

            cancel = aio.Event()
            assert await run_preprocessors(preprocess, nodes2, base, progress, cancel, models)
            assert os.path.isfile(os.path.join(base, "raws", "a.png"))
            assert not os.path.exists(os.path.join(base, "raws.zip"))
            assert nodes2[0]["options"]["model"] == os.path.join(models, "4x_b.pth")
            cancel.set()
            assert not await run_preprocessors(preprocess, nodes2, base, progress, cancel, models)

        asyncio.run(run())
        assert percents == sorted(percents) and percents[-1] == 50, percents
        ok("run_preprocessors: download+unarchive+rewrite, cancel honored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nALL {len(checks)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
