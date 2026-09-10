"""Config handling and the per-image chain of the pipeline runner."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from _harness import Checks, finish

from reline_ws.pipeline import PipelineWs, PlanStep, label_of, resolve_path, split_config
from reline_ws.pipeline import _run_chain


class FakeNode:
    """Stands in for a real node: sleeps, then hands the image on."""

    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.calls = 0

    def single_process(self, image: Any) -> Any:
        self.calls += 1
        time.sleep(self.cost)
        return image


def fake_steps(costs: list[float]) -> tuple[list[PlanStep], list[FakeNode]]:
    nodes = [FakeNode(cost) for cost in costs]
    steps = [
        PlanStep(type=f"fake_{index}", label=f"Fake {index}", node=node)
        for index, node in enumerate(nodes)
    ]
    return steps, nodes


def main() -> int:
    checks = Checks("pipeline")
    print("pipeline:", flush=True)

    # -- config --------------------------------------------------------
    nodes, preprocess = split_config({"nodes": [{"type": "level"}], "preprocess": [{"type": "unarchive"}]})
    checks.eq("map config splits", (len(nodes), len(preprocess)), (1, 1))
    legacy_nodes, legacy_preprocess = split_config([{"type": "level"}])
    checks.eq("legacy flat config still runs", (len(legacy_nodes), legacy_preprocess), (1, []))
    checks.raises("a non-map config is rejected", ValueError, split_config, "nope")
    checks.raises("node lists must be lists", ValueError, split_config, {"nodes": "x"})

    checks.eq("paths join onto root", resolve_path("in", "/data"), "/data/in")
    checks.eq("absolute paths survive", resolve_path("/data/in", "/data"), "/data/in")
    checks.eq("…even with a trailing slash on root", resolve_path("/data", "/data/"), "/data")
    checks.eq("no root -> untouched", resolve_path("in", None), "in")

    prepared, preprocess_left = PipelineWs.prepare_config(
        {
            "nodes": [
                {"type": "folder_reader", "options": {"path": "in"}},
                {"type": "level", "options": {"low_input": 0}, "meta": {"disabled": True}},
                {"type": "folder_writer", "options": {"path": "out"}},
            ],
            "preprocess": [{"type": "download", "options": {"name": "x"}, "meta": {"disabled": True}}],
        },
        "/data",
    )
    checks.eq("disabled nodes are dropped", [item["type"] for item in prepared], ["folder_reader", "folder_writer"])
    checks.eq("…and their preprocessors too", preprocess_left, [])
    checks.eq("reader path resolved against root", prepared[0]["options"]["path"], "/data/in")

    # -- upscale `model`: a file or a download request -------------------
    def model_of(value: str, extra: dict[str, Any] | None = None) -> str:
        prepared, _ = PipelineWs.prepare_config(
            {"nodes": [{"type": "upscale", "options": {"model": value, **(extra or {})}}]},
            "/data",
        )
        return prepared[0]["options"]["model"]

    checks.eq("a bare model name waits for its download", model_of("4x_fake"), "4x_fake")
    checks.eq("a mounted path is left alone", model_of("/mnt/models/4x_a.pth"), "/mnt/models/4x_a.pth")
    checks.eq("a relative model path lands under root", model_of("weights/4x_a.pth"), "/data/weights/4x_a.pth")
    checks.eq("…also without a slash", model_of("4x_a.pth"), "/data/4x_a.pth")
    checks.eq("a model url is not a path", model_of("https://cdn.example/4x_a.pth"), "https://cdn.example/4x_a.pth")
    checks.eq("nothing is invented without root", PipelineWs.prepare_config({"nodes": [{"type": "upscale", "options": {"model": "weights/4x_a.pth"}}]}, None)[0][0]["options"]["model"], "weights/4x_a.pth")

    checks.eq("wire type becomes a label", label_of("folder_reader"), "Folder Reader")
    checks.eq("single word types stay", label_of("upscale"), "Upscale")

    # -- per-image chain ------------------------------------------------
    steps, nodes = fake_steps([0.01, 0.06, 0.01])
    timings = [0.0] * len(steps)
    cancel = asyncio.Event()
    hot = _run_chain(steps, object(), timings, cancel)
    checks.eq("every node saw the image", [node.calls for node in nodes], [1, 1, 1])
    checks.eq("the slowest node is reported", hot.label, "Fake 1")
    checks.true("timings are measured", all(timing > 0 for timing in timings))

    cancel.set()
    hot = _run_chain(steps, object(), timings, cancel)
    checks.eq("a cancelled chain stops before the first node", sum(node.calls for node in nodes), 3)
    checks.eq("…and reports nothing", hot, None)

    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
