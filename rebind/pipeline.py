from reline.nodes import (
    FileReaderNode,
    FolderReaderNode,
    FolderWriterNode,
    FileWriterNode,
)
from reline.pipeline import Pipeline
from starlette.websockets import WebSocket
import asyncio


class PipelineWs(Pipeline):
    @classmethod
    def from_json(cls, data: dict) -> "PipelineWs":
        base = super().from_json(data)
        instance = cls.__new__(cls)
        instance.__dict__ = base.__dict__
        return instance

    async def process_linear_ws(
        self, ws: WebSocket, cancel_event: asyncio.Event
    ) -> None:
        data = []
        data_len = 0

        async def send_status(status: str, progress: int) -> None:
            await ws.send_json(
                {
                    "status": status,
                    "progress": progress,
                    "data_len": data_len,
                }
            )

        nodes_index = 0

        while nodes_index < len(self.nodes):
            if cancel_event.is_set():
                await send_status("cancelled", 0)
                return

            node = self.nodes[nodes_index]

            if isinstance(node, (FileReaderNode, FolderReaderNode)):
                data = node.single_process(data)
                data_len = len(data)
                writer_index: int | None = None
                for i, n in enumerate(
                    self.nodes[nodes_index + 1 :], start=nodes_index + 1
                ):
                    if isinstance(n, (FolderWriterNode, FileWriterNode)):
                        writer_index = i
                        break

                for img_index, img in enumerate(data):
                    if cancel_event.is_set():
                        await send_status("cancelled", img_index)
                        return

                    if img is None:
                        continue

                    await send_status("running", img_index)
                    await asyncio.sleep(0)

                    for inner_index in range(nodes_index + 1, len(self.nodes)):
                        if cancel_event.is_set():
                            await send_status("cancelled", img_index)
                            return

                        inner_node = self.nodes[inner_index]
                        img = inner_node.single_process(img)

                        if isinstance(inner_node, (FolderWriterNode, FileWriterNode)):
                            break
                nodes_index = (
                    (writer_index + 1) if writer_index is not None else len(self.nodes)
                )
            else:
                nodes_index += 1

        del data
        await send_status("done", data_len)
