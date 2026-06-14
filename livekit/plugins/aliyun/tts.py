import os
from dataclasses import dataclass
from typing import AsyncIterable, Optional, Dict
import time
import aiohttp
import asyncio
import json
import uuid

from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, utils
from osc_data.text_stream import TextStreamSentencizer

from .log import logger

STREAM_EOS = "EOS"


@dataclass
class TTSOptions:
    api_key: str
    model: str
    rate: float
    voice: str
    speech_rate: int
    volume: int
    sample_rate: int
    pitch: float = 1.0

    def get_ws_url(self) -> str:
        return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    def get_ws_header(self) -> Dict[str, str]:
        return {
            "Authorization": f"bearer {self.api_key}",
            "X-DashScope-DataInspection": "enable",
        }

    def _task_header(self, action: str, task_id: str) -> Dict[str, str]:
        return {
            "action": action,
            "task_id": task_id,
            "streaming": "duplex",
        }

    def get_run_task_params(self, task_id: str) -> Dict[str, object]:
        params = {
            "header": self._task_header("run-task", task_id),
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": self.model,
                "parameters": {
                    "text_type": "PlainText",
                    "voice": self.voice,
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "volume": self.volume,
                    "rate": self.rate,
                    "pitch": self.pitch,
                },
                "input": {},
            },
        }
        return params

    def get_continue_task_params(self, task_id: str, text: str) -> Dict[str, object]:
        params = {
            "header": self._task_header("continue-task", task_id),
            "payload": {
                "input": {
                    "text": text,
                }
            },
        }
        return params

    def get_finish_task_params(self, task_id: str) -> Dict[str, object]:
        params = {
            "header": self._task_header("finish-task", task_id),
            "payload": {"input": {}},
        }
        return params


class TTS(tts.TTS):
    def __init__(
            self,
            *,
            api_key: Optional[str] = None,
            sample_rate: int = 24000,
            voice: str = "longcheng",
            model: str = "cosyvoice-v2",
            speech_rate: int = 1,
            volume: int = 100,
            rate: float = 1.0,
            pitch: float = 1.0,
            http_session: aiohttp.ClientSession | None = None,
            max_session_duration: float = 600,  
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")
        self._session = http_session
        self._opts = TTSOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            speech_rate=speech_rate,
            volume=volume,
            sample_rate=sample_rate,
            rate=rate,
            pitch=pitch,
        )
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=max_session_duration,
            mark_refreshed_on_get=True,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url()
        headers = self._opts.get_ws_header()
        return await asyncio.wait_for(
            session.ws_connect(
                url,
                headers=headers,
                autoping=True,
                heartbeat=15.0
            ),
            timeout=timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def synthesize(
            self,
            text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        raise NotImplementedError

    def stream(
            self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
            self,
            *,
            tts: TTS,
            opts: TTSOptions,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _acquire_ws(self) -> aiohttp.ClientWebSocketResponse:
        last_error = None
        for attempt in range(2):
            ws = await self._tts._pool.get(timeout=self._conn_options.timeout)
            if not ws.closed:
                if attempt > 0:
                    logger.warning("recovered from stale pooled tts websocket")
                return ws

            last_error = RuntimeError("WebSocket connection was already closed")
            logger.warning(
                "discarding stale pooled tts websocket",
                extra={"attempt": attempt + 1},
            )
            self._tts._pool.remove(ws)

        raise last_error or RuntimeError("Unable to acquire a valid TTS websocket")

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        task_id = uuid.uuid4().hex
        emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            mime_type="audio/pcm",
            stream=True,
            num_channels=1,
            frame_size_ms=200,
        )

        # 🔴 [修复点] 必须在 push 之前调用 start_segment
        emitter.start_segment(segment_id=utils.shortuuid())

        ws: aiohttp.ClientWebSocketResponse | None = None
        reuse_ws = False
        try:
            # 1. 整个流只获取一次连接；如果池中连接已经被服务端关闭，则丢弃并重连一次
            ws = await self._acquire_ws()

            # 2. 一次完整的流式会话，只发送一次 run-task
            run_task_params = self._opts.get_run_task_params(task_id=task_id)
            logger.info(
                "tts starting task",
                extra={"task_id": task_id, "model": self._opts.model, "voice": self._opts.voice},
            )
            await ws.send_json(run_task_params)

            start_time = time.perf_counter()
            task_started = asyncio.Event()
            finish_sent = asyncio.Event()
            task_finished = asyncio.Event()
            sent_any_text = False

            # 发送任务：不断将 LLM 产出的文本发送给阿里云
            async def _send_task():
                splitter = TextStreamSentencizer(remove_emoji=True)

                async def _send_sentence(sentence: str) -> None:
                    nonlocal sent_any_text

                    cleaned_sentence = "".join(char for char in sentence if char.isalnum())
                    if not cleaned_sentence:
                        return

                    sent_any_text = True
                    logger.info(
                        "tts sending sentence",
                        extra={"task_id": task_id, "sentence": sentence},
                    )
                    await ws.send_json(
                        self._opts.get_continue_task_params(task_id=task_id, text=sentence)
                    )

                try:
                    await asyncio.wait_for(task_started.wait(), timeout=10.0)

                    async for token in self._input_ch:
                        if isinstance(token, self._FlushSentinel):
                            sentences = splitter.flush()
                        else:
                            sentences = splitter.push(text=token)

                        for sentence in sentences:
                            await _send_sentence(sentence)

                    for sentence in splitter.flush():
                        await _send_sentence(sentence)

                    if not sent_any_text:
                        logger.info(
                            "tts stream finished without valid text, sending finish-task to close gracefully",
                            extra={"task_id": task_id},
                        )
                        # 发送 finish-task 让服务端正常关闭，而非直接断开 websocket
                        # 避免服务端返回 task-failed 错误
                        try:
                            if not ws.closed:
                                await ws.send_json(self._opts.get_finish_task_params(task_id=task_id))
                        except Exception:
                            pass
                        finish_sent.set()
                        return

                    logger.info("llm output finished, sending finish-task", extra={"task_id": task_id})
                    await ws.send_json(self._opts.get_finish_task_params(task_id=task_id))
                    finish_sent.set()
                except asyncio.CancelledError:
                    logger.debug("send_task cancelled")
                    return
                except asyncio.TimeoutError as e:
                    logger.error("tts task-started timeout", extra={"task_id": task_id})
                    raise Exception("TTS task-started timeout") from e
                except Exception as e:
                    logger.error(f"Error while sending tts text: {e}")
                    raise e

            # 接收任务：持续接收阿里云返回的音频片段
            async def _recv_task():
                is_first_response = True
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.receive(), timeout=30.0)

                        if msg.type == aiohttp.WSMsgType.BINARY:
                            if is_first_response:
                                elapsed = time.perf_counter() - start_time
                                logger.info(
                                    "tts first response",
                                    extra={"task_id": task_id, "spent": round(elapsed, 4)},
                                )
                                is_first_response = False
                            emitter.push(data=msg.data)
                            continue

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            msg_json = json.loads(msg.data)
                            header = msg_json.get("header", {})
                            event = header.get("event")
                            recv_task_id = header.get("task_id")

                            if recv_task_id and recv_task_id != task_id:
                                logger.warning(
                                    "ignoring unexpected tts event task_id",
                                    extra={
                                        "expected_task_id": task_id,
                                        "received_task_id": recv_task_id,
                                        "event": event,
                                    },
                                )
                                continue

                            if event == "task-started":
                                logger.info("tts task started", extra={"task_id": task_id})
                                task_started.set()
                                continue

                            if event == "result-generated":
                                continue

                            if event == "task-finished":
                                request_uuid = header.get("attributes", {}).get("request_uuid")
                                logger.info(
                                    "tts task finished successfully",
                                    extra={"task_id": task_id, "request_uuid": request_uuid},
                                )
                                task_finished.set()
                                break
                            if event == "task-failed":
                                error_code = header.get("error_code", "")
                                error_message = header.get("error_message", "")
                                # InvalidParameter 通常是空文本/无效文本导致，无需抛异常，优雅结束即可
                                if error_code == "InvalidParameter":
                                    logger.warning(
                                        "tts task failed due to invalid/empty text, ending synthesis gracefully",
                                        extra={
                                            "task_id": task_id,
                                            "error_code": error_code,
                                            "error_message": error_message,
                                        },
                                    )
                                    task_finished.set()
                                    break
                                logger.error(
                                    "tts task failed event received",
                                    extra={
                                        "task_id": task_id,
                                        "error_code": error_code,
                                        "error_message": error_message,
                                        "request_uuid": header.get("attributes", {}).get("request_uuid"),
                                    },
                                )
                                raise Exception(f"TTS task failed: {msg_json}")
                            continue

                        if msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            if finish_sent.is_set() or task_finished.is_set() or ws.closed:
                                logger.info("tts websocket closed after synthesis finished")
                                break
                            raise Exception(f"WebSocket closed unexpectedly: {msg.type}")

                        if msg.type == aiohttp.WSMsgType.ERROR:
                            if finish_sent.is_set() or task_finished.is_set():
                                logger.info("tts websocket errored after synthesis finished")
                                break
                            raise Exception(f"WebSocket error: {ws.exception()}")

                    return
                except asyncio.TimeoutError:
                    logger.error("tts receive timeout")
                    raise Exception("TTS task timeout")
                except asyncio.CancelledError:
                    logger.debug("recv_task cancelled")
                    return
                except Exception as e:
                    logger.error(f"tts receive error: {e}")
                    raise e

            tasks = [
                asyncio.create_task(_send_task()),
                asyncio.create_task(_recv_task()),
            ]

            try:
                await asyncio.gather(*tasks)
                reuse_ws = task_finished.is_set() and not ws.closed
            except asyncio.CancelledError:
                logger.warning("tts synthesis cancelled (user interrupted), closing connection.")
                if not ws.closed:
                    await ws.close()
                raise
            except Exception:
                reuse_ws = False
                raise
            finally:
                await utils.aio.gracefully_cancel(*tasks)

        finally:
            if ws is not None:
                if reuse_ws and not ws.closed:
                    self._tts._pool.put(ws)
                else:
                    self._tts._pool.remove(ws)
            # 🔴 这里对应的 end_segment() 前提是前面已经 start_segment()
            emitter.end_segment()
