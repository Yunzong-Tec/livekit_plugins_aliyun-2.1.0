import os
from dataclasses import dataclass
from typing import AsyncIterable, Optional, Dict
import time
import aiohttp
import asyncio
import json

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

    def get_run_task_params(self) -> Dict[str, str]:
        params = {
            "header": {
                "action": "run-task",
                "task_id": utils.shortuuid(),
                "streaming": "duplex",
            },
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

    def get_continue_task_params(self, text: str) -> Dict[str, str]:
        params = {
            "header": {
                "action": "continue-task",
                "task_id": utils.shortuuid(),
                "streaming": "duplex",
            },
            "payload": {
                "input": {
                    "text": text,
                }
            },
        }
        return params

    def get_finish_task_params(self) -> Dict[str, str]:
        params = {
            "header": {
                "action": "finish-task",
                "task_id": utils.shortuuid(),
                "streaming": "duplex",
            },
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

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
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

        try:
            # 1. 整个流只获取一次连接
            async with self._tts._pool.connection(
                    timeout=self._conn_options.timeout
            ) as ws:
                if ws.closed:
                    raise Exception("WebSocket connection was closed")

                # 2. 一次完整的流式会话，只发送一次 run-task
                run_task_params = self._opts.get_run_task_params()
                await ws.send_json(run_task_params)

                start_time = time.perf_counter()

                # 发送任务：不断将 LLM 产出的文本发送给阿里云
                async def _send_task():
                    splitter = TextStreamSentencizer(remove_emoji=True)
                    try:
                        async for token in self._input_ch:
                            if isinstance(token, self._FlushSentinel):
                                sentences = splitter.flush()
                            else:
                                sentences = splitter.push(text=token)

                            for sentence in sentences:
                                cleaned_sentence = "".join(char for char in sentence if char.isalnum())
                                if not cleaned_sentence:
                                    continue

                                logger.info("tts sending sentence", extra={"sentence": sentence})
                                # 持续发送文本
                                await ws.send_json(self._opts.get_continue_task_params(text=sentence))

                        # 文本全部生成完毕，发送 finish-task
                        logger.info("llm output finished, sending finish-task")
                        await ws.send_json(self._opts.get_finish_task_params())
                    except asyncio.CancelledError:
                        logger.debug("send_task cancelled")
                        return
                    except Exception as e:
                        logger.error(f"Error while sending tts text: {e}")
                        raise e

                # 接收任务：持续接收阿里云返回的音频片段
                async def _recv_task():
                    is_first_response = True
                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.receive(), timeout=15.0)

                            if msg.type in (
                            aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING):
                                raise Exception(f"WebSocket closed unexpectedly: {msg.type}")

                            if msg.type == aiohttp.WSMsgType.BINARY:
                                if is_first_response:
                                    elapsed = time.perf_counter() - start_time
                                    logger.info("tts first response", extra={"spent": round(elapsed, 4)})
                                    is_first_response = False
                                # 这里如果之前没调用 start_segment，就会报错 RuntimeError
                                emitter.push(data=msg.data)

                            elif msg.type == aiohttp.WSMsgType.TEXT:
                                msg_json = json.loads(msg.data)
                                header = msg_json.get("header", {})
                                event = header.get("event")

                                if event == "task-finished":
                                    logger.info("tts task finished successfully")
                                    break  # 整个流式任务顺利结束
                                elif event == "task-failed":
                                    raise Exception(f"TTS task failed: {msg_json}")

                    except asyncio.TimeoutError:
                        logger.error("tts receive timeout")
                        raise Exception("TTS task timeout")
                    except asyncio.CancelledError:
                        logger.debug("recv_task cancelled")
                        return
                    except Exception as e:
                        logger.error(f"tts receive error: {e}")
                        raise e

                # 并发执行收发任务
                tasks = [
                    asyncio.create_task(_send_task()),
                    asyncio.create_task(_recv_task()),
                ]

                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    # 3. 处理用户打断 (LiveKit VAD 检测到用户说话)
                    logger.warning("tts synthesis cancelled (user interrupted), closing connection.")
                    # 强制关闭 websocket 可以切断阿里云那边的生成，防止继续扣费和产生音频
                    if not ws.closed:
                        await ws.close()
                    raise
                finally:
                    await utils.aio.gracefully_cancel(*tasks)

        finally:
            # 🔴 这里对应的 end_segment() 前提是前面已经 start_segment()
            emitter.end_segment()