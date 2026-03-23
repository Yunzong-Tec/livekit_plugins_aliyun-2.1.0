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
    # 语速，取值范围：0.5~2。
    rate: float
    # 音色
    voice: str
    # 合成音频的语速，取值范围：0.5~2。
    speech_rate: int
    # 合成音频的音量，取值范围：0~100。
    volume: int
    # 采样率，取值范围：8000, 16000, 22050, 24000, 44100, 48000
    sample_rate: int
    # 音调，取值范围：0.5~2。
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

        async def _send_task(sentence: str, ws: aiohttp.ClientWebSocketResponse):
            try:
                run_task_params = self._opts.get_run_task_params()
                await ws.send_json(run_task_params)
                continue_task_params = self._opts.get_continue_task_params(text=sentence)
                await ws.send_json(continue_task_params)
                finish_task_params = self._opts.get_finish_task_params()
                await ws.send_json(finish_task_params)
            except Exception as e:
                logger.error(f"Error while sending tts task: {e}")
                if not ws.closed:
                    await ws.close()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            is_first_response = True
            start_time = time.perf_counter()
            while True:
                try:
                    # 增加超时限制，防止阿里云服务挂起导致卡死进程
                    msg = await asyncio.wait_for(ws.receive(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.error("tts task timeout: Aliyun server did not respond in 15 seconds")
                    if not ws.closed:
                        await ws.close()
                    raise Exception("TTS task timeout: Aliyun server did not respond in 15 seconds")
                except Exception as e:
                    logger.warning(f"Error while receiving bytes: {e}")
                    if not ws.closed:
                        await ws.close()
                    raise

                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING):
                    logger.warning(f"WebSocket closed unexpectedly with type: {msg.type}")
                    raise Exception(f"WebSocket closed unexpectedly with type: {msg.type}")

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if is_first_response:
                        elapsed_time = time.perf_counter() - start_time
                        logger.info(
                            "tts first response",
                            extra={"spent": round(elapsed_time, 4)},
                        )
                        is_first_response = False
                    emitter.push(data=msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        msg_json = json.loads(msg.data)
                        if "header" in msg_json:
                            header = msg_json["header"]
                            if "event" in header:
                                event = header["event"]
                                if event == "task-finished":
                                    break
                                if event == "task-failed":
                                    logger.error(f"tts task failed: {msg_json}")
                                    if not ws.closed:
                                        await ws.close()
                                    raise Exception(f"TTS task failed: {msg_json}")
                    except Exception as e:
                        logger.error(f"Failed to parse json msg: {e}")
                        if not ws.closed:
                            await ws.close()
                        raise

        splitter = TextStreamSentencizer(remove_emoji=True)
        is_first_sentence = True
        start_time = time.perf_counter()
        emitter.start_segment(segment_id=utils.shortuuid())
        
        try:
            async for token in self._input_ch:
                if isinstance(token, self._FlushSentinel):
                    sentences = splitter.flush()
                else:
                    sentences = splitter.push(text=token)
                for sentence in sentences:
                    # 过滤掉仅包含空格或标点符号的无效文本，防止触发 Aliyun TTS InvalidParameter 报错
                    cleaned_sentence = "".join(char for char in sentence if char.isalnum())
                    if not cleaned_sentence:
                        continue
                    if is_first_sentence:
                        first_sentence_spend = time.perf_counter() - start_time
                        logger.info(
                            "llm first sentence",
                            extra={"spent": str(first_sentence_spend)},
                        )
                        is_first_sentence = False
                    logger.info("tts start", extra={"sentence": sentence})
                    
                    async with self._tts._pool.connection(
                        timeout=self._conn_options.timeout
                    ) as ws:
                        # 检查连接是否有效，如果已关闭则抛出异常触发连接池重建
                        if ws.closed:
                            logger.warning(f"WebSocket connection is closed, will reconnect for sentence: {sentence[:30]}...")
                            raise Exception("WebSocket connection was closed, triggering reconnection")
                        tasks = [
                            asyncio.create_task(_send_task(sentence=sentence, ws=ws)),
                            asyncio.create_task(_recv_task(ws=ws)),
                        ]
                        try:
                            # 增加整体超时控制，防止任何意外导致的死锁
                            await asyncio.wait_for(asyncio.gather(*tasks), timeout=60.0)
                            
                            # 如果子任务内部遇到错误通过 break 退出，而没有抛出异常，
                            # wait_for/gather 会认为任务已正常完成，从而使得连接在 ws.closed=True 的状态下被连接池回收。
                            # 必须在这里主动检测闭合状态并抛出异常，触发 ConnectionPool 回收该死连接。
                            if ws.closed:
                                raise Exception("WebSocket was closed unexpectedly during synthesis tasks")
                        except asyncio.TimeoutError as e:
                            logger.error(f"tts synthesis timeout for sentence: {sentence}")
                            if not ws.closed:
                                await ws.close()
                            raise e
                        except asyncio.CancelledError:
                            logger.warning(f"tts synthesis cancelled (user interrupted), closing connection to prevent dirty state.")
                            if not ws.closed:
                                await ws.close()
                            raise
                        except Exception as e:
                            logger.error(f"tts synthesis failed: {e}")
                            if not ws.closed:
                                await ws.close()
                            raise e
                        finally:
                            logger.info("tts end", extra={"sentence": sentence})
                            await utils.aio.gracefully_cancel(*tasks)
        finally:
            emitter.end_segment()
