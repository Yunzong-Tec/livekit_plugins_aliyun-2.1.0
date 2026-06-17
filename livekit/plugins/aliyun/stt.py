from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List
import json

import asyncio
import aiohttp

from livekit import rtc
from livekit.agents import (
    stt,
    utils,
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    APIStatusError,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from .log import logger


@dataclass
class STTOptions:
    api_key: str | None
    language: str | None
    detect_language: bool
    interim_results: bool
    punctuate: bool
    model: str
    max_sentence_silence: int = 500
    sample_rate: int = 16000
    workspace: str | None = None

    # 增加热词表 提供热词识别
    # 参考url 创建 热词表
    # https://help.aliyun.com/zh/model-studio/custom-hot-words?spm=a2c4g.11186623.0.0.1a7c2fc2CeNIxu
    vocabulary_id: str | None = None
    # 过滤语气词
    disfluency_removal_enabled: bool = False
    # 设置是否开启语义断句，默认关闭。
    semantic_punctuation_enabled: bool = False
    # 设置是否开启标点预测，默认关闭。
    punctuation_prediction_enabled: bool = True
    # 设置是否开启文本逆归一化，默认关闭。
    inverse_text_normalization_enabled: bool = True

    def get_ws_url(self):
        return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    def get_header(self):
        header = {
            "Authorization": f"bearer {self.api_key}",
            "X-DashScope-DataInspection": "enable",
        }
        if self.workspace is not None:
            header["X-DashScope-WorkSpace"] = self.workspace
        return header

    def get_run_task_params(self, task_id: str):
        params = {
            "header": {
                "action": "run-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": self.model,
                "parameters": {
                    "format": "wav",
                    "sample_rate": self.sample_rate,
                    "vocabulary_id": self.vocabulary_id,
                    "disfluency_removal_enabled": self.disfluency_removal_enabled,
                    "semantic_punctuation_enabled": self.semantic_punctuation_enabled,
                    "punctuation_prediction_enabled": self.punctuation_prediction_enabled,
                    "inverse_text_normalization_enabled": self.inverse_text_normalization_enabled,
                    "max_sentence_silence": self.max_sentence_silence,
                    "heartbeat": True,
                    "language_hints": [self.language],
                },
                "input": {},
            },
        }
        return params

    def get_finish_task_params(self, task_id: str):
        params = {
            "header": {
                "action": "finish-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {"input": {}},
        }
        return params


class STT(stt.STT):
    def __init__(
        self,
        *,
        language="zh",
        detect_language: bool = False,
        interim_results: bool = True,
        punctuate: bool = True,
        model: str = "paraformer-realtime-v2",
        api_key: str | None = None,
        max_sentence_silence: int = 500,
        disfluency_removal_enabled: bool = False,
        semantic_punctuation_enabled: bool = False,
        punctuation_prediction_enabled: bool = True,
        inverse_text_normalization_enabled: bool = True,
        vocabulary_id: str | None = None,
        workspace: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=interim_results
            )
        )
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("DASHSCOPE API key is required")
        self._opts = STTOptions(
            api_key=api_key,
            language=language,
            detect_language=detect_language,
            interim_results=interim_results,
            punctuate=punctuate,
            model=model,
            max_sentence_silence=max_sentence_silence,
            disfluency_removal_enabled=disfluency_removal_enabled,
            semantic_punctuation_enabled=semantic_punctuation_enabled,
            punctuation_prediction_enabled=punctuation_prediction_enabled,
            inverse_text_normalization_enabled=inverse_text_normalization_enabled,
            vocabulary_id=vocabulary_id,
            workspace=workspace,
        )

        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("not implemented")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStream":
        return SpeechStream(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
            http_session=self._ensure_session(),
        )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)

        if opts.language is None:
            raise ValueError("language detection is not supported in streaming mode")
        self._opts: STTOptions = opts
        self._config = opts
        self._speaking = False
        self._closed = False
        self._request_id = utils.shortuuid()
        self._reconnect_event = asyncio.Event()
        self._session = http_session
        # 累计已发送给阿里云的音频时长（秒），作为 SpeechData.start_time/end_time
        # 的稳定时间源（见 _process_stream_event 中的说明）
        self._audio_elapsed_s: float = 0.0

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        ws = await asyncio.wait_for(
            self._session.ws_connect(
                self._opts.get_ws_url(), 
                headers=self._opts.get_header(),
                autoping=True,
                heartbeat=15.0
            ),
            self._conn_options.timeout,
        )
        logger.info("connected to stt websocket successfully")
        return ws

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse, task_id: str):
            nonlocal closing_ws

            # 计算 100ms 音频帧的采样数 (16000Hz 下是 1600 个采样)
            samples_100ms = self._opts.sample_rate // 10
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_100ms,
            )

            # 【新增】：计算 100ms 静音音频的字节数据 (16kHz, 16bit位深=2字节, 单声道)
            bytes_per_sample = 2
            silent_bytes = b'\x00' * (samples_100ms * bytes_per_sample)

            # 获取音频输入管道的异步迭代器
            iterator = self._input_ch.__aiter__()

            while True:
                try:
                    if closing_ws or ws.closed:
                        break

                    # 【核心修改】：使用 wait_for 设置 0.1 秒 (100ms) 超时时间
                    # 如果这 100ms 内用户说话了，就会正常读取到 data；如果没有，则抛出 TimeoutError
                    data = await asyncio.wait_for(iterator.__anext__(), timeout=0.1)

                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        # 将收到的音频数据写入缓冲流，生成固定时长的帧
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        # LiveKit flush 仅表示一句话结束，不应直接结束阿里云 task。
                        # 这里仅把 AudioByteStream 内部尚未对齐的尾帧推出，保持长连接识别任务继续存活。
                        frames.extend(audio_bstream.flush())

                    # 发送组装好的真实语音数据
                    for frame in frames:
                        await ws.send_bytes(frame.data.tobytes())
                        self._audio_elapsed_s += frame.samples_per_channel / frame.sample_rate

                except asyncio.TimeoutError:
                    # 【核心修改】：100ms 内没有收到 LiveKit 的音频（说明处于静默期或 VAD 截断）
                    # 主动向服务端发送提前准备好的静音空白帧，保持心跳与服务端长连接
                    if not closing_ws and not ws.closed:
                        await ws.send_bytes(silent_bytes)
                        self._audio_elapsed_s += samples_100ms / self._opts.sample_rate

                except StopAsyncIteration:
                    # input_ch 管道被关闭时（通常是 Livekit 断开或者调用了 aclose），
                    # 这里才真正结束阿里云 task，避免每次 utterance flush 都把整条识别任务结束掉。
                    try:
                        for frame in audio_bstream.flush():
                            await ws.send_bytes(frame.data.tobytes())
                        if not ws.closed:
                            await ws.send_json(self._opts.get_finish_task_params(task_id))
                    except Exception as e:
                        logger.warning(f"stt finish_task on close failed: {e}")
                    break
                except Exception as e:
                    logger.error(f"stt send_task error: {e}")
                    break

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            while True:
                try:
                    # 增加接收超时，防止底层 websocket 假死导致进程被看门狗杀掉 (10分钟超时防止正常静默被切断)
                    msg = await asyncio.wait_for(ws.receive(), timeout=600.0)
                except asyncio.TimeoutError:
                    if closing_ws:
                        return
                    logger.warning("stt websocket receive timeout, forcing reconnect")
                    if not ws.closed:
                        await ws.close()
                    raise APIStatusError(message="stt connection timeout")
                except Exception as e:
                    if closing_ws:
                        return
                    logger.warning(f"stt connection error: {e}")
                    if not ws.closed:
                        await ws.close()
                    raise APIStatusError(message=f"stt connection error: {e}")

                if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(message="connection closed unexpectedly")

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                task_id = utils.shortuuid()
                ws = await self._connect_ws()
                await ws.send_json(self._opts.get_run_task_params(task_id=task_id))
                tasks = [
                    asyncio.create_task(send_task(ws, task_id)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    closing_ws = True  # 通知子任务准备退出
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            except Exception as e:
                if closing_ws:
                    break
                logger.warning(f"stt connection error, reconnecting: {e}")
                await asyncio.sleep(1.0)
            finally:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass

    def _process_stream_event(self, data: dict) -> None:
        event_type = data["header"]["event"]
        if event_type == "result-generated":
            output = data["payload"]["output"]["sentence"]
            is_sentence_end = output["sentence_end"]
            # ⚠️ 时间戳处理（修复"语音时长/计费爆炸 + 识别失效"问题）：
            # 阿里云 begin_time/end_time 单位为毫秒，interim 阶段可能为 None，STT 重连后
            # 还会从 0 重新计数；而新版 livekit-agents 把 SpeechData.start_time/end_time
            # 当作"相对音频流起点的秒数"，与 _input_started_at(墙钟)相加得到
            # started/stopped_speaking_at。若传 0/None/毫秒，框架会把说话起点映射到
            # "音频流起点"(进程已跑十几小时 → 66574s 前)，导致语音时长与计费爆炸、
            # turn 无法完成、出现 "speech scheduling is paused"。
            # 这里统一使用累计音频时钟(秒)：它始终 ≈ now - _input_started_at，与框架期望
            # 严格对齐，且天然免疫 None / 单位 / 重连复位 三类问题。
            start_time = self._audio_elapsed_s
            end_time = self._audio_elapsed_s
            text = output["text"]
            if not self._speaking:
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)
                logger.info("transcription start")
                self._speaking = True
            if text and not is_sentence_end:
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)
            if text and is_sentence_end:
                alternatives = [
                    stt.SpeechData(
                        language=self._opts.language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ]
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=alternatives,
                )
                self._event_ch.send_nowait(interim_event)
                end_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH, request_id=self._request_id
                )
                self._event_ch.send_nowait(end_event)
                self._speaking = False
                logger.info(
                    "transcription end",
                    extra={
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                )


def live_transcription_to_speech_data(
    language: str,
    data,
) -> List[stt.SpeechData]:
    return [
        stt.SpeechData(
            language=language,
            start_time=data.get("begin_time") or 0,
            end_time=data.get("end_time") or 0,
            confidence=0.0,
            text=data["text"],
        )
    ]
