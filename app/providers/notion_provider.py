# app/providers/notion_provider.py
import json
import time
import logging
import uuid
import re
import asyncio
import cloudscraper
from typing import Dict, Any, AsyncGenerator, List, Optional, Tuple
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.providers.base_provider import BaseProvider
from app.utils.sse_utils import create_sse_data, create_chat_completion_chunk, DONE_CHUNK

# 设置日志记录器
logger = logging.getLogger(__name__)


class _IncrementalCleaner:
    """流式内容清洗器，实时过滤 <thinking> 等内部推理标签。"""

    SUPPRESSED_TAGS = frozenset({
        'thinking', 'thought', 'reflection',
        'internal_monologue', 'reasoning', 'analysis', 'scratchpad',
        'inner_monologue', 'meta', 'plan', 'internal', 'response_planning',
    })

    def __init__(self):
        self._buffer = ""
        self._suppressing = False
        self._suppress_tag = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        return self._process()

    def _process(self) -> str:
        output_parts = []

        while self._buffer:
            if self._suppressing:
                close_tag = f'</{self._suppress_tag}>'
                idx = self._buffer.lower().find(close_tag.lower())
                if idx >= 0:
                    self._buffer = self._buffer[idx + len(close_tag):]
                    self._suppressing = False
                    self._buffer = self._buffer.lstrip('\n\r')
                else:
                    self._buffer = ""
                    break
            else:
                lang_match = re.search(r'<lang\s+primary="[^"]*"\s*/>\n*', self._buffer)
                if lang_match:
                    output_parts.append(self._buffer[:lang_match.start()])
                    self._buffer = self._buffer[lang_match.end():]
                    continue

                earliest_match = None
                earliest_tag = ""
                for tag in self.SUPPRESSED_TAGS:
                    m = re.search(f'<{tag}>', self._buffer, re.IGNORECASE)
                    if m and (earliest_match is None or m.start() < earliest_match.start()):
                        earliest_match = m
                        earliest_tag = tag

                if earliest_match:
                    output_parts.append(self._buffer[:earliest_match.start()])
                    self._suppressing = True
                    self._suppress_tag = earliest_tag
                    self._buffer = self._buffer[earliest_match.end():]
                else:
                    partial = re.search(r'<[a-zA-Z_][^>]*$', self._buffer)
                    if partial:
                        output_parts.append(self._buffer[:partial.start()])
                        self._buffer = self._buffer[partial.start():]
                        break
                    else:
                        output_parts.append(self._buffer)
                        self._buffer = ""

        return "".join(output_parts)

    def flush(self) -> str:
        if self._suppressing:
            return ""
        result = self._buffer
        self._buffer = ""
        return result


class NotionAIProvider(BaseProvider):
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.api_endpoints = {
            "runInference": "https://www.notion.so/api/v3/runInferenceTranscript",
            "saveTransactions": "https://www.notion.so/api/v3/saveTransactionsFanout"
        }
        
        if not all([settings.NOTION_COOKIE, settings.NOTION_SPACE_ID, settings.NOTION_USER_ID]):
            raise ValueError("配置错误: NOTION_COOKIE, NOTION_SPACE_ID 和 NOTION_USER_ID 必须在 .env 文件中全部设置。")

        self._warmup_session()

    def _warmup_session(self):
        try:
            logger.info("正在进行会话预热 (Session Warm-up)...")
            headers = self._prepare_headers()
            headers.pop("Accept", None)
            response = self.scraper.get("https://www.notion.so/", headers=headers, timeout=30)
            response.raise_for_status()
            logger.info("会话预热成功。")
        except Exception as e:
            logger.error(f"会话预热失败: {e}", exc_info=True)

    def _create_request_scraper(self):
        scraper = cloudscraper.create_scraper()
        scraper.cookies.update(self.scraper.cookies)
        return scraper
            
    async def _create_thread(self, thread_type: str, scraper=None) -> str:
        thread_id = str(uuid.uuid4())
        request_scraper = scraper or self.scraper
        payload = {
            "requestId": str(uuid.uuid4()),
            "transactions": [{
                "id": str(uuid.uuid4()),
                "spaceId": settings.NOTION_SPACE_ID,
                "operations": [{
                    "pointer": {"table": "thread", "id": thread_id, "spaceId": settings.NOTION_SPACE_ID},
                    "path": [],
                    "command": "set",
                    "args": {
                        "id": thread_id, "version": 1, "parent_id": settings.NOTION_SPACE_ID,
                        "parent_table": "space", "space_id": settings.NOTION_SPACE_ID,
                        "created_time": int(time.time() * 1000),
                        "created_by_id": settings.NOTION_USER_ID, "created_by_table": "notion_user",
                        "messages": [], "data": {}, "alive": True, "type": thread_type
                    }
                }]
            }]
        }
        try:
            logger.info(f"正在创建新的对话线程 (type: {thread_type})...")
            response = await run_in_threadpool(
                lambda: request_scraper.post(
                    self.api_endpoints["saveTransactions"],
                    headers=self._prepare_headers(),
                    json=payload,
                    timeout=20
                )
            )
            response.raise_for_status()
            logger.info(f"对话线程创建成功, Thread ID: {thread_id}")
            return thread_id
        except Exception as e:
            logger.error(f"创建对话线程失败: {e}", exc_info=True)
            raise Exception("无法创建新的对话线程。")

    async def chat_completion(self, request_data: Dict[str, Any]):
        stream = request_data.get("stream", True)

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            request_id = f"chatcmpl-{uuid.uuid4()}"
            final_message: Optional[str] = None

            try:
                model_name = request_data.get("model", settings.DEFAULT_MODEL)
                mapped_model = settings.MODEL_MAP.get(model_name, "anthropic-sonnet-alt")

                thread_type = "markdown-chat" if mapped_model.startswith("vertex-") else "workflow"

                request_scraper = self._create_request_scraper()
                thread_id = await self._create_thread(thread_type, scraper=request_scraper)
                payload = self._prepare_payload(request_data, thread_id, mapped_model, thread_type)
                headers = self._prepare_headers()

                role_chunk = create_chat_completion_chunk(request_id, model_name, role="assistant")
                yield create_sse_data(role_chunk)

                cleaner = _IncrementalCleaner()
                PREFIX_MAX = 8000
                prefix_parts: List[str] = []
                prefix_len = 0
                prefix_flushed = False

                def sync_stream_iterator():
                    try:
                        logger.info(f"请求 Notion AI URL: {self.api_endpoints['runInference']}")
                        logger.info(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")

                        response = request_scraper.post(
                            self.api_endpoints['runInference'], headers=headers, json=payload, stream=True,
                            timeout=settings.API_REQUEST_TIMEOUT
                        )
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                yield line
                    except Exception as e:
                        yield e

                sync_gen = sync_stream_iterator()
                line_queue: asyncio.Queue = asyncio.Queue()

                async def _bg_reader():
                    try:
                        while True:
                            item = await run_in_threadpool(lambda: next(sync_gen, None))
                            await line_queue.put(item)
                            if item is None:
                                break
                    except Exception as e:
                        await line_queue.put(e)
                        await line_queue.put(None)

                reader_task = asyncio.create_task(_bg_reader())

                try:
                    while True:
                        try:
                            line = await asyncio.wait_for(line_queue.get(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.debug("发送心跳保活")
                            yield b": heartbeat\n\n"
                            continue

                        if line is None:
                            break
                        if isinstance(line, Exception):
                            raise line

                        parsed_results = self._parse_ndjson_line_to_texts(line)
                        for text_type, content in parsed_results:
                            if text_type == 'incremental':
                                if not prefix_flushed:
                                    prefix_parts.append(content)
                                    prefix_len += len(content)
                                    if prefix_len >= PREFIX_MAX:
                                        cleaned_prefix = self._clean_content("".join(prefix_parts))
                                        if cleaned_prefix:
                                            chunk = create_chat_completion_chunk(request_id, model_name, content=cleaned_prefix)
                                            yield create_sse_data(chunk)
                                        prefix_flushed = True
                                else:
                                    cleaned = cleaner.feed(content)
                                    if cleaned:
                                        chunk = create_chat_completion_chunk(request_id, model_name, content=cleaned)
                                        yield create_sse_data(chunk)
                            elif text_type == 'final':
                                final_message = content
                finally:
                    if not reader_task.done():
                        reader_task.cancel()
                        try:
                            await reader_task
                        except asyncio.CancelledError:
                            pass

                if not prefix_flushed:
                    text = "".join(prefix_parts) if prefix_parts else (final_message or "")
                    if text:
                        cleaned_response = self._clean_content(text)
                        if cleaned_response:
                            chunk = create_chat_completion_chunk(request_id, model_name, content=cleaned_response)
                            yield create_sse_data(chunk)
                        else:
                            fallback = re.sub(r'<(thinking|thought|reflection|reasoning|analysis|scratchpad|internal_monologue)>[\s\S]*?</\1>', '', text, flags=re.IGNORECASE).strip()
                            fallback = self._extract_after_eq_delimiter(fallback)
                            fallback = self._strip_leading_reasoning(fallback)
                            fallback = fallback.strip()
                            if fallback:
                                chunk = create_chat_completion_chunk(request_id, model_name, content=fallback)
                                yield create_sse_data(chunk)
                    elif not final_message:
                        logger.warning("警告: Notion 返回的数据流中未提取到任何有效文本。")
                else:
                    remaining = cleaner.flush()
                    if remaining:
                        chunk = create_chat_completion_chunk(request_id, model_name, content=remaining)
                        yield create_sse_data(chunk)

                final_chunk = create_chat_completion_chunk(request_id, model_name, finish_reason="stop")
                yield create_sse_data(final_chunk)
                yield DONE_CHUNK

            except Exception as e:
                error_message = f"处理 Notion AI 流时发生意外错误: {str(e)}"
                logger.error(error_message, exc_info=True)
                error_chunk = {"error": {"message": error_message, "type": "internal_server_error"}}
                yield create_sse_data(error_chunk)
                yield DONE_CHUNK

        if stream:
            headers = {
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)
        else:
            raise HTTPException(status_code=400, detail="此端点当前仅支持流式响应 (stream=true)。")

    def _prepare_headers(self) -> Dict[str, str]:
        cookie_source = (settings.NOTION_COOKIE or "").strip()
        cookie_header = cookie_source if "=" in cookie_source else f"token_v2={cookie_source}"

        return {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "Cookie": cookie_header,
            "x-notion-space-id": settings.NOTION_SPACE_ID,
            "x-notion-active-user-header": settings.NOTION_USER_ID,
            "x-notion-client-version": settings.NOTION_CLIENT_VERSION,
            "notion-audit-log-platform": "web",
            "Origin": "https://www.notion.so",
            "Referer": "https://www.notion.so/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }

    def _normalize_block_id(self, block_id: str) -> str:
        if not block_id: return block_id
        b = block_id.replace("-", "").strip()
        if len(b) == 32 and re.fullmatch(r"[0-9a-fA-F]{32}", b):
            return f"{b[0:8]}-{b[8:12]}-{b[12:16]}-{b[16:20]}-{b[20:]}"
        return block_id

    def _prepare_payload(self, request_data: Dict[str, Any], thread_id: str, mapped_model: str, thread_type: str) -> Dict[str, Any]:
        req_block_id = request_data.get("notion_block_id") or settings.NOTION_BLOCK_ID
        normalized_block_id = self._normalize_block_id(req_block_id) if req_block_id else None

        context_value: Dict[str, Any] = {
            "timezone": "Asia/Shanghai",
            "spaceId": settings.NOTION_SPACE_ID,
            "userId": settings.NOTION_USER_ID,
            "userEmail": settings.NOTION_USER_EMAIL,
            "currentDatetime": datetime.now().astimezone().isoformat(),
        }
        if normalized_block_id:
            context_value["blockId"] = normalized_block_id

        config_value: Dict[str, Any]
        
        if mapped_model.startswith("vertex-"):
            logger.info(f"检测到 Gemini 模型 ({mapped_model})，应用特定的 config 和 context。")
            context_value.update({
                "userName": f" {settings.NOTION_USER_NAME}",
                "spaceName": f"{settings.NOTION_USER_NAME}的 Notion",
                "spaceViewId": "2008eefa-d0dc-80d5-9e67-000623befd8f",
                "surface": "ai_module"
            })
            config_value = {
                "type": thread_type,
                "model": mapped_model,
                "useWebSearch": True,
                "enableAgentAutomations": False, "enableAgentIntegrations": False,
                "enableBackgroundAgents": False, "enableCodegenIntegration": False,
                "enableCustomAgents": False, "enableExperimentalIntegrations": False,
                "enableLinkedDatabases": False, "enableAgentViewVersionHistoryTool": False,
                "searchScopes": [{"type": "everything"}], "enableDatabaseAgents": False,
                "enableAgentComments": False, "enableAgentForms": False,
                "enableAgentMakesFormulas": False, "enableUserSessionContext": False,
                "modelFromUser": True, "isCustomAgent": False
            }
        else:
            context_value.update({
                "userName": settings.NOTION_USER_NAME,
                "surface": "workflows"
            })
            config_value = {
                "type": thread_type,
                "model": mapped_model,
                "useWebSearch": True,
            }

        transcript = [
            {"id": str(uuid.uuid4()), "type": "config", "value": config_value},
            {"id": str(uuid.uuid4()), "type": "context", "value": context_value}
        ]
      
        for msg in request_data.get("messages", []):
            if msg.get("role") == "user":
                transcript.append({
                    "id": str(uuid.uuid4()),
                    "type": "user",
                    "value": [[msg.get("content")]],
                    "userId": settings.NOTION_USER_ID,
                    "createdAt": datetime.now().astimezone().isoformat()
                })
            elif msg.get("role") == "assistant":
                transcript.append({"id": str(uuid.uuid4()), "type": "agent-inference", "value": [{"type": "text", "content": msg.get("content")}]})

        payload = {
            "traceId": str(uuid.uuid4()),
            "spaceId": settings.NOTION_SPACE_ID,
            "transcript": transcript,
            "threadId": thread_id,
            "createThread": False,
            "isPartialTranscript": True,
            "asPatchResponse": True,
            "generateTitle": True,
            "saveAllThreadOperations": True,
            "threadType": thread_type
        }

        if mapped_model.startswith("vertex-"):
            logger.info("为 Gemini 请求添加 debugOverrides。")
            payload["debugOverrides"] = {
                "emitAgentSearchExtractedResults": True,
                "cachedInferences": {},
                "annotationInferences": {},
                "emitInferences": False
            }
        
        return payload

    def _extract_after_eq_delimiter(self, content: str) -> str:
        """处理 ="[语言代码] 分隔符，提取分隔符后的实际回复内容。
        模型有时会输出: [推理文本]="zh-[实际回复] 或 [推理文本]="[实际回复]
        """
        match = re.search(r'="(?:[a-z]{2}(?:-[a-zA-Z]{2,8})?)?[-\s]*(?=\S)', content)
        if not match:
            return content

        before = content[:match.start()]
        after = content[match.end():].strip()

        if len(before) < 50 or len(after) < 2:
            return content

        lower = before.lower()
        indicators = [
            'user', 'message', 'should', 'respond', 'context',
            'chinese', 'incomplete', 'unclear', 'search', 'reply',
            'appears', 'looking', 'according', 'instructions',
            'need to', "i should", "i don't", 'this is',
            'the character', 'conversation', 'timezone',
            'acknowledge', 'clarification', 'brief', 'i\'ll',
            'given the', 'based on', 'since ',
        ]
        if sum(1 for ind in indicators if ind in lower) >= 2:
            return after

        return content

    def _strip_leading_reasoning(self, content: str) -> str:
        """去除开头的英文推理文本块，只保留实际回复内容。"""
        if not content:
            return content

        first_non_space = content.lstrip()
        if first_non_space and '\u4e00' <= first_non_space[0] <= '\u9fff':
            return content

        lower_start = content[:500].lower().strip()
        reasoning_starters = [
            'user\'s message', 'the user', 'user message', 'users message',
            'looking at', 'given the context', 'given that',
            'this is a', 'this appears', 'this seems', 'this is very',
            'the message', 'the character', 'the word', 'the text',
            'i should', 'i need to', 'i will', 'i don\'t', 'i\'ll',
            'let me', 'according to', 'based on',
            'since this', 'since the', 'since we',
            'sent "', 'sent \'', 'sent \u201c',
        ]

        starts_with_reasoning = False
        for starter in reasoning_starters:
            if lower_start.startswith(starter):
                starts_with_reasoning = True
                break

        if not starts_with_reasoning and re.match(r'^["\u201c\u300c\u2018]', lower_start):
            starts_with_reasoning = True

        if not starts_with_reasoning:
            return content

        lines = content.split('\n')
        total_reasoning_chars = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            chinese_chars = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
            total_chars = max(len(stripped), 1)

            if chinese_chars > 3 and chinese_chars / total_chars > 0.3 and total_reasoning_chars > 80:
                return '\n'.join(lines[i:]).strip()

            if chinese_chars / total_chars < 0.3:
                total_reasoning_chars += len(stripped)

        return content

    def _clean_content(self, content: str) -> str:
        """清洗模型输出，移除推理过程、思考标签等，只保留实际回复内容。"""
        if not content:
            return ""

        content = re.sub(r'<lang\s+primary="[^"]*"\s*/>\n*', '', content)

        for tag in ['thinking', 'thought', 'reflection', 'internal_monologue',
                    'reasoning', 'analysis', 'scratchpad', 'inner_monologue',
                    'meta', 'plan', 'internal', 'response_planning']:
            content = re.sub(rf'<{tag}(?:\s[^>]*)?>[\s\S]*?</{tag}>\s*', '', content, flags=re.IGNORECASE)

        content = self._extract_after_eq_delimiter(content)

        content = self._strip_leading_reasoning(content)

        reasoning_patterns = [
            r'(?:The\s+)?user[\'\'"]?s?\s+message\s+(?:is|was|says?|reads?)\s+[^\n]*\.\s*',
            r'(?:The\s+)?user\s+(?:has\s+)?(?:asked|is\s+asking|wants|sent|said)\s+[^\n]*\.\s*',
            r'(?:I\s+)?(?:should|need\s+to|will|must)\s+(?:respond|reply|answer|acknowledge)[^\n]*\.\s*',
            r'(?:Since|Because|As)\s+[^\n]*?(?:Chinese|request|message|query|question)[^\n]*\.\s*',
            r'This\s+is\s+(?:a\s+)?(?:straightforward|simple|basic|very\s+brief|unclear|minimal|incomplete|short)[^\n]*\.\s*',
            r'I\s+(?:don\'t|do\s+not)\s+need\s+to\s+(?:use|search|look)[^\n]*\.\s*',
            r'(?:I\s+)?should\s+(?:identify|not\s+make|not\s+reveal|just|probably|NOT)[^\n]*\.\s*',
            r'(?:This|It)\s+(?:is|requires|appears|seems|looks|could)[^\n]*?(?:response|answer|reply|message)[^\n]*\.\s*',
            r'(?:Let\s+me\s+)?respond\s+(?:directly\s+)?(?:to\s+)?[^\n]*\.\s*',
            r'(?:The|Their|Its|His|Her)\s+(?:name|timezone|current|previous|message|language)[^\n]*\n',
            r'(?:Given|Looking\s+at)\s+(?:the\s+)?(?:context|conversation|history)[^\n]*:?\s*\n',
            r'According\s+to\s+(?:my\s+)?(?:instructions|guidelines|rules)[^\n]*\.\s*',
            r'(?:An?\s+)?(?:incomplete|unclear|brief|minimal|ambiguous)\s+message\s*\n?',
            r'A\s+typo\s*\n?',
            r'Testing\s+if\s+[^\n]*\n?',
            r'The\s+beginning\s+of\s+[^\n]*\n?',
            r'(?:First|Second|Third|Fourth)\s+message:?\s+[^\n]*\n',
            r'The\s+(?:character|word|phrase|text)\s+["\u201c][^\n]*\.\s*',
            r'(?:I\s+)?(?:responded|asked|told|said|replied)\s+(?:in\s+)?(?:Chinese)?[^\n]*\n',
            r'The\s+system\s+also\s+called\s+[^\n]*\n',
            r'I\'m\s+not\s+sure\s+why\s+[^\n]*\n',
            r'(?:They|He|She)\s+(?:could|might|may)\s+be\s+[^\n]*\n?',
            r'(?:They\'re|He\'s|She\'s)\s+(?:saying|asking|testing|checking)[^\n]*\n?',
        ]

        for pattern in reasoning_patterns:
            content = re.sub(r'^[\s]*' + pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)

        chinese_reasoning_patterns = [
            r'用?户(?:说|问)了?\s*[「"\'"\s].*?[。.]\s*',
            r'(?:意思是|也就是说)[「"\'"]?.*?[。.]\s*',
            r'(?:但是?|不过)用户(?:之前|之后|还没|并没|只是)[^。\n]*[。.]\s*',
            r'现在用户(?:要求|想要|需要|希望)[^。\n]*[。.]\s*',
            r'我(?:应该|需要|不需要)[^。\n]*?(?:回[答复]|告诉|响应|使用|输出|直接)[^。\n]*[。.]\s*',
            r'根据(?:语言指南|系统提示|上述指令|指示)[，,]?[^。\n]*[。.]?\s*',
            r'这是一个(?:关于|简单|直接|基本|一般性)[^。\n]*?(?:的问题|的请求|的查询)[^。\n]*[。.]\s*',
            r'(?:我)?不需要使用任何(?:工具|tools?)[^。\n]*[。.]\s*',
            r'(?:我可以|这(?:是|个)我?可以?)直接(?:回答|回复)[^。\n]*[。.]\s*',
            r'[，,]\s*(?:并|然后)(?:询问|回[答复])(?:他们?|用户)[^。\n]*[。.]\s*',
        ]

        for pattern in chinese_reasoning_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE)

        content = re.sub(r'\s*primary\s*', '', content)
        content = re.sub(r'^\s*\n', '', content)

        return content.strip()

    def _parse_ndjson_line_to_texts(self, line: bytes) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        try:
            s = line.decode("utf-8", errors="ignore").strip()
            if not s: return results
            
            data = json.loads(s)
            logger.debug(f"原始响应数据: {json.dumps(data, ensure_ascii=False)}")
            
            # 格式1: Gemini 返回的 markdown-chat 事件
            if data.get("type") == "markdown-chat":
                content = data.get("value", "")
                if content:
                    logger.info("从 'markdown-chat' 直接事件中提取到内容。")
                    results.append(('final', content))

            # 格式2: Claude 和 GPT 返回的补丁流，以及 Gemini 的 patch 格式
            elif data.get("type") == "patch" and "v" in data:
                for operation in data.get("v", []):
                    if not isinstance(operation, dict): continue
                    
                    op_type = operation.get("o")
                    path = operation.get("p", "")
                    value = operation.get("v")
                    
                    # 【修改】Gemini 的完整内容 patch 格式
                    if op_type == "a" and path.endswith("/s/-") and isinstance(value, dict) and value.get("type") == "markdown-chat":
                        content = value.get("value", "")
                        if content:
                            logger.info("从 'patch' (Gemini-style) 中提取到完整内容。")
                            results.append(('final', content))
                    
                    # 【修改】Gemini 的增量内容 patch 格式
                    elif op_type == "x" and "/s/" in path and path.endswith("/value") and isinstance(value, str):
                        content = value
                        if content:
                            logger.info(f"从 'patch' (Gemini增量) 中提取到内容: {content}")
                            results.append(('incremental', content))
                    
                    # 【修改】Claude 和 GPT 的增量内容 patch 格式
                    elif op_type == "x" and "/value/" in path and isinstance(value, str):
                        content = value
                        if content:
                            logger.info(f"从 'patch' (Claude/GPT增量) 中提取到内容: {content}")
                            results.append(('incremental', content))
                    
                    # 【修改】Claude 和 GPT 的完整内容 patch 格式
                    elif op_type == "a" and path.endswith("/value/-") and isinstance(value, dict) and value.get("type") == "text":
                        content = value.get("content", "")
                        if content:
                            logger.info("从 'patch' (Claude/GPT-style) 中提取到完整内容。")
                            results.append(('final', content))

            # 格式3: 处理record-map类型的数据（取最后一条有效AI消息）
            elif data.get("type") == "record-map" and "recordMap" in data:
                record_map = data["recordMap"]
                if "thread_message" in record_map:
                    last_content = None
                    last_step_type = None
                    for msg_id, msg_data in record_map["thread_message"].items():
                        value_data = msg_data.get("value", {}).get("value", {})
                        step = value_data.get("step", {})
                        if not step:
                            continue

                        content = ""
                        step_type = step.get("type")

                        if step_type == "markdown-chat":
                            content = step.get("value", "")
                        elif step_type == "agent-inference":
                            agent_values = step.get("value", [])
                            if isinstance(agent_values, list):
                                for item in agent_values:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        content = item.get("content", "")
                                        break

                        if content and isinstance(content, str):
                            last_content = content
                            last_step_type = step_type

                    if last_content:
                        logger.info(f"从 record-map (type: {last_step_type}) 提取到最终内容。")
                        results.append(('final', last_content))
    
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"解析NDJSON行失败: {e} - Line: {line.decode('utf-8', errors='ignore')}")
        
        return results

    async def get_models(self) -> JSONResponse:
        model_data = {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": int(time.time()), "owned_by": "lzA6"}
                for name in settings.KNOWN_MODELS
            ]
        }
        return JSONResponse(content=model_data)
