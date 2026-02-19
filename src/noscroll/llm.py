"""LLM API module with serial queue enabled by default for rate limiting."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any
from asyncio import AbstractEventLoop
from datetime import datetime, timezone

import httpx


# Language code to name mapping
_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "tr": "Turkish (Türkçe)",
    "vi": "Vietnamese (Tiếng Việt)",
    "th": "Thai (ไทย)",
    "id": "Indonesian (Bahasa Indonesia)",
}


def _get_language_name(code: str) -> str:
    """Get language name from ISO 639-1 code."""
    return _LANGUAGE_NAMES.get(code.lower(), code)


def _apply_top_n_to_prompt(system_prompt: str, top_n: int) -> str:
    """
    Apply top_n limit to system prompt by replacing placeholders.
    
    Placeholders in system prompt:
    - {{TOTAL_ITEMS}}: Total number of items (e.g., "7 items")
    - {{AI_COUNT}}: AI category count (e.g., "3 items")
    - {{OTHER_COUNT}}: Other News category count (e.g., "2 items")
    - {{LIFE_COUNT}}: Life & Health category count (e.g., "2 items")
    - {{DISTRIBUTION}}: Distribution string (e.g., "AI 3 + Other News 2 + Life & Health 2")
    
    Distribution ratio: AI ~43%, Other ~29%, Life ~29%
    """
    if top_n <= 0:
        return system_prompt
    
    def plural(n: int) -> str:
        """Return 'item' or 'items' based on count."""
        return "item" if n == 1 else "items"
    
    # Calculate proportional distribution
    # Default: AI 3/7 ≈ 43%, Other 2/7 ≈ 29%, Life 2/7 ≈ 29%
    if top_n >= 3:
        ai_count = max(1, round(top_n * 3 / 7))
        remaining = top_n - ai_count
        other_count = max(1, remaining // 2)
        life_count = max(1, remaining - other_count)
        # Adjust if total doesn't match
        total = ai_count + other_count + life_count
        if total > top_n:
            if ai_count > 1:
                ai_count -= (total - top_n)
            else:
                other_count -= (total - top_n)
        elif total < top_n:
            ai_count += (top_n - total)
    elif top_n == 2:
        ai_count, other_count, life_count = 1, 1, 0
    else:  # top_n == 1
        ai_count, other_count, life_count = 1, 0, 0
    
    total = ai_count + other_count + life_count
    
    # Replace placeholders
    prompt = system_prompt
    prompt = prompt.replace("{{TOTAL_ITEMS}}", f"{total} {plural(total)}")
    prompt = prompt.replace("{{AI_COUNT}}", f"{ai_count} {plural(ai_count)}")
    prompt = prompt.replace("{{OTHER_COUNT}}", f"{other_count} {plural(other_count)}")
    prompt = prompt.replace("{{LIFE_COUNT}}", f"{life_count} {plural(life_count)}")
    prompt = prompt.replace("{{DISTRIBUTION}}", f"AI {ai_count} + Other News {other_count} + Life & Health {life_count}")
    
    # Handle fallback rules - if a category has 0 items, adjust the fallback message
    import re
    if life_count == 0:
        prompt = re.sub(
            r'- If Life & Health < 2:.*?explain the reason\.',
            '- Life & Health section is not required for this output.',
            prompt,
            flags=re.DOTALL
        )
    
    if other_count == 0:
        prompt = re.sub(
            r'- If Other News < 2:.*?explain the reason\.',
            '- Other News section is not required for this output.',
            prompt,
            flags=re.DOTALL
        )
    
    # Add explicit instruction at the end
    top_n_instruction = f"\n\n**CRITICAL: Your output MUST contain exactly {total} {plural(total)} total ({ai_count} AI + {other_count} Other News + {life_count} Life & Health). Do not output more or fewer items.**"
    prompt = prompt + top_n_instruction
    
    return prompt


@dataclass
class LLMRequest:
    """A single LLM request."""
    api_url: str
    api_key: str
    model: str
    mode: str
    system_prompt: str
    user_prompt: str
    timeout_ms: int
    log_path: Optional[str] = None
    tag: str = ""  # Optional tag for identifying request source (rss, web, hn)


@dataclass
class LLMResponse:
    """Result of an LLM request."""
    content: str
    elapsed_ms: int
    success: bool
    error: Optional[str] = None
    tag: str = ""


class LLMClient:
    """
    LLM client with optional serial queue for rate limiting.
    
    Usage:
        # Parallel mode (default)
        client = LLMClient()
        result = await client.call(request)
        
        # Serial mode with delay
        client = LLMClient(serial=True, delay_ms=1000)
        result = await client.call(request)
        
        # Or use the global client
        set_llm_client(LLMClient(serial=True, delay_ms=500))
        result = await call_llm_queued(...)
    """
    
    def __init__(
        self,
        serial: bool = True,
        delay_ms: int = 0,
        lang: str = "en",
        top_n: int = 0,
        on_request: Optional[Callable[[LLMRequest], None]] = None,
        on_response: Optional[Callable[[LLMResponse], None]] = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            serial: If True, requests are processed one at a time (default: True)
            delay_ms: Delay in milliseconds between requests (only in serial mode)
            lang: Output language code (e.g., 'en', 'zh', 'ja'). Default: 'en'
            top_n: Keep only top N most important items. 0 = no limit. Default: 0
            on_request: Callback before each request
            on_response: Callback after each response
        """
        self.serial = serial
        self.delay_ms = delay_ms
        self.lang = lang
        self.top_n = top_n
        self.on_request = on_request
        self.on_response = on_response
        
        # Queue for serial mode (bound lazily to current event loop)
        self._queue: Optional[asyncio.Queue[tuple[LLMRequest, asyncio.Future]]] = None
        self._queue_loop: Optional[AbstractEventLoop] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._last_request_time: float = 0
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_elapsed_ms": 0,
        }
    
    @property
    def stats(self) -> dict:
        """Get request statistics."""
        return self._stats.copy()
    
    async def _ensure_worker(self) -> None:
        """Ensure the worker task is running in serial mode."""
        if not self.serial:
            return

        current_loop = asyncio.get_running_loop()
        
        async with self._lock:
            # Re-bind queue to current event loop when needed
            if self._queue is None or self._queue_loop is not current_loop:
                if self._worker_task and not self._worker_task.done():
                    self._worker_task.cancel()
                self._queue = asyncio.Queue()
                self._queue_loop = current_loop

            if self._worker_task is None or self._worker_task.done():
                self._worker_task = asyncio.create_task(self._worker())
    
    async def _worker(self) -> None:
        """Worker that processes requests serially."""
        queue = self._queue
        if queue is None:
            return

        while True:
            try:
                request, future = await asyncio.wait_for(
                    queue.get(), 
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                # No requests for 60 seconds, exit worker
                break
            
            try:
                # Apply delay if needed
                if self.delay_ms > 0:
                    elapsed_since_last = (time.time() - self._last_request_time) * 1000
                    if elapsed_since_last < self.delay_ms:
                        await asyncio.sleep((self.delay_ms - elapsed_since_last) / 1000)
                
                # Execute request
                response = await self._execute_request(request)
                self._last_request_time = time.time()
                
                if not future.done():
                    future.set_result(response)
                    
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                queue.task_done()
    
    async def _execute_request(self, request: LLMRequest) -> LLMResponse:
        """Execute a single LLM request."""
        self._stats["total_requests"] += 1
        
        if self.on_request:
            self.on_request(request)
        
        start = time.time()
        try:
            content = await _call_llm_internal(
                api_url=request.api_url,
                api_key=request.api_key,
                model=request.model,
                mode=request.mode,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                timeout_ms=request.timeout_ms,
                log_path=request.log_path,
            )
            elapsed_ms = int((time.time() - start) * 1000)
            
            self._stats["successful_requests"] += 1
            self._stats["total_elapsed_ms"] += elapsed_ms
            
            response = LLMResponse(
                content=content,
                elapsed_ms=elapsed_ms,
                success=True,
                tag=request.tag,
            )
            
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            self._stats["failed_requests"] += 1
            self._stats["total_elapsed_ms"] += elapsed_ms
            
            response = LLMResponse(
                content="",
                elapsed_ms=elapsed_ms,
                success=False,
                error=str(e),
                tag=request.tag,
            )
        
        if self.on_response:
            self.on_response(response)
        
        return response
    
    async def call(self, request: LLMRequest) -> LLMResponse:
        """
        Call LLM API. In serial mode, request is queued.
        
        Args:
            request: The LLM request to execute
            
        Returns:
            LLMResponse with result or error
        """
        if self.serial:
            await self._ensure_worker()
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            if self._queue is None:
                raise RuntimeError("Serial queue is not initialized")
            await self._queue.put((request, future))
            return await future
        else:
            return await self._execute_request(request)
    
    async def call_and_raise(self, request: LLMRequest) -> str:
        """
        Call LLM API and raise exception on error.
        
        Args:
            request: The LLM request to execute
            
        Returns:
            Response content string
            
        Raises:
            RuntimeError: If request fails
        """
        response = await self.call(request)
        if not response.success:
            raise RuntimeError(response.error or "LLM request failed")
        return response.content
    
    async def shutdown(self) -> None:
        """Shutdown the client and wait for pending requests."""
        if self._queue is not None:
            await self._queue.join()

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass


# Global client instance
_global_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client, creating one if needed."""
    global _global_client
    if _global_client is None:
        _global_client = LLMClient()
    return _global_client


def set_llm_client(client: LLMClient) -> None:
    """Set the global LLM client."""
    global _global_client
    _global_client = client


def configure_llm_client(
    serial: bool = True,
    delay_ms: int = 0,
    lang: str = "en",
    top_n: int = 0,
    on_request: Optional[Callable[[LLMRequest], None]] = None,
    on_response: Optional[Callable[[LLMResponse], None]] = None,
) -> LLMClient:
    """
    Configure and set the global LLM client.
    
    Args:
        serial: If True, requests are processed one at a time (default: True)
        delay_ms: Delay in milliseconds between requests
        lang: Output language code (e.g., 'en', 'zh', 'ja'). Default: 'en'
        top_n: Keep only top N most important items. 0 = no limit. Default: 0
        on_request: Callback before each request
        on_response: Callback after each response
        
    Returns:
        The configured client
    """
    client = LLMClient(
        serial=serial,
        delay_ms=delay_ms,
        lang=lang,
        top_n=top_n,
        on_request=on_request,
        on_response=on_response,
    )
    set_llm_client(client)
    return client


def resolve_api_url(api_url: str, mode: str) -> str:
    """Resolve API URL based on mode."""
    if not api_url:
        return api_url

    base = api_url.rstrip("/")
    looks_like_base = base.endswith("/v1")

    if looks_like_base:
        if mode == "completions":
            return f"{base}/completions"
        elif mode == "responses":
            return f"{base}/responses"
        else:
            return f"{base}/chat/completions"

    if mode == "completions" and "/chat/completions" in base:
        return base.replace("/chat/completions", "/completions")
    if mode == "responses" and "/chat/completions" in base:
        return base.replace("/chat/completions", "/responses")
    if mode == "chat" and base.endswith("/completions"):
        return base.replace("/completions", "/chat/completions")
    if mode == "responses" and base.endswith("/completions"):
        return base.replace("/completions", "/responses")

    return base


def _append_log(log_path: Optional[str], entry: dict) -> None:
    """Append log entry to file."""
    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if entry.get("type") == "request":
        if path.exists() and path.stat().st_size > 0:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _truncate_text(text: str | None, limit: int) -> str:
    """Truncate text to limit."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


async def _call_llm_internal(
    api_url: str,
    api_key: str,
    model: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    timeout_ms: int,
    log_path: Optional[str] = None,
) -> str:
    """Internal LLM API call implementation."""
    if not api_url or not api_key or not model:
        raise ValueError(
            "LLM config missing: LLM_API_URL, LLM_API_KEY, LLM_MODEL are required."
        )

    # Infer API mode from URL if not specified
    if "/responses" in api_url:
        inferred_mode = "responses"
    elif "/chat/completions" in api_url:
        inferred_mode = "chat"
    elif api_url.endswith("/completions"):
        inferred_mode = "completions"
    else:
        inferred_mode = "responses"

    api_mode = mode if mode in ("completions", "chat", "responses") else inferred_mode
    resolved_url = resolve_api_url(api_url, api_mode)
    start = time.time()
    timeout_sec = timeout_ms / 1000

    # Build request body based on mode
    if api_mode == "completions":
        body = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "temperature": 0.2,
        }
    elif api_mode == "responses":
        body = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
        }
    else:  # chat
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

    _append_log(
        log_path,
        {
            "type": "request",
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": api_mode,
            "model": model,
            "url": resolved_url,
            "body": body,
            "body_json": json.dumps(body),
        },
    )

    # trust_env=True makes httpx use HTTP_PROXY, HTTPS_PROXY, ALL_PROXY env vars
    async with httpx.AsyncClient(timeout=timeout_sec, trust_env=True) as client:
        response = await client.post(
            resolved_url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=body,
        )
        response_text = response.text
        elapsed_ms = int((time.time() - start) * 1000)

        if not response.is_success:
            _append_log(
                log_path,
                {
                    "type": "error",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "mode": api_mode,
                    "model": model,
                    "url": resolved_url,
                    "status": response.status_code,
                    "raw_response": response_text,
                },
            )
            raise RuntimeError(
                f"LLM request failed ({response.status_code}) "
                f"[mode={api_mode}, model={model}, url={resolved_url}]: {response_text}"
            )

        _append_log(
            log_path,
            {
                "type": "response_raw",
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": api_mode,
                "model": model,
                "url": resolved_url,
                "status": response.status_code,
                "raw_response": response_text,
                "elapsed_ms": elapsed_ms,
            },
        )

        try:
            data = response.json()
        except json.JSONDecodeError as err:
            _append_log(
                log_path,
                {
                    "type": "response_parse_error",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "mode": api_mode,
                    "model": model,
                    "url": resolved_url,
                    "status": response.status_code,
                    "error": str(err),
                    "raw_response": response_text,
                },
            )
            raise RuntimeError(f"LLM response parse error: {err}")

        # Extract content based on mode
        if api_mode == "completions":
            choices = data.get("choices", [])
            content = choices[0].get("text", "") if choices else ""
        elif api_mode == "responses":
            # Try output_text first
            output_text = data.get("output_text")
            if isinstance(output_text, str) and output_text:
                content = output_text
            else:
                # Parse output array
                content = ""
                for output in data.get("output", []):
                    if not output or not isinstance(output.get("content"), list):
                        continue
                    for block in output["content"]:
                        if block and isinstance(block.get("text"), str):
                            content += block["text"]
        else:  # chat
            choices = data.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content", "")

        _append_log(
            log_path,
            {
                "type": "response",
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": api_mode,
                "model": model,
                "url": resolved_url,
                "status": 200,
                "elapsed_ms": elapsed_ms,
                "output_excerpt": _truncate_text(content, 2000),
                "output_length": len(content),
            },
        )

        return content


async def call_llm(
    api_url: str,
    api_key: str,
    model: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    timeout_ms: int,
    log_path: Optional[str] = None,
    tag: str = "",
) -> str:
    """
    Call LLM API and return response text.
    
    This is the main entry point for LLM calls. If a global LLMClient is configured
    with serial=True, requests will be queued and processed one at a time.
    
    Args:
        api_url: LLM API URL
        api_key: API key
        model: Model name
        mode: API mode (chat, completions, responses)
        system_prompt: System prompt
        user_prompt: User prompt
        timeout_ms: Timeout in milliseconds
        log_path: Optional path for logging
        tag: Optional tag for identifying request source (rss, web, hn)
        
    Returns:
        Response content string
    """
    client = get_llm_client()
    
    # Inject language instruction for configured language (including English)
    final_system_prompt = system_prompt
    if client.lang:
        lang_name = _get_language_name(client.lang)
        lang_instruction = f"\n\n**IMPORTANT: You MUST write your entire response in {lang_name}.**"
        final_system_prompt = system_prompt + lang_instruction
    
    # Inject top_n instruction if specified
    if client.top_n and client.top_n > 0:
        final_system_prompt = _apply_top_n_to_prompt(final_system_prompt, client.top_n)
    
    request = LLMRequest(
        api_url=api_url,
        api_key=api_key,
        model=model,
        mode=mode,
        system_prompt=final_system_prompt,
        user_prompt=user_prompt,
        timeout_ms=timeout_ms,
        log_path=log_path,
        tag=tag,
    )
    
    return await client.call_and_raise(request)


async def call_llm_queued(
    api_url: str,
    api_key: str,
    model: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    timeout_ms: int,
    log_path: Optional[str] = None,
    tag: str = "",
) -> LLMResponse:
    """
    Call LLM API through the global client, returning full response.
    
    Unlike call_llm(), this returns an LLMResponse object which includes
    success status and error information without raising exceptions.
    
    Args:
        api_url: LLM API URL
        api_key: API key
        model: Model name
        mode: API mode (chat, completions, responses)
        system_prompt: System prompt
        user_prompt: User prompt
        timeout_ms: Timeout in milliseconds
        log_path: Optional path for logging
        tag: Optional tag for identifying request source (rss, web, hn)
        
    Returns:
        LLMResponse with content, success status, and error info
    """
    client = get_llm_client()
    
    request = LLMRequest(
        api_url=api_url,
        api_key=api_key,
        model=model,
        mode=mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_ms=timeout_ms,
        log_path=log_path,
        tag=tag,
    )
    
    return await client.call(request)