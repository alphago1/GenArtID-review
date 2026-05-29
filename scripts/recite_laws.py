import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional
from tqdm import tqdm
import httpx


def _now() -> float:
    return time.monotonic()


class TokenBucket:
    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self._capacity = float(capacity)
        self._refill_per_second = float(refill_per_second)
        self._tokens = float(capacity)
        self._updated_at = _now()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = _now()
        elapsed = max(0.0, now - self._updated_at)
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
        self._updated_at = now

    async def acquire(self, amount: float) -> None:
        amount = float(amount)
        if amount <= 0:
            return
        
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                missing = amount - self._tokens
                rate = self._refill_per_second
            
            if rate <= 0:
                await asyncio.sleep(1.0)
            else:
                wait_s = missing / rate
                wait_s = max(0.01, min(wait_s, 2.0))
                await asyncio.sleep(wait_s)

    async def refund(self, amount: float) -> None:
        amount = float(amount)
        if amount <= 0:
            return
        async with self._lock:
            self._refill()
            self._tokens = min(self._capacity, self._tokens + amount)


@dataclass(frozen=True)
class Limits:
    rpm: int
    tpm: int


class RateLimiter:
    def __init__(self, limits: Limits) -> None:
        self._req_bucket = TokenBucket(capacity=limits.rpm, refill_per_second=limits.rpm / 60.0)
        self._tok_bucket = TokenBucket(capacity=limits.tpm, refill_per_second=limits.tpm / 60.0)

    async def acquire(self, token_estimate: int) -> None:
        await self._req_bucket.acquire(1.0)
        await self._tok_bucket.acquire(float(token_estimate))

    async def refund_tokens(self, token_amount: int) -> None:
        await self._tok_bucket.refund(float(token_amount))


def estimate_tokens(text: str) -> int:
    if not text:
        return 1
    return max(1, len(text))


def load_done_articles(md_path: str) -> set[int]:
    if not os.path.exists(md_path):
        return set()
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return set()
    done: set[int] = set()
    for m in re.finditer(r"^##\s+第(\d+)条\s*$", content, flags=re.MULTILINE):
        try:
            done.add(int(m.group(1)))
        except Exception:
            continue
    return done


def build_messages(article_no: int, law_name: str) -> list[dict[str, str]]:
    system = (
        f"你现在要逐条背诵《{law_name}》。"
        "必须只输出指定条文的正文内容，不要输出标题、目录、注释、解释、来源或任何额外文字。"
        "尽量保持官方常见标点与换行风格。"
    )
    user = f"请背诵《{law_name}》第{article_no}条全文。只输出正文。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def post_chat_completions(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, Optional[int], dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "max_tokens": max_tokens,
    }
    if "flash" in model.lower() or "pro" in model.lower() or "r1" in model.lower():
        payload["thinking"] = {"type": "disabled"}
    r = await client.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    content = (
        (data.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    usage = data.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    try:
        total_tokens_int = int(total_tokens) if total_tokens is not None else None
    except Exception:
        total_tokens_int = None
    return content, total_tokens_int, data


async def fetch_article(
    limiter: RateLimiter,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: Optional[str],
    model: str,
    law_name: str,
    article_no: int,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
) -> str:
    messages = build_messages(article_no, law_name)
    prompt_text = "\n".join([m["content"] for m in messages])
    token_est = estimate_tokens(prompt_text) + max_tokens
    await limiter.acquire(token_est)

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            content, total_tokens, _raw = await post_chat_completions(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            if total_tokens is not None:
                delta = token_est - total_tokens
                if delta > 0:
                    await limiter.refund_tokens(delta)
            content = (content or "").strip()
            return content if content else "【缺失】"
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            body = ""
            try:
                body = e.response.text[:2000] if e.response is not None else ""
            except Exception:
                body = ""
            last_err = f"HTTP {status}: {body}".strip()
            if status in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_retries:
                backoff = min(30.0, (2 ** attempt) + random.random())
                await asyncio.sleep(backoff)
                continue
            return f"【请求失败】{last_err}"
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
            last_err = f"{type(e).__name__}: {str(e)}".strip()
            if attempt < max_retries:
                backoff = min(30.0, (2 ** attempt) + random.random())
                await asyncio.sleep(backoff)
                continue
            return f"【请求失败】{last_err}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)}".strip()
            return f"【请求失败】{last_err}"

    return f"【请求失败】{last_err or 'unknown'}"


def ensure_md_header(md_path: str, law_name: str, model_name: str) -> None:
    if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
        return
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"# {law_name}（{model_name} 背诵输出）\n\n")


async def run(args: argparse.Namespace) -> int:
    # 确定输出文件和参数
    law_name = args.law
    model_alias = args.model_alias
    
    # 默认文件名生成规则
    if not args.out:
        filename_law = {
            "中华人民共和国民法典": "civil_code",
            "中华人民共和国刑法": "criminal_law",
            "中华人民共和国教育法": "education_law",
            "工伤保险条例": "work_injury_laws",
            "中华人民共和国刑事诉讼法": "criminal_procedure",
        }.get(law_name, "unknown_law")
        
        filename_model = {
            "qwen": "recite",
            "deepseek": "deepseek",
            "deepseek-sf": "deepseek_sf",
            "deepseek-flash": "deepseek_flash",
        }.get(model_alias, "recite")
        
        args.out = f"{filename_law}_{filename_model}.md"

    # 设置不同模型的默认参数
    sf_key = os.environ.get("siliconflow_key", "") or os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("SILICON", "")
    if model_alias == "deepseek":
        if not args.base_url: args.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not args.api_key: args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not args.model: args.model = "deepseek-chat"
    elif model_alias == "qwen":
        if not args.base_url: args.base_url = "https://api.siliconflow.cn/v1"
        if not args.api_key: args.api_key = sf_key
        if not args.model: args.model = "Qwen/Qwen3-8B"
    elif model_alias == "deepseek-sf":
        if not args.base_url: args.base_url = "https://api.siliconflow.cn/v1"
        if not args.api_key: args.api_key = sf_key
        if not args.model: args.model = "deepseek-ai/DeepSeek-V3.2"
    elif model_alias == "deepseek-flash":
        if not args.base_url: args.base_url = "https://api.deepseek.com"
        if not args.api_key: args.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not args.model: args.model = "deepseek-v4-flash"

    if not args.api_key:
        print(f"❌ 错误: 未找到 {model_alias} 的 API Key，请检查环境变量或参数。")
        return 1

    out_path: str = os.path.abspath(args.out)
    ensure_md_header(out_path, law_name, args.model)

    done = load_done_articles(out_path)
    all_articles = list(range(1, args.total + 1))
    pending = [n for n in all_articles if n not in done]
    
    print(f"🚀 开始背诵任务: {law_name}")
    print(f"🤖 使用模型: {args.model}")
    print(f"📄 输出文件: {out_path}")
    print(f"📊 进度: 已完成 {len(done)}/{args.total} 条，剩余 {len(pending)} 条")

    if not pending:
        print("✅ 所有条目已背诵完成！")
        return 0

    limits = Limits(rpm=args.rpm, tpm=args.tpm)
    limiter = RateLimiter(limits)

    queue: asyncio.Queue[Optional[int]] = asyncio.Queue()
    for n in pending:
        queue.put_nowait(n)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    results: dict[int, str] = {}
    cv = asyncio.Condition()

    async def worker() -> None:
        async with httpx.AsyncClient() as client:
            while True:
                n = await queue.get()
                try:
                    if n is None:
                        return
                    try:
                        text = await fetch_article(
                            limiter=limiter,
                            client=client,
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            law_name=law_name,
                            article_no=n,
                            max_tokens=args.max_tokens,
                            timeout_s=args.timeout_s,
                            max_retries=args.max_retries,
                        )
                    except Exception as e:
                        text = f"【请求失败】{str(e)}"

                    async with cv:
                        results[n] = text
                        cv.notify_all()
                finally:
                    queue.task_done()

    async def writer() -> None:
        with tqdm(total=len(pending), desc=f"📜 {law_name}背诵中", unit="条", ncols=80) as pbar:
            with open(out_path, "a", encoding="utf-8") as f:
                for n in pending:
                    async with cv:
                        while n not in results:
                            await cv.wait()
                        text = results.pop(n)
                    
                    f.write(f"## 第{n}条\n\n")
                    f.write(text.rstrip() + "\n\n")
                    f.flush()
                    
                    pbar.update(1)

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    writer_task = asyncio.create_task(writer())
    await asyncio.gather(*workers)
    await writer_task
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="法律条文背诵工具 (Unified Version)")
    
    # 核心选择参数
    p.add_argument("--law", required=True, choices=["中华人民共和国民法典", "中华人民共和国刑法", "中华人民共和国教育法", "工伤保险条例", "中华人民共和国刑事诉讼法"], help="选择要背诵的法律名称")
    p.add_argument("--model-alias", required=True, choices=["qwen", "deepseek", "deepseek-sf", "deepseek-flash"], help="选择使用的模型别名 (qwen/deepseek/deepseek-sf/deepseek-flash)")
    
    # 自动推导但可覆盖的参数
    p.add_argument("--out", help="输出文件路径 (默认根据法律和模型自动生成)")
    p.add_argument("--total", type=int, default=0, help="总条数 (默认根据法律自动设定)")
    
    # 性能参数
    p.add_argument("--concurrency", type=int, default=10, help="并发请求数")
    p.add_argument("--rpm", type=int, default=1000, help="每分钟请求限制")
    p.add_argument("--tpm", type=int, default=50000, help="每分钟 Token 限制")
    
    # API 参数 (通常由 model-alias 自动设置，但可覆盖)
    p.add_argument("--model", help="具体模型名称 (覆盖默认值)")
    p.add_argument("--base-url", help="API Base URL (覆盖默认值)")
    p.add_argument("--api-key", help="API Key (覆盖默认值)")
    
    # 其他参数
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=5)
    
    args = p.parse_args()
    
    # 自动设置 total
    if args.total == 0:
        if args.law == "中华人民共和国民法典":
            args.total = 1260
        elif args.law == "中华人民共和国刑法":
            args.total = 452
        elif args.law == "中华人民共和国教育法":
            args.total = 86
        elif args.law == "工伤保险条例":
            args.total = 67
        elif args.law == "中华人民共和国刑事诉讼法":
            args.total = 307
            
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
