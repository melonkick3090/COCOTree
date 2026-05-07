from __future__ import annotations

import ast
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


class OpenRouterRetryableError(RuntimeError):
    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code


TOOL_TAG_PAIRS: List[Tuple[str, str]] = [
    ("<tool>", "</tool>"),
    ("<tool_call>", "</tool_call>"),
    ("<tool_code>", "</tool_code>"),
]

NODE_ID_RE = re.compile(r"^n\d{5}$")


@dataclass
class OpenRouterUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
        }


def dedupe_texts(xs: List[Any], max_items: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs or []:
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _normalize_tool_obj(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if "name" in obj and "parameters" in obj:
        return obj
    if "text_prompts" in obj or "items" in obj:
        return {"name": "propose_prompts", "parameters": obj}
    return None


def parse_llm_tool_call(response: str) -> Optional[Dict[str, Any]]:
    if not response:
        return None

    try:
        for start_tag, end_tag in TOOL_TAG_PAIRS:
            if start_tag not in response or end_tag not in response:
                continue

            matches = re.finditer(
                rf"{re.escape(start_tag)}\s*(.*?)\s*{re.escape(end_tag)}",
                response,
                flags=re.DOTALL,
            )

            for match in matches:
                tool_str = match.group(1).strip()
                tool_str = re.sub(r"^\s*```\s*(?:json)?\s*", "", tool_str, flags=re.IGNORECASE)
                tool_str = re.sub(r"\s*```\s*$", "", tool_str)

                candidates = [tool_str]
                if "{" in tool_str and "}" in tool_str:
                    candidates.append(tool_str[tool_str.find("{") : tool_str.rfind("}") + 1])

                last_err: Optional[Exception] = None
                for cand in candidates:
                    try:
                        obj = json.loads(cand)
                        norm = _normalize_tool_obj(obj)
                        if norm is not None:
                            return norm
                    except json.JSONDecodeError as exc:
                        last_err = exc

                try:
                    obj = ast.literal_eval(candidates[-1])
                    norm = _normalize_tool_obj(obj)
                    if norm is not None:
                        return norm
                except Exception as exc:  # pragma: no cover - best-effort parser
                    last_err = exc

                print(f"[OpenRouter] Tool parse error: {last_err}")
                print(f"[OpenRouter] Failed tool payload head: {tool_str[:300]}")
        return None
    except Exception as exc:  # pragma: no cover - best-effort parser
        print(f"[OpenRouter] Tool parse exception: {exc}")
        return None


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()

    if "```" in s:
        parts = s.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                return part
            if part.startswith("json"):
                cand = part[4:].strip()
                if cand.startswith("{") and cand.endswith("}"):
                    return cand

    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        return s[i : j + 1]
    return None


def safe_load_json_obj(text: str) -> Optional[Any]:
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("\t", " ").strip())
        except Exception:
            return None


def normalize_batch_mapping(obj: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(obj, dict):
        return out

    if "items" in obj and isinstance(obj["items"], list):
        for item in obj["items"]:
            if not isinstance(item, dict):
                continue
            nid = item.get("id")
            prompts = item.get("text_prompts", [])
            if not isinstance(nid, str) or not NODE_ID_RE.match(nid):
                continue
            if isinstance(prompts, str):
                out[nid] = [prompts]
            elif isinstance(prompts, list):
                out[nid] = [str(x) for x in prompts if x and str(x).strip()]
            else:
                out[nid] = []
        return out

    for key, value in obj.items():
        if not isinstance(key, str) or not NODE_ID_RE.match(key):
            continue
        if isinstance(value, str):
            out[key] = [value]
        elif isinstance(value, list):
            out[key] = [str(x) for x in value if x and str(x).strip()]
        else:
            out[key] = []
    return out


def parse_llm_batch_mapping(resp: str) -> Dict[str, List[str]]:
    if not resp:
        return {}

    tool = parse_llm_tool_call(resp)
    if tool is not None:
        if tool.get("name") != "propose_prompts":
            return {}
        params = tool.get("parameters", {}) or {}

        if isinstance(params.get("items"), list):
            return normalize_batch_mapping({"items": params["items"]}) or {}

        text_prompts = params.get("text_prompts")
        json_text = None
        if isinstance(text_prompts, str):
            json_text = text_prompts
        elif isinstance(text_prompts, list) and len(text_prompts) == 1 and isinstance(text_prompts[0], str):
            json_text = text_prompts[0]

        if json_text:
            obj = safe_load_json_obj(json_text)
            return normalize_batch_mapping(obj) or {}
        return {}

    block = extract_json_block(resp)
    if block:
        obj = safe_load_json_obj(block)
        return normalize_batch_mapping(obj) or {}
    return {}


def parse_llm_single_prompts(resp: str) -> List[str]:
    if not resp:
        return []

    tool = parse_llm_tool_call(resp)
    if tool is not None:
        if tool.get("name") != "propose_prompts":
            return []
        params = tool.get("parameters", {}) or {}
        prompts = params.get("text_prompts", [])
        if isinstance(prompts, str):
            return dedupe_texts([prompts])
        if isinstance(prompts, list):
            return dedupe_texts(prompts)
        return []

    block = extract_json_block(resp)
    if block:
        obj = safe_load_json_obj(block)
        if isinstance(obj, dict) and "text_prompts" in obj:
            prompts = obj.get("text_prompts", [])
            if isinstance(prompts, str):
                return dedupe_texts([prompts])
            if isinstance(prompts, list):
                return dedupe_texts(prompts)
        if isinstance(obj, list):
            return dedupe_texts(obj)
    return []


def send_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_sec: int = 300,
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
) -> Tuple[str, Dict[str, int]]:
    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            },
            timeout=int(timeout_sec),
        )

        if resp.status_code == 429 or (500 <= resp.status_code <= 599):
            raise OpenRouterRetryableError(resp.status_code, resp.text)

        if resp.status_code != 200:
            print(f"[OpenRouter] error {resp.status_code}: {resp.text}")
            return "", {}

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            err = data.get("error")
            raise OpenRouterRetryableError(200, f"200 but no choices | error={err}")

        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        usage = data.get("usage", {}) or {}
        usage_obj = OpenRouterUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
        return str(content), usage_obj.to_dict()
    except OpenRouterRetryableError:
        raise
    except requests.RequestException as exc:
        raise OpenRouterRetryableError(None, f"requests error: {exc}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[OpenRouter] exception (non-retry): {exc}")
        return "", {}


def send_openrouter_with_retry(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_retry: int = 3,
    tag: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_sec: int = 300,
) -> Tuple[str, Dict[str, int]]:
    max_retry = max(1, int(max_retry))
    for attempt in range(1, max_retry + 1):
        try:
            return send_openrouter(
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_sec=timeout_sec,
            )
        except OpenRouterRetryableError as exc:
            if attempt >= max_retry:
                print(
                    f"[OpenRouter] retry exhausted ({max_retry}) | {tag} | "
                    f"status={exc.status_code} | {exc}"
                )
                return "", {}
            wait = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            print(
                f"[OpenRouter] retry {attempt}/{max_retry} in {wait:.1f}s | "
                f"{tag} | status={exc.status_code}"
            )
            time.sleep(wait)
    return "", {}
