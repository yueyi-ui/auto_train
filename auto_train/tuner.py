"""LLM tuner: proposes the next hyperparameter config via an OpenAI-compatible API."""

import json
import time
import urllib.request


SYSTEM_PROMPT = """You are a senior engineer specializing in YOLO11 instance segmentation training.
You participate in an automated hyperparameter tuning loop. Given the dataset summary, the
experiment history, the allowed search space, and the current best metric, propose the next
hyperparameter configuration most likely to improve the target metric.

Rules:
- Return ONLY a JSON object, no markdown fences, no extra text.
- The JSON must contain exactly these keys: lr0, batch, epochs, imgsz, warmup, patience,
  workers, amp, reason.
- Values must respect the search space. Use one of the listed choices for list-type fields.
- reason is a short one-sentence explanation in English.
"""


class Tuner:
    def __init__(self, base_url, model, api_key, timeout=90):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.last_raw = None

    def _chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 600,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError(f"LLM API failed after 3 attempts: {last_err}")

    @staticmethod
    def _parse_json(content):
        if not content:
            return None
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None

    def propose(self, dataset_summary, history, search_space, defaults,
                target_metric, target_value):
        rows = []
        for t in history:
            rows.append(
                {
                    "id": t["id"],
                    "config": t["config"],
                    "fitness": t.get("fitness"),
                    "status": t.get("status"),
                }
            )
        user = (
            "Dataset:\n"
            f"{json.dumps(dataset_summary, ensure_ascii=False)}\n\n"
            "Experiment history:\n"
            f"{json.dumps(rows, ensure_ascii=False)}\n\n"
            "Search space:\n"
            f"{json.dumps(search_space, ensure_ascii=False)}\n\n"
            "Baseline defaults:\n"
            f"{json.dumps(defaults, ensure_ascii=False)}\n\n"
            f"Target: maximize {target_metric}, target value {target_value}.\n"
            "Propose the next configuration as JSON."
        )
        content = self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        self.last_raw = content
        return self._parse_json(content)
