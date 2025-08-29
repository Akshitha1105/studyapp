from __future__ import annotations

import os
from typing import List, Dict, Any

import requests

try:
	from transformers import AutoModelForCausalLM, AutoTokenizer
	exists_transformers = True
except Exception:
	exists_transformers = False

def _format_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]


class HFInferenceClient:
	def __init__(self, model_id: str) -> None:
		self.model_id = model_id
		self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
		if not self.api_token:
			raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set in environment")

	def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
		url = f"https://api-inference.huggingface.co/models/{self.model_id}"
		headers = {"Authorization": f"Bearer {self.api_token}"}
		# Use text-generation-inference style payload where supported
		payload = {
			"inputs": _format_messages(system_prompt, user_prompt),
			"parameters": {
				"max_new_tokens": 512,
				"temperature": temperature,
				"return_full_text": False,
			}
		}
		resp = requests.post(url, headers=headers, json=payload, timeout=120)
		resp.raise_for_status()
		data = resp.json()
		# Try common response shapes
		if isinstance(data, list) and data and "generated_text" in data[0]:
			return data[0]["generated_text"]
		if isinstance(data, dict) and "generated_text" in data:
			return data["generated_text"]
		# Fallback to raw string
		return str(data)


class LocalTransformersClient:
	def __init__(self, model_id: str) -> None:
		if not exists_transformers:
			raise RuntimeError("transformers not installed")
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

	def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
		messages = _format_messages(system_prompt, user_prompt)
		prompt = "\n".join([m["content"] for m in messages])
		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
		outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=temperature)
		text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		return text