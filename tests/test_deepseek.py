import unittest
from dataclasses import dataclass

from research_agent.errors import ConfigurationError, LLMTransportError, MalformedLLMResponseError
from research_agent.llm.client import LLMMessage, LLMRequest
from research_agent.llm.config import DeepSeekConfig
from research_agent.llm.deepseek import DeepSeekClient


@dataclass
class Answer:
    answer: str


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        return self.response


class DeepSeekClientTests(unittest.TestCase):
    def test_construction_is_offline_and_request_uses_injected_transport(self) -> None:
        transport = RecordingTransport(
            {
                "id": "request-1",
                "choices": [{"message": {"content": '{"answer":"yes"}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )
        config = DeepSeekConfig(
            api_key="secret",
            base_url="https://example.test",
            model="model-x",
        )
        response = DeepSeekClient(config, transport).complete_structured(
            LLMRequest((LLMMessage("user", "return JSON"),)), Answer
        )
        self.assertEqual(response.output.answer, "yes")
        self.assertEqual(response.metadata.total_tokens, 5)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][0], "https://example.test/chat/completions")
        self.assertEqual(transport.calls[0][2]["response_format"], {"type": "json_object"})

    def test_missing_request_config_fails_before_transport(self) -> None:
        transport = RecordingTransport({})
        client = DeepSeekClient(DeepSeekConfig(), transport)
        with self.assertRaises(ConfigurationError):
            client.complete_structured(
                LLMRequest((LLMMessage("user", "return JSON"),)), Answer
            )
        self.assertEqual(transport.calls, [])

    def test_malformed_provider_content_is_explicit(self) -> None:
        transport = RecordingTransport(
            {"choices": [{"message": {"content": "not-json"}}]}
        )
        config = DeepSeekConfig("secret", "https://example.test", "model-x")
        with self.assertRaises(MalformedLLMResponseError):
            DeepSeekClient(config, transport).complete_structured(
                LLMRequest((LLMMessage("user", "return JSON"),)), Answer
            )

    def test_malformed_provider_envelope_is_explicit(self) -> None:
        transport = RecordingTransport({"choices": []})
        config = DeepSeekConfig("secret", "https://example.test", "model-x")
        with self.assertRaises(LLMTransportError):
            DeepSeekClient(config, transport).complete_structured(
                LLMRequest((LLMMessage("user", "return JSON"),)), Answer
            )

    def test_gateway_wrapper_around_json_object_is_accepted_and_schema_checked(self) -> None:
        transport = RecordingTransport(
            {"choices": [{"message": {"content": "Here is the JSON:\n{\"answer\":\"yes\"}"}}]}
        )
        config = DeepSeekConfig("secret", "https://example.test", "model-x")
        response = DeepSeekClient(config, transport).complete_structured(
            LLMRequest((LLMMessage("user", "return JSON"),)), Answer
        )
        self.assertEqual(response.output.answer, "yes")


if __name__ == "__main__":
    unittest.main()
