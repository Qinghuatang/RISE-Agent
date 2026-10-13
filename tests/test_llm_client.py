import unittest
from dataclasses import dataclass

from research_agent.errors import LLMError, StructuredOutputError
from research_agent.llm.client import (
    LLMMessage,
    LLMRequest,
    coerce_structured_output,
)
from research_agent.llm.fake import FakeLLMClient


@dataclass
class Answer:
    answer: str


class LLMClientTests(unittest.TestCase):
    def test_fake_client_returns_structured_output_and_records_call(self) -> None:
        client = FakeLLMClient([{"answer": "ok"}])
        request = LLMRequest((LLMMessage("user", "return JSON"),))
        response = client.complete_structured(request, Answer)
        self.assertEqual(response.output.answer, "ok")
        self.assertEqual(response.metadata.provider, "fake")
        self.assertEqual(len(client.calls), 1)

    def test_fake_client_rejects_malformed_or_exhausted_output(self) -> None:
        request = LLMRequest((LLMMessage("user", "return JSON"),))
        with self.assertRaises(StructuredOutputError):
            FakeLLMClient([{"wrong": "field"}]).complete_structured(request, Answer)
        with self.assertRaises(LLMError):
            FakeLLMClient().complete_structured(request, Answer)

    def test_json_string_is_decoded_but_non_object_is_rejected(self) -> None:
        self.assertEqual(coerce_structured_output('{"answer":"ok"}', Answer).answer, "ok")
        with self.assertRaises(StructuredOutputError):
            coerce_structured_output('["not", "an", "object"]', Answer)

    def test_single_json_object_can_be_extracted_from_transport_wrapper(self) -> None:
        response = coerce_structured_output(
            "The structured response is below:\n{\"answer\":\"ok\"}\nEnd.",
            Answer,
        )
        self.assertEqual(response.answer, "ok")

    def test_multiple_json_objects_are_not_ambiguously_repaired(self) -> None:
        with self.assertRaises(Exception):
            coerce_structured_output(
                '{"answer":"first"}\n{"answer":"second"}',
                Answer,
            )


if __name__ == "__main__":
    unittest.main()
