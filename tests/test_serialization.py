import unittest
from dataclasses import dataclass

from research_agent.errors import SerializationError
from research_agent.serialization import dumps_json, dumps_jsonl, loads_json, loads_jsonl


@dataclass
class Payload:
    name: str
    count: int


class SerializationTests(unittest.TestCase):
    def test_json_round_trip_uses_schema_version_envelope(self) -> None:
        text = dumps_json("test.payload", Payload("x", 2))
        decoded = loads_json(text, expected_schema="test.payload")
        self.assertEqual(decoded["version"], 1)
        self.assertEqual(decoded["payload"], {"name": "x", "count": 2})

    def test_jsonl_round_trip(self) -> None:
        text = dumps_jsonl("test.event", [{"step": 1}, {"step": 2}])
        records = loads_jsonl(text, expected_schema="test.event")
        self.assertEqual([record["payload"]["step"] for record in records], [1, 2])

    def test_invalid_json_and_envelope_are_not_silently_accepted(self) -> None:
        with self.assertRaises(SerializationError):
            loads_json("not-json")
        with self.assertRaises(SerializationError):
            loads_json('{"schema":"x","version":1}', expected_schema="x")
        with self.assertRaises(SerializationError):
            loads_json(dumps_json("x", {}), expected_schema="other")
        with self.assertRaises(SerializationError):
            loads_jsonl('{"schema":"x"}\n', expected_schema="x")

    def test_unserializable_payload_is_explicit(self) -> None:
        with self.assertRaises(SerializationError):
            dumps_json("test.bad", object())


if __name__ == "__main__":
    unittest.main()

