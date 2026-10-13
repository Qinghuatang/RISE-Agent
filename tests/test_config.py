import os
import unittest
from unittest.mock import patch

from research_agent.errors import ConfigurationError
from research_agent.llm.config import DeepSeekConfig


class DeepSeekConfigTests(unittest.TestCase):
    def test_reads_required_values_and_optional_runtime_config(self) -> None:
        values = {
            "DEEPSEEK_API_KEY": "secret",
            "DEEPSEEK_BASE_URL": "https://example.test/",
            "DEEPSEEK_MODEL": "model-x",
            "DEEPSEEK_TIMEOUT_SECONDS": "12.5",
            "DEEPSEEK_MAX_RETRIES": "2",
        }
        config = DeepSeekConfig.from_env(values)
        self.assertEqual(config.base_url, "https://example.test")
        self.assertEqual(config.model, "model-x")
        self.assertEqual(config.timeout_seconds, 12.5)
        self.assertEqual(config.max_retries, 2)
        self.assertTrue(config.public_dict()["api_key_configured"])
        self.assertNotIn("api_key", config.public_dict())

    def test_offline_environment_can_construct_config_without_secret(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = DeepSeekConfig.from_env()
        self.assertIsNone(config.api_key)
        with self.assertRaises(ConfigurationError):
            config.validate_for_request()

    def test_invalid_runtime_config_is_explicit(self) -> None:
        with self.assertRaises(ConfigurationError):
            DeepSeekConfig.from_env({"DEEPSEEK_TIMEOUT_SECONDS": "not-a-number"})
        with self.assertRaises(ConfigurationError):
            DeepSeekConfig.from_env({"DEEPSEEK_MAX_RETRIES": "-1"})


if __name__ == "__main__":
    unittest.main()

