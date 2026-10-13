import unittest
from contextlib import redirect_stdout
from io import StringIO

from research_agent.__main__ import main


class CLITests(unittest.TestCase):
    def test_help_smoke(self) -> None:
        with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()):
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_offline_smoke_does_not_need_services(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["offline-smoke"]), 0)
        self.assertIn("offline smoke: ok", output.getvalue())


if __name__ == "__main__":
    unittest.main()

