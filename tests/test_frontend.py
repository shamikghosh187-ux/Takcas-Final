from pathlib import Path
import unittest


class FrontendRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (Path(__file__).resolve().parents[1] / "index.html").read_text(
            encoding="utf-8"
        )
        cls.script = cls.html.split("<script>", 1)[1].split("</script>", 1)[0]

    def test_history_json_parse_is_guarded(self):
        self.assertIn("function loadHistory(", self.script)
        self.assertIn("try {", self.script)
        self.assertIn("JSON.parse(raw)", self.script)
        self.assertIn("localStorage.removeItem(getHistoryKey(persona))", self.script)

    def test_history_is_validated_and_bounded(self):
        self.assertIn("function normalizeHistory(value)", self.script)
        self.assertIn("item.role === 'user' || item.role === 'assistant'", self.script)
        self.assertIn(".slice(-100)", self.script)
        self.assertIn("content.slice(0, 12000)", self.script)

    def test_storage_failures_do_not_crash_chat(self):
        self.assertIn("localStorage.setItem(getHistoryKey(persona)", self.script)
        self.assertIn("Unable to save chat history.", self.script)

    def test_send_requests_are_serialized(self):
        self.assertIn("let isSending = false", self.script)
        self.assertIn("if (isSending) return", self.script)
        self.assertIn("sendButton.disabled = sending", self.script)
        self.assertIn("finally {", self.script)
        self.assertIn("setSending(false)", self.script)

    def test_request_keeps_original_persona_when_user_switches(self):
        self.assertIn("const persona = currentPersona", self.script)
        self.assertIn("const lang = currentLang", self.script)
        self.assertIn("loadHistory(persona)", self.script)
        self.assertIn("if (currentPersona === persona)", self.script)

    def test_non_success_http_responses_are_reported(self):
        self.assertIn("response.json().catch(() => ({}))", self.script)
        self.assertIn("if (!response.ok)", self.script)


if __name__ == "__main__":
    unittest.main()
