"""Run the shared-app checks in Google Colab before deployment."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from project_chat import ask_assistant, ChatUnavailable, validate_action
from public_backend import load_example, result_context, validate_selection, drawing_bounds, map_overlay

PROJECT = Path(__file__).resolve().parents[1]


class SharedDashboardChecks(unittest.TestCase):
    def test_free_router_and_bounded_history(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "model": "free-provider/model", "choices": [{"message": {"content": "Six predictions."}}]}
        with patch("project_chat.requests.post", return_value=response) as post:
            history = [{"role": "user", "content": "A previous question"}] * 20
            answer = ask_assistant("How many ships?", history, {"predictions": 6}, "test-key")
        self.assertEqual(answer["text"], "Six predictions.")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "openrouter/free")
        self.assertEqual(len(payload["messages"]), 11)
        self.assertNotIn("test-key", json.dumps(payload))

    def test_provider_limits_and_malformed_response(self):
        with patch("project_chat.requests.post", return_value=Mock(status_code=429)) as post:
            with self.assertRaisesRegex(ChatUnavailable, "usage limit"):
                ask_assistant("Explain", [], {}, "test-key")
            self.assertEqual(post.call_count, 1)
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": None}]}
        with patch("project_chat.requests.post", return_value=response):
            with self.assertRaises(ChatUnavailable):
                ask_assistant("Explain", [], {}, "test-key")
        with patch("project_chat.requests.post") as post:
            with self.assertRaises(ChatUnavailable):
                ask_assistant("Explain", [], {}, "")
            post.assert_not_called()

    def test_only_defined_map_actions(self):
        call = {"function": {"name": "show_example", "arguments": '{"location":"Vancouver"}'}}
        self.assertEqual(validate_action(call)["location"], "Vancouver")
        reference_filter = {"function": {"name": "filter_ports",
                            "arguments": '{"province":"Yukon","query":""}'}}
        action = validate_action(reference_filter)
        self.assertEqual(action, {"name": "filter_ports", "province": "YT", "query": ""})
        call["function"] = {"name": "run_code", "arguments": '{"code":"anything"}'}
        with self.assertRaises(ChatUnavailable):
            validate_action(call)

    def test_examples_remain_immutable_and_georeferenced(self):
        result = load_example(PROJECT, "Halifax")
        before = copy.deepcopy(result["metadata"])
        context = result_context(result, 1, "YT", "", [-123.115, 49.285, -123.08, 49.31])
        self.assertEqual(result["metadata"], before)
        self.assertEqual(context["displayed_run"]["predicted_ships"], 6)
        self.assertFalse(context["requested_area_has_been_processed"])
        image, bounds = map_overlay(result)
        self.assertTrue(image.startswith(b"\x89PNG"))
        self.assertLess(bounds[0][0], bounds[1][0])
        self.assertLess(bounds[0][1], bounds[1][1])

    def test_selection_limits(self):
        with self.assertRaises(ValueError):
            validate_selection([-140, 42, -53, 83])
        feature = {"geometry": {"type": "Polygon", "coordinates": [[
            [-63.58, 44.615], [-63.545, 44.615], [-63.545, 44.64], [-63.58, 44.64], [-63.58, 44.615]]]}}
        self.assertEqual(drawing_bounds(feature), [-63.58, 44.615, -63.545, 44.64])

    def test_app_startup_and_chat_navigation(self):
        from streamlit.testing.v1 import AppTest
        with patch("streamlit_folium.st_folium", return_value={}):
            app = AppTest.from_file(str(PROJECT / "code/app.py")).run(timeout=45)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "6")
            app.session_state["pending_chat"] = {
                "prompt": "Show Vancouver", "response": {
                    "text": "", "actions": [{"name": "show_example", "location": "Vancouver"}],
                    "model": "test/free"}}
            app.run(timeout=45)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "20")
            self.assertEqual(app.session_state["messages"][-1]["role"], "assistant")
            app.session_state["pending_chat"] = {
                "prompt": "Filter to Yukon", "response": {
                    "text": "", "actions": [{"name": "filter_ports", "province": "YT", "query": ""}],
                    "model": "test/free"}}
            app.run(timeout=45)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["province"], "YT")
            self.assertIn("Port reference filter set to Yukon.", app.session_state["messages"][-1]["content"])
            other = AppTest.from_file(str(PROJECT / "code/app.py")).run(timeout=45)
            self.assertFalse(other.exception)
            self.assertEqual(other.metric[0].value, "6")
            self.assertEqual(other.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
