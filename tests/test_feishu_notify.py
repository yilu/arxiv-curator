import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import feishu_notify


def paper(paper_id, score, similarity=0.0, abstract="Abstract text."):
    return {
        "id": paper_id,
        "title": f"Title {paper_id}",
        "authors": [{"name": "A. Researcher"}, "B. Researcher"],
        "summary": abstract,
        "score": score,
        "vector_matches": [{"score": similarity}],
    }


class FeishuNotifyTests(unittest.TestCase):
    def test_every_paper_is_ranked_by_score_then_similarity(self):
        papers = [
            paper("2611.00001v1", 0.9, 0.2),
            paper("2611.00002v1", 0.7, 0.9),
            paper("2611.00003v1", 0.9, 0.8),
        ]
        messages = feishu_notify.format_messages(papers)
        text = "\n".join(messages)
        self.assertLess(text.index("1. Title 2611.00003v1"), text.index("2. Title 2611.00001v1"))
        self.assertLess(text.index("2. Title 2611.00001v1"), text.index("3. Title 2611.00002v1"))
        self.assertEqual(text.count("A. Researcher, B. Researcher"), 3)
        self.assertEqual(text.count("Abstract text."), 3)
        self.assertIn("https://arxiv.org/abs/2611.00003v1", text)

    def test_long_abstract_is_preserved_and_parts_fit_the_byte_limit(self):
        abstract = "量子 correlations " * 1200
        messages = feishu_notify.format_messages([paper("2611.00001v1", 0.9, abstract=abstract)])
        self.assertGreater(len(messages), 1)
        for n, msg in enumerate(messages, start=1):
            self.assertIn(f"{n}/{len(messages)}", msg.splitlines()[0])
            self.assertLessEqual(len(msg.encode()), feishu_notify.MAX_TEXT_BYTES)
        body = "".join(msg.split("\n\n", 1)[1] for msg in messages)
        self.assertTrue(body.endswith(abstract))

    def test_empty_list_neither_authenticates_nor_sends(self):
        with patch.object(feishu_notify, "post_json") as post:
            self.assertEqual(feishu_notify.notify([], "publication"), 0)
        post.assert_not_called()

    def test_request_ids_are_stable_and_content_matches(self):
        messages = feishu_notify.format_messages([paper("2611.00001v1", 0.9)])
        with patch.object(feishu_notify, "tenant_access_token", return_value="test-token"), \
             patch.object(feishu_notify, "post_json", return_value={"code": 0}) as post, \
             patch.dict("os.environ", {"FEISHU_RECIPIENT_OPEN_ID": "ou_test"}):
            feishu_notify.notify([paper("2611.00001v1", 0.9)], "publication")
            feishu_notify.notify([paper("2611.00001v1", 0.9)], "publication")
        first, retry = [call.args[1] for call in post.call_args_list]
        self.assertEqual(first["uuid"], retry["uuid"])
        self.assertEqual(json.loads(first["content"])["text"], messages[0])
        self.assertEqual(first["receive_id"], "ou_test")


if __name__ == "__main__":
    unittest.main()
