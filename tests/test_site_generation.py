import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import recommend


def paper(paper_id, published_date):
    return {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "authors": [{"name": "A. Researcher"}],
        "summary": "An abstract.",
        "published_date": published_date,
        "categories": ["cond-mat.str-el"],
        "score": 0.8,
    }


def changed_pages(before, after):
    return {
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    }


class SiteGenerationTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(scratch.name)
        Path("templates").symlink_to(PROJECT_ROOT / "templates", target_is_directory=True)
        self.archive = {
            "2610.00001v1": paper("2610.00001v1", "2026-10-01"),
            "2611.00001v1": paper("2611.00001v1", "2026-11-01"),
        }

    def render(self, archive, date="2026-11-20", added=()):
        with patch.object(recommend, "datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime.fromisoformat(date)
            with redirect_stdout(io.StringIO()):
                recommend.generate_site(added, archive)
        return {page.name: page.read_bytes() for page in Path("dist").glob("*.html")}

    def test_unchanged_archive_renders_the_same_pages_on_different_days(self):
        before = self.render(self.archive, added=("2611.00001v1",))
        after = self.render(self.archive, date="2026-11-21")
        self.assertEqual(changed_pages(before, after), set())

    def test_new_paper_changes_only_its_own_month(self):
        before = self.render(self.archive)
        for month, paper_id in (("2026-10", "2610.00002v1"), ("2026-11", "2611.00002v1")):
            with self.subTest(month=month):
                after = self.render(
                    self.archive | {paper_id: paper(paper_id, f"{month}-02")},
                    date="2026-11-21",
                    added=(paper_id,),
                )
                self.assertEqual(changed_pages(before, after), {f"{month}.html"})
                self.assertIn(f"https://arxiv.org/abs/{paper_id}".encode(), after[f"{month}.html"])

    def test_index_points_to_the_latest_month_after_rollover(self):
        before = self.render(self.archive)
        after = self.render(
            self.archive | {"2612.00001v1": paper("2612.00001v1", "2026-12-01")},
            date="2026-12-02",
            added=("2612.00001v1",),
        )
        self.assertLess(len(before["index.html"]), 1024)
        self.assertIn(b'url=2026-11.html', before["index.html"])
        self.assertIn(b'url=2026-12.html', after["index.html"])
        self.assertIn(b'href="2026-12.html"', after["index.html"])
        self.assertIn(b'href="2026-12.html"', after["2026-10.html"])
        self.assertIn("2026-12.html", after)

    def test_tied_papers_stay_in_order_when_archive_insertion_order_changes(self):
        archive = self.archive | {"2611.00002v1": paper("2611.00002v1", "2026-11-01")}
        before = self.render(archive)
        after = self.render(dict(reversed(list(archive.items()))))
        self.assertEqual(changed_pages(before, after), set())


if __name__ == "__main__":
    unittest.main()
