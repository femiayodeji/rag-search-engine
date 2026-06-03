import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.lib.chunk_semantic_search import semantic_chunk_text


class TestSemanticChunkTextEdgeCases(unittest.TestCase):
    def test_trims_whitespace_before_split(self):
        chunks = semantic_chunk_text(" Leading and trailing spaces. ")
        self.assertEqual(chunks, ["Leading and trailing spaces."])

    def test_single_sentence_without_punctuation(self):
        chunks = semantic_chunk_text("Text without punctuation")
        self.assertEqual(chunks, ["Text without punctuation"])

    def test_whitespace_only_returns_empty_list(self):
        chunks = semantic_chunk_text(" ")
        self.assertEqual(chunks, [])

    def test_empty_string_returns_empty_list(self):
        chunks = semantic_chunk_text("")
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
