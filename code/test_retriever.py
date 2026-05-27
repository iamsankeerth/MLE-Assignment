import unittest

from retriever import InMemoryDocumentProvider, SupportRetriever


class TestSupportRetrieverDeterminism(unittest.TestCase):
    def test_equal_scores_use_path_tie_break(self):
        docs = [
            {"path": "data/devplatform/z-last.md", "content": "reset password support guidance"},
            {"path": "data/devplatform/a-first.md", "content": "reset password support guidance"},
        ]
        retriever = SupportRetriever(provider=InMemoryDocumentProvider(docs))

        results = retriever.retrieve("reset password support", company="DevPlatform", top_k=2)

        self.assertEqual([doc["path"] for doc in results], [
            "data/devplatform/a-first.md",
            "data/devplatform/z-last.md",
        ])


if __name__ == "__main__":
    unittest.main()
