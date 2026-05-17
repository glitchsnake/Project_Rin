import unittest
from unittest.mock import MagicMock, patch
from memory import (
    count_tokens,
    count_history_tokens,
    _is_important,
    is_memory_available,
    DUPLICATE_THRESHOLD
)
import memory

class TestMemory(unittest.TestCase):
    def test_count_tokens(self):
        """Verify token counting using tiktoken is accurate and doesn't fail."""
        text = "Hello, world!"
        tokens_count = count_tokens(text)
        self.assertGreater(tokens_count, 0)
        self.assertEqual(count_tokens(""), 0)

    def test_count_history_tokens(self):
        """Verify summation of tokens in a simulated message history list."""
        history = [
            {"role": "user", "content": "Hi, how are you?"},
            {"role": "assistant", "content": "Fine."}
        ]
        total = count_history_tokens(history)
        self.assertGreater(total, 0)
        # Verify it accounts for overhead per message
        self.assertGreater(total, count_tokens("Hi, how are you?") + count_tokens("Fine."))

    def test_is_important(self):
        """Verify importance filter logic based on role and text length."""
        # System messages are never important
        self.assertFalse(_is_important("system", "This is an important system instruction."))

        # Short messages are not important
        self.assertFalse(_is_important("user", "Hi"))
        self.assertFalse(_is_important("assistant", "Ok"))

        # Long messages are important
        self.assertTrue(_is_important("user", "Tell me a detailed story about how this bot is designed."))
        self.assertTrue(_is_important("assistant", "I am sitting at home listening to music because it is raining outside."))

    @patch('memory._init_memory')
    def test_is_memory_available(self, mock_init):
        """Verify that memory availability check queries lazy initializer."""
        memory._memory_available = False
        mock_init.return_value = True
        self.assertTrue(is_memory_available())
        mock_init.assert_called_once()

    @patch('memory.is_memory_available')
    def test_save_to_memory_deduplication(self, mock_available):
        """Verify duplicate detection and metadata update logic inside save_to_memory."""
        mock_available.return_value = True
        
        # Mock database collection
        mock_collection = MagicMock()
        memory._collection = mock_collection
        
        # Mock sentence transformer embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        memory._embedder = mock_embedder

        # Configure query to return a duplicate (distance < DUPLICATE_THRESHOLD)
        mock_collection.query.return_value = {
            "ids": [["doc_123"]],
            "distances": [[0.05]],  # Less than DUPLICATE_THRESHOLD (0.08)
            "metadatas": [[{"role": "user", "user_id": "test_user", "frequency": 1}]]
        }

        # Save duplicate message
        memory.save_to_memory(role="user", content="Some important repeating phrase.", user_id="test_user")

        # Check that duplicate branch was hit: collection.update called, not add
        mock_collection.update.assert_called_once()
        mock_collection.add.assert_not_called()

        # Check metadata update parameters
        updated_meta = mock_collection.update.call_args[1]["metadatas"][0]
        self.assertEqual(updated_meta["frequency"], 2)
        self.assertIn("last_seen_timestamp", updated_meta)
