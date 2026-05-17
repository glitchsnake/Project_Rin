import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import semantic_cache
import semantic_router_engine

class TestRoutingCaching(unittest.TestCase):
    @patch('semantic_cache._init_cache_collection')
    @patch('memory._embed')
    def test_semantic_cache_hit(self, mock_embed, mock_init_coll):
        """Verify semantic cache hit returns cached response with composite filtering."""
        mock_coll = MagicMock()
        mock_init_coll.return_value = mock_coll
        mock_embed.return_value = [0.1, 0.2, 0.3]
        
        # Mock Chroma query returning a close match (distance 0.04 < 0.08)
        mock_coll.count.return_value = 1
        mock_coll.query.return_value = {
            "documents": [["Как дела?"]],
            "metadatas": [[{"ai_response": "Все отлично, спасибо!"}]],
            "distances": [[0.04]]
        }
        
        res = semantic_cache.get_semantic_cache("Как дела?", "нейтральное", "neutral")
        self.assertEqual(res, "Все отлично, спасибо!")
        mock_coll.query.assert_called_once()

    @patch('semantic_cache._init_cache_collection')
    @patch('memory._embed')
    def test_semantic_cache_miss(self, mock_embed, mock_init_coll):
        """Verify semantic cache miss returns None."""
        mock_coll = MagicMock()
        mock_init_coll.return_value = mock_coll
        mock_embed.return_value = [0.1, 0.2, 0.3]
        
        # Mock Chroma query returning a distant match (distance 0.12 > 0.08)
        mock_coll.count.return_value = 1
        mock_coll.query.return_value = {
            "documents": [["Как дела?"]],
            "metadatas": [[{"ai_response": "Все отлично, спасибо!"}]],
            "distances": [[0.12]]
        }
        
        res = semantic_cache.get_semantic_cache("Как дела?", "нейтральное", "neutral")
        self.assertIsNone(res)

    @patch('semantic_router_engine.init_router')
    @patch('memory._embed')
    def test_semantic_router_tools(self, mock_embed, mock_init_router):
        """Verify semantic router correctly identifies 'tools' route."""
        semantic_router_engine._initialized = True
        
        # Set anchor embeddings
        semantic_router_engine._anchor_embeddings = {
            "tools": [[1.0, 0.0, 0.0]],
            "deep_thought": [[0.0, 1.0, 0.0]]
        }
        
        # Query embedding is identical to tools anchor
        mock_embed.return_value = [1.0, 0.0, 0.0]
        
        route = semantic_router_engine.route_message("выполни код")
        self.assertEqual(route, "tools")

    @patch('semantic_router_engine.init_router')
    @patch('memory._embed')
    def test_semantic_router_general(self, mock_embed, mock_init_router):
        """Verify semantic router falls back to 'general' for low similarity."""
        semantic_router_engine._initialized = True
        semantic_router_engine._anchor_embeddings = {
            "tools": [[1.0, 0.0, 0.0]],
            "deep_thought": [[0.0, 1.0, 0.0]]
        }
        
        # Query embedding has low similarity to both (0.707 < 0.80 threshold)
        mock_embed.return_value = [0.707, 0.707, 0.0]
        
        route = semantic_router_engine.route_message("приветик")
        self.assertEqual(route, "general")
