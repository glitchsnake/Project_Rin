import os
import unittest
from pathlib import Path
import database
from database import (
    init_db,
    ensure_user,
    get_user,
    update_user_warmth,
    update_user_attitude,
    update_core_memory,
    update_persona_narrative,
    save_message,
    load_history,
    touch_message_time,
    get_last_message_time
)

# Use a separate test database file to isolate testing from production data
TEST_DB_PATH = Path("./test_rin_sessions.db")

class TestDatabase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare the database before running tests."""
        # Patch database.DB_PATH
        cls.original_db_path = database.DB_PATH
        database.DB_PATH = TEST_DB_PATH
        
        # Ensure temporary test db does not exist from an aborted run
        if TEST_DB_PATH.exists():
            os.remove(TEST_DB_PATH)
            
        # Initialize schema
        init_db()

    @classmethod
    def tearDownClass(cls):
        """Clean up the test database file to keep the workspace clean."""
        database.DB_PATH = cls.original_db_path
        if TEST_DB_PATH.exists():
            os.remove(TEST_DB_PATH)

    async def test_01_ensure_and_get_user(self):
        """Verify that ensure_user creates a new record and get_user retrieves it."""
        session_id = "test_user_123"
        user_name = "Alex"
        
        # Create user
        await ensure_user(session_id, user_name)
        
        # Retrieve user profile
        profile = await get_user(session_id)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], user_name)
        self.assertEqual(profile["warmth"], 0.0)
        self.assertEqual(profile["base_attitude"], "neutral")
        self.assertEqual(profile["core_memory"], "")
        self.assertEqual(profile["persona_narrative"], "")

    async def test_02_update_user_warmth(self):
        """Verify that update_user_warmth increments or decrements the warmth attribute correctly."""
        session_id = "test_user_123"
        
        # Increment by +0.2
        await update_user_warmth(session_id, 0.2)
        profile = await get_user(session_id)
        self.assertAlmostEqual(profile["warmth"], 0.2)

        # Decrement by -0.5
        await update_user_warmth(session_id, -0.5)
        profile = await get_user(session_id)
        self.assertAlmostEqual(profile["warmth"], -0.3)

    async def test_03_update_user_attitude_and_memory(self):
        """Verify updating user stance, core memory, and persona narratives."""
        session_id = "test_user_123"
        
        # Stance / attitude shift
        await update_user_attitude(session_id, "warm and trusting")
        # Core memory
        await update_core_memory(session_id, "Knows programming; Likes sushi")
        # Narrative
        await update_persona_narrative(session_id, "Interesting companion, writes regularly")

        profile = await get_user(session_id)
        self.assertEqual(profile["base_attitude"], "warm and trusting")
        self.assertEqual(profile["core_memory"], "Knows programming; Likes sushi")
        self.assertEqual(profile["persona_narrative"], "Interesting companion, writes regularly")

    async def test_04_save_and_load_history(self):
        """Verify saving and loading message history lists."""
        session_id = "test_user_123"
        
        # Save messages
        await save_message(session_id, "user", "Hello, Rin!")
        await save_message(session_id, "assistant", "Hi.")

        # Load history
        history = await load_history(session_id, system_prompt="INSTRUCTION", limit=10)
        self.assertEqual(len(history), 3, "Should contain system prompt + 2 messages")
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[0]["content"], "INSTRUCTION")
        self.assertEqual(history[1]["role"], "user")
        self.assertEqual(history[1]["content"], "Hello, Rin!")
        self.assertEqual(history[2]["role"], "assistant")
        self.assertEqual(history[2]["content"], "Hi.")

    async def test_05_touch_and_get_message_time(self):
        """Verify touching last message timestamp works correctly."""
        session_id = "test_user_123"
        
        # Record current message time
        await touch_message_time(session_id)
        
        # Read it back
        last_time = await get_last_message_time(session_id)
        self.assertIsNotNone(last_time)
