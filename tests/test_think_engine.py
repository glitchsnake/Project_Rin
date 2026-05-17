import unittest
from unittest.mock import MagicMock, patch
from think_engine import (
    _node_router,
    _node_system_1_fast_track,
    _check_vulnerability,
    _build_persona_block,
    LLMThinkOutput,
    ThinkState,
    ThinkGraph,
    IdleGraph
)

class TestThinkEngine(unittest.TestCase):
    def test_node_router_fast_track(self):
        """Verify that simple greeting or small-talk phrases trigger System 1 (fast track)."""
        fast_inputs = [
            "hello",
            "hi",
            "how are you",
            "fine",
            "ok",
            "yep",
            "um",
            "good night",
            "good morning"
        ]
        for inp in fast_inputs:
            state = ThinkState(user_text=inp)
            route = _node_router(state)
            self.assertEqual(route, "system_1", f"Failed for input: {inp}")

    def test_node_router_deep_track(self):
        """Verify that longer or non-greeting messages trigger System 2 (deep track)."""
        deep_inputs = [
            "Tell me about string theory and quantum gravity.",
            "Why did you not reply to my message yesterday?",
            "What do you think of Dostoevsky's works?",
            "Buy milk on the way home"
        ]
        for inp in deep_inputs:
            state = ThinkState(user_text=inp)
            route = _node_router(state)
            self.assertEqual(route, "system_2", f"Failed for input: {inp}")

    def test_vulnerability_override(self):
        """Verify that vulnerability signals override hostile/aggressive emotions to 'indifference'."""
        # Vulnerable input with aggressive emotion should map to 'indifference'
        vuln_inputs = ["i feel sad", "i am so tired", "i am lonely and hurt"]
        aggressive_emotions = ["dry sarcasm", "quiet contempt", "irritation"]
        
        for inp in vuln_inputs:
            for emotion in aggressive_emotions:
                res = _check_vulnerability(inp, warmth=0.2, rin_emotion=emotion)
                self.assertEqual(res, "indifference", f"Failed to override {emotion} for input: {inp}")

        # When warmth <= 0, no overrides should occur
        for inp in vuln_inputs:
            for emotion in aggressive_emotions:
                res = _check_vulnerability(inp, warmth=-0.1, rin_emotion=emotion)
                self.assertEqual(res, emotion, f"Should not override when warmth <= 0")

        # Non-vulnerable input should not override any emotions
        safe_input = "the weather is nice today"
        for emotion in aggressive_emotions:
            res = _check_vulnerability(safe_input, warmth=0.5, rin_emotion=emotion)
            self.assertEqual(res, emotion, "Should not override non-vulnerable inputs")

    def test_llm_think_output_validators(self):
        """Verify that LLMThinkOutput sanitizes and validates emotions, attitudes, and tactics."""
        # Valid data passes through untouched
        valid_data = {
            "fact_analysis": "User asked about the weather",
            "hidden_intent": "Desire to start a conversation",
            "rin_inner_conflict": "FACT -> ANALYSIS -> REACTION",
            "confidence": 0.85,
            "rin_emotion": "pensiveness",
            "rin_attitude": "neutral",
            "response_tactic": "short reaction"
        }
        output = LLMThinkOutput(**valid_data)
        self.assertEqual(output.rin_emotion, "pensiveness")
        self.assertEqual(output.rin_attitude, "neutral")
        self.assertEqual(output.response_tactic, "short reaction")

        # Invalid data triggers validator fallbacks
        invalid_data = {
            "fact_analysis": "User is aggressive",
            "hidden_intent": "Provoke",
            "rin_inner_conflict": "FACT -> ANALYSIS -> REACTION",
            "confidence": 0.9,
            "rin_emotion": "UNKNOWN_EMOTION",
            "rin_attitude": "UNKNOWN_ATTITUDE",
            "response_tactic": "UNKNOWN_TACTIC"
        }
        output_invalid = LLMThinkOutput(**invalid_data)
        self.assertEqual(output_invalid.rin_emotion, "detachment", "Invalid emotion must default to 'detachment'")
        self.assertEqual(output_invalid.rin_attitude, "neutral", "Invalid attitude must default to 'neutral'")
        self.assertEqual(output_invalid.response_tactic, "short reaction", "Invalid tactic must default to 'short reaction'")

    def test_build_persona_block_gradients(self):
        """Verify that _build_persona_block renders custom persona segments according to warmth gradients."""
        # 1. Cold styling (warmth < 0)
        persona_cold = _build_persona_block(
            warmth=-0.2,
            base_attitude="guarded",
            user_name="Alex",
            core_memory="likes cats",
            persona_narrative="he is weird"
        )
        self.assertIn("Alex", persona_cold)
        self.assertIn("guarded", persona_cold)
        self.assertIn("likes cats", persona_cold)
        self.assertIn("he is weird", persona_cold)
        self.assertIn("DYNAMIC: The companion is unpleasant. Dry, detached.", persona_cold)

        # 2. Medium styling (0 <= warmth <= 0.5)
        persona_mid = _build_persona_block(
            warmth=0.3,
            base_attitude="neutral",
            user_name="Alex",
            core_memory="",
            persona_narrative=""
        )
        self.assertIn("DYNAMIC: Acquaintance. Distance, brevity.", persona_mid)
        self.assertNotIn("[Facts]", persona_mid)

        # 3. Warm styling (warmth > 0.5)
        persona_warm = _build_persona_block(
            warmth=0.8,
            base_attitude="warm and trusting",
            user_name="Alex",
            core_memory="creator",
            persona_narrative="he is important"
        )
        self.assertIn("DYNAMIC: Close person. You can be slightly warmer.", persona_warm)
