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
            "привет",
            "как дела",
            "ясно",
            "понятно",
            "ок",
            "угу",
            "мм",
            "спокойной ночи",
            "доброе утро"
        ]
        for inp in fast_inputs:
            state = ThinkState(user_text=inp)
            route = _node_router(state)
            self.assertEqual(route, "system_1", f"Failed for input: {inp}")

    def test_node_router_deep_track(self):
        """Verify that longer or non-greeting messages trigger System 2 (deep track)."""
        deep_inputs = [
            "Расскажи мне о теории струн и квантовой гравитации.",
            "Почему ты вчера не ответила мне на сообщение?",
            "Что ты думаешь о творчестве Достоевского?",
            "Купи молоко по дороге домой"
        ]
        for inp in deep_inputs:
            state = ThinkState(user_text=inp)
            route = _node_router(state)
            self.assertEqual(route, "system_2", f"Failed for input: {inp}")

    def test_vulnerability_override(self):
        """Verify that vulnerability signals override hostile/aggressive emotions to 'равнодушие'."""
        # Vulnerable input with aggressive emotion should map to 'равнодушие'
        vuln_inputs = ["мне грустно", "я так устал", "мне одиноко и больно"]
        aggressive_emotions = ["сухой сарказм", "тихое презрение", "раздражённая скука", "раздражение"]
        
        for inp in vuln_inputs:
            for emotion in aggressive_emotions:
                res = _check_vulnerability(inp, warmth=0.2, rin_emotion=emotion)
                self.assertEqual(res, "равнодушие", f"Failed to override {emotion} for input: {inp}")

        # When warmth <= 0, no overrides should occur
        for inp in vuln_inputs:
            for emotion in aggressive_emotions:
                res = _check_vulnerability(inp, warmth=-0.1, rin_emotion=emotion)
                self.assertEqual(res, emotion, f"Should not override when warmth <= 0")

        # Non-vulnerable input should not override any emotions
        safe_input = "сегодня хорошая погода"
        for emotion in aggressive_emotions:
            res = _check_vulnerability(safe_input, warmth=0.5, rin_emotion=emotion)
            self.assertEqual(res, emotion, "Should not override non-vulnerable inputs")

    def test_llm_think_output_validators(self):
        """Verify that LLMThinkOutput sanitizes and validates emotions, attitudes, and tactics."""
        # Valid data passes through untouched
        valid_data = {
            "fact_analysis": "Юзер спросил про погоду",
            "hidden_intent": "Желание начать беседу",
            "rin_inner_conflict": "ФАКТ -> АНАЛИЗ -> РЕАКЦИЯ",
            "confidence": 0.85,
            "rin_emotion": "задумчивость",
            "rin_attitude": "нейтральное",
            "response_tactic": "короткая реакция"
        }
        output = LLMThinkOutput(**valid_data)
        self.assertEqual(output.rin_emotion, "задумчивость")
        self.assertEqual(output.rin_attitude, "нейтральное")
        self.assertEqual(output.response_tactic, "короткая реакция")

        # Invalid data triggers validator fallbacks
        invalid_data = {
            "fact_analysis": "Юзер агрессивен",
            "hidden_intent": "Спровоцировать",
            "rin_inner_conflict": "ФАКТ -> АНАЛИЗ -> РЕАКЦИЯ",
            "confidence": 0.9,
            "rin_emotion": "НЕВЕДОМАЯ_ЭМОЦИЯ",
            "rin_attitude": "НЕВЕДОМОЕ_ОТНОШЕНИЕ",
            "response_tactic": "НЕВЕДОМАЯ_ТАКТИКА"
        }
        output_invalid = LLMThinkOutput(**invalid_data)
        self.assertEqual(output_invalid.rin_emotion, "отстраненность", "Invalid emotion must default to 'отстраненность'")
        self.assertEqual(output_invalid.rin_attitude, "нейтральное", "Invalid attitude must default to 'нейтральное'")
        self.assertEqual(output_invalid.response_tactic, "короткая реакция", "Invalid tactic must default to 'короткая реакция'")

    def test_build_persona_block_gradients(self):
        """Verify that _build_persona_block renders custom persona segments according to warmth gradients."""
        # 1. Cold styling (warmth < 0)
        persona_cold = _build_persona_block(
            warmth=-0.2,
            base_attitude="настороженное",
            user_name="Алексей",
            core_memory="любит кошек",
            persona_narrative="он странный"
        )
        self.assertIn("Алексей", persona_cold)
        self.assertIn("настороженное", persona_cold)
        self.assertIn("любит кошек", persona_cold)
        self.assertIn("он странный", persona_cold)
        self.assertIn("ДИНАМИКА: Собеседник неприятен. Сухо, отстраненно.", persona_cold)

        # 2. Medium styling (0 <= warmth <= 0.5)
        persona_mid = _build_persona_block(
            warmth=0.3,
            base_attitude="нейтральное",
            user_name="Алексей",
            core_memory="",
            persona_narrative=""
        )
        self.assertIn("ДИНАМИКА: Знакомый. Дистанция, краткость.", persona_mid)
        self.assertNotIn("[Факты]", persona_mid)

        # 3. Warm styling (warmth > 0.5)
        persona_warm = _build_persona_block(
            warmth=0.8,
            base_attitude="тёплое и доверительное",
            user_name="Алексей",
            core_memory="создатель",
            persona_narrative="он важен"
        )
        self.assertIn("ДИНАМИКА: Близкий человек. Можно быть чуть теплее.", persona_warm)
