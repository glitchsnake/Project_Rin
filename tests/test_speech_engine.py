import unittest
from speech_engine import (
    _clean_output,
    build_generation_logit_bias,
    init_speech_engine,
    _PRECOMPUTED_COLD_BIAS,
    _PRECOMPUTED_WARM_BIAS
)
import speech_engine

class TestSpeechEngine(unittest.TestCase):
    def test_clean_output_basic(self):
        """Verify that _clean_output strips basic whitespaces and returns standard text."""
        self.assertEqual(_clean_output(" Hello. "), "Hello.")
        self.assertEqual(_clean_output(""), ".")

    def test_clean_output_thinking_tags(self):
        """Verify that <thinking> tags and their contents are successfully stripped."""
        raw_text = "<thinking>Must answer dryly.</thinking>Hello."
        self.assertEqual(_clean_output(raw_text), "Hello.")

        nested_text = "<thinking>FACT: Question\nANALYSIS: Simple\nREACTION: Cold</thinking><response>Fine.</response>"
        self.assertEqual(_clean_output(nested_text), "Fine.")

    def test_clean_output_response_tags(self):
        """Verify that <response> tags are parsed and only inner text is retrieved."""
        self.assertEqual(_clean_output("<response>Sure.</response>"), "Sure.")
        self.assertEqual(_clean_output("   <response>  Okay.   </response> "), "Okay.")

    def test_clean_output_leakage_markers(self):
        """Verify that prompt leakage markers trigger fallback to the first sentence or dot."""
        # Contains "[ref: " marker, should extract first sentence
        leaked_text = "Sure. [ref: Wikipedia]"
        self.assertEqual(_clean_output(leaked_text), "Sure")

        # Contains "my tactic" marker, should extract first sentence
        leaked_text_2 = "I'm here. My tactic - keep silent."
        self.assertEqual(_clean_output(leaked_text_2), "I'm here")

        # Single word leaked with marker should fall back to "."
        leaked_text_3 = "[ref: ]"
        self.assertEqual(_clean_output(leaked_text_3), "[ref: ]")

    def test_build_generation_logit_bias(self):
        """Verify logit bias retrieval based on warmth level and initialization status."""
        # Uninitialized engine should return empty bias
        speech_engine._ENGINE_INITIALIZED = False
        self.assertEqual(build_generation_logit_bias(-0.5), {})
        self.assertEqual(build_generation_logit_bias(0.8), {})
        self.assertEqual(build_generation_logit_bias(0.0), {})

        # Inject fake precomputed values to test retrieval when initialized
        speech_engine._ENGINE_INITIALIZED = True
        speech_engine._PRECOMPUTED_COLD_BIAS = {"123": -5.0}
        speech_engine._PRECOMPUTED_WARM_BIAS = {"456": 1.5}

        # Cold warmth (< 0) -> cold bias
        self.assertEqual(build_generation_logit_bias(-0.1), {"123": -5.0})
        # Warm warmth (> 0.5) -> warm bias
        self.assertEqual(build_generation_logit_bias(0.6), {"456": 1.5})
        # Neutral warmth (0.0 to 0.5) -> empty bias
        self.assertEqual(build_generation_logit_bias(0.2), {})

        # Reset initialized state
        speech_engine._ENGINE_INITIALIZED = False
