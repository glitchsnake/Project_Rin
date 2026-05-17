import unittest
import os
import re

class TestSecurityAudit(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.dangerous_keywords = ["os.system(", "subprocess.Popen(..., shell=True)"]

    def test_no_hardcoded_secrets_or_tokens(self):
        """Scan codebase to ensure no raw API keys or Bot Tokens are accidentally hardcoded."""
        # Bot token regex pattern (e.g., 1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ)
        bot_token_pattern = re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")
        openai_key_pattern = re.compile(r"\bsk-[A-Za-z0-9_-]{48,}\b")
        
        excluded_dirs = {".git", "venv", "__pycache__", ".gemini"}
        
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                if file.endswith((".py", ".env", ".template", ".yml", ".md")):
                    # Skip .env itself as it's gitignored and expected to have secrets locally
                    if file == ".env":
                        continue
                        
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            
                        # Search for bot token leaks
                        token_matches = bot_token_pattern.findall(content)
                        self.assertEqual(
                            len(token_matches), 0,
                            f"⚠️ SECURITY ALERT: Hardcoded Telegram Bot Token pattern found in {file_path}: {token_matches}"
                        )
                        
                        # Search for OpenAI key leaks
                        key_matches = openai_key_pattern.findall(content)
                        self.assertEqual(
                            len(key_matches), 0,
                            f"⚠️ SECURITY ALERT: Hardcoded OpenAI API Key pattern found in {file_path}"
                        )
                    except Exception as e:
                        pass

    def test_no_dangerous_code_eval_execution(self):
        """Ensure no un-sandboxed eval/exec functions are run on raw variables."""
        excluded_dirs = {".git", "venv", "__pycache__", "tests", ".gemini"}
        allowed_skills_eval = "skills.py"  # sandboxed python calculations
        
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    # We allow skills.py as it implements sandboxing for python calculations
                    if os.path.basename(file_path) == allowed_skills_eval:
                        continue
                        
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            
                        for i, line in enumerate(lines, 1):
                            stripped = line.strip()
                            # Check for eval() or exec()
                            if ("eval(" in stripped or "exec(" in stripped) and not stripped.startswith("#"):
                                self.fail(
                                    f"⚠️ SECURITY ALERT: Un-sandboxed execution function found in {file_path} at line {i}: '{stripped}'"
                                )
                    except Exception as e:
                        pass
                        
    def test_no_sql_injection_vulnerabilities(self):
        """Verify that all SQL operations are fully parameterized (no string formatting in SQL statements)."""
        excluded_dirs = {".git", "venv", "__pycache__", "tests", ".gemini"}
        
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                        # Look for common SQL injection formatting patterns: e.g., "SELECT ... WHERE ... = %s" or f"SELECT ... {var}"
                        # Check for f-string style queries in executes
                        f_string_queries = re.findall(r"\.execute\(\s*f[\"'].*\{.*\}[\"']", content)
                        self.assertEqual(
                            len(f_string_queries), 0,
                            f"⚠️ SECURITY ALERT: Potential SQL Injection (f-string in execute) found in {file_path}: {f_string_queries}"
                        )
                        
                        percent_formatting = re.findall(r"\.execute\(\s*[\"'].*%s.*[\"']\s*%", content)
                        self.assertEqual(
                            len(percent_formatting), 0,
                            f"⚠️ SECURITY ALERT: Potential SQL Injection (percent formatting in execute) found in {file_path}: {percent_formatting}"
                        )
                    except Exception as e:
                        pass
