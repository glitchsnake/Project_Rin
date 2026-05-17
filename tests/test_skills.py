import asyncio
import unittest
from unittest.mock import patch, MagicMock
from skills import execute_python, execute_tool_async

class TestSkills(unittest.IsolatedAsyncioTestCase):
    async def test_execute_python_security_blacklist(self):
        """Verify that blacklisted Python modules/commands are securely blocked."""
        unsafe_snippets = [
            "import os\nos.system('ls')",
            "import sys\nsys.exit()",
            "eval('print(1)')",
            "exec('x = 10')",
            "open('/etc/passwd', 'r')",
            "__import__('subprocess').run(['ls'])",
            "import urllib.request"
        ]
        for snippet in unsafe_snippets:
            res = await execute_python(snippet)
            self.assertIn("SECURITY BLOCKED", res, f"Failed to block unsafe snippet: {snippet}")

    async def test_execute_python_syntax_error(self):
        """Verify that code with syntax errors returns a descriptive error message."""
        bad_code = "print('hello'"
        res = await execute_python(bad_code)
        self.assertIn("Error", res)

    async def test_execute_python_valid_code(self):
        """Verify execution of safe basic mathematical calculations."""
        safe_code = "print(2 + 2)"
        res = await execute_python(safe_code)
        self.assertIn("4", res)

    @patch('skills.asyncio.wait_for')
    async def test_execute_python_timeout(self, mock_wait_for):
        """Verify that infinite loops or slow scripts are stopped by timeout."""
        mock_wait_for.side_effect = asyncio.TimeoutError()
        slow_code = "import time\nwhile True:\n    time.sleep(1)"
        res = await execute_python(slow_code)
        self.assertIn("timeout", res.lower())

    async def test_execute_tool_async_unregistered(self):
        """Verify that requesting an unregistered tool returns an error."""
        res = await execute_tool_async("non_existent_tool", {"arg": 1})
        self.assertIn("not found", res.lower())

    @patch('skills.aiohttp.ClientSession.get')
    async def test_search_wikipedia_mocked(self, mock_get):
        """Verify wikipedia searching using dynamic mock aiohttp request."""
        # Mock ClientSession context and JSON response
        mock_response = MagicMock()
        mock_response.status = 200
        
        async def mock_json():
            return {
                "query": {
                    "search": [
                        {"title": "Python", "snippet": "Python is a high-level programming language."}
                    ]
                }
            }
        mock_response.json = mock_json
        
        # Setup session.get context manager return value
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value = mock_response
        mock_get.return_value = mock_ctx

        res = await execute_tool_async("search_wikipedia", {"query": "Python"})
        self.assertIn("Python", res)
        self.assertIn("Python is a high-level programming language.", res)
