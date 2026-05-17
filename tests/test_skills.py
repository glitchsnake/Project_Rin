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

    @patch('skills.is_docker_available')
    @patch('skills.asyncio.wait_for')
    async def test_execute_python_timeout(self, mock_wait_for, mock_is_docker):
        """Verify that infinite loops or slow scripts are stopped by timeout."""
        mock_is_docker.return_value = False
        mock_wait_for.side_effect = asyncio.TimeoutError()
        slow_code = "import time\nwhile True:\n    time.sleep(1)"
        res = await execute_python(slow_code)
        self.assertIn("Превышен лимит времени выполнения", res)

    async def test_execute_tool_async_unregistered(self):
        """Verify that requesting an unregistered tool returns an error."""
        res = await execute_tool_async("non_existent_tool", {"arg": 1})
        self.assertIn("не найден", res)

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

    @patch('skills.is_docker_available')
    @patch('skills.asyncio.create_subprocess_exec')
    async def test_execute_python_docker_success(self, mock_create, mock_is_docker):
        """Verify that Python execution via Docker sandbox runs cleanly."""
        mock_is_docker.return_value = True
        
        # Mock docker process
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        async def mock_communicate():
            return b"docker output\n", b""
        mock_proc.communicate = mock_communicate
        
        mock_create.return_value = mock_proc
        
        res = await execute_python("print('hello')")
        # Ensure it tried to run "docker"
        self.assertEqual(mock_create.call_args[0][0], "docker")
        self.assertIn("docker output", res)

    @patch('skills.is_docker_available')
    @patch('skills.asyncio.create_subprocess_exec')
    async def test_execute_python_docker_fallback(self, mock_create, mock_is_docker):
        """Verify that if Docker daemon is missing or errors out, sandbox falls back to subprocess."""
        mock_is_docker.return_value = True
        
        # Mock docker call raising FileNotFoundError (e.g. docker not installed)
        # And next subprocess call succeeding
        mock_docker_proc = MagicMock()
        mock_docker_proc.returncode = 125 # Docker error
        async def mock_docker_communicate():
            return b"", b"docker failed"
        mock_docker_proc.communicate = mock_docker_communicate
        
        mock_sub_proc = MagicMock()
        mock_sub_proc.returncode = 0
        async def mock_sub_communicate():
            return b"subprocess fallback output\n", b""
        mock_sub_proc.communicate = mock_sub_communicate
        
        mock_create.side_effect = [mock_docker_proc, mock_sub_proc]
        
        res = await execute_python("print('hello')")
        self.assertIn("subprocess fallback output", res)
