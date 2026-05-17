# Rin — Your Digital Companion 🌌

Rin is a personal AI companion project with deep emotional processing. Unlike generic assistants, Rin strives for maximum human-likeness through a multi-layered character, variable moods, and the ability to simulate complex psychological states.

## ✨ Features

- **Complex Character**: Rin can be depressive, supportive, or dismissive depending on the context and communication history.
- **Dynamic Emotionality**: The system changes states (pensiveness, confusion, dry sarcasm) on the fly, creating the effect of live conversation.
- **Long-Term Memory**: Uses a vector database (ChromaDB) to remember important details about the user.
- **Multi-Stage Cognition (Think Graph)**: A built-in "Think Engine" allows the model to analyze its thoughts and plan its tactics before responding, dramatically reducing hallucinations.
- **Voice Interaction**: Built-in support for audio transcription via Whisper.
- **Skills Integration**: The ability to securely use external tools such as Wikipedia search and sandboxed Python calculations.

## 🛠 Tech Stack

- **Core**: Python 3.10+
- **Interface**: [aiogram 3.x](https://github.com/aiogram/aiogram) (Telegram Bot API)
- **AI Engine**: OpenAI Compatible API (optimized for LM Studio / Local LLMs)
- **Memory**: ChromaDB + Sentence-Transformers + Tiktoken
- **Skills**: Wikipedia API, wttr.in, Custom Async Skill Sandbox
- **Voice**: Whisper (local or API)

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Project_Rin.git
   cd Project_Rin
   ```

2. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   # venv\Scripts\activate   # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r req.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory (see `.env.template` for reference) and set your tokens:
   ```env
   TELEGRAM_BOT_TOKEN="your_token_here"
   ```

5. **Run the bot**:
   ```bash
   python main.py
   ```

## 📝 Project Status

The project is under active development and research into minimizing hallucinations and optimizing response speeds.

---
**License**: [Apache License 2.0](LICENSE)  
**Author**: Loki  

> [!IMPORTANT]
> **Important Note**: The project does not contain a built-in pre-trained model. The project acts strictly as a tool (framework), not as a ready service or product. The model must be connected separately (e.g., via LM Studio or any other OpenAI-compatible API).

был многократно использован исскуственный интелект для кода
