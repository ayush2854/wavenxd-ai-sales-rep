"""Chat with the assistant in the terminal.  Run from the backend folder:  python cli.py"""
import os
import sys

from dotenv import load_dotenv

from app.agent import SalesAgent

load_dotenv()


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    agent = SalesAgent()
    print("WaveNxD AI Sales Representative (type 'quit' to exit)\n")
    # The client speaks first; this greeting trigger makes the assistant introduce itself.
    print("Aarya:", agent.reply("cli", "Hello"), "\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            print("\nAarya:", agent.reply("cli", text), "\n")


if __name__ == "__main__":
    main()