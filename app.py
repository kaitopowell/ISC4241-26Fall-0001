from pathlib import Path

from flask import Flask, send_from_directory

app = Flask(__name__)

# match-replay/ lives next to this file, at the repo root.
REPLAY_DIR = Path(__file__).resolve().parent / "match-replay"


@app.get("/")
def home():
    return send_from_directory(REPLAY_DIR, "index.html")


if __name__ == "__main__":
    app.run(debug=True)
