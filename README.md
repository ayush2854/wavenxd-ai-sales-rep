# WaveNxD AI Sales Representative

An AI sales representative for **WaveNxD Technologies Pvt. Ltd.** (Kolhapur), maker of no-pressure
ultrasonic spray nozzle systems. It chats with a prospective client the way a good sales engineer would:
it introduces itself, explains the product, finds out the client's application, recommends the right
nozzle **from the product catalog only**, and books the next step (a quotation email and a follow-up call).

Built for the NexaPrime Intern Evaluation Task (19 to 23 October 2026).

## Required features and where they live

| # | Feature | How it is done | Code |
|---|---------|----------------|------|
| 1 | Chat assistant | React chat UI talking to a FastAPI backend, multi-turn memory per session | `frontend/`, `backend/app/main.py`, `agent.py` |
| 2 | Self-introduction | The assistant speaks first: name, company, short pitch | `backend/app/prompts.py` |
| 3 | Multilingual replies | English by default; replies in the client's language and script when they switch (tested with Marathi and Hindi) | `prompts.py`, `extractor.py` |
| 4 | Product knowledge | Catalog loaded from `backend/data/catalog.json` into the prompt; no invented specs, prices or delivery times | `prompts.py`, `data/catalog.json` |
| 5 | Requirement discovery | Conversation state tracks industry, application, material, viscosity, flow rate, droplet size; the next question comes from what is still unknown | `state.py`, `agent.py` |
| 6 | Product recommendation | Deterministic recommender scores catalog nozzles and explains why | `recommender.py` |
| 7 | Quotation and follow-up | Collects name, company, email, phone, call time; stores the lead; emails a quotation request with a calendar invite (`.ics`) | `quotation.py`, `emailer.py`, `leads.py` |

Bonus items included: a clean UI, real email sending (SMTP), a calendar invite for the follow-up call, and
optional **voice input and spoken replies** in the browser (see below).

## How it works

Each client message goes through three steps (details and diagrams in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. **Extract:** one low-temperature Gemini call turns the latest message into structured facts
   (application, viscosity, flow rate, contact details, ...).
2. **Decide, in plain code:** the conversation state is updated, the recommender runs when the requirement
   is known, and the quotation is submitted when all details are valid.
3. **Reply:** a second Gemini call writes the answer, given the verified results as "live context".
   A guard rewrites any reply that claims something was booked or emailed when it was not.

Why not let the model call tools itself? In testing, the model often skipped tool calls and answered from
its prompt. Moving the decisions into code makes recommendations and bookings reliable, and the model can
only report what really happened.

## Tech stack

- **Backend:** Python 3.10+, FastAPI, `google-genai` (Gemini), SQLite, `smtplib`
- **Frontend:** React 18 + TypeScript + Vite
- **Models:** Gemini, configurable in `.env` (main, optional fallback, optional extraction model)
- **Tests:** pytest, no network needed (a scripted fake Gemini is used)

## Setup

You need Python 3.10+, Node.js 18+ and a Gemini API key.

```bash
git clone https://github.com/ayush2854/wavenxd-ai-sales-rep.git
cd wavenxd-ai-sales-rep

# backend
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env        # Windows: copy ..\.env.example .env
# edit backend/.env and set GEMINI_API_KEY (and optionally the email settings)
```

### Run: demo mode (one server)

```bash
cd frontend
npm install
npm run build
cd ../backend
uvicorn app.main:app
```
Open http://localhost:8000

### Run: development mode (two terminals)

```bash
# terminal 1
cd backend && uvicorn app.main:app --reload
# terminal 2
cd frontend && npm install && npm run dev
```
Open http://localhost:5173

### Terminal chat (no UI)

```bash
cd backend && python cli.py
```
The terminal shows `Controller:` log lines for every recommendation and quotation, which makes the
behaviour easy to follow.

### Tests

```bash
cd backend
python -m pytest -q
```

### See the stored leads

```bash
cd backend
python show_leads.py
```

## Configuration (`backend/.env`)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Your Gemini key (required) |
| `GEMINI_MODEL` | Main model for replies |
| `GEMINI_FALLBACK_MODEL` | Optional backup used automatically if the main model is overloaded (503) |
| `GEMINI_EXTRACT_MODEL` | Optional stronger model for reading the client's message |
| `GEMINI_TIMEOUT_SECONDS` | Wait time before a Gemini call is retried (default 30) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | Email sending, for example Gmail with an App Password |
| `SENDER_NAME`, `SENDER_EMAIL` | The "From" name and address of quotation emails |
| `SALES_TEAM_EMAIL` | Optional hidden copy of every quotation request |
| `CORS_ORIGINS` | Allowed origins in development |

If the SMTP settings are empty, emails are **saved** to `backend/data/outbox/` instead of sent, and the
assistant says the team will send the quotation (it never claims an email was sent when it was not).
Secrets live only in `backend/.env`, which is git-ignored.

## Project structure

```
backend/
  app/
    agent.py        controller: extract, decide, reply, honesty guard
    extractor.py    structured extraction of facts from the client's message
    state.py        what is known about the client so far
    recommender.py  deterministic nozzle recommender (reads the catalog only)
    prompts.py      persona, rules and the catalog rendered for the prompt
    quotation.py    validation, lead storage, email, follow-up booking
    emailer.py      quotation email and .ics calendar invite
    leads.py        SQLite lead storage
    main.py         FastAPI app (also serves the built UI)
  data/catalog.json product knowledge base (single source of truth)
  tests/            unit and flow tests
  cli.py, show_leads.py
frontend/           React chat UI
docs/               architecture, demo script, presentation notes
```

## Voice (optional bonus)

In Chrome or Edge the chat shows a voice bar: choose a language (English, Hindi, Marathi, Tamil), press
**Speak**, talk, and the message is sent automatically. Tick **Read replies aloud** to hear the answers.
It uses the browser's Web Speech API, so there is nothing to install, but:

- voice input needs an internet connection and microphone permission;
- spoken replies depend on the voices installed on the device (if a language has no voice, the UI says so
  and the browser default is used);
- the voice bar is hidden in browsers without speech support, and typing always works.

## Honesty and safety rules

- Product answers come from `catalog.json` only. Price, delivery time and any unlisted detail get
  "the WaveNxD team will confirm".
- Out-of-range requests (for example 5000 cP at 500 ml/min) get an honest "no listed product covers that".
- Off-topic requests (jokes, politics) are politely declined.
- Only sample or test details are used; no real customer data is stored or committed.

## Known limitations and ideas for improvement

- Sessions are kept in memory (a restart clears them); a database or Redis would fix that.
- Two Gemini calls per message add some latency; streaming replies would feel faster.
- Call times are proposed by the assistant; connecting a real calendar (availability and booking) is the
  next step.
- The guard against false "booked/emailed" claims matches English wording only.
- Voice input and output, and a larger catalog with retrieval, are natural extensions.

## Development note

Built with the help of AI coding tools, as permitted by the task. All parts of the project are understood
and can be explained.