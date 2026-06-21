# Hotel Summary Web

Next.js dashboard for the SPACE hotel summarization outputs.

## Local frontend

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

## Optional backend

The frontend works without a backend by reading `public/data/*.json`. To use
the FastAPI service:

```bash
cd ../backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then set `web/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Deploy

- Frontend: deploy `web/` to Vercel or any Node host.
- Backend: deploy `backend/` to a Python host if dynamic API/search is needed.
- Model inference is intentionally not part of the frontend. Put model-serving
  endpoints behind the FastAPI service on GPU/CPU infrastructure.
