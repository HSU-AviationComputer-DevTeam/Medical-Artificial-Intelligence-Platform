# Korean PHI Masking Agent

This agent uses Microsoft Presidio to identify and mask sensitive information such as Korean phone numbers.
It exposes a `/a2a` endpoint compatible with Google A2A.

## Usage
- Run with: `uvicorn app.main:app --reload`
- POST to `http://localhost:8000/a2a` with JSON: `{ "input": "..." }`