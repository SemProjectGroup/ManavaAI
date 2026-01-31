from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/humanize")
async def humanize(data: dict):
    return {
        "output_text": "This is a humanized version of your text.",
        "score": 95.5
    }
#from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware
#>>>>>>> Stashed changes