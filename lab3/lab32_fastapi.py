from fastapi import FastAPI
import uvicorn
app = FastAPI()
@app.get("/")
def root():
    return {'Key': 'Salam  tt monde'}
@app.post("/detect")
def detect():
    return {'Key': 'detect'}

uvicorn.run(app, port=8000, host= '0.0.0.0')