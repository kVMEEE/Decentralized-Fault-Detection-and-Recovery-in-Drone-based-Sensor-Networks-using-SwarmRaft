from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import os

from web_app.headless_sim import run_simulation

app = FastAPI(title="SwarmRaft Web Demo")

# Serve static files for the frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Run the simulation and stream data to the client
        async for state in run_simulation(fps=30):
            await websocket.send_json(state)
            
            # If simulation completes, we could break or wait
            if state.get("status") == "complete":
                break
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
