from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from shutil import make_archive

import numpy as np
import uvicorn
import os

from DQNAgent import Agent

app = FastAPI()
agent = Agent()

class LogSample(BaseModel):
    episode_reward_mean: float
    episode_reward_min: int
    episode_reward_max: int
    epsilon: float

class Sample(BaseModel):
    state: list[float]
    action: int = None
    reward: int = None 
    next_state: list[float] = None
    done: bool = None
    epsilon: float = None

@app.post("/")
def home():
    return "ok"

@app.post("/action")
def action(sample: Sample):
    action = agent.get_action(
        np.array(sample.state), 
        sample.epsilon
    )
    return JSONResponse(content=jsonable_encoder(action))

@app.post("/memorise")
def memorise(sample: Sample):
    agent.append_sample(
        sample.state,
        sample.action,
        sample.reward,
        sample.next_state,
        sample.done   
    )
    return "ok"

@app.post("/update")
def memorise():
    agent.update()
    return "ok"

@app.post("/nextepisode")
def nextepisode():
    agent.next_episode()
    return "ok"

@app.post("/savemodel")
def savemodel(name : str):
    agent.save(name)
    make_archive(name, 'zip', name)
    return FileResponse(path=os.path.join(os.getcwd(),f"{name}.zip"), filename=f"{name}.zip", media_type="application/zip")

@app.post("/loadmodel")
def savemodel(name : str):
    agent.load(name)
    return "ok"

@app.post("/log")
def log(logsample: LogSample):
    agent.log(episode_reward_mean=logsample.episode_reward_mean, 
              episode_reward_min=logsample.episode_reward_min, 
              episode_reward_max=logsample.episode_reward_max, 
              epsilon=logsample.epsilon)
    return "ok"

@app.post("/downloadlogs")
def downloadlogs():
    name = "logs"
    make_archive(name, 'zip', name)
    return FileResponse(path=os.path.join(os.getcwd(),f"{name}.zip"), filename=f"{name}.zip", media_type="application/zip")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)