from fastapi import FastAPI
from contextlib import asynccontextmanager



@asynccontextmanager
def lifespan(app: FastAPI):
    load_model()


app = FastAPI(lifespan=lifespan)