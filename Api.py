from __future__ import annotations

import io
import os
import logging
import pandas as pd

from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File

from Tools.SentimentAnalyzer import AnalyzePosts
from Tools.ProfileInsights import GenerateProfileInsights

load_dotenv()
FRONTEND_URL = os.getenv("FRONTEND_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

Logger = logging.getLogger(__name__)

App = FastAPI(
    title="Insta-Motions",
    description="API For Analyzing Sentiments Of Instagram Posts And Comments Using NLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
App.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PostsRequest(BaseModel):
    data: List[Dict]

@App.get("/", tags=["Health"])
def HealthCheck():
    try:
        Logger.info("Health Check Request Received")
        return {"status": "ok"}
    except Exception as e:
        Logger.error("Error In Health Check", exc_info=e)
        raise e

@App.post("/api/analyze", tags=["Analysis"])
def AnalyzeFromJson(Payload: PostsRequest):
    try:
        if not Payload.data:
            Logger.warning("Empty Data Payload Received")
            raise HTTPException(status_code=400, detail="No Data Provided. Please Send At Least One Post.")

        Df = pd.DataFrame(Payload.data)
        ValidateDataFrame(Df)

        Logger.info("Received JSON Payload For Analysis")
        Results = AnalyzePosts(Payload.data)
        Insights = GenerateProfileInsights(Results)
        Logger.info("JSON Analysis Completed Successfully")
        return {"results": Results, "insights": Insights}
    except HTTPException:
        raise
    except Exception as e:
        Logger.error("Error During Sentiment Analysis", exc_info=e)
        raise HTTPException(status_code=400, detail="An Unexpected Error Occurred During Analysis. Please Try Again.")

REQUIRED_COLUMNS = ["Post_ID", "Post_Text", "Post_Date", "Likes_Count", "Comments_Count", "Comments", "Post_URL"]

def ValidateDataFrame(Df: pd.DataFrame):
    try:
        if Df.empty:
            Logger.warning("Uploaded File Is Empty")
            raise HTTPException(status_code=400, detail="The Uploaded File Is Empty. Please Upload A File With Data.")

        MissingColumns = [Col for Col in REQUIRED_COLUMNS if Col not in Df.columns]
        if MissingColumns:
            Logger.warning("Missing Columns: %s", MissingColumns)
            raise HTTPException(
                status_code=400,
                detail=f"Missing Required Columns: {', '.join(MissingColumns)}. Please Check The Data Format Guide."
            )

        if Df["Post_Text"].isnull().all():
            Logger.warning("Post_Text Column Is Entirely Empty")
            raise HTTPException(status_code=400, detail="The Post_Text Column Is Empty. Each Row Must Have Caption Text.")

        if Df["Comments"].isnull().all():
            Logger.warning("Comments Column Is Entirely Empty")
            raise HTTPException(status_code=400, detail="The Comments Column Is Empty. Each Row Must Have Comments Data.")

        Logger.info("Data Validation Passed: %d Rows", len(Df))
    except HTTPException:
        raise
    except Exception as e:
        Logger.error("Error During Data Validation", exc_info=e)
        raise e

@App.post("/api/analyze/upload", tags=["Analysis"])
async def AnalyzeFromFile(FileUpload: UploadFile = File(...)):
    try:
        if not FileUpload.filename.endswith((".xlsx", ".xls")):
            Logger.warning("Invalid File Format Uploaded: %s", FileUpload.filename)
            raise HTTPException(status_code=400, detail="Only Excel Files (.xlsx) Are Supported. Please Upload A Valid .xlsx File.")

        Logger.info("Received File Upload: %s", FileUpload.filename)
        Contents = await FileUpload.read()

        if len(Contents) == 0:
            Logger.warning("Uploaded File Is Empty (0 Bytes)")
            raise HTTPException(status_code=400, detail="The Uploaded File Is Empty. Please Select A Valid Excel File.")

        try:
            Df = pd.read_excel(io.BytesIO(Contents), engine="openpyxl")
        except Exception:
            Logger.warning("Failed To Parse Excel File")
            raise HTTPException(status_code=400, detail="Unable To Read The File. Please Make Sure It Is A Valid .xlsx Excel File.")

        ValidateDataFrame(Df)

        Data = Df.to_dict(orient="records")
        Results = AnalyzePosts(Data)
        Insights = GenerateProfileInsights(Results)
        Logger.info("File Analysis Completed Successfully")
        return {"results": Results, "insights": Insights}
    except HTTPException:
        raise
    except Exception as e:
        Logger.error("Error Processing Uploaded File", exc_info=e)
        raise HTTPException(status_code=400, detail="An Unexpected Error Occurred While Processing The File. Please Try Again.")

if __name__ == "__main__":
    try:
        import uvicorn

        Logger.info("Starting Server")
        uvicorn.run("Api:App", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        Logger.error("Error Starting Server", exc_info=e)
        raise e