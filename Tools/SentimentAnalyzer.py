from __future__ import annotations

import os
import re
import ast
import logging
import pandas as pd

import requests
from dotenv import load_dotenv

load_dotenv()

Logger = logging.getLogger(__name__)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


def ClassifyText(Text: str) -> list[float]:
    try:
        Headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        Response = requests.post(HF_API_URL, headers=Headers, json={"inputs": Text})
        Response.raise_for_status()

        Result = Response.json()

        if isinstance(Result, list) and isinstance(Result[0], list):
            Result = Result[0]

        LabelMap = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
        Scores = [0.0, 0.0, 0.0]
        for Item in Result:
            Index = LabelMap.get(Item["label"])
            if Index is not None:
                Scores[Index] = Item["score"]

        return Scores
    except Exception as e:
        Logger.error("Error In Text Classification", exc_info=e)
        raise e


def RefineText(Text: str) -> str:
    try:
        Text = re.sub(r"@\w*", "", Text)
        Text = re.sub(r"#\w+", "", Text)
        Text = re.sub(r"[^A-Za-z0-9\s]", "", Text)
        Text = re.sub(r"\s+", " ", Text).strip()
        return Text
    except Exception as e:
        Logger.error("Error In Text Refinement", exc_info=e)
        raise e


def ParseComments(Comments):
    try:
        if isinstance(Comments, str):
            try:
                Comments = ast.literal_eval(Comments)
            except (ValueError, SyntaxError):
                Comments = [Comments]
        return [RefineText(Comment) for Comment in Comments]
    except Exception as e:
        Logger.error("Error In Parsing Comments", exc_info=e)
        raise e


def AnalyzePosts(Data: list[dict]) -> list[dict]:
    try:
        Df = pd.DataFrame(Data)

        Df["Post_Text"] = Df["Post_Text"].astype(str).apply(RefineText)
        Df["Post_Text_Label"] = Df["Post_Text"].apply(ClassifyText)

        Df["Comments"] = Df["Comments"].apply(ParseComments)

        Df["Comments_Label"] = Df["Comments"].apply(
            lambda Comments: {Comment: ClassifyText(Comment) for Comment in Comments}
        )

        Logger.info("Sentiment Analysis Completed Successfully")
        return Df.to_dict(orient="records")
    except Exception as e:
        Logger.error("Error In Analyzing Posts", exc_info=e)
        raise e
