from __future__ import annotations

import os
import re
import ast
import logging
import pandas as pd

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

Logger = logging.getLogger(__name__)
MODEL_NAME = os.getenv("MODEL_NAME")
HF_API_URL = os.getenv("HF_API_URL") + MODEL_NAME
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
BATCH_SIZE = 32
MAX_WORKERS = 8


def ClassifyBatch(Texts: list[str]) -> list[list[float]]:
    try:
        Headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        Response = requests.post(HF_API_URL, headers=Headers, json={"inputs": Texts})
        Response.raise_for_status()

        Results = Response.json()
        LabelMap = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

        Scores = []
        for Result in Results:
            S = [0.0, 0.0, 0.0]
            for Item in Result:
                Index = LabelMap.get(Item["label"])
                if Index is not None:
                    S[Index] = Item["score"]
            Scores.append(S)

        return Scores
    except Exception as e:
        Logger.error("Error In Batch Text Classification", exc_info=e)
        raise e


def ClassifyAll(Texts: list[str]) -> list[list[float]]:
    Batches = [Texts[i:i + BATCH_SIZE] for i in range(0, len(Texts), BATCH_SIZE)]
    Results = [None] * len(Batches)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as Executor:
        Futures = {Executor.submit(ClassifyBatch, Batch): Idx for Idx, Batch in enumerate(Batches)}
        for Future in as_completed(Futures):
            Idx = Futures[Future]
            Results[Idx] = Future.result()

    return [Score for Batch in Results for Score in Batch]


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


def ParseComments(Comments) -> list[str]:
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


def ClassifyTexts(Texts: list[str]) -> dict[str, list[float]]:
    UniqueTexts = list({T for T in Texts if T.strip()})
    Labels = ClassifyAll(UniqueTexts)
    TextLabelMap = dict(zip(UniqueTexts, Labels))
    Default = [0.0, 1.0, 0.0]
    return {T: TextLabelMap.get(T, Default) for T in Texts}


def AnalyzePosts(Data: list[dict]) -> list[dict]:
    try:
        Df = pd.DataFrame(Data)

        # Refine post texts
        Df["Post_Text"] = Df["Post_Text"].astype(str).apply(RefineText)

        # Batch classify all post texts
        PostTexts = Df["Post_Text"].tolist()
        PostLabelMap = ClassifyTexts(PostTexts)
        Df["Post_Text_Label"] = Df["Post_Text"].map(PostLabelMap)

        # Parse all comments
        Df["Comments"] = Df["Comments"].apply(ParseComments)

        # Collect all unique comments and batch classify
        AllComments = list({Comment for Comments in Df["Comments"] for Comment in Comments if Comment.strip()})
        AllLabels = ClassifyAll(AllComments)
        CommentLabelMap = dict(zip(AllComments, AllLabels))
        Default = [0.0, 1.0, 0.0]

        # Map back to each post
        Df["Comments_Label"] = Df["Comments"].apply(
            lambda Comments: {Comment: CommentLabelMap.get(Comment, Default) for Comment in Comments}
        )

        Logger.info("Sentiment Analysis Completed Successfully")
        return Df.to_dict(orient="records")
    except Exception as e:
        Logger.error("Error In Analyzing Posts", exc_info=e)
        raise e
