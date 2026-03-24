from __future__ import annotations

import re
import ast
import logging
import pandas as pd

from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

Logger = logging.getLogger(__name__)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

Tokenizer = None
Model = None


def LoadModel():
    global Tokenizer, Model
    try:
        if Tokenizer is None or Model is None:
            Logger.info("Loading Model: %s", MODEL_NAME)
            Tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            Model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            Logger.info("Model Loaded Successfully")
    except Exception as e:
        Logger.error("Failed To Load Model", exc_info=e)
        raise e


def ClassifyText(Text: str) -> list[float]:
    try:
        LoadModel()
        Encoded = Tokenizer(Text, return_tensors="pt", truncation=True, max_length=512)
        Output = Model(**Encoded)
        Scores = Output.logits[0].detach().numpy()
        return softmax(Scores).tolist()
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
