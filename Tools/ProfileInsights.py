from __future__ import annotations

import logging
import numpy as np

Logger = logging.getLogger(__name__)

def CalculateOverallSentiment(Posts: list[dict]) -> dict:
    try:
        AllScores = [Post["Post_Text_Label"] for Post in Posts]
        AvgNegative = float(np.mean([S[0] for S in AllScores]))
        AvgNeutral = float(np.mean([S[1] for S in AllScores]))
        AvgPositive = float(np.mean([S[2] for S in AllScores]))

        Labels = ["Negative", "Neutral", "Positive"]
        Averages = [AvgNegative, AvgNeutral, AvgPositive]
        DominantSentiment = Labels[int(np.argmax(Averages))]

        Logger.info("Overall Sentiment Calculated Successfully")
        return {
            "AverageNegative": round(AvgNegative, 4),
            "AverageNeutral": round(AvgNeutral, 4),
            "AveragePositive": round(AvgPositive, 4),
            "DominantSentiment": DominantSentiment,
        }
    except Exception as e:
        Logger.error("Error In Calculating Overall Sentiment", exc_info=e)
        raise e

def CalculateEngagementMetrics(Posts: list[dict]) -> dict:
    try:
        TotalPosts = len(Posts)
        TotalLikes = sum(int(Post.get("Likes_Count", 0)) for Post in Posts)
        TotalComments = sum(int(Post.get("Comments_Count", 0)) for Post in Posts)
        TotalEngagement = TotalLikes + TotalComments

        AvgLikes = round(TotalLikes / TotalPosts, 2) if TotalPosts > 0 else 0
        AvgComments = round(TotalComments / TotalPosts, 2) if TotalPosts > 0 else 0
        AvgEngagement = round(TotalEngagement / TotalPosts, 2) if TotalPosts > 0 else 0

        LikesList = [int(Post.get("Likes_Count", 0)) for Post in Posts]
        CommentsList = [int(Post.get("Comments_Count", 0)) for Post in Posts]
        MaxLikes = max(LikesList) if LikesList else 0
        MinLikes = min(LikesList) if LikesList else 0
        MaxComments = max(CommentsList) if CommentsList else 0
        MinComments = min(CommentsList) if CommentsList else 0

        Logger.info("Engagement Metrics Calculated Successfully")
        return {
            "TotalPosts": TotalPosts,
            "TotalLikes": TotalLikes,
            "TotalComments": TotalComments,
            "TotalEngagement": TotalEngagement,
            "AverageLikes": AvgLikes,
            "AverageComments": AvgComments,
            "AverageEngagement": AvgEngagement,
            "MaxLikes": MaxLikes,
            "MinLikes": MinLikes,
            "MaxComments": MaxComments,
            "MinComments": MinComments,
        }
    except Exception as e:
        Logger.error("Error In Calculating Engagement Metrics", exc_info=e)
        raise e

def CalculateSentimentTrend(Posts: list[dict]) -> list[dict]:
    try:
        Trend = []
        SortedPosts = sorted(Posts, key=lambda P: str(P.get("Post_Date", "")))

        for Post in SortedPosts:
            Scores = Post["Post_Text_Label"]
            Trend.append({
                "PostDate": str(Post.get("Post_Date", "")),
                "PostId": str(Post.get("Post_ID", "")),
                "Negative": round(Scores[0], 4),
                "Neutral": round(Scores[1], 4),
                "Positive": round(Scores[2], 4),
            })

        Logger.info("Sentiment Trend Calculated Successfully")
        return Trend
    except Exception as e:
        Logger.error("Error In Calculating Sentiment Trend", exc_info=e)
        raise e

def CalculateEngagementSentimentCorrelation(Posts: list[dict]) -> list[dict]:
    try:
        DataPoints = []
        for Post in Posts:
            Scores = Post["Post_Text_Label"]
            Likes = int(Post.get("Likes_Count", 0))
            Comments = int(Post.get("Comments_Count", 0))
            PositiveScore = Scores[2]

            DataPoints.append({
                "PostId": str(Post.get("Post_ID", "")),
                "PositiveScore": round(PositiveScore, 4),
                "Likes": Likes,
                "Comments": Comments,
                "TotalEngagement": Likes + Comments,
            })

        Logger.info("Engagement Sentiment Correlation Calculated Successfully")
        return DataPoints
    except Exception as e:
        Logger.error("Error In Calculating Engagement Sentiment Correlation", exc_info=e)
        raise e

def CalculateCommentsSentimentSummary(Posts: list[dict]) -> dict:
    try:
        AllNegative = []
        AllNeutral = []
        AllPositive = []

        for Post in Posts:
            CommentsLabel = Post.get("Comments_Label", {})
            for Scores in CommentsLabel.values():
                AllNegative.append(Scores[0])
                AllNeutral.append(Scores[1])
                AllPositive.append(Scores[2])

        TotalComments = len(AllNegative)

        if TotalComments == 0:
            Logger.info("No Comments Found For Sentiment Summary")
            return {
                "TotalComments": 0,
                "AverageNegative": 0,
                "AverageNeutral": 0,
                "AveragePositive": 0,
                "DominantSentiment": "N/A",
            }

        AvgNeg = float(np.mean(AllNegative))
        AvgNeu = float(np.mean(AllNeutral))
        AvgPos = float(np.mean(AllPositive))

        Labels = ["Negative", "Neutral", "Positive"]
        Averages = [AvgNeg, AvgNeu, AvgPos]
        DominantSentiment = Labels[int(np.argmax(Averages))]

        Logger.info("Comments Sentiment Summary Calculated Successfully")
        return {
            "TotalComments": TotalComments,
            "AverageNegative": round(AvgNeg, 4),
            "AverageNeutral": round(AvgNeu, 4),
            "AveragePositive": round(AvgPos, 4),
            "DominantSentiment": DominantSentiment,
        }
    except Exception as e:
        Logger.error("Error In Calculating Comments Sentiment Summary", exc_info=e)
        raise e

def CalculateTopPosts(Posts: list[dict]) -> dict:
    try:
        if not Posts:
            Logger.info("No Posts Found For Top Posts Calculation")
            return {"MostPositive": None, "MostNegative": None, "MostEngaged": None}

        MostPositive = max(Posts, key=lambda P: P["Post_Text_Label"][2])
        MostNegative = max(Posts, key=lambda P: P["Post_Text_Label"][0])
        MostEngaged = max(Posts, key=lambda P: int(P.get("Likes_Count", 0)) + int(P.get("Comments_Count", 0)))

        def FormatPost(Post):
            return {
                "PostId": str(Post.get("Post_ID", "")),
                "PostText": str(Post.get("Post_Text", "")),
                "PostUrl": str(Post.get("Post_URL", "")),
                "Likes": int(Post.get("Likes_Count", 0)),
                "Comments": int(Post.get("Comments_Count", 0)),
                "SentimentScores": Post["Post_Text_Label"],
            }

        Logger.info("Top Posts Calculated Successfully")
        return {
            "MostPositive": FormatPost(MostPositive),
            "MostNegative": FormatPost(MostNegative),
            "MostEngaged": FormatPost(MostEngaged),
        }
    except Exception as e:
        Logger.error("Error In Calculating Top Posts", exc_info=e)
        raise e

def GenerateProfileInsights(Posts: list[dict]) -> dict:
    try:
        Insights = {
            "OverallSentiment": CalculateOverallSentiment(Posts),
            "EngagementMetrics": CalculateEngagementMetrics(Posts),
            "SentimentTrend": CalculateSentimentTrend(Posts),
            "EngagementSentimentCorrelation": CalculateEngagementSentimentCorrelation(Posts),
            "CommentsSentimentSummary": CalculateCommentsSentimentSummary(Posts),
            "TopPosts": CalculateTopPosts(Posts),
        }

        Logger.info("Profile Insights Generated Successfully")
        return Insights
    except Exception as e:
        Logger.error("Error In Generating Profile Insights", exc_info=e)
        raise e