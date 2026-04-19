# Insta-Motions - Backend

FastAPI Backend For Insta-Motions. Analyzes Instagram Post Sentiments And Generates Profile Insights Using NLP.

## How It Works

Upload An Excel File With Instagram Post Data, And It Runs Sentiment Analysis On Every Caption And Comment Using A Pre-Trained RoBERTa Model. It Also Crunches Profile-Level Statistics Like Engagement Trends, Sentiment Distribution, And Top Posts.

## Setup

1. Install Dependencies:

```bash
pip install -r requirements.txt
```

2. Run The Server:

```bash
uvicorn Api:App --reload --port 8000
```

Server Starts At `http://localhost:8000`.

## Dependencies

| Package          | Version |
| ---------------- | ------- |
| fastapi          | 0.115.6 |
| uvicorn          | 0.34.0  |
| python-dotenv    | 1.0.1   |
| pandas           | 2.2.1   |
| numpy            | 1.26.4  |
| openpyxl         | 3.1.5   |
| transformers     | 4.44.2  |
| torch            | 2.5.1   |
| scipy            | 1.14.1  |
| python-multipart | 0.0.20  |

## API

**`GET /`** — Health Check.

**`POST /api/analyze`** — Analyze Posts From JSON Payload.

**`POST /api/analyze/upload`** — Analyze Posts From Excel File Upload.

Request Body For `/api/analyze`:

```json
{
  "data": [
    {
      "Post_ID": "B1abc123",
      "Post_Text": "Having a Good Day!",
      "Post_Date": "2024-09-10",
      "Likes_Count": 123,
      "Comments_Count": 12,
      "Comments": "['Beautiful!', 'So lovely!']",
      "Image_URL": "https://image1.jpg",
      "Post_URL": "https://www.instagram.com/p/B1abc123/"
    }
  ]
}
```

## Project Structure

```
Backend/
├── Api.py                      — Routes And Server Setup
├── Tools/
│   ├── SentimentAnalyzer.py    — NLP Sentiment Analysis Pipeline
│   └── ProfileInsights.py      — Statistical Profile Insights
├── .env                        — Environment Config
├── requirements.txt            — Dependencies
├── .gitignore
└── README.md
```

## NLP Model

Uses **cardiffnlp/twitter-roberta-base-sentiment** From HuggingFace, Fine-Tuned On ~58M Tweets.

Outputs Three Scores Per Text: `[Negative, Neutral, Positive]` — Normalized Via Softmax, Summing To 1.0.


## Modules

### Text Preprocessing

Before Classification, All Text Goes Through The RefineText Pipeline:

1. Remove @Mentions
2. Remove #Hashtags
3. Remove Non-Alphanumeric Characters (Except Spaces)
4. Collapse Multiple Spaces Into One
5. Trim Whitespace

### Excel File Format

| Column         | Type    | Description                  |
| -------------- | ------- | ---------------------------- |
| Post_ID        | String  | Unique Post Identifier       |
| Post_Text      | String  | Caption Text                 |
| Post_Date      | String  | Post Date (YYYY-MM-DD)       |
| Likes_Count    | Integer | Number Of Likes              |
| Comments_Count | Integer | Number Of Comments           |
| Comments       | String  | List Of Comments As String   |
| Image_URL      | String  | Post Image URL               |
| Post_URL       | String  | Instagram Post URL           |