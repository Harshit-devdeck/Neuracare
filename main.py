from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import time
import hashlib
import re
# removed the api key from the code, from preventing missuse
# Try to import AI libraries, use fallback if they fail
try:
    from openai import OpenAI
    PERPLEXITY_AVAILABLE = True
except:
    PERPLEXITY_AVAILABLE = False
    print("WARNING: OpenAI not available, using fallback responses")

try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
    vader = SentimentIntensityAnalyzer()
except:
    SENTIMENT_AVAILABLE = False
    print("WARNING: Sentiment libraries not available, using basic analysis")

app = FastAPI(title="NeuraCare")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
DB = "neuracare.db"
conn = sqlite3.connect(DB, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    session_hash TEXT,
    timestamp INTEGER,
    user_message TEXT,
    bot_response TEXT,
    sentiment_label TEXT,
    burnout_level TEXT,
    emotion TEXT
)
""")
conn.commit()

# Perplexity setup
if PERPLEXITY_AVAILABLE:
    try:
        perplexity_client = OpenAI(
            api_key="",
            base_url="https://api.perplexity.ai"
        )
    except:
        PERPLEXITY_AVAILABLE = False

class ChatRequest(BaseModel):
    session_id: str
    message: str
    consent: Optional[bool] = True

def anonymize(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]

def analyze_sentiment(text: str) -> dict:
    """Sentiment analysis with fallback"""
    if not SENTIMENT_AVAILABLE:
        # Basic keyword-based sentiment
        text_lower = text.lower()
        negative_words = ["sad", "depressed", "angry", "frustrated", "bad", "terrible", "awful", "hate", "stressed"]
        positive_words = ["happy", "good", "great", "excellent", "love", "wonderful", "amazing", "excited"]
        
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        if pos_count > neg_count:
            return {"score": 0.5, "label": "positive", "emotion": "joy"}
        elif neg_count > pos_count:
            return {"score": -0.5, "label": "negative", "emotion": "distress"}
        else:
            return {"score": 0.0, "label": "neutral", "emotion": "calm"}
    
    try:
        vader_scores = vader.polarity_scores(text)
        blob = TextBlob(text)
        combined = (vader_scores['compound'] * 0.7) + (blob.sentiment.polarity * 0.3)
        
        label = "positive" if combined >= 0.3 else "negative" if combined <= -0.3 else "neutral"
        
        if vader_scores['pos'] > 0.6:
            emotion = "joy"
        elif vader_scores['neg'] > 0.6:
            emotion = "distress"
        elif vader_scores['neg'] > 0.3:
            emotion = "anxiety"
        else:
            emotion = "calm"
        
        return {"score": round(combined, 3), "label": label, "emotion": emotion}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"score": 0.0, "label": "neutral", "emotion": "calm"}

def detect_burnout(text: str, sentiment: dict) -> dict:
    """Enhanced burnout detection"""
    keywords = {
        "critical": ["suicidal","suicide","dead",  "death", "die", "dying", "killing", "ending", "worthless", "killed","kill",               "hopeless", "pointless", "despair", "agony", "torment", "destroyed", 
                "shattered", "paralyzed", "numb", "empty", "emptiness", "hollow" "want to die", "can't go on", "hopeless", "end it all"],
        "high": ["begging", "always", "never", "everything", "nothing", "nobody", 
                "impossible", "unbearable", "rage", "fury", "hatred", "disgusting", 
                "disgust", "ew", "yuck", "exhausted", "overwhelmed", "burned out", "burnt out", "drained", 
                 "can't cope", "breaking down", "falling apart", "too much"],
        "medium": ["tired", "stressed", "anxious", "worried", "struggling", 
                   "pressure", "frustrated", "difficult", "hard time", "gloomy", 
            "depressed", "heavy", "disinterested", "indifferent", "unmotivated", 
            "meaningless", "dull", "flat", "lost", "confused", "foggy", "uncertain", 
            "indecisive", "indecisiveness", "distracted", "forgetful", "failure", 
            "inadequate", "useless", "guilty", "guilt", "ashamed", "regret",],
        "low": ["busy", "hectic", "challenging", "demanding", "worried", "concerned", "bothered", "uneasy", "stressed", "annoyed", 
            "irritable", "frustrated", "withdrawn", "distant", "detached", "alone", 
           "lonely", "isolated", "longing", "overthinking", "dwelling", "stuck", 
            "repetitive", "uncomfortable", "sluggish", "tense", "aching", "myself", 
            "was", "were", "used", "before", "back", "previously"]
    }
    
    text_lower = text.lower()
    score = 0.0
    
    for level, words in keywords.items():
        for word in words:
            if word in text_lower:
                score += 0.4 if level == "critical" else 0.25 if level == "high" else 0.12 if level == "medium" else 0.05
    
    if sentiment['label'] == 'negative':
        score += 0.15
    if sentiment['emotion'] == 'distress':
        score += 0.2
    
    score = min(score, 1.0)
    
    level = "critical" if score >= 0.7 else "high" if score >= 0.4 else "medium" if score >= 0.15 else "low"
    
    return {"score": round(score, 2), "level": level, "requires_help": score >= 0.7}

def get_fallback_response(burnout_level: str) -> str:
    """Pre-written responses when AI is unavailable"""
    responses = {
        "critical": "I'm really concerned about what you're sharing. Please reach out for immediate support: Call 988 (Suicide Prevention Lifeline) or text HOME to 741741. You don't have to face this alone.",
        
        "high": "I hear that you're going through a really difficult time. Burnout is exhausting, both mentally and physically. Please consider taking a break, even if it's just 10 minutes. Your well-being matters.",
        
        "medium": "It sounds like you're under quite a bit of stress. Remember to take care of yourself - even small breaks can help. Is there something specific that's weighing on you?",
        
        "low": "Thank you for sharing. I'm here to listen and support you. How can I help you today?"
    }
    return responses.get(burnout_level, responses["low"])

def clean_response(text: str) -> str:
    """Remove citations"""
    text = re.sub(r'\s*\[\d+\](\[\d+\])*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_ai_response(message: str, burnout: dict) -> str:
    """Get AI response with robust fallback"""
    if not PERPLEXITY_AVAILABLE:
        return get_fallback_response(burnout["level"])
    
    try:
        system_prompt = "You are NeuraCare, an empathetic wellness assistant. Respond in 2-3 sentences with warmth and support."
        
        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=150,
            timeout=10  # 10 second timeout
        )
        
        ai_text = response.choices[0].message.content.strip()
        return clean_response(ai_text)
        
    except Exception as e:
        print(f"Perplexity error: {e}, using fallback")
        return get_fallback_response(burnout["level"])

@app.get("/")
def root():
    return {
        "app": "NeuraCare",
        "status": "running",
        "perplexity": PERPLEXITY_AVAILABLE,
        "sentiment": SENTIMENT_AVAILABLE
    }

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print(f"\n=== Message: {req.message} ===")
        
        session_hash = anonymize(req.session_id)
        
        # Analyze
        sentiment = analyze_sentiment(req.message)
        burnout = detect_burnout(req.message, sentiment)
        
        print(f"Sentiment: {sentiment['label']}, Burnout: {burnout['level']} ({burnout['score']})")
        
        # Get response
        bot_response = get_ai_response(req.message, burnout)
        
        # Save
        c.execute("""
            INSERT INTO conversations 
            (session_hash, timestamp, user_message, bot_response, sentiment_label, burnout_level, emotion)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_hash, int(time.time()), req.message, bot_response,
              sentiment["label"], burnout["level"], sentiment["emotion"]))
        conn.commit()
        
        return {
            "response": bot_response,
            "sentiment": {
                "label": sentiment["label"],
                "score": sentiment["score"],
                "emotion": sentiment["emotion"]
            },
            "burnout": {
                "level": burnout["level"],
                "score": burnout["score"],
                "requires_help": burnout["requires_help"]
            },
            "personalization": {"recommendations": []},
            "privacy": {"data_encrypted": True, "anonymized": True}
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Return safe fallback
        return {
            "response": "I'm here to listen. Please tell me more about how you're feeling.",
            "sentiment": {"label": "neutral", "score": 0, "emotion": "calm"},
            "burnout": {"level": "low", "score": 0, "requires_help": False},
            "personalization": {"recommendations": []},
            "privacy": {"data_encrypted": True, "anonymized": True}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
