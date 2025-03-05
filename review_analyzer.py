import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from config import ReviewAnalysisConfig
from mlops_tracker import MLOpsTracker
from langchain_core.runnables import RunnableSequence 
from transformers import pipeline
from langchain_core.runnables import Runnable

class EnterpriseReviewAnalyzer:
    def __init__(self, config: ReviewAnalysisConfig = ReviewAnalysisConfig()):
        self.config = config
        self.mlops_tracker = MLOpsTracker()
        self._initialize_models()

    def _initialize_models(self):
        # Sentiment model
        self.sentiment_model = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            batch_size=16, 
            truncation=True, 
            padding=True,     
            max_length=512    
        )

        # Summarization model
        self.summarization_pipeline = pipeline(
            "summarization", 
            model=self.config.model_name,
            max_length=self.config.max_length,
            min_length=self.config.min_length,
            temperature=self.config.temperature,
            truncation=True
        )

        self.llm = HuggingFacePipeline(pipeline=self.summarization_pipeline)

        # Correct way to create a prompt template using langchain
        self.review_analysis_prompt = PromptTemplate(
            input_variables=["review"],
            template="Summarize this review: {review}"
        )

        # Now create a runnable for the prompt template
        self.analysis_chain = self.review_analysis_prompt | self.llm


    def analyze_reviews(self, df):
        try:
            # Sample dataframe to avoid overloading the model
            df = df.sample(n=1000, random_state=42).reset_index(drop=True)
            df['review'] = df['review'].astype(str).str.strip()
            df = df[df['review'] != '']

            reviews = df['review'].tolist()

            # Sentiment analysis
            sentiments = self.sentiment_model(reviews)
            sentiment_stats = {
                'positive_count': sum(1 for s in sentiments if s['label'] == 'POSITIVE'),
                'negative_count': sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
            }

            # Summarization
            summaries = self.summarization_pipeline(reviews)
            summaries = [summary['summary_text'] for summary in summaries]

            # Metrics logging
            metrics = {
                'total_reviews': len(reviews),
                'verified_purchases': df['verified_purchaser'].sum(),
                'avg_rating': df['rating'].mean(),
                'positive_reviews': sentiment_stats['positive_count'],
                'negative_reviews': sentiment_stats['negative_count']
            }

            self.mlops_tracker.log_model_performance(metrics)

            return {
                'summaries': summaries,
                'metrics': metrics
            }

        except Exception as e:
            print(f"Error during review analysis: {e}")
            self.mlops_tracker.log_error(str(e))

            return {
                'summaries': [],
                'metrics': {'total_reviews': 0, 'verified_purchases': 0, 'avg_rating': 0, 'positive_reviews': 0, 'negative_reviews': 0}
            }


