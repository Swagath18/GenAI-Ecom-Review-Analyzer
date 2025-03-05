from config import ReviewAnalysisConfig
from data_loader import DataLoader
from review_analyzer import EnterpriseReviewAnalyzer

def main():
    data_loader = DataLoader("cleaneddata.csv")  # Change to your file path
    df = data_loader.load_data()

    config = ReviewAnalysisConfig()
    analyzer = EnterpriseReviewAnalyzer(config)

    # sample_reviews = df['review'].head(20) # Analyze
    # analysis_result = analyzer.analyze_reviews(sample_reviews)
    analysis_result = analyzer.analyze_reviews(df)

    print(analysis_result)

if __name__ == "__main__":
    main()
