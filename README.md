# E-Commerce Customer Review Insights

## Overview
The **E-Commerce Customer Review Insights** is a tool designed to analyze and process Walmart product reviews using state-of-the-art **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The tool extracts valuable insights from customer feedback, including sentiment analysis, review summaries, and key metrics related to customer satisfaction and product performance.

By leveraging advanced libraries like **MLflow**, **Langchain**, and **Transformers**, the project offers a robust pipeline for analyzing and tracking performance, as well as utilizing cutting-edge pre-trained language models for review analysis. This tool is ideal for researchers, data scientists, and e-commerce professionals looking to analyze large-scale product reviews efficiently.

## Key Features
- **Sentiment Analysis**: Categorizes reviews as positive or negative using NLP models.
- **Review Summaries**: Generates concise summaries of individual reviews using **Transformers** for better readability.
- **Key Metrics Tracking**: Logs essential review metrics such as total reviews, verified purchases, average rating, and sentiment distributions (positive/negative).
- **Performance Tracking with MLflow**: Utilizes **MLflow** to log and track performance metrics, ensuring reproducibility and efficient experimentation.
- **Automated Data Handling**: Efficient data processing with **Langchain**, structuring and processing data for different NLP tasks.

## Dataset
This project uses a Walmart product review dataset, sourced from **Kaggle**. The dataset includes user reviews of Walmart products, containing fields such as ratings, review text, and whether the purchase was verified. The data undergoes **Exploratory Data Analysis (EDA)** to uncover trends and provide insights before being processed in the analysis pipeline.

## Technologies Used
### MLflow
**MLflow** is integrated into this project for logging and managing the performance of machine learning models. It helps track:
- Model training performance (accuracy, loss, etc.)
- Hyperparameters used during experimentation
- Sentiment analysis metrics and other performance indicators
- Reproducibility of experiments and models

With **MLflow**, model development is transparent and version-controlled, enabling continuous improvement.

### Langchain
**Langchain** structures and manages the natural language processing pipeline. It facilitates:
- **Text summarization**: Automatically generating concise summaries of reviews.
- **Sentiment analysis**: Classifying reviews as positive or negative.
- **Data structuring**: Organizing and pre-processing review data for effective downstream processing.

Langchain enhances pipeline extensibility by integrating various data sources and models, improving overall efficiency.

### Transformers
**Transformers** by Hugging Face is a powerful library for state-of-the-art NLP models. In this project, it enables:
- **Text Summarization**: Summarizing lengthy reviews into digestible pieces.
- **Sentiment Classification**: Using pre-trained models to classify reviews as positive, negative, or neutral.

These advanced NLP models allow for accurate large-scale review analysis, delivering timely insights.

## MLOps Integration
### Performance Tracking with MLflow
Integrating **MLflow** ensures that all models' performance metrics, hyperparameters, and other relevant data are logged. This enables:
- Tracking sentiment analysis model performance.
- Comparing different model versions and configurations.
- Visualizing metrics such as accuracy, loss, and other relevant values.
- Achieving model versioning for reproducibility and consistency.

### Continuous Monitoring
With **MLflow**, users can set up continuous monitoring to assess model performance over time. This is crucial in e-commerce, where review sentiment trends may evolve.

## Dependencies
This project requires the following Python libraries:
- `torch==1.13.1`: For deep learning model training and inference.
- `numpy==1.23.0`: For numerical operations.
- `pandas==2.2.3`: For handling and manipulating datasets.
- `transformers==4.49.0`: For using pre-trained NLP models from Hugging Face.
- `langchain==0.3.20`: For structuring and managing NLP tasks.
- `mlflow==2.20.3`: For tracking machine learning model performance and experiments.
- Python 3.9.21: The required Python version to run the project.

## Output
After running the analysis, the script outputs key metrics and review summaries in a structured JSON format.

### Example Output:
```json
{
    "summaries": [
        "The product is great, I love it! Highly recommend it to others.",
        "Not worth the price. I had issues with the quality right away."
    ],
    "metrics": {
        "total_reviews": 1000,
        "verified_purchases": 557,
        "avg_rating": 4.066,
        "positive_reviews": 654,
        "negative_reviews": 346
    }
}
```

"# GenAI-Ecom-Review-Analyzer" 
