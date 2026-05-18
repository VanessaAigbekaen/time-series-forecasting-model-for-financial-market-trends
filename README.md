# time-series-forecasting-model-for-financial-market-trends
AI Market Forecasting System

A machine learning-based market forecasting system developed using historical SPY financial data to analyze market trends and predict potential future market direction. The project combines financial analysis, technical indicators, and machine learning models to identify hidden market patterns and generate probabilistic trading signals.

Project Overview

This project focuses on predicting market direction (UP / DOWN / NEUTRAL) using historical stock market data and technical indicators. The system was designed to explore how machine learning can be applied to financial forecasting while also understanding the practical limitations and unpredictability of financial markets.

The project includes:
Data cleaning and preprocessing
Feature engineering using technical indicators
Machine learning model training and evaluation
Risk and confidence analysis
GUI-based prediction system for market forecasting

Features:
Historical SPY market data analysis
Technical indicators:
True Range (TR)
Average True Range (ATR)
Relative Strength Index (RSI)
Machine learning classification models:
K-Nearest Neighbors (KNN)
Logistic Regression
Naive Bayes
Random Forest
Gradient Boosting
Model evaluation using:
Accuracy Scores
Confusion Matrix
ROC Curves
AUC Scores
GUI interface for live market direction prediction
Confidence and risk analysis integration

Technologies Used:
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Tkinter
Joblib
Yahoo Finance API (yfinance)


Machine Learning Workflow:
Data Collection
Historical SPY market data collected from Yahoo Finance.
Data Cleaning & Preprocessing
Removed missing/invalid values and prepared the dataset for training.
Feature Engineering
Added financial technical indicators such as ATR, RSI, and True Range to improve pattern recognition.
Model Training
Trained and compared multiple machine learning algorithms.
Model Evaluation
Evaluated models using ROC curves, AUC scores, confusion matrices, and validation/test accuracy.
GUI Development
Built a Python GUI system capable of predicting market direction and displaying confidence/risk levels.

Disclaimer

This project was created for educational and research purposes only. Financial markets are highly unpredictable and no machine learning model can guarantee profits or eliminate trading risk. This system should not be considered financial advice.

Future Improvements:
Add MACD indicator support
Improve model balancing and prediction confidence
Add candlestick visualization dashboard
Introduce multi-timeframe analysis (Daily + 4H)
Deploy as a web application
Implement backtesting and paper trading simulation

Author
Developed by Vanessa Aigbekaen as an Artificial Intelligence and Financial Market Forecasting project.
