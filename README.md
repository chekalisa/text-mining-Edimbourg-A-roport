# Analysis of Customer Feedback at Edinburgh Airport
## Description du projet
This project was carried out in collaboration with Edinburgh Airport and the University Panthéon-Sorbonne under the supervision of Armand L’Huiller and Aryan Razaghi. The objective is to analyze travelers' feedback regarding their experience at the airport in order to identify areas for improvement and service opportunities.

As one of the busiest airports in the United Kingdom, it is crucial to understand passengers' perceptions of aspects such as cleanliness, convenience, customer service, and security. Analyzing this data will help enhance the traveler experience and optimize the services offered by the airport.

## Analysis Objectives

The study focuses on several key aspects:

- **Analysis of cleaned databases** : extraction and preparation of data from traveler responses.

- **Analysis of categorical variables** :  study of relationships between different categorical variables and their impact on satisfaction.

- **Predictive modeling** : implementation of machine learning models, including logistic regression and ensemble models, to predict factors influencing passenger satisfaction.

- **Sentiment analysis** : classification of comments based on their tone (positive, neutral, or negative) to understand key dissatisfaction points.

- **Topic Modeling** : identification of main topics mentioned by passengers in their comments.



## Data Used
The dataset contains 24,934 responses with 62 columns, including quantitative and qualitative variables collected through a questionnaire. Two open-ended questions allow travelers to freely express their satisfaction and suggestions for improvement.

## Methodology

**1.Data Preparation**

- Cleaning of missing and anomalous values.
- Transformation of categorical variables and creation of new derived variables.
- Normalization and encoding of variables.

**2. Exploratory Data Analysis (EDA)**

- Visualization of variable distributions.
- Study of correlations between variables.
- Identification of trends in passenger responses.

**3. Modeling**

- Logistic regression to identify the most influential factors.
- Random Forest and SVM to improve prediction accuracy.
- Cross-validation and hyperparameter tuning.

**4.Sentiment Analysis**

- Use of the VADER method to evaluate comment tone.
- Comparison of sentiment distribution over time.

**5. Topic Modeling**

- Extraction of recurring topics in comments using the LDA (Latent Dirichlet Allocation) method and bigram analysis.

## Running the code

- Creating the virtual environment
```
python -m venv .venv
```
- Activating the environment
```
source .venv/bin/activate
```
- Installing libraries
```
pip install -r requirements.txt
```
- Running the code
```
jupyter notebook main.ipynb
```

## Authors

- [Alisa Chekalina](https://github.com/chekalisa)
- [Carmen Cristea](https://github.com/CarmenParis)
- [Vo Nguyen Thao Nhi](https://github.com/vonguyenthaonhi)

