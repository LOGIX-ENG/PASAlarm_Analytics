# Industrial Alarm Analytics and Machine Learning for Operational Pattern Discovery

## Overview

This project is a web-based data analytics application developed as a Computer Science Capstone project.

The application analyzes historical industrial alarm-event data to identify recurring and unusual patterns in alarm activity. It combines descriptive analytics with unsupervised machine learning to provide an interactive decision-support tool.

The application is built with Python and Streamlit and uses HDBSCAN for density-based pattern discovery.

> **Important:** This application is an analytical decision-support tool. HDBSCAN results identify statistical patterns and unusual observations. They do not diagnose equipment failures, determine alarm causes, or replace engineering judgment.

---

## Project Question

**Can historical industrial alarm data be analyzed using descriptive analytics and density-based machine learning to identify recurring and unusual alarm patterns across Areas and Fields?**

---

## Business Problem

Industrial SCADA systems can generate large numbers of alarm events across different Areas, Fields, Assets, and alarm tags.

Reviewing large historical alarm datasets manually can make it difficult to quickly identify:

* Areas with high alarm activity
* Changes in alarm activity over time
* Recurring alarm patterns
* Groups of similar alarm observations
* Unusual periods of alarm activity

This application provides a simple way to explore the historical data and identify patterns that may warrant additional engineering investigation.

---

## Application Features

The application provides:

* Interactive Area filtering
* Interactive Priority filtering
* Alarm event totals
* Alarm activity by Area
* Alarm activity over time
* Filtered alarm-record viewing
* HDBSCAN pattern discovery
* HDBSCAN cluster and noise statistics
* HDBSCAN cluster visualization
* Chronological 80/20 training and holdout evaluation
* HDBSCAN cluster-membership classification for held-out observations
* Membership-strength reporting for held-out observations
* Automated pytest testing

The application intentionally uses a small number of features and controls to keep the interface simple and easy to use.

---

## Visualizations

The application contains three primary visualizations.

### 1. Alarm Events by Area

A bar chart showing the number of alarm records associated with each Area.

This visualization helps identify where alarm activity is concentrated.

### 2. Alarm Events Over Time

A line chart showing changes in alarm activity over time.

This visualization helps identify periods of increased or decreased alarm activity.

### 3. HDBSCAN Cluster Plot

A scatter plot showing HDBSCAN clustering results.

Each point represents a five-minute Area/Field observation. PCA is used only to reduce the feature space to two dimensions for visualization.

The plot shows recurring groups of observations and observations identified by HDBSCAN as noise.

---

## Dataset

The application uses a historical industrial alarm-event dataset containing:

* **14,521 alarm records**
* Data covering **January through June 2026**

The dataset contains fields such as:

* Time / Date
* Alarm Tag
* Priority
* Type
* Quality
* Alarm State
* Area
* Field
* Related Value

The public dataset has been sanitized and anonymized.

Company, facility, asset, field, area, and tag identifiers have been replaced with non-identifying values.

The repository does not require access to a live SCADA system or industrial control network.

---

## Data Preparation

Before analysis, the application performs basic data preparation:

1. Loads the CSV dataset.
2. Converts the Time / Date field to datetime.
3. Converts Priority values to numeric values.
4. Removes records with invalid timestamps.
5. Applies the selected Area and Priority filters.
6. Creates five-minute Area/Field observations for machine-learning analysis.
7. Generates numerical features.
8. Standardizes the features before HDBSCAN clustering.

---

## HDBSCAN Analysis

The project uses **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** for unsupervised pattern discovery.

Each five-minute Area/Field observation is represented using features derived from the alarm records.

### Features

The model uses:

* `AlarmCount`
* `UniqueTags`
* `AveragePriority`
* `MaximumPriority`
* `ActiveCount`
* `AckedCount`
* `NormalCount`
* `HighPriorityCount`
* `AlarmRatePerMinute`

Numerical features are standardized before clustering.

### Selected Configuration

The final application uses a fixed configuration:

| Parameter            |     Value |
| -------------------- | --------: |
| Window duration      | 5 minutes |
| Minimum cluster size |        10 |

The application does not expose HDBSCAN parameters to the end user.

This keeps the application simple and provides a consistent analysis.

---

## Model Results

### Descriptive HDBSCAN Results

Using the complete dataset with the selected five-minute window and `min_cluster_size=10`, the application produced:

| Metric | Result |
| --- | ---: |
| Alarm records | 14,521 |
| Five-minute observations | 9,130 |
| Clusters | 34 |
| Noise observations | 247 |
| Noise percentage | 2.71% |
| Silhouette Score | 0.9922 |
| Davies-Bouldin Index | 0.1847 |

These metrics evaluate the density-based cluster structure. Because HDBSCAN is an unsupervised clustering algorithm, traditional classification accuracy is not applicable.

### Chronological Holdout Evaluation

All five-minute Area/Field observations are sorted chronologically. The earliest approximately 80% are used for training, and the remaining approximately 20% are held out. `StandardScaler` is fit only on the training observations. HDBSCAN is fit with `prediction_data=True`, and `hdbscan.approximate_predict()` classifies held-out observations against the learned cluster structure.

| Metric | Result |
| --- | ---: |
| Training observations | 7,304 |
| Holdout observations | 1,826 |
| Holdout matches | 1,724 |
| Holdout noise | 102 |
| Training Silhouette Score | 0.9908 |
| Training Davies-Bouldin Index | 0.1768 |

A holdout observation assigned to an established cluster is treated as matching a learned historical pattern. An observation classified as noise did not sufficiently match the learned density structure.

### Interpretation

Clustered observations represent groups of five-minute Area/Field observations with similar feature characteristics. HDBSCAN noise represents observations that do not sufficiently fit the identified density structure.

Noise should be interpreted as **statistically unusual**, not automatically as equipment failure, equipment malfunction, operator error, alarm-system failure, or a specific process condition. Additional engineering investigation is required to determine why an observation is unusual.

The holdout workflow predicts **cluster membership**, not future alarms, equipment failures, equipment condition, or root cause.

| Information |   Totals |
| ------------------------ |---------:|
| Alarm records            |   14,521 |
| Five-minute observations |    9,130 |
| Clusters                 |       34 |
| Noise observations       |      247 |
| Noise percentage         |    2.71% |
| Silhouette Score         | 0.992416 |
| Davies-Bouldin Score     | 0.191195 |


These metrics are used to evaluate the clustering structure.

Because HDBSCAN is an unsupervised clustering algorithm, traditional classification accuracy is not applicable.

### Interpretation

Clustered observations represent groups of five-minute Area/Field observations with similar feature characteristics.

Observations classified as HDBSCAN noise do not fit the density structure of the identified clusters.

Noise should be interpreted as **statistically unusual**, not automatically as:

* Equipment failure
* Equipment malfunction
* Operator error
* Alarm-system failure
* A specific process condition

Additional engineering investigation is required to determine why an observation is unusual.

---

## Technology Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Web Application      | Streamlit                 |
| Data Processing      | Pandas                    |
| Numerical Processing | NumPy                     |
| Machine Learning     | HDBSCAN                   |
| ML Preprocessing     | scikit-learn              |
| Testing              | pytest                    |
| Version Control      | Git / GitHub              |
| Deployment           | Streamlit Community Cloud |

---

## Project Structure

```text
PASAlarm_Analytics/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── alarms.csv
│
└── tests/
    └── test_app.py
```

---

## Hosted Application

The application is deployed using Streamlit Community Cloud.

**Hosted application:** `https://pasalarmanalytics.streamlit.app`

The hosted version can be used directly in a modern web browser and does not require local installation. If the application has been inactive, Streamlit may require a short period to wake before it becomes available.

---

## Source Code

The complete project source is maintained in this repository. For the WGU assessment submission, matching source files, the sanitized dataset, and dependency information are also submitted directly so the project can be recreated locally.

---

## Installation

### Requirements

Python is required to run the application locally.

A compatible Python version should be used with the versions specified in `requirements.txt`.

### Clone the Repository

```bash
   git clone <repository-url>
   cd PASAlarm_Analytics
```

### Install Dependencies

```bash
   pip install -r requirements.txt
```

---

## Run the Application

From the project root:

```bash
   python -m streamlit run app.py
```

Streamlit will provide a local URL where the application can be opened in a web browser.

---

## Using the Application

### Step 1 — Review the Dataset Summary
Review the displayed totals for alarm records, Areas, and Fields.

### Step 2 — Select an Area
Use the Area filter to limit the descriptive analysis to a specific Area or view all Areas.

### Step 3 — Select Priority
Use the Priority filter to examine selected priority levels.

### Step 4 — Review the Descriptive Analysis
Review **Alarm Events by Area** and **Alarm Events Over Time**.

### Step 5 — Review the Alarm Records
Review the filtered historical alarm-record table.

### Step 6 — Run Descriptive HDBSCAN
Select **Run HDBSCAN**. Review Clusters, Noise Observations, Noise %, Silhouette Score, Davies-Bouldin Index, and the PCA cluster visualization.

### Step 7 — Classify Holdout Observations
Navigate to **Classify Holdout Observations Against Historical Patterns** and select **Train Model & Classify Holdout**.

The application creates five-minute Area/Field observations from the complete dataset, sorts them chronologically, uses the earliest approximately 80% for training, reserves the remaining approximately 20% as holdout observations, fits feature scaling only on training data, fits HDBSCAN, and uses `approximate_predict()` to classify the holdout observations.

Review Training Windows, Holdout Matches, Holdout Noise, Training Silhouette Score, Training Davies-Bouldin Index, and any displayed noise observations and membership-strength values.

A holdout noise classification indicates that the observation did not sufficiently match an established cluster. It does not establish an equipment failure or identify an alarm cause.

---

## Testing

The project uses `pytest` for automated testing. Run the test suite from the project root:

```bash
   pytest
```

The final automated test suite contains **23 tests**. The verified final result was:

```text
23 passed, 0 failed
```

The tests cover the primary data-processing and machine-learning functions, including source-column validation, timestamp and Priority conversion, five-minute Area/Field observation creation, all nine engineered features, HDBSCAN execution, cluster evaluation, chronological training/holdout separation, holdout cluster-membership prediction, and membership-strength validation.

---

## Security and Data Protection

The public project uses sanitized and anonymized data.

The repository should not contain:

* Proprietary raw alarm data
* Personal identifying information
* Passwords
* API keys
* Authentication tokens
* SCADA credentials
* Network credentials
* Connection strings
* Private configuration files

The application does not connect to live industrial control systems and does not send commands to industrial equipment.

Original proprietary/raw data should remain outside the public repository.

---

## Limitations

This project has several limitations.

### Historical Data

The application analyzes historical alarm records and does not process live SCADA alarm streams.

### Available Features

The machine-learning model can only use information contained in the available dataset.

### Unsupervised Learning

HDBSCAN identifies groups based on statistical similarity. There is no predefined correct cluster label.

### Alarm State Interpretation

Alarm states represent recorded alarm-state information. A `Normal` state does not by itself prove that an operator acknowledged or otherwise acted on the alarm.

### No Root-Cause Analysis

The application identifies patterns but does not determine the cause of an alarm.

### No Failure Prediction

A record classified as HDBSCAN noise does not mean that equipment will fail or that a failure has occurred.

### Cluster-Membership Prediction

The chronological holdout workflow classifies previously unseen observations according to their similarity to density-based patterns learned from earlier historical observations. It does not predict future alarm occurrences, equipment failures, equipment condition, or operational causes.

---

## ISA-18.2 Context

The project is **ISA-18.2-oriented** and focuses on historical alarm monitoring and assessment.

The application is not intended to establish or certify ISA-18.2 compliance.

The analytical results can provide information that may support further alarm-management investigation.

---

## Academic Purpose

This application was developed as a Computer Science Capstone project to demonstrate the practical application of:

* Python programming
* Data cleaning
* Data transformation
* Feature engineering
* Descriptive analytics
* Machine learning
* HDBSCAN clustering
* Data visualization
* Interactive web application development
* Software testing
* Documentation

---

## License

This project is developed for academic purposes.

The dataset and application should not be used to make operational decisions involving industrial equipment without appropriate engineering review and validation.