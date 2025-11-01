Seoul Bike Sharing Demand Prediction
Project Overview:
This project predicts the number of public bikes rented per hour in Seoul, South Korea, using machine learning.  
The dataset includes weather information, time factors, and holiday indicators, allowing us to model how environmental and temporal features influence rental demand.

Objective of this project:
To build and evaluate a predictive model that estimates hourly bike rental counts using key features such as temperature, humidity, season, and time of day.

Dataset Source: UCI Machine Learning Repository – Seoul Bike Sharing Dataset: https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand

Key Features:

| Feature | Description |

| `Date` | Date of observation |

| `Rented Bike Count` | Number of bikes rented per hour (target variable) |

| `Hour` | Hour of the day (0–23) |

| `Temperature(°C)` | Temperature in Celsius |

| `Humidity(%)` | Percentage of humidity |

| `Wind speed (m/s)` | Wind speed |

| `Visibility (10m)` | Visibility distance |

| `Dew point temperature(°C)` | Dew point temperature |

| `Solar Radiation (MJ/m2)` | Solar radiation level |

| `Rainfall(mm)` | Amount of rainfall |

| `Snowfall (cm)` | Amount of snowfall |

| `Seasons` | One of Winter, Spring, Summer, Autumn |

| `Holiday` | Whether it was a holiday or not |

| `Functioning Day` | Whether the system was operating normally |

Tech Stack:
- Python 3.10+
- Pandas, NumPy: Data processing  
- Matplotlib, Seaborn, Plotly: Visualization  
- Scikit-learn: Machine Learning  
- Jupyter Notebook: Development Environment  

Future Work:
- Incorporate real-time weather APIs for live predictions.
- Experiment with deep learning models (LSTM/GRU) for time-series forecasting.
- Develop a Streamlit dashboard for interactive prediction.

Author:
- Chukwuemeka Eugene Obiyo
- Data Scientist | Machine Learning Engineer
- praise609@gmail.com 
