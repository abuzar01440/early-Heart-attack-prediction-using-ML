<div align="center">
  
# 💓 Early Heart Attack Prediction - Code Implementation 🔬

![Machine Learning](https://img.shields.io/badge/Machine_Learning-FF6F61?style=for-the-badge&logo=python&logoColor=white)
![Medical AI](https://img.shields.io/badge/Medical_AI-2B9EB3?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEwLDIwSDZWNEg5VjdIMTFWNEgxNFYyMEgxMFYxMUg5VjE2SDEwVjIwWk0xMCw5SDlWMTBIMTBWOVoiIGZpbGw9IndoaXRlIi8+PC9zdmc+)
![Healthcare](https://img.shields.io/badge/Healthcare-4CAF50?style=for-the-badge&logo=heart&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3499CD?style=for-the-badge&logo=lightgbm&logoColor=white)

```
  _    _                 _      _   _   _             _      ____               _ _      _   _             
 | |  | |               | |    | | | | | |           | |    |  _ \             | (_)    | | (_)            
 | |__| | ___  __ _ _ __| |_   | | | | | |_ __ _  ___| | __ | |_) |_ __ ___  __| |_  ___| |_ _  ___  _ __  
 |  __  |/ _ \/ _` | '__| __|  | | | | | __/ _` |/ __| |/ / |  _ <| '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \ 
 | |  | |  __/ (_| | |  | |_   | |_| | | || (_| | (__|   <  | |_) | | |  __/ (_| | | (__| |_| | (_) | | | |
 |_|  |_|\___|\__,_|_|   \__|   \___/|_|\__\__,_|\___|_|\_\ |____/|_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|
```

</div>

<p align="center">
  <i>🔍 Applying Machine Learning to Predict Heart Attacks Before They Happen 🔍</i>
</p>

---

## 📋 Table of Contents
- [📝 Project Overview](#-project-overview)
- [📖 Abstract of the Research](#-abstract-of-the-research)
- [🎯 Project Goals](#-project-goals)
- [⚙️ Prerequisites](#️-prerequisites)
- [🔗 Resources](#-resources)

---

## 📝 Project Overview

<img align="right" width="180" src="https://img.icons8.com/color/452/heart-with-pulse.png">

This repository contains the code implementation for the research paper: **"EARLY HEART ATTACK PREDICTION BY USING MACHINE LEARNING"** by:
- 👨‍🔬 **Sarang Ali**
- 👨‍💻 **Abuzar Shahid**
- 👩‍⚕️ **Gohar Mumtaz**

The project leverages advanced machine learning models, with specific focus on the **LightGBM classifier**, to predict the early threat of heart attacks and potentially save lives through early intervention.

---

## 📖 Abstract of the Research

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #ff6b6b;">

🔍 Machine learning is increasingly applied across various domains, significantly advancing fields like digital healthcare. 

🏥 This research leverages machine learning models to predict the early threat of heart attacks, aiming to support healthcare systems and enhance human well-being. 

❤️ Heart attacks are a significant cause of mortality, often remaining undetected until a critical event occurs, which underscores the need for early and accurate risk assessment. 

🎯 This study focuses on identifying individuals at high risk of cardiac events with high accuracy, intending to alert users to consult with medical professionals promptly. 

📊 The study utilizes a publicly available "Heart Attack Prediction" dataset.

🔄 The data was split into 90% for training and 10% for testing. 

⚙️ Python's Optuna library was employed for hyperparameter tuning to identify the best machine learning model and its parameters.

✅ The LightGBM classifier was found to be the most effective for this binary classification task, demonstrating promising results on the test data.
</div>

### 🔑 Keywords
- 🤖 **Machine learning**
- 🌲 **LightGBM classifier**
- 🎛️ **Parameter tuning**

---

## 🎯 Project Goals

<div align="center">
  <table>
    <tr>
      <td align="center">🔎</td>
      <td>Practical implementation of the heart attack prediction model</td>
    </tr>
    <tr>
      <td align="center">🧹</td>
      <td>Data preprocessing and cleaning</td>
    </tr>
    <tr>
      <td align="center">🔧</td>
      <td>Feature engineering based on dataset analysis</td>
    </tr>
    <tr>
      <td align="center">⚙️</td>
      <td>Model training with hyperparameter optimization using Optuna</td>
    </tr>
    <tr>
      <td align="center">📊</td>
      <td>Evaluation of the LightGBM classifier performance</td>
    </tr>
  </table>
</div>

---

## ⚙️ Prerequisites

<div align="center">
  <h3>🐍 Python Libraries 📚</h3>
</div>

Ensure you have Python installed on your system (the paper implies Python usage). The following Python libraries are essential, as suggested by the paper's methodology:

<table align="center">
  <tr>
    <td align="center"><img width="40" src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg"></td>
    <td><b>pandas</b> - for data manipulation</td>
  </tr>
  <tr>
    <td align="center"><img width="40" src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg"></td>
    <td><b>numpy</b> - for numerical operations</td>
  </tr>
  <tr>
    <td align="center"><img width="40" src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"></td>
    <td><b>scikit-learn</b> - for general machine learning tasks and metrics</td>
  </tr>
  <tr>
    <td align="center"><img width="40" src="https://lightgbm.readthedocs.io/en/latest/_static/LightGBM_logo_black_text.svg"></td>
    <td><b>LightGBM</b> - the chosen classifier</td>
  </tr>
  <tr>
    <td align="center"><img width="40" src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png"></td>
    <td><b>Optuna</b> - for hyperparameter tuning</td>
  </tr>
  <tr>
    <td align="center">⚖️</td>
    <td><b>imbalanced-learn</b> - for handling imbalanced datasets</td>
  </tr>
</table>

---

<div align="center">
  
## 🔗 Resources

[📄 Link to Paper](https://sesjournal.com/index.php/1/article/view/239/221)

📧 **Contact**: [abuzarbhutta@gmail.com](mailto:abuzarbhutta@gmail.com)

<img src="https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge&logo=heart" alt="Made with love">

</div>

---

<div align="center">
  <p>
    <a href="https://github.com/abuzar01440">
      <img src="https://img.shields.io/github/followers/abuzar01440?label=Follow&style=social" alt="GitHub Follow">
    </a>
    ⭐ Star this repository if you found it helpful! ⭐
  </p>
  
  <p>
    <img src="https://forthebadge.com/images/badges/built-with-science.svg" alt="Built with Science">
  </p>

  <p>💝 Stay healthy! Take care of your heart! 💝</p>
</div>
