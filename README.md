# ⚡ AutoML Nexus

> **The No-Code Machine Learning Pipeline Builder.**
> *Turn raw CSVs into deployed models in seconds. Zero coding required.*

-----

## 🔮 Overview

**AutoML Nexus** is a web-based, end-to-end Machine Learning dashboard designed with a **futuristic Glassmorphism UI**. It guides users through the entire ML lifecycle—from data ingestion to live inference—using an interactive "Wizard" interface.

Whether you are a data science beginner testing a hypothesis or a developer needing a quick baseline model, AutoML Nexus handles the heavy lifting (preprocessing, splitting, training) so you can focus on the results.

## 🚀 Key Features

  * **👾 Drag & Drop Ingestion:** Upload CSV files instantly or load the sample "Heart/Iris" dataset for quick testing.
  * **📊 Automated EDA:** Visualize data distributions with Scatter plots and Histograms without writing Matplotlib code.
  * **⚖️ Smart Preprocessing:** Apply **Standard Scaling** (Z-Score) or **MinMax Scaling** with a single click.
  * **✂️ Dynamic Data Splitting:** Interactive slider to partition your data into Training and Testing sets (80/20, 70/30, etc.).
  * **🧠 Multi-Model Training:** Choose from 5 powerful algorithms:
      * Logistic Regression
      * Linear Regression
      * Support Vector Machines (SVM)
      * Random Forest
      * Decision Trees
  * **📉 Real-time Metrics:** Instant feedback on Accuracy, Precision, Recall, F1-Score, MAE, and MSE.
  * **🔮 Live Inference Engine:** Automatically generates input fields based on your dataset for real-time predictions.
  * **💾 Export & Deploy:** Download the **Trained Model (.pkl)** or the **Generated Python Script (.py)** to use offline.
  * **🎨 Cyberpunk UI:** Fully responsive, dark-mode interface with neon accents and glassmorphism effects.

-----

## 🛠️ Installation & Setup

Follow these steps to get the app running locally on your machine.

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/automl-nexus.git
cd automl-nexus
```

### 2\. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

Create a file named `requirements.txt` with the list below, then run the install command.

```bash
pip install -r requirements.txt
```

**Content of `requirements.txt`:**

```text
Flask
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### 4\. Run the Application

```bash
python app.py
```

Visit **`http://127.0.0.1:5000/`** in your browser to launch the Nexus.

-----

## 🎮 How to Use

1.  **Step 1: Initialize Data**
      * Drag and drop your `.csv` file into the drop zone.
      * *Note:* Ensure your CSV contains only numerical values for features (Categorical encoding is handled for targets only).
2.  **Step 2: Visual Analysis**
      * Select X and Y columns to generate plots. Check for patterns or outliers.
3.  **Step 3: Normalization**
      * Choose a scaler to normalize your data range. This improves model performance for algorithms like SVM.
4.  **Step 4: Partition**
      * Use the neon slider to set your Test size (e.g., 20%).
5.  **Step 5: Training**
      * Select an algorithm and hit "Initiate Training".
      * Review the performance metrics cards.
6.  **Step 6: Live Inference**
      * Enter values into the generated fields and click "Run Prediction" to see the model's output instantly.

-----

## 📂 Project Structure

```bash
AutoML-Nexus/
├── app.py                 # Main Flask Application (Backend Logic)
├── model.pkl              # Saved model (auto-generated)
├── generated_model.py     # Downloadable script (auto-generated)
├── templates/
│   └── index.html         # Frontend UI (HTML/CSS/JS)
├── static/
│   └── plot_*.png         # Generated plots
├── uploads/
│   └── data.csv           # Persisted dataset
└── README.md              # Documentation
```

-----

## ⚠️ Troubleshooting

**Issue: "Matplotlib UserWarning: Starting a GUI..."**

  * **Fix:** Ensure your `app.py` includes `matplotlib.use('Agg')` **before** importing `pyplot`. This allows plots to generate in the background without trying to open a window.

**Issue: "ValueError: Could not interpret value..."**

  * **Fix:** If you see this after restarting the server, simply click the **"↻ RESET PIPELINE"** button in the top right corner to clear the session cache.

-----

## 🤝 Contributing

Contributions are welcome\! If you have ideas for new features (e.g., Categorical Encoding support, Neural Networks, etc.):

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

-----

<center>Made with ❤️ by Ankush Yadav</center>
