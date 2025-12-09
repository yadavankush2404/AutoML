from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import matplotlib
import wget
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

app = Flask(__name__)

#globals
uploaded_data = None
scaled_data = None
scaler_type = None
model = None
plot_path = None
metrics = None
uploaded_filename = None
plot_details = []
feature_names = []
pred_res = None
preview_html = None
test_size = 0.2
split_stats = None
error_message = None


@app.route('/', methods=['GET', 'POST'])
def index():
    global uploaded_data, plot_path, metrics, uploaded_filename, plot_details, scaled_data, scaler_type
    global model, scaler, feature_names, pred_res, preview_html, test_size, split_stats, error_message

    #File upload Block
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                uploaded_data = pd.read_csv(file)
                uploaded_filename = file.filename

                plot_details = []
                plot_path = None
                scaled_data = None
                scaler_type = None
                metrics = None
                error_message = None
                print("🔄 New file uploaded. All previous states reset.")

        elif 'use_sample' in request.form:
            drive_url = 'https://raw.githubusercontent.com/yadavankush2404/ces_git_test/refs/heads/main/heart.csv'
            uploaded_data = pd.read_csv(drive_url)
            print(uploaded_data.shape)
            
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            uploaded_data.to_csv(os.path.join('uploads', 'data.csv'), index=False)
            
            uploaded_filename = "Google_Drive_Data.csv"
            
            # Reset State (Same as upload)
            plot_details = []
            metrics = None
            scaled_data = None
            scaler_type = None
            error_message = None
        
            # Clear old plots
            if os.path.exists('static'):
                for f in os.listdir('static'):
                    if f.startswith('plot_'):
                        os.remove(os.path.join('static', f))
            
            print("✅ Loaded Sample Dataset")
        
        # Data Plotting Block
        elif 'plot_type' in request.form:
            if uploaded_data is None:
                return "Error: Data lost. Please upload the CSV again."
            x_col = request.form['x_column']
            y_col = request.form['y_column']
            plot_type = request.form['plot_type']

            plt.figure()
            if plot_type == 'scatter':
                sns.scatterplot(data=uploaded_data, x=x_col, y=y_col)

            elif plot_type == 'histogram':
                sns.histplot(data=uploaded_data[x_col])

            if not os.path.exists('static'):
                os.makedirs('static')

            plot_path = f'static/plot_{len(plot_details)}.png'
            plt.savefig(plot_path)
            plt.close()
            plot_details.append({'x_col': x_col, 'y_col': y_col, 'plot_type': plot_type})

        #Scaling block
        elif 'scaler' in request.form:
            if uploaded_data is None:
                return "Error: Data missing. Please upload CSV first."
            
            scaler_type = request.form['scaler']
            scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            scaled_data = scaler.fit_transform(uploaded_data.iloc[:,:-1].select_dtypes(include='number'))
            print(f"⚖️ Scaling ({scaler_type}) complete. Data shape:", scaled_data.shape)
            metrics = None
            error_message = None

        # setting split sizes
        elif 'set_split' in request.form:
            if uploaded_data is None:
                return "Error: Data missing"
            slider_val = int(request.form['split_ratio'])
            test_size = slider_val / 100.0

            total_rows = len(uploaded_data)
            n_test = int(total_rows*test_size)
            n_train = total_rows - n_test
            split_stats = {
                'train': n_train, 
                'test': n_test, 
                'percent': slider_val
            }
            
            # Reset future steps if split changes
            metrics = None
            pred_res = None

        # Model Training Block
        elif 'model' in request.form:
            try:
                if scaled_data is None:
                    raise ValueError("Please apply scaling to the data before training the model.")
            
                feature_names = uploaded_data.iloc[:,:-1].columns.tolist()

                model_type = request.form['model']

                X_train, X_test, y_train, y_test = train_test_split(scaled_data, uploaded_data.iloc[:,-1], test_size=test_size)

                model = {
                    'logistic_regression': LogisticRegression(),
                    'linear_regression': LinearRegression(),
                    'decision_tree': DecisionTreeClassifier(),
                    'random_forest': RandomForestClassifier(),
                    'svm': SVC()
                }[model_type]

                try: 
                    model.fit(X_train, y_train)
                    print("📈 Model Training Complete.. ")
                    predictions = model.predict(X_test)
                    if model_type == 'linear_regression':
                        metrics ={
                            'Model': model_type,
                            'MAE' : mean_absolute_error(y_test,predictions),
                            'MSE' : mean_squared_error(y_test,predictions),
                            'R2_Score' : r2_score(y_test, predictions)
                        }
                    else:
                        metrics = {
                            'Model': model_type,
                            'Accuracy': accuracy_score(y_test, predictions),
                            'Precision': precision_score(y_test, predictions, average='weighted'),
                            'Recall': recall_score(y_test, predictions, average='weighted'),
                            'F1-Score': f1_score(y_test, predictions, average='weighted')
                        }
                    generate_ml_script(uploaded_filename, plot_details, scaler_type, model_type, metrics)
                    pred_res = None
                except Exception as e:
                    return f"Training Error: {str(e)}"
            except Exception as e:
                error_message = str(e)
                print(f"Error encountered: {error_message}")
            
        # Prediction Block
        elif 'predict' in request.form:
            try:
                input_data= []
                for col in feature_names:
                    val = request.form.get(col)
                    if val is None: 
                        return f"Error: Missing value for {col}"
                    input_data.append(float(val))

                df_input = pd.DataFrame([input_data],columns=feature_names)
                scaled_input = scaler.transform(df_input)
                pred = model.predict(scaled_input)
                pred_res = pred[0]
                print(pred_res)
            except Exception as e:
                pred_res = f"Error:{str(e)}"

    preview_html = None
    if uploaded_data is not None:
        preview_html = uploaded_data.head().to_html(classes='table table-glass',index=False,border=0)

    columns = uploaded_data.columns.tolist() if uploaded_data is not None else []
    return render_template(
        'index.html', 
        columns=columns, 
        plot_url=plot_path, 
        metrics=metrics,
        feature_names=feature_names,
        pred_res = pred_res,
        preview_html= preview_html,
        split_stats = split_stats,
        error_message = error_message
        )

# Model pickel saving route
@app.route('/save_model')
def save_model():
    global model
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    fpath = os.path.join('downloads','trained_model.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(model, f)
    return send_file(fpath, as_attachment=True)

# Download python script file.
@app.route('/download_ml_script')
def download_ml_script():
    fpath = os.path.join('downloads','ml_model_script.py')
    return send_file(fpath, as_attachment=True)

def generate_ml_script(filename, plot_details, scaler_type, model_type, metrics):
    plot_code = ""
    for plot in plot_details:
        # 1. Start the figure
        plot_code += "plt.figure()\n"
        
        # 2. Python logic decides which line to write (Scatter vs Histogram)
        if plot['plot_type'] == 'scatter':
            # Note: I am assuming the dataframe variable in your generated script is named 'df'
            plot_code += f"sns.scatterplot(data=data, x='{plot['x_col']}', y='{plot['y_col']}')\n"
        elif plot['plot_type'] == 'histogram':
            plot_code += f"sns.histplot(data['{plot['x_col']}'])\n"
            
        # 3. Close the figure
        plot_code += "plt.show()\nplt.close()\n\n"

    if {model_type} == 'linear_regression':
        evaluation_code = """
print("Model Performance (Regression):")
print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

"""

    else:
        evaluation_code = """
print("Model Performance (Classification):")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, predictions, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, predictions, average='weighted'):.4f}")
"""        

    script_content = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import {'StandardScaler' if scaler_type == 'standard' else 'MinMaxScaler'}
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

# Load Data
data = pd.read_csv('{filename}')

# Plotting
{plot_code}

# Preprocessing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scaling
scaler = {'StandardScaler()' if scaler_type == 'standard' else 'MinMaxScaler()'}
scaled_data = scaler.fit_transform(X.select_dtypes(include='number'))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2)

# Model Training
model = {{
    'logistic_regression': LogisticRegression(),
    'linear_regression': LinearRegression(),
    'svm': SVC(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier()
}}['{model_type}']

model.fit(X_train, y_train)


# Evaluation
predictions = model.predict(X_test)

{evaluation_code}

"""
    fpath = os.path.join('downloads','ml_model_script.py')
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    with open(fpath, 'w') as script_file:
        script_file.write(script_content)

from flask import redirect, url_for # <--- Make sure these are imported

# to get everything from start refresh(hard)
@app.route('/reset')
def reset_session():
    global uploaded_data, plot_details, metrics, scaled_data, scaler_type,test_size
    global feature_names, pred_res, model, scaler, uploaded_filename,error_message,split_stats

    uploaded_data = None
    plot_details = []
    metrics = None
    scaled_data = None
    scaler_type = None
    feature_names = []
    pred_res = None
    model = None
    scaler = None
    uploaded_filename = None
    error_message = None
    test_size = None
    split_stats = None
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("server is running..!")
    app.run(debug=False)