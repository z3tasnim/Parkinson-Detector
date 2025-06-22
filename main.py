import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings('ignore')


class ParkinsonPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Parkinson's Disease Prediction System")
        self.root.geometry("1150x750")
        self.root.configure(bg="#f2fbff")

        self.models_trained = False
        self.logistic_model = None
        self.knn_model = None
        self.tree_model = None
        self.scaler = None

        self.feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]

        self.entries = {}
        self.build_interface()

    def build_interface(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f2fbff")
        style.configure("TLabel", background="#f2fbff", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#3282b8", foreground="white")

        header = ttk.Label(self.root, text="🧠 Parkinson's Disease Prediction", font=("Segoe UI", 18, "bold"), foreground="#1b262c")
        header.pack(pady=20)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        ttk.Button(control_frame, text="Train Models", style="Accent.TButton", command=self.train_models).grid(row=0, column=0, padx=10)
        ttk.Button(control_frame, text="Load Sample Data", command=self.load_sample_data).grid(row=0, column=1, padx=10)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).grid(row=0, column=2, padx=10)
        ttk.Button(control_frame, text="Predict", style="Accent.TButton", command=self.make_prediction).grid(row=0, column=3, padx=10)
        ttk.Button(control_frame, text="Predict from Test File", command=self.predict_from_test_file).grid(row=0, column=4, padx=10)

        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Input section
        input_frame = ttk.LabelFrame(content_frame, text="Voice Feature Inputs", padding=10)
        input_frame.pack(side="left", fill="both", expand=True, padx=10)

        canvas = tk.Canvas(input_frame, bg="#ffffff", highlightthickness=1, highlightbackground="#bbb", width=320)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, feature in enumerate(self.feature_names):
            ttk.Label(scrollable_frame, text=feature).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            entry = ttk.Entry(scrollable_frame, width=20)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[feature] = entry

        # Output & Graph section
        output_frame = ttk.LabelFrame(content_frame, text="Results and Graph", padding=10)
        output_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.status_label = ttk.Label(output_frame, text="Status: Ready", font=("Segoe UI", 10, "bold"), foreground="#00796b")
        self.status_label.pack(anchor="w", pady=5)

        self.results_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=50, height=15, font=("Consolas", 10))
        self.results_text.pack(fill="both", expand=False)

        # Matplotlib Figure for graph
        self.fig, self.ax = plt.subplots(figsize=(6, 3.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=output_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)

    def train_models(self):
        try:
            self.status_label.config(text="Status: Training models...")
            self.root.update()

            if not os.path.exists("parkinson_disease.csv"):
                messagebox.showerror("Error", "CSV file 'parkinson_disease.csv' not found.")
                self.status_label.config(text="Status: CSV not found")
                return

            df = pd.read_csv("parkinson_disease.csv")
            df.drop(columns="name", inplace=True)

            X = df.drop(columns="status")
            y = df["status"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.logistic_model = LogisticRegression(random_state=42)
            self.logistic_model.fit(X_train_scaled, y_train)

            self.knn_model = KNeighborsClassifier(n_neighbors=5)
            self.knn_model.fit(X_train_scaled, y_train)

            self.tree_model = DecisionTreeClassifier(random_state=42, max_depth=4)
            self.tree_model.fit(X_train_scaled, y_train)

            def metrics(true, pred):
                return [
                    accuracy_score(true, pred),
                    precision_score(true, pred),
                    recall_score(true, pred),
                    f1_score(true, pred),
                    roc_auc_score(true, pred)
                ]

            result_str = "✅ MODEL PERFORMANCE\n" + "-"*45 + "\n"
            result_str += "Model            Accuracy  Precision  Recall   F1 Score  AUC\n"
            result_str += "-"*60 + "\n"

            for name, model in {
                "Logistic": self.logistic_model,
                "KNN     ": self.knn_model,
                "Tree    ": self.tree_model
            }.items():
                m = metrics(y_test, model.predict(X_test_scaled))
                result_str += f"{name:<15} {m[0]:.4f}    {m[1]:.4f}    {m[2]:.4f}   {m[3]:.4f}   {m[4]:.4f}\n"

            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, result_str)
            self.status_label.config(text="Status: Training completed")
            self.models_trained = True
            messagebox.showinfo("Training Complete", "Models trained successfully!")

            # Clear previous graph
            self.ax.clear()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.status_label.config(text="Status: Training failed")

    def load_sample_data(self):
        sample = {
            'MDVP:Fo(Hz)': '119.992', 'MDVP:Fhi(Hz)': '157.302', 'MDVP:Flo(Hz)': '74.997',
            'MDVP:Jitter(%)': '0.00784', 'MDVP:Jitter(Abs)': '0.00007', 'MDVP:RAP': '0.00370',
            'MDVP:PPQ': '0.00554', 'Jitter:DDP': '0.01109', 'MDVP:Shimmer': '0.04374',
            'MDVP:Shimmer(dB)': '0.426', 'Shimmer:APQ3': '0.02182', 'Shimmer:APQ5': '0.03130',
            'MDVP:APQ': '0.02971', 'Shimmer:DDA': '0.06545', 'NHR': '0.02211', 'HNR': '21.033',
            'RPDE': '0.414783', 'DFA': '0.815285', 'spread1': '-4.813031', 'spread2': '0.266482',
            'D2': '2.301442', 'PPE': '0.284654'
        }
        for k, v in sample.items():
            if k in self.entries:
                self.entries[k].delete(0, tk.END)
                self.entries[k].insert(0, v)
        self.status_label.config(text="Status: Sample data loaded")

    def clear_all(self):
        for e in self.entries.values():
            e.delete(0, tk.END)
        self.results_text.delete("1.0", tk.END)
        self.status_label.config(text="Status: Cleared")
        self.ax.clear()
        self.canvas.draw()

    def make_prediction(self):
        try:
            if not self.models_trained:
                messagebox.showwarning("Not Trained", "Please train the models first.")
                return

            values = []
            for feat in self.feature_names:
                val = self.entries[feat].get().strip()
                if not val:
                    messagebox.showerror("Missing Input", f"Please enter a value for {feat}")
                    return
                values.append(float(val))

            arr = np.array([values])
            scaled = self.scaler.transform(arr)

            preds = [
                self.logistic_model.predict(scaled)[0],
                self.knn_model.predict(scaled)[0],
                self.tree_model.predict(scaled)[0],
            ]
            vote = max(set(preds), key=preds.count)
            conf = sum(preds) / len(preds)

            result = "🧠 Ensemble Prediction: " + ("Parkinson's Detected" if vote == 1 else "Healthy")
            conf_text = f"Confidence Score: {conf:.2f} (0 = Healthy, 1 = Parkinson's)"

            self.results_text.insert(tk.END, f"\n\n{result}\n{conf_text}\n")
            self.status_label.config(text="Status: Prediction completed")

            # Plot bar chart of feature values for this input
            self.ax.clear()
            self.ax.barh(self.feature_names, values, color="#3282b8")
            self.ax.set_title("Feature Values for Input Patient")
            self.ax.set_xlabel("Value")
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_label.config(text="Status: Prediction failed")

    def predict_from_test_file(self):
        if not self.models_trained:
            messagebox.showwarning("Not Trained", "Please train the models first.")
            return
        file_path = filedialog.askopenfilename(
            title="Select Test File",
            filetypes=[("CSV files", "*.csv"), ("Data files", "*.data"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            # Try reading with comma delimiter first
            try:
                df = pd.read_csv(file_path)
            except Exception:
                # Try tab delimiter if comma fails
                df = pd.read_csv(file_path, delimiter='\t')

            # Verify required columns present
            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                messagebox.showerror("File Error", f"Missing features in file: {', '.join(missing_features)}")
                return
            if 'status' not in df.columns:
                messagebox.showwarning("Warning", "No 'status' column found in file, skipping performance metrics.")

            X = df[self.feature_names]
            y_true = df['status'] if 'status' in df.columns else None

            X_scaled = self.scaler.transform(X)

            preds_logistic = self.logistic_model.predict(X_scaled)
            preds_knn = self.knn_model.predict(X_scaled)
            preds_tree = self.tree_model.predict(X_scaled)

            # Ensemble vote
            preds_ensemble = []
            for i in range(len(X)):
                votes = [preds_logistic[i], preds_knn[i], preds_tree[i]]
                pred = max(set(votes), key=votes.count)
                preds_ensemble.append(pred)

            # Display results
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, f"🧪 Predictions from file: {os.path.basename(file_path)}\n")
            self.results_text.insert(tk.END, "-"*60 + "\n")

            for i, pred in enumerate(preds_ensemble):
                status = "Parkinson's Detected" if pred == 1 else "Healthy"
                self.results_text.insert(tk.END, f"Sample {i+1}: {status}\n")

            if y_true is not None:
                acc = accuracy_score(y_true, preds_ensemble)
                prec = precision_score(y_true, preds_ensemble)
                rec = recall_score(y_true, preds_ensemble)
                f1 = f1_score(y_true, preds_ensemble)
                auc = roc_auc_score(y_true, preds_ensemble)
                self.results_text.insert(tk.END, "-"*60 + "\n")
                self.results_text.insert(tk.END, "Performance on test file:\n")
                self.results_text.insert(tk.END, f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\nAUC: {auc:.4f}\n")

            self.status_label.config(text="Status: File prediction completed")

            # Optionally plot average feature values from test file
            avg_values = X.mean(axis=0).values
            self.ax.clear()
            self.ax.barh(self.feature_names, avg_values, color="#3282b8")
            self.ax.set_title("Average Feature Values from Test File")
            self.ax.set_xlabel("Value")
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("File Prediction Error", str(e))
            self.status_label.config(text="Status: File prediction failed")


def main():
    root = tk.Tk()
    app = ParkinsonPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
