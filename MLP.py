import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


RESULTS_DB = "experiment_results.json"

class OptimizedMLP:
    def __init__(self, input_size, hidden_size=64, output_size=1, 
                 learning_rate=0.01, lambda_l2=0.0001):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.loss_history = []

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.leaky_relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X, y, class_weights):
        m = X.shape[0]
        
        
        dZ2 = (self.A2 - y) * class_weights
        dW2 = (1/m) * np.dot(self.A1.T, dZ2) + (self.lambda_l2 * self.W2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.leaky_relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1) + (self.lambda_l2 * self.W1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def compute_loss(self, y, y_pred, class_weights):
        bce = -np.mean(class_weights * (y * np.log(y_pred + 1e-7) + (1-y) * np.log(1-y_pred + 1e-7)))
        l2_loss = 0.5 * self.lambda_l2 * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return bce + l2_loss
    
    def fit(self, X, y, epochs=2000, batch_size=128, verbose=True):
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y.flatten())
        sample_weights = np.where(y == 0, class_weights[0], class_weights[1])
        
        best_loss = float('inf')
        patience = 30
        wait = 0
        
        for epoch in range(epochs):
            
            indices = np.random.permutation(len(X))
            epoch_loss = 0
            
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                w_batch = sample_weights[batch_idx]
                
                
                y_pred = self.forward(X_batch)
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch, w_batch)
                
                
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                
            
                batch_loss = self.compute_loss(y_batch, y_pred, w_batch)
                epoch_loss += batch_loss * len(batch_idx)
            
            epoch_loss /= len(X)
            self.loss_history.append(epoch_loss)
            
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and (epoch % 100 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    def predict(self, X, threshold=0.5):
        return (self.forward(X) > threshold).astype(int)
    
    def predict_proba(self, X):
        return self.forward(X)

def load_or_create_db():
    if os.path.exists(RESULTS_DB):
        with open(RESULTS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_to_db(db):
    with open(RESULTS_DB, 'w') as f:
        json.dump(db, f, indent=2)

def record_experiment(seed, train_path, test_path, metrics):    
    db = load_or_create_db()
    
    db[str(seed)] = {
        "train_path": train_path,
        "test_path": test_path,
        "metrics": metrics,
    }
    
    save_to_db(db)
    print(f"\nResults saved under seed: {seed}")

def retrieve_experiment(seed):
    
    db = load_or_create_db()
    if str(seed) in db:
        return db[str(seed)]
    return None

def get_user_input():
    while True:
        try:
            seed = int(input("\nEnter seed value (or 0 for new experiment): ").strip())
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    if seed != 0 and retrieve_experiment(seed):
        print(f"\nFound existing experiment with seed {seed}")
        return seed, None, None
    
    train_path = input("Enter training dataset path: ").strip()
    test_path = input("Enter test dataset path: ").strip()
    return seed, train_path, test_path

def load_and_prepare_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit(1)
    
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def print_clean_metrics(y_true, y_pred, set_name):
    accuracy = np.mean(y_true == y_pred) * 100  
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']  
    recall = report['weighted avg']['recall']      
    f1 = report['weighted avg']['f1-score']        
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n=== {set_name} Metrics ===")
    print(f"{'Accuracy:':<12} {accuracy:.2f}%")   
    print(f"{'F1 Score:':<12} {f1:.4f}")         
    print(f"{'Precision:':<12} {precision:.4f}")  
    print(f"{'Recall:':<12} {recall:.4f}")         
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               |   0   |   1")
    print("         ---------------------")
    print(f"Actual 0     |  {cm[0,0]:^5} | {cm[0,1]:^5}")
    print(f"Actual 1     |  {cm[1,0]:^5} | {cm[1,1]:^5}")

def run_new_experiment(seed, train_path, test_path):
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = load_and_prepare_data(train_path, test_path)
    
    model = OptimizedMLP(
        input_size=X_train.shape[1],
        hidden_size=16,
        learning_rate=0.03,
        lambda_l2=0.001,
        output_size=1 
    )
    
    print("\nTraining started...")
    model.fit(X_train, y_train)
    
    
    metrics = {
    "training": {
        "accuracy": np.mean(y_train == model.predict(X_train)) * 100,  
        "f1": float(classification_report(y_train, model.predict(X_train), output_dict=True)['weighted avg']['f1-score']), 
        "precision": float(classification_report(y_train, model.predict(X_train), output_dict=True)['weighted avg']['precision']),  
        "recall": float(classification_report(y_train, model.predict(X_train), output_dict=True)['weighted avg']['recall']),  
        "confusion": confusion_matrix(y_train, model.predict(X_train)).tolist()
    },
    "test": {
        "accuracy": np.mean(y_test == model.predict(X_test)) * 100,  
        "f1": float(classification_report(y_test, model.predict(X_test), output_dict=True)['weighted avg']['f1-score']), 
        "precision": float(classification_report(y_test, model.predict(X_test), output_dict=True)['weighted avg']['precision']),  
        "recall": float(classification_report(y_test, model.predict(X_test), output_dict=True)['weighted avg']['recall']),  
        "confusion": confusion_matrix(y_test, model.predict(X_test)).tolist()
    }
}
    
    record_experiment(seed, train_path, test_path, metrics)
    return metrics

def display_results(metrics):
    print("\n=== Results ===")
    for dataset in ['training', 'test']:
        print(f"\n{dataset.capitalize()} Metrics:")
        data = metrics[dataset]
        print(f"Accuracy:  {data['accuracy']:.2f}%")  
        print(f"F1 Score:  {data['f1']:.4f}")       
        print(f"Precision: {data['precision']:.4f}")  
        print(f"Recall:    {data['recall']:.4f}")     
        
        print("\nConfusion Matrix:")
        print("        Predicted 0  Predicted 1")
        print(f"Actual 0 {data['confusion'][0][0]:^11} {data['confusion'][0][1]:^11}")
        print(f"Actual 1 {data['confusion'][1][0]:^11} {data['confusion'][1][1]:^11}")

def main():
    print("=== MLP Experiment Manager ===")
    print(f"Results database: {Path(RESULTS_DB).absolute()}")
    
    seed, train_path, test_path = get_user_input()
    
    if train_path: 
        metrics = run_new_experiment(seed, train_path, test_path)
    else:  
        metrics = retrieve_experiment(seed)['metrics']
    
    display_results(metrics)

if __name__ == "__main__":
    main()