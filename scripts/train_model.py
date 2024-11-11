'''将所有特征都进行训练
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


# 加载预处理后的数据集
def load_preprocessed_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    return X_train, y_train


# 训练随机森林模型，调用所有 CPU 核心
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # 使用所有 CPU 核心
    rf_model.fit(X_train, y_train)
    return rf_model


# 保存模型
def save_model(model, output_dir, model_name='random_forest_model.pkl'):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # 指定数据目录和模型保存目录
    data_dir = os.path.abspath('../data/processed')
    model_dir = os.path.abspath('../models')

    # 加载预处理后的数据
    print("Loading preprocessed data...")
    X_train, y_train = load_preprocessed_data(data_dir)

    # 训练模型
    print("Training Random Forest model with all CPU cores...")
    rf_model = train_random_forest(X_train, y_train)

    # 保存模型
    print("Saving model...")
    save_model(rf_model, model_dir)

    print("Model training and saving completed successfully.")
'''


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 定义前30个重要特征的列表
TOP_30_FEATURES = [
    'Packet Length Variance', 'Packet Length Std', 'Max Packet Length', 'Subflow Bwd Bytes',
    'Avg Bwd Segment Size', 'Average Packet Size', 'Packet Length Mean', 'Bwd Packet Length Max',
    'Total Length of Bwd Packets', 'Bwd Packet Length Std', 'Init_Win_bytes_backward',
    'Flow Bytes/s', 'Min Packet Length', 'Bwd Packet Length Mean', 'Flow IAT Max',
    'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Flow Duration', 'Flow IAT Min',
    'Init_Win_bytes_forward', 'act_data_pkt_fwd', 'Total Fwd Packets', 'Fwd Header Length',
    'Subflow Fwd Bytes', 'ACK Flag Count', 'Avg Fwd Segment Size', 'Flow IAT Std',
    'Bwd IAT Mean', 'Fwd IAT Min', 'Idle Max'
]

# 加载并筛选前30个重要特征的数据集
def load_preprocessed_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    # 只保留前30个重要特征
    X_train = X_train[TOP_30_FEATURES]
    return X_train, y_train

# 训练随机森林模型，使用所有 CPU 核心
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # 使用所有 CPU 核心
    rf_model.fit(X_train, y_train)
    return rf_model

# 保存模型
def save_model(model, output_dir, model_name='random_forest_model_top30.pkl'):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # 指定数据目录和模型保存目录
    data_dir = os.path.abspath('../data/processed')
    model_dir = os.path.abspath('../models')

    # 加载预处理后的数据（只保留前30个重要特征）
    print("Loading preprocessed data with top 30 features...")
    X_train, y_train = load_preprocessed_data(data_dir)

    # 训练模型
    print("Training Random Forest model with all CPU cores on top 30 features...")
    rf_model = train_random_forest(X_train, y_train)

    # 保存模型
    print("Saving model...")
    save_model(rf_model, model_dir)

    print("Model training and saving completed successfully.")
