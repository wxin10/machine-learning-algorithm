'''
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# 加载测试数据集
def load_test_data(data_dir):
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_test, y_test


# 加载模型
def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


# 绘制并保存混淆矩阵
def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


# 绘制并保存分类报告
def plot_classification_report(report, output_dir):
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 0.05, str(report), {'fontsize': 12}, fontproperties='monospace')
    plt.axis('off')
    plt.title('Classification Report')
    plt.savefig(os.path.join(output_dir, 'classification_report.png'))
    plt.close()


# 评估模型
def evaluate_model(model, X_test, y_test, output_dir):
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 分类报告
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 保存评估结果为图片
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(cm, output_dir)
    plot_classification_report(report, output_dir)

    # 保存分类报告文本
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    return accuracy, cm


if __name__ == "__main__":
    # 指定数据目录、模型路径和输出目录
    data_dir = os.path.abspath('../data/processed')  # 测试数据目录
    model_path = os.path.abspath('../models/random_forest_model.pkl')  # 训练好的模型路径
    output_dir = os.path.abspath('../results')

    # 加载测试数据
    print("Loading test data...")
    X_test, y_test = load_test_data(data_dir)

    # 加载模型
    print("Loading model...")
    model = load_model(model_path)

    # 评估模型
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, output_dir)

    print("Model evaluation completed successfully.")
'''

import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 前30个重要特征的列表，与训练时一致
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


# 加载测试数据集，只选择前30个重要特征
def load_test_data(data_dir):
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))[TOP_30_FEATURES]
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_test, y_test


# 加载模型
def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


# 绘制并保存混淆矩阵
def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix1.png'))
    plt.close()


# 绘制并保存分类报告
def plot_classification_report(report, output_dir):
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 0.05, str(report), {'fontsize': 12}, fontproperties='monospace')
    plt.axis('off')
    plt.title('Classification Report')
    plt.savefig(os.path.join(output_dir, 'classification_report1.png'))
    plt.close()


# 评估模型
def evaluate_model(model, X_test, y_test, output_dir):
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 分类报告
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 保存评估结果为图片
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(cm, output_dir)
    plot_classification_report(report, output_dir)

    # 保存报告文本文件
    with open(os.path.join(output_dir, 'classification_report1.txt'), 'w') as f:
        f.write(report)

    return accuracy, cm


if __name__ == "__main__":
    # 指定数据目录、模型路径和输出目录
    data_dir = os.path.abspath('../data/processed')  # 测试数据目录
    model_path = os.path.abspath('../models/random_forest_model_top30.pkl')  # 模型路径
    output_dir = os.path.abspath('../results')

    # 加载测试数据
    print("Loading test data with top 30 features...")
    X_test, y_test = load_test_data(data_dir)

    # 加载模型
    print("Loading model...")
    model = load_model(model_path)

    # 评估模型
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, output_dir)

    print("Evaluation completed successfully.")
