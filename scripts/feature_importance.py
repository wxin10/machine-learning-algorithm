import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# 加载预处理后的数据集
def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    return X_train, y_train


# 计算并绘制特征重要性
def feature_importance_analysis(X_train, y_train, output_dir):
    # 使用随机森林来计算特征重要性，启用所有 CPU 核心
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # n_jobs=-1 启用所有 CPU 核心
    model.fit(X_train, y_train)

    # 获取特征重要性
    importance = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # 输出特征重要性数据到CSV
    os.makedirs(output_dir, exist_ok=True)
    feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    print("Feature importance analysis completed and saved.")


if __name__ == "__main__":
    data_dir = os.path.abspath('../data/processed')  # 确保指向预处理后的数据文件夹
    output_dir = os.path.abspath('../results')  # 指向存储结果的文件夹

    # 加载数据
    print("Loading preprocessed data for feature importance analysis...")
    X_train, y_train = load_data(data_dir)

    # 分析特征重要性
    print("Analyzing feature importance...")
    feature_importance_analysis(X_train, y_train, output_dir)

    print("Feature importance analysis completed successfully.")
