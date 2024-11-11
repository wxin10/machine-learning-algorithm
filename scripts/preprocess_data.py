import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 读取和合并CSV文件
def load_and_merge_csv(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    combined_data = pd.DataFrame()

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"正在读取文件: {file}")
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()  # 去除列名中的空格
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    return combined_data


# 处理缺失值
def handle_missing_values(data):
    numeric_columns = data.select_dtypes(include=['number']).columns  # 筛选数值列
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())  # 填充数值列的缺失值
    return data


# 处理无效值
def handle_invalid_values(data):
    data = data.replace([float('inf'), float('-inf')], float('nan'))
    data = data.fillna(0)  # 填充NaN
    return data


# 标签编码
def encode_labels(data, label_column='Label'):
    if label_column in data.columns:
        label_encoder = LabelEncoder()
        data[label_column] = label_encoder.fit_transform(data[label_column])
        print(f"已对标签 '{label_column}' 进行编码。")
        return data, label_encoder
    else:
        print(f"警告：数据中没有找到 '{label_column}' 列，跳过标签编码。")
        return data, None


# 特征标准化
def scale_features(data, feature_columns):
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data, scaler


# 数据分割
def split_data(data, label_column='Label', test_size=0.2):
    X = data.drop(columns=[label_column])
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    folder_path = '../data/CICIDS2017/'  # 确保路径正确
    output_dir = '../data/processed'

    combined_data = load_and_merge_csv(folder_path)

    # 处理缺失值和无效值
    combined_data = handle_missing_values(combined_data)
    combined_data = handle_invalid_values(combined_data)

    feature_columns = combined_data.select_dtypes(include=['number']).columns  # 仅选择数值列进行标准化

    # 标签编码
    combined_data, label_encoder = encode_labels(combined_data)

    # 如果存在 'Label' 列则继续处理
    if label_encoder is not None:
        combined_data, scaler = scale_features(combined_data, feature_columns)

        # 分割数据集
        X_train, X_test, y_train, y_test = split_data(combined_data)

        # 保存处理后的数据
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
        print("数据预处理完成并保存！")
    else:
        print("由于缺少 'Label' 列，跳过预处理。")
