# model.py
import numpy as np
import joblib
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class CNN1D(nn.Module):
    def __init__(self, input_dim=171, hidden_channels=64, kernel_size=3, epochs=10,
                 batch_size=32, learning_rate=0.001):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels,
                                kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # x shape: (batch, 1, input_dim)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)  # (batch, hidden_channels)
        x = self.fc(x)                 # (batch, 1)
        return x.squeeze(-1)



class BrainAgeModel:
    def __init__(self, model_type='lasso', **kwargs):
        """
        初始化模型。

        参数:
            model_type (str): 'lasso', 'svr', 'elasticnet', 'cnn'
            **kwargs: 传递给具体模型的参数（例如 alpha, C, etc.）
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        if self.model_type == 'lasso':
            return Lasso(**self.kwargs)
        elif self.model_type == 'svr':
            return SVR(**self.kwargs)
        elif self.model_type == 'elasticnet':
            return ElasticNet(**self.kwargs)
        elif self.model_type == 'cnn':
            return CNN1D(**self.kwargs).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def fit(self, X, y):
        """训练模型"""
        if self.model_type == 'cnn':
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # 转换为 PyTorch 张量并添加通道维度 (batch, 1, features)
            X_tensor = torch.tensor(X).unsqueeze(1).to(self.device)  # (batch, 1, input_dim)
            y_tensor = torch.tensor(y).to(self.device)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.model.batch_size, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.model.learning_rate)

            self.model.train()
            for epoch in tqdm(range(self.model.epochs)):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.model.epochs}, Loss: {total_loss/len(dataloader):.4f}")


        else:
            self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """预测年龄"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        if self.model_type == 'cnn':
            X = np.array(X, dtype=np.float32)
            X_tensor = torch.tensor(X).unsqueeze(1).to(self.device)
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy()
            return y_pred
        return self.model.predict(X)

    def evaluate(self, X, y):
        """评估模型，返回 MAE 和 R²"""
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {'MAE': mae, 'R2': r2}

    def save(self, path):
        """保存模型和参数到文件"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，无法保存")
        if self.model_type == 'cnn':
            # 对于CNN，我们需要保存模型状态字典和初始化参数
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'kwargs': self.kwargs
            }, path)
            print(f"CNN模型已保存至 {path}")
        else:
            joblib.dump({
                'model_type': self.model_type,
                'kwargs': self.kwargs,
                'model': self.model,
                'is_fitted': self.is_fitted
            }, path)
        print(f"模型已保存至 {path}")

    def load(self, path):
        """从文件加载模型"""
        if self.model_type == 'cnn':
            checkpoint = torch.load(path, map_location=self.device)
            self.model_type = checkpoint['model_type']
            self.kwargs = checkpoint['kwargs']
            self.model = self._build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_fitted = True
        else:
            data = joblib.load(path)
            self.model_type = data['model_type']
            self.kwargs = data['kwargs']
            self.model = data['model']
            self.is_fitted = data['is_fitted']
        print(f"模型已从 {path} 加载")
        return self

    def age_bias_correction(self, y_true, y_pred):
        """
        年龄偏差校正：基于训练集拟合线性回归，校正预测年龄的系统性偏差。
        返回校正后的预测年龄。
        注意：通常应在训练集上拟合校正参数，然后应用于测试集。
        """
        from sklearn.linear_model import LinearRegression
        # 拟合线性回归：y_pred ~ y_true
        lr = LinearRegression()
        lr.fit(y_true.reshape(-1, 1), y_pred)
        # 计算偏差
        bias = lr.predict(y_true.reshape(-1, 1)) - y_pred
        # 校正
        corrected = y_pred + bias
        return corrected

# 示例用法
if __name__ == "__main__":
    # 简单测试
    X_demo = np.random.rand(100, 120)
    y_demo = np.random.rand(100) * 50 + 50
    model = BrainAgeModel(model_type='lasso', alpha=0.01)
    model.fit(X_demo, y_demo)
    print(model.evaluate(X_demo, y_demo))