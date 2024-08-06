# free-GPU-memory
pytorchでGPUに転送したモデルをどうにかして削除し，メモリ解放する

## 環境構築
```
python -m venv venv
.\venv\Scripts\activate
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ipykernel
ipython kernel install --user --name=venv
```

## 使用方法
以下のコマンドでプログラムを実行するだけ．
```
python main.py
```
