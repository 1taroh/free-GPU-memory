# free-GPU-memory
pytorchでGPUに転送したモデルをどうにかして削除し，メモリ解放する

## 環境構築
```
python -m venv venv
.\venv\Scripts\activate
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法
以下のコマンドでプログラムを実行するだけ．
```
python main.py
```
