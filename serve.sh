#!/bin/bash
echo "ローカルサーバーを起動します..."
echo "ブラウザで http://localhost:8080 を開いてください"
echo "停止するには Ctrl+C を押してください"
cd "$(dirname "$0")/docs"
python3 -m http.server 8080 --bind 127.0.0.1
