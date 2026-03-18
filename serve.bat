@echo off
echo ローカルサーバーを起動します...
echo ブラウザで http://localhost:9090 を開いてください
echo 停止するには Ctrl+C を押してください
cd /d "%~dp0docs"
python -m http.server 9090
