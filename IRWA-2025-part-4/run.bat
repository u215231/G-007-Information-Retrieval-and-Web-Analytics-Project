@echo off
set FLASK_APP=web_app.py
set FLASK_ENV=development
flask run --host=0.0.0.0 --port=8088
pause