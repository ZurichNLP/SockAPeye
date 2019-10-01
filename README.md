# SockAPeye

This project requires Python 3.

Install requirements as follows:

    pip install -r requirements.txt

Run with gunicorn as follows:

    gunicorn app:app --workers=1 --timeout 100
