# crimepredict/tabs/home/__init__.py
from .view import render

def register():
    return {
        "key": "home",
        "title": "Ana Sayfa",
        "icon": "🏠",
        "render": render,
    }
