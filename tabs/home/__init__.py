# crimepredict/tabs/home/__init__.py
from .view import render
import tabs.home.diagnostics

def register():
    return {
        "key": "home",
        "title": "Ana Sayfa",
        "icon": "🏠",
        "render": render,
    }
