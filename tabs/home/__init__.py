from .view import render

def register():
    return {
        "key": "home",
        "title": "Ana Sayfa",
        "icon": "🏠",
        "render": render,
    }
