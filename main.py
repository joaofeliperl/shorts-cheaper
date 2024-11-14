from flask import Flask
from app.routes import routes
import os  # Adicione esta linha para importar o módulo 'os'

def create_app():
    app = Flask(__name__, template_folder=os.path.join('app', 'templates'))
    app.register_blueprint(routes)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
