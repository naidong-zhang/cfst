from flask import Flask
from flask_bootstrap import Bootstrap
# from flask_mail import Mail
from flask_moment import Moment
# from flask_sqlalchemy import SQLAlchemy
from config import config

bootstrap = Bootstrap()
# mail = Mail()
moment = Moment()
# db = SQLAlchemy()


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    bootstrap.init_app(app)
    # mail.init_app(app)
    moment.init_app(app)
    # db.init_app(app)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .cfst import cfst as cfst_blueprint
    app.register_blueprint(cfst_blueprint, url_prefix='/cfst')

    from .api.v0 import api as apiv0_blueprint
    app.register_blueprint(apiv0_blueprint, url_prefix='/api/v0')

    @app.route('/hello-world')
    def hello():
        return 'Hello World!'

    return app

