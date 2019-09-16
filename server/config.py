import os
from models import create_singleton_models

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY')
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in \
        ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
    FLASKY_MAIL_SENDER = 'Flasky Admin <flasky@example.com>'
    FLASKY_ADMIN = os.environ.get('FLASKY_ADMIN')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # models
    TF_CONFIGFILE = os.path.join(basedir, 'models', 'opencv_face_detector.pbtxt')
    TF_MODELFILE = os.path.join(basedir, 'models' ,'opencv_face_detector_uint8.pb')
    MTCNN_PATH = os.path.join(basedir, 'models', 'mtcnn-model')
    MODEL_PATH = os.path.join(basedir, 'models', 'model-y1-test2', 'model-0000')


    @classmethod
    def init_app(cls, app):
        assert not hasattr(app, 'models')
        app.models = create_singleton_models(tf_configfile=cls.TF_CONFIGFILE,
                                            tf_modelfile=cls.TF_MODELFILE,
                                            mtcnn_path=cls.MTCNN_PATH,
                                            model_path=cls.MODEL_PATH)


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data-dev.sqlite')


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        'sqlite://'


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data.sqlite')


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}
