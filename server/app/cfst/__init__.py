from flask import Blueprint

cfst = Blueprint('cfst', __name__)

from . import views, errors
