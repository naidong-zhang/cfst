# import os
# import base64

from flask import render_template, session, redirect, url_for, current_app, flash, request, jsonify
# from werkzeug.utils import secure_filename
from . import main
from .errors import bad_request
# from .forms import FaceForm

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/projects')
def projects():
    return render_template('projects.html')

@main.route('/contact')
def contact():
    return render_template('contact.html')
