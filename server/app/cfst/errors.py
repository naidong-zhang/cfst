from flask import render_template
from . import cfst

@cfst.app_errorhandler(400)
def bad_request(e):
    return render_template('400.html', msg=e), 400

@cfst.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html', msg=e), 404

@cfst.app_errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', msg=e), 500
