from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

from wtforms import SubmitField, HiddenField, StringField, IntegerField
from wtforms.validators import InputRequired

class FaceForm(FlaskForm):
    face0 = FileField('input my photo(face)', validators=[FileRequired()])
    face1 = FileField("input my spouse's photo(face)", validators=[FileRequired()])
    submit = SubmitField('Submit')
