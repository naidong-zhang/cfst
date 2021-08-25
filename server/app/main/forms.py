from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

from wtforms import SubmitField, HiddenField, StringField, IntegerField
from wtforms import validators

class FaceForm(FlaskForm):
    face0 = FileField('Input a photo containing my face:', validators=[FileRequired()])
    face1 = FileField("Input a photo containing my spouse's face:", validators=[FileRequired()])
