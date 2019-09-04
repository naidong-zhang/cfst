from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired

from wtforms import SubmitField, HiddenField, StringField, IntegerField
from wtforms.validators import InputRequired

class PortraitForm(FlaskForm):
    portrait = FileField('input my photo', validators=[FileRequired()])
    bbox_id = IntegerField('input bbox id', validators=[InputRequired()])
    face = FileField("input my spouse's photo(face)", validators=[FileRequired()])
    submit = SubmitField('Submit')
