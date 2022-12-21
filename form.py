from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, SelectField, BooleanField, StringField, TelField

class profileForm(FlaskForm):
    fname= StringField('fname')
    lname= StringField('lname')
    phone = TelField('phone')
    age = IntegerField('age')
    # dependents = IntegerField('Dependents')
    gender = SelectField('gender', choices=[(0, 0), (1, 1)])
    income = IntegerField('income')
    # residence = SelectField('residence', choices=[(1,0), (0, 0)])
    # diabetes = BooleanField('diabetes')
    # heart = BooleanField('heart')
    # bp = BooleanField('bp')
    # covid = BooleanField('covid')
    # surg = BooleanField('Surgery')
    # other = BooleanField('other')
    submit = SubmitField('Sign Up')