import os
import torch
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField
from wtforms import SubmitField
from wtforms import FloatField
from wtforms import HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io
from utils.models import VGGEncoder
from utils.models import Decoder
from utils.utils import adaptive_instance_normalization
from utils.utils import calculate_mean_std
from werkzeug import run_simple
from PIL import ImageOps

app = Flask(__name__)
app.config['SECRET_KEY'] = "supersecreatkey"
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {"png","jpg","jpeg"}
os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)
class Uploadform(FlaskForm):
    content = FileField("Content image")
    style = FileField("Style image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha",default=1.0)
    submit = SubmitField("Transfer style")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VGGEncoder("vgg_normalised.pth").to(device)
decoder = Decoder().to(device)
checkpoint = torch.load("C:/Desktop-1/NST_PROJECT_TEST/experiment/checkpoint_epoch_10.pth",map_location=device)
state_dict = checkpoint["decoder"]
new_state_dict = {}
for key,value in state_dict.items():
    new_key = key.replace("decoder.","net.")
    new_state_dict[new_key] = value
decoder.load_state_dict(new_state_dict)
encoder.eval()
decoder.eval()
def allowed_file(filename):
    return (
        "." in filename and filename.rsplit(".",1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )
def style_transfer(content_image,style_image,encoder,decoder,alpha,device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)
    with torch.no_grad():
        content_feats = encoder(content_image,is_test=True)
        style_feats = encoder(style_image,is_test=True)
        stylized_feats = adaptive_instance_normalization(content_feats,style_feats)
        stylized_feats = alpha*stylized_feats + (1-alpha)*content_feats
        stylized_image = decoder(stylized_feats)
        return stylized_image
def save_image(image,path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0,1)
    image = transforms.ToPILImage()(image)
    image.save(path)
@app.route("/",methods=["GET","POST"])
def index():
    form = Uploadform()
    result_image = None
    content_filename = None
    style_filename = None
    error = None
    if request.method=="POST":
        if not form.validate_on_submit():
            error = "Form valdation failed"
        elif not form.content.data or not form.content.data.filename:
            error = "Please upload a content image"
        elif not form.style.data or not form.style.data.filename:
            error = "Please upload a style image"
        else:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                content_path = os.path.join(app.config["UPLOAD_FOLDER"],content_filename)
                form.content.data.save(content_path)
            else:
                error = "Invalid content image format"
            if error is None:
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    style_path = os.path.join(app.config["UPLOAD_FOLDER"],style_filename)
                    form.style.data.save(style_path)
                else:
                    error = "Invalid style image format"
            if error is None:
                try:
                    content_image = ImageOps.exif_transpose(Image.open(content_path)).convert("RGB")
                    style_image = ImageOps.exif_transpose(Image.open(style_path)).convert("RGB")
                    alpha = float(form.alpha.data)
                    stylized_image = style_transfer(content_image,style_image,encoder,decoder,alpha,device)
                    result_filename = "stylized_" + content_filename
                    result_path = os.path.join(app.config["UPLOAD_FOLDER"],result_filename)
                    save_image(stylized_image,result_path)
                    result_image = result_filename
                except Exception as e:
                    error = str(e)
    return render_template("index.html",form=form,result_image=result_image,content_image=content_filename,style_image=style_filename,error=error)
@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],filename)
@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples",filename)
if __name__=="__main__":
    run_simple("localhost",5000,app,use_reloader=True,use_debugger=True)