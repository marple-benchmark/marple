import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from flask import Flask, render_template, request


app = Flask(__name__)

# Route for the index page
@app.route('/')
def index():
    return render_template('demo.html')

@app.after_request
def add_header(response):
    response.cache_control.no_cache = True
    response.cache_control.max_age = 0
    return response

# tb replaced with image generator
def generate_program():
    # Generate a random noise image and save it as JPG
    img = np.random.randn(200, 400, 3) * 255
    img = img.astype(np.uint8)
    os.makedirs('static/images/', exist_ok=True)
    img_file = os.path.join('static/images/', 'random_noise.jpg')
    Image.fromarray(img).save(img_file)
    return f'<img src=static/images/random_noise.jpg width="300" height="300"/>'
 

@app.route('/generate', methods=['POST'])
def generate():
    # Get the data from the POST request
    task = request.form['task']
    num_agents = request.form['num_agents']
    num_rooms = request.form['num_rooms']
    num_furniture = request.form['num_furniture']
    num_objects = request.form['num_objects']

    print(f"Selected values: {task}, {num_agents}, {num_rooms}, {num_furniture}, {num_objects}")

    # Generate the program

    program = generate_program()

    data = {
        'program': program,
    }
    print(data)
    return data


# Route for submitting the data

if __name__ == '__main__':
    app.run(debug=True)