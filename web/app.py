# Flask server: loads model on startup, serves API routes and Jinja2 frontend.

from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
	return "hi"

if __name__ == "__main__":
	app.run()
