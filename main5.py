from flask import Flask, render_template
app = Flask(__name__)

@app.route("/user/<float:username>")
def display_name(username):
    return f"This page is about {username} "

@app.route("/about")
def about_page():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)

