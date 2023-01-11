from flask import Flask, redirect, url_for

app = Flask(__name__)

@app.route("/")
def Intro():
    return "Welcome"


@app.route("/admin")
def Hello_a():
    return "Hello Admin"

@app.route("/<guest>")
def hello_g(guest):
    return"Hello %s" %guest

@app.route("/user/<name>")
def hello_n(name):
    if name == "admin":
        return redirect(url_for("Hello_a"))
    else:
        return  redirect(url_for("hello_g", guest = name))



if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
