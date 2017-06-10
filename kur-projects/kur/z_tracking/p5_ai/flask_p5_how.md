**How to install flask?**
- `source activate dlnd-tf-lab`
- `conda|pip install flask`

**Flask hello world**
```python
# hello.py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()
```
```bash
python hello.py
# Running on http://localhost:5000/
```

**01_simple_flask_p5**
- structure of p5-flask project
	- p5 library(p5.dom.js, p5.sound.js, p5.js)
	- index.html (structure the webpage, and bring in libraries)
	- sketch.js (like p5 editor page)
	- server.py (flask moving data from python to p5?)
- Atom assistant
	- use browser-plus package to open html file from atom
	- open web on html file: `ctrl option o`
- make sketch performance into video or gif using `docode` [here](https://www.npmjs.com/package/docode)

**Tasks to do**
- shiffman dl week5: bring keras, p5, flask together
- dissect `server.py`
- learn p5 with simple [pure p5 examples](https://p5js.org/examples/)
