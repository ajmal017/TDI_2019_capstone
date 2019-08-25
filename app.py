# app.py
from flask import Flask, flash, redirect, render_template, render_template_string, request, session, abort
from bokeh.client import pull_session
from bokeh.embed import server_session
from bokeh.embed import server_document
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__)

# grab the static resources
js_resources = INLINE.render_js()
css_resources = INLINE.render_css()

htmlbits="""
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Flask-Embedded Bokeh Server - Jack Rory Staunton</title>
  {{ js_resources|indent(4)|safe }}
  {{ css_resources|indent(4)|safe }}
</head>

<body>
  <div>
    The interactive chart interface above is a Bokeh app served by a Bokeh server that has been embedded
    in the Flask web app microframework. For more information see the section
    <a  target="_blank" href="https://bokeh.pydata.org/en/latest/docs/user_guide/server.html#embedding-bokeh-server-as-a-library">Embedding Bokeh Server as a Library</a>
    in the User's Guide. 
  </div>
  <div>{{ script|safe }}</div>
</body>
</html>"""

@app.route("/")
def hello():
    session = pull_session(session_id=None, url='http://localhost:5006/bokeh_plot')
    script = server_session(session_id=session.id, url='http://localhost:5006/bokeh_plot')
    return render_template('htmlbits.html', script=script, js_resources=js_resources,
        css_resources=css_resources)

if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)