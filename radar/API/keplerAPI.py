# import main Flask class and request object
from flask import Flask, request
from orbit.kepler import svFromJsonArgs
import json

# create the Flask app
app = Flask(__name__)

@app.route('/query-example')
def query_example():
    return 'Query String Example'

@app.route('/form-example')
def form_example():
    return 'Form Data Example'

# GET requests will be blocked
@app.route('/json-example', methods=['POST'])
def json_example():
    request_data = request.get_json()
    sv_data = svFromJsonArgs(request_data)
    
    return json.dumps(sv_data, indent=2)


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)