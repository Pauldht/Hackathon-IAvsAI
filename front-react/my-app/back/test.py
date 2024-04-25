from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/get_prediction', methods=['GET'])
def get_data():
    data = {'message': 'Ceci est un message provenant du serveur !'}
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3001')
    return response



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
    print("RUN SERVER")