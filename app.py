# Updated app.py with name input and leaderboard functionality
from flask import Flask, render_template, Response, request, jsonify
from backend.game import BuzzWireGame

app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')
game = BuzzWireGame()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        frame = game.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    success = game.start_game()
    if success:
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "error", "message": "Enter player name first"})

@app.route('/restart', methods=['POST'])
def restart():
    game.reset_game()
    return jsonify({"status": "restarted"})

@app.route('/set_name', methods=['POST'])
def set_name():
    data = request.get_json()
    name = data.get('name', '').strip()
    if name:
        game.set_player_name(name)
        return jsonify({"status": "success", "name": name})
    else:
        return jsonify({"status": "error", "message": "Name cannot be empty"})

@app.route('/get_leaderboard', methods=['GET'])
def get_leaderboard():
    leaderboard = game.get_leaderboard()
    return jsonify({"leaderboard": leaderboard})

@app.route('/can_start', methods=['GET'])
def can_start():
    return jsonify({"can_start": game.can_start_game()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)