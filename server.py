from flask import Flask, jsonify, render_template

app = Flask(__name__)

# Route להצגת הדף HTML
@app.route('/')
def index():
    return render_template('ui.html')  # זה מציג את הדף niv.html

# endpoint שיבצע את הקוד של ה-"Hello"
@app.route('/generate', methods=['GET'])
def generate_music():
    print("hello")  # הדפסת "hello" בשרת
    return jsonify({'message': 'hello printed successfully'})

if __name__ == '__main__':
    app.run(debug=True, port=5003)
