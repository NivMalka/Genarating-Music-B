import os, glob, subprocess
from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # This route serves the main "ui.html" file to the user.
    return render_template('ui.html')

@app.route('/generateMusic', methods=['GET'])
def generate_music():
    try:
        # 1. Run the script that generates music files in "static/audio/mix"
        subprocess.run(["python", "generateMusic.py"], check=True)

        # 2. Find all mp3 files in the "static/audio/mix" directory
        #    that match the pattern "generated_music*.mp3"
        mp3_files = glob.glob('static/audio/mix/generated_music*.mp3')
        if not mp3_files:
            # If no mp3 file is found, return an error response
            return jsonify({'success': False, 'error': 'No MP3 files found'}), 500

        # 3. Choose the last created/modified mp3 file
        latest_mp3 = max(mp3_files, key=os.path.getctime)

        # Extract only the filename (without the full path)
        filename = os.path.basename(latest_mp3)

        # Build a path starting with "/static/..." so the browser can access it
        songUrl = f'/static/audio/mix/{filename}'

        return jsonify({'success': True, 'songUrl': songUrl})

    except Exception as e:
        # If an error occurs, print it and return a JSON with success=False
        print(e)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode on port 5003
    app.run(debug=True, port=5003)
