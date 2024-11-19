from flask import Blueprint, render_template, request, url_for, send_file, jsonify, redirect
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import whisper
import os

routes = Blueprint('routes', __name__)

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = whisper.load_model("base")

def create_text_image(text, video_size, color='white', stroke_color='black', font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size=30):
    """
    Cria uma imagem de texto com bordas.
    """
    max_width = video_size[0]
    max_height = 100  # Altura da área de texto
    img = Image.new('RGBA', (max_width, max_height), (0, 0, 0, 0))  # Fundo transparente
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Divide o texto em linhas
    words = text.split(' ')
    current_line = []
    lines = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        text_bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width <= max_width - 20:  # Margem lateral
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))

    # Centraliza as linhas verticalmente dentro da área de texto
    line_height = font_size + 5  # Espaçamento entre linhas
    total_height = len(lines) * line_height
    y_offset = (max_height - total_height) // 2

    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x_position = (max_width - text_width) // 2
        
        # Adiciona a borda ao texto
        draw.text((x_position, y_offset), line, font=font, fill=stroke_color, stroke_width=2, stroke_fill=stroke_color)
        
        # Adiciona o texto principal
        draw.text((x_position, y_offset), line, font=font, fill=color, stroke_width=0)
        y_offset += line_height

    return np.array(img)

def split_text_into_phrases(text, max_length=50):
    """
    Divide um texto em frases menores com base no comprimento máximo.
    """
    words = text.split(' ')
    phrases = []
    current_phrase = []

    for word in words:
        if len(' '.join(current_phrase + [word])) <= max_length:
            current_phrase.append(word)
        else:
            phrases.append(' '.join(current_phrase))
            current_phrase = [word]

    if current_phrase:
        phrases.append(' '.join(current_phrase))

    return phrases

@routes.route('/')
def index():
    return redirect(url_for('routes.upload_video'))

@routes.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                processed_filename = f"processed_{file.filename}"
                processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

                clip = VideoFileClip(filepath)
                transcription_segments = model.transcribe(filepath)["segments"]

                subtitle_clips = []

                for segment in transcription_segments:
                    text = segment["text"]
                    start_time = segment["start"]
                    end_time = segment["end"]

                    # Divide o texto do segmento em frases menores
                    phrases = split_text_into_phrases(text, max_length=50)

                    # Calcula a duração de cada frase proporcionalmente ao tempo do segmento
                    segment_duration = end_time - start_time
                    phrase_duration = segment_duration / len(phrases)

                    for i, phrase in enumerate(phrases):
                        phrase_start = start_time + i * phrase_duration
                        phrase_end = phrase_start + phrase_duration

                        # Cria imagem para cada frase
                        text_image = create_text_image(phrase, (clip.w, 100))
                        text_clip = ImageClip(text_image).set_start(phrase_start).set_end(phrase_end).set_position(('center', clip.h - 200))
                        subtitle_clips.append(text_clip)

                # Combina o vídeo com as legendas
                video_with_subtitles = CompositeVideoClip([clip, *subtitle_clips])
                video_with_subtitles.write_videofile(processed_path, codec="libx264")

                processed_video_url = url_for('routes.download_video', filename=processed_filename, _external=True)
                return jsonify({"processed_video_url": processed_video_url})

            except Exception as e:
                print(f"Erro no processamento do vídeo: {e}")
                return jsonify({"error": "Erro ao processar o vídeo"}), 500

    return render_template('upload.html')

@routes.route('/download/<filename>')
def download_video(filename):
    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(processed_path, as_attachment=True)
