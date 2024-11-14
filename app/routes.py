from flask import Blueprint, render_template, request, url_for, send_file, jsonify, redirect
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import whisper
import os

# Inicialização do Blueprint
routes = Blueprint('routes', __name__)

# Pastas para upload e processamento de vídeos
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

# Criação das pastas, se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Carregar o modelo Whisper para reconhecimento de fala
model = whisper.load_model("base")

# Função para criar imagem de texto usando Pillow e converter para NumPy
def create_text_image(text, size, color='white'):
    img = Image.new('RGB', size, color=(0, 0, 0))  # Fundo preto
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Fonte padrão do Pillow
    
    # Centraliza o texto usando textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Desenha o texto centralizado
    draw.text(text_position, text, fill=color, font=font)
    return np.array(img)  # Converte a imagem Pillow para um array NumPy

# Rota para upload e processamento do vídeo
@routes.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                # Processamento do vídeo
                processed_filename = f"processed_{file.filename}"
                processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

                clip = VideoFileClip(filepath)
                transcription = model.transcribe(filepath)["text"]

                # Criação da imagem de texto com Pillow e conversão para NumPy
                text_image = create_text_image(transcription, (clip.w, 100))
                text_clip = ImageClip(text_image).set_duration(clip.duration).set_position('bottom')

                # Combina o vídeo com a imagem de texto
                video_with_subtitles = CompositeVideoClip([clip, text_clip])

                # Salva o vídeo processado
                video_with_subtitles.write_videofile(processed_path, codec="libx264")

                processed_video_url = url_for('routes.download_video', filename=processed_filename, _external=True)
                return jsonify({"processed_video_url": processed_video_url})

            except Exception as e:
                print(f"Erro no processamento do vídeo: {e}")
                return jsonify({"error": "Erro ao processar o vídeo"}), 500

    return render_template('upload.html')

# Rota para download do vídeo processado
@routes.route('/download/<filename>')
def download_video(filename):
    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(processed_path, as_attachment=True)
