from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import os
from urllib.parse import quote as url_quote

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress():
    content = request.json
    data = content.get('data', '')
    algorithm = content.get('algorithm', '')

    if not data or not algorithm:
        return jsonify({'error': 'Data or algorithm is missing'}), 400

    # Алгоритм сонгох
    if algorithm == 'runlength':
        compressed, extra_info = run_length_encoding(data)
        return jsonify({'compressed': compressed, 'extra_info': extra_info})
    elif algorithm == 'huffman':
        compressed, tree, extra_info = huffman_coding(data)
        return jsonify({'compressed': compressed, 'tree': tree, 'extra_info': extra_info})
    elif algorithm == 'lempelziv':
        compressed, extra_info = lz78_encoded(data)
        return jsonify({'compressed': compressed, 'extra_info': extra_info})
    elif algorithm == 'arithmetic':
        compressed, extra_info = arithmetic_coding(data)
        return jsonify({'compressed': compressed, 'extra_info': extra_info})
    elif algorithm == 'jpeg':
        compressed, extra_info = lossless_jpeg(data)
        return jsonify({'compressed': compressed, 'extra_info': extra_info})
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400

# Run-Length Encoding (RLE)
def run_length_encoding(data):
    encoded = ""
    count = 1
    extra_info = ""

    extra_info += f"Анхны өгөгдөл: {data}\n"

    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            extra_info += f"{data[i-1]} тэмдэг {count} удаа давтагдсан\n"
            encoded += data[i - 1] + str(count)
            count = 1

    # Сүүлчийн тэмдэгтийг нэмэх
    extra_info += f"{data[-1]} тэмдэг {count} удаа давтагдсан.\n"
    encoded += data[-1] + str(count)

    extra_info += f"\nНийтэд нь харвал {encoded} болж байна.\n\n"
    extra_info += "Run-Length Encoding нь дараалсан адил тэмдэгтүүдийг тэмдэгт+тоо гэсэн хослолоор орлуулдаг. Энэ нь адил тэмдэгт олон удаа давтагдаж байвал хамгийн үр дүнтэй байдаг."

    return encoded, extra_info

# Huffman Coding
def huffman_coding(data):
    from collections import Counter
    import heapq
    
    class HuffmanNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.freq < other.freq
    
    extra_info = f"Анхны өгөгдөл: {data}\n"
    extra_info += "Тэмдэгтийн давталт\n"
    
    freq_map = Counter(data)
    
    for char, count in freq_map.items():
        extra_info += f"{char} тэмдэгт {count} удаа\n"
    
    heap = []
    for char, freq in freq_map.items():
        heapq.heappush(heap, HuffmanNode(char, freq))
    

    if len(heap) == 1:
        node = heapq.heappop(heap)
        node.left = HuffmanNode(None, 0)  
        heapq.heappush(heap, node)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
    
        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        
        heapq.heappush(heap, internal)
    
    root = heap[0]
    
    codes = {}
    
    def generate_codes(node, code=""):
        if node:
            if node.char is not None: 
                codes[node.char] = code
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")
    
    generate_codes(root)
    
    for char, code in sorted(codes.items()):
        extra_info += f"{char}: {code}\n"
    
    encoded = "".join(codes[char] for char in data)
    extra_info += "\nЭнэ алгоритм нь олон давталттай тэмдэгтэд богино код, цөөн давталттай тэмдэгтэд урт код өгөх зарчмаар ажилладаг."
    
    huffman_tree = []
    for char, code in codes.items():
        huffman_tree.append([char, code])
    
    return encoded, huffman_tree, extra_info

def lz78_encoded(data):
    dictionary = {}
    encoded = []
    current_string = ""
    code = 1
    step = 1  # Step counter for the explanation
    
    extra_info = f"Анхны өгөгдөл: {data}\n\n"
    extra_info += "LZ78 шахалт нь толь бичиг үүсгэх зарчмаар ажилладаг:\n\n"
    extra_info += "Толь | Кодчилол\n"
    extra_info += "----------------\n"

    for char in data:
        combined = current_string + char
        if combined in dictionary:
            current_string = combined
        else:
            if current_string:
                encoded.append(f"{dictionary[current_string]}{char}")
                extra_info += f"{step}. {combined} -> {dictionary[current_string]}{char}\n"
                step += 1
            else:
                encoded.append(f"{char}")
                extra_info += f"{step}. {char} -> {char}\n"
                step += 1
                
            dictionary[combined] = code
            code += 1
            current_string = ""
            
    if current_string:
        encoded.append(str(dictionary[current_string]))
        extra_info += f"{step}. {current_string} -> {dictionary[current_string]}\n"
        step += 1

    final_encoded = "".join(encoded)
    extra_info += f"\nЭцсийн шахалт: {final_encoded}\n"
    extra_info += "\nLZ78 нь өмнө таарсан үгийн хэсгүүдээр  толь үүсгэж хадгалаад, давталтуудыг индексээр орлуулж шахалт хийдэг."

    return final_encoded, extra_info

def arithmetic_coding(data):
    step = 1
    extra_info = f"Анхны өгөгдөл: {data}\n\n"
    extra_info += "Arithmetic coding нь бүх өгөгдлийг нэг тоо болгон шахдаг:\n\n"
    
    extra_info += f"{step}. Тэмдэгт бүрийн давталтыг тоолж магадлал тооцоолох\n"
    step += 1
    
    frequency = {}
    for char in data:
        frequency[char] = frequency.get(char, 0) + 1
    
    total_characters = sum(frequency.values())
    probabilities = {char: freq / total_characters for char, freq in frequency.items()}
    
    extra_info += "Тэмдэгтүүдийн давталт болон магадлал:\n"
    for char in sorted(frequency.keys()):
        extra_info += f"{char}: {frequency[char]} удаа (магадлал = {probabilities[char]:.4f})\n"
    
    extra_info += f"\n{step}. Хуримтлагдсан магадлалыг тооцоолох\n"
    step += 1
    
    cumulative_prob = {}
    current_sum = 0
    for char in sorted(probabilities.keys()):
        cumulative_prob[char] = current_sum
        current_sum += probabilities[char]
    
    extra_info += "Хуримтлагдсан магадлал:\n"
    for char in sorted(cumulative_prob.keys()):
        extra_info += f"{char}: эхлэх магадлал = {cumulative_prob[char]:.4f}, " \
                    f"төгсөх магадлал = {cumulative_prob[char] + probabilities[char]:.4f}\n"
    
    extra_info += f"\n{step}. Дараалсан интервал тооцоолол хийх\n"
    step += 1
    
    low = 0.0
    high = 1.0
    extra_info += "Эхлэх интервал: [0.0000, 1.0000]\n\n"
    
    for i, char in enumerate(data):
        range_size = high - low
        new_high = low + range_size * (cumulative_prob[char] + probabilities[char])
        new_low = low + range_size * cumulative_prob[char]
        
        extra_info += f"Алхам {i+1}: {char} тэмдэгт кодлох\n"
        extra_info += f"  Одоогийн интервал: [{low:.3f}, {high:.3f}] (хэмжээ: {range_size:.3f})\n"
        extra_info += f"  '{char}' тэмдэгтийн интервал: [{cumulative_prob[char]:.4f}, {cumulative_prob[char] + probabilities[char]:.4f}]\n"
        
        extra_info += f"  Шинэ доод хязгаар тооцоолол: {low:.3f} + {range_size:.3f} * {cumulative_prob[char]:.4f} = {new_low:.3f}\n"
        extra_info += f"  Шинэ дээд хязгаар тооцоолол: {low:.3f} + {range_size:.3f} * {(cumulative_prob[char] + probabilities[char]):.4f} = {new_high:.3f}\n"
        extra_info += f"  Шинэ интервал: [{new_low:.3f}, {new_high:.3f}]\n\n"

        low = new_low
        high = new_high

    arithmetic_code = (low + high) / 2
    extra_info += f"{step}. Эцсийн интервалын дундаж тоог авах: [{low:.3f}, {high:.3f}]\n"
    extra_info += f"Тооцоолол: ({low:.3f} + {high:.3f}) / 2 = {arithmetic_code:.10f}\n\n"
    extra_info += f"Эцсийн тоо: {arithmetic_code:.10f}\n\n"
    
    extra_info += "Arithmetic coding нь өгөгдлийг интервалын нарийвчлал ашиглан нэг тоонд шахдаг:\n"
    extra_info += "- Өгөгдөл урт болох тусам интервал улам нарийсдаг\n"
    extra_info += "- Энэ нь өгөгдлийн магадлалаас хамаарч маш үр дүнтэй шахалт хийх боломжтой\n"
    extra_info += "- Олон давталттай тэмдэгт нь шахалтын үр дүнд илүү нөлөөлдөг\n"
    
    return str(arithmetic_code), extra_info

def lossless_jpeg(data):
    extra_info = "Lossless JPEG нь зургийг чанарыг нь алдалгүйгээр багасгах шахалтын арга\n\n"
    extra_info += "1 Зургийг өнгөний тусдаа давхаргуудад хуваадаг (гэрэл, өнгөний бүрдэл)\n"
    extra_info += "2 Хөрш пикселүүд хэр адилхан байгааг харж, ялгааг нь тэмдэглэдэг\n"
    extra_info += "3 Ирээдүйн пикселийг урьдчилан таамаглаж, зөвхөн ялгааг нь хадгалдаг\n"
    extra_info += "4 Хадгалах мэдээллийг илүү багасгах код (жишээ нь, Huffman код) ашигладаг\n\n"

    try:
        if data.startswith('/9j/') or data.startswith('iVBORw0') or data.startswith('R0lGOD'):
            image_data = base64.b64decode(data)
            img = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Original
            extra_info += f"Эх зургийн хэмжээ: {img.size}, Өнгөний горим: {img.mode}\n"

            # Save as JPEG
            jpeg_bytes = io.BytesIO()
            img.save(jpeg_bytes, format="JPEG", quality=100, subsampling=0, optimize=True)
            jpeg_bytes.seek(0)

            compressed_img = Image.open(jpeg_bytes)

            # Size comparison
            original_size = len(image_data)
            compressed_size = len(jpeg_bytes.getvalue())
            ratio = round(original_size / compressed_size, 2)
            extra_info += f"Шахалт: {original_size} байтаас → {compressed_size} байт (~{ratio}:1 харьцаа)\n"

            # Create comparison image instead of showing
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(np.array(img))
            axes[0].set_title("Эх зураг")
            axes[0].axis('off')

            axes[1].imshow(np.array(compressed_img))
            axes[1].set_title("Шахагдсан зураг")
            axes[1].axis('off')

            plt.tight_layout()
            
            # Instead of plt.show(), save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(fig)  # Important: close the figure to free memory
            
            extra_info += f"<img_data>{img_data}</img_data>"
            compressed = "Шахалт амжилттай хийгдлээ."
        else:
            extra_info += "Энэ формат нь зураг биш байна. Base64 зураг оруулна уу.\n"
            compressed = "Шахалт боломжгүй"

    except Exception as e:
        extra_info += f"Алдаа гарлаа: {str(e)}\n"
        compressed = "Шахалт хийгдсэнгүй"

    return compressed, extra_info

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)