from diffusers import StableDiffusionPipeline
import pandas as pd
import os
import torch
import random
from PIL import Image
import clip
import matplotlib.pyplot as plt
import numpy as np
import time

# NSFW güvenlik filtresini devre dışı bırakmak için bir dummy checker işlevi oluşturun
def dummy_safety_checker(images, clip_input, **kwargs):
    return images, [False for _ in images]

def get_random_prompts(folder_path):
    """
    Belirtilen klasördeki kategoriye göre ayrılmış CSV dosyalarından, 
    token boyutu 77'den büyük olmayan rastgele promptlar seçer.

    Argümanlar:
        folder_path (str): CSV dosyalarının bulunduğu klasörün yolu.

    Dönüş:
        Her bir prompt için category:prompt eşleşmesinden oluşturulan bir sözlük döner.
        Eğer bir CSV dosyasında uygun prompt yoksa o dosya atlanır.
    """
    result = {}
    max_token_length = 77  # CLIP'in token boyutu sınırı

    # Klasördeki tüm CSV dosyalarını listele
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in csv_files:
        # Dosyanın tam yolunu oluştur
        file_path = os.path.join(folder_path, csv_file)
        
        # CSV dosyasını oku ve uygun promptları seç
        try:
            df = pd.read_csv(file_path)

            # Token boyutu sınırını aşmayan promptları filtrele
            valid_prompts = df["Prompts"].dropna().tolist()
            valid_prompts = [prompt for prompt in valid_prompts if len([prompt]) <= max_token_length]

            # Eğer uygun prompt yoksa bu dosyayı atla
            if not valid_prompts:
                print(f"Uygun prompt bulunamadı, dosya atlanıyor: {csv_file}")
            else:
                # Uygun promptlardan rastgele bir tanesini seç
                random_sample = pd.Series(valid_prompts).sample().to_list()
                print(csv_file, random_sample)
                result[os.path.splitext(csv_file)[0]] = random_sample
        except Exception as e:
            print(f"Dosya okunurken hata oluştu: {csv_file}, Hata: {e}")
    
    return result

def create_and_save_image(model_dir, category, prompt, pipeline):
    """
    Belirtilen kategori ve metin girdisine göre bir görsel oluşturur ve modeli belirtilen klasör yoluna kaydeder.

    Argümanlar:
        model_dir (str): Modellerin ve görsellerin kaydedileceği temel dizin yolu.
        category (str): Görselin ait olduğu kategori ismi.
        prompt (str): Görüntünün oluşturulması için kullanılan metin girdisi.
        pipeline (Pipeline): Modeli çalıştırır

    Dönüş:
        image_path: Üretilen görselin kaydedildiği dosya yolu.
    """
    # Kategoriye özel klasör oluştur
    category_dir = os.path.join(model_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Görüntü oluşturma boyutları
    height = 512
    width = 512
    
    # NSFW filtresini devre dışı bırakın
    pipeline.safety_checker = dummy_safety_checker
    
    # Süre loglama için
    start_time = time.time()
    
    # Görüntüyü üret
    with torch.autocast(device):
        image = pipeline(prompt, height=height, width=width, num_interface_steps=5000).images[0]
        
    # Süre loglama için
    end_time = time.time()
    
    # Görüntüyü kategori altına kaydet
    image_path = os.path.join(category_dir, f"{category}-{width}x{height}.png")
    image.save(image_path)
    print(f"Görsel üretildi ve kaydedildi: {image_path}")
    return image_path, end_time-start_time

def get_clip_score(image_path, prompt, model, preprocess):
    """
    Bir görüntü ile metin girdisi arasındaki CLIP skorunu hesaplar.

    Argümanlar:
        image_path (str): Giriş görüntüsünün dosya yolu.
        prompt (str): Görüntüyle karşılaştırılacak metin girdisi.

    Dönüş:
        Görüntü ile metin girdisi arasındaki CLIP benzerlik skorunu döner.
    """
    # Preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Tokenize the text prompt
    text = clip.tokenize([prompt]).to(device)
    
    # Extract features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = (image_features @ text_features.T).item()
    
    return similarity

def create_clip_scores_table(scores):
    clip_mean_scores = []
    mNames = []
    for mName, ss in scores:
        clip_mean_scores.append(np.mean(ss))
        mNames.append(mName)

    # Bar grafiği oluştur
    plt.figure(figsize=(10, 6))
    plt.bar(mNames, clip_mean_scores, alpha=0.7)
    plt.title("Modellere Göre Ortalama CLIP Skorları", fontsize=16)
    plt.xlabel("Modeller", fontsize=14)
    plt.ylabel("Ortalama CLIP Skoru", fontsize=14)
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Grafik gösterimi
    plt.tight_layout()
    plt.show()
        

device = "mps" if torch.backends.mps.is_available() else "cpu"

prompts = get_random_prompts('./Drawbench/')
images_with_prompts = []

models = {
    "Model-1.4":"CompVis/stable-diffusion-v1-4", # official modeller
    "Model-1.5":"runwayml/stable-diffusion-v1-5",
    "Model-2.1":"stabilityai/stable-diffusion-2-1",
    "Model-2-fine-tuned":"stabilityai/stable-diffusion-2-depth" # foto-gerçekçi fine-tuned model
}

clipModel, clipPreprocess = clip.load("ViT-B/32", device=device)
scores = {}
times = []
for mName, model_ckpt in models.items():
    # Model yüklenir       
    # Model için çıktı klasörü oluştur
    model_dir = os.path.join(os.getcwd(), "generated_images/"+mName)
    scores[mName] = []
    os.makedirs(model_dir, exist_ok=True)
    pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to(device)
    for category, prompt in prompts.items():
        image_path, total_time = create_and_save_image(model_dir,category, prompt[0], pipeline)
        score = get_clip_score(image_path, prompt[0], clipModel, clipPreprocess)
        print(prompt[0]+":"+str(score))
        scores[mName].append(score)
        times.append(total_time)
    print("---------------------------")
    print(scores[mName])
    print("---------------------------")

create_clip_scores_table(scores)