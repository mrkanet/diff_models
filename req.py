# requirements.txt oluşturmak için kullanılan kod
import subprocess

# Mevcut ortamda yüklü paketleri listele ve requirements.txt dosyasına yaz
def create_requirements_file(output_file="requirements.txt"):
    try:
        # 'pip freeze' komutunu çalıştır ve çıktıyı dosyaya yaz
        with open(output_file, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)
        print(f"{output_file} başarıyla oluşturuldu.")
    except Exception as e:
        print(f"Hata oluştu: {e}")

# requirements.txt dosyasını oluştur
create_requirements_file()