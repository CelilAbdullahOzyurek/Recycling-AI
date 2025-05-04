from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

# 📂 Dataset yolu
dataset_path = "waste/"  # flow_from_directory ile kullandığın klasör

# 📌 DataGenerator (etiket sırasını alacağız)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Sadece class_indices için generator oluşturuyoruz
dummy_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=1,
    class_mode='categorical',
    subset='training'
)

# Gerçek sıralama (alfabetik dizin adına göre)
true_class_indices = dummy_generator.class_indices
sorted_classes = sorted(true_class_indices.items(), key=lambda x: x[1])
true_class_names = [k for k, v in sorted_classes]

print("✅ Modelin beklediği class_names sırası:")
for i, name in enumerate(true_class_names):
    print(f"{i}: {name}")

# 🎯 Kullanacağın class_names bu sıra olmalı:
print("\n⚠️ Lütfen ana kodundaki class_names listeni buna göre güncelle.")
