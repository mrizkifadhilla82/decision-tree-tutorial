# decision-tree-tutorial
This is my simple decision tree tutorial.

## Cara Kerja Decision Tree
Decision Tree adalah algoritma pembelajaran mesin yang menggunakan struktur pohon untuk membuat keputusan berdasarkan fitur input. Setiap node internal mewakili fitur, cabang mewakili aturan keputusan, dan setiap daun mewakili hasil.

## Komentar dalam Kode
Berikut adalah penjelasan kode dengan komentar:
```python
import warnings
warnings.filterwarnings('ignore')  # Mengabaikan peringatan yang tidak diperlukan

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Memuat dataset iris
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Memisahkan fitur (X) dan target (y)
X = data.drop('target', axis=1)
y = data[['target']]

from sklearn.model_selection import train_test_split
# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # Melatih model dengan data latih

# Memprediksi data uji
y_pred_dt = model.predict(X_test)

# Menghitung akurasi model
print('Akurasi', accuracy_score(y_test, y_pred_dt))
```

## Langkah-langkah Membangun Decision Tree
1. **Memuat Dataset**: Gunakan dataset yang relevan, seperti dataset iris dari `sklearn`.
2. **Memisahkan Fitur dan Target**: Pisahkan data menjadi fitur (X) dan target (y).
3. **Membagi Data**: Bagi dataset menjadi data latih dan data uji menggunakan `train_test_split`.
4. **Membuat Model**: Inisialisasi model Decision Tree menggunakan `DecisionTreeClassifier`.
5. **Melatih Model**: Latih model dengan data latih menggunakan `fit`.
6. **Memprediksi**: Gunakan model untuk memprediksi data uji.
7. **Evaluasi**: Hitung akurasi model menggunakan `accuracy_score`.

Dengan langkah-langkah ini, Anda dapat membangun dan mengevaluasi model Decision Tree sederhana.
