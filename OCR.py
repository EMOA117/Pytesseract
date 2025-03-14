import pytesseract as pyt
import cv2
from PIL import Image
import numpy as np

# Cargar la imagen
imagen = cv2.imread('boletoD.jpeg')

# Convertir a escala de grises (mejora la detección de texto)
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Aplicar un filtro de umbralización (opcional, mejora OCR)
_, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detección de bordes usando Canny
bordes = cv2.Canny(umbral, 50, 150, apertureSize=3)


# Guardar la imagen procesada (opcional, para ver cómo queda)
cv2.imwrite('procesado.jpg', bordes)

# Extraer texto usando Tesseract
texto = pyt.image_to_string(umbral, lang='spa', config='--oem 3 --psm 6')  # 'eng' = inglés, usa 'spa' para español

# Mostrar el texto extraído
print("Texto extraído:")
print(texto)

# Mostrar la imagen procesada (opcional)
# cv2.imshow('Imagen procesada', umbral)
# cv2.waitKey(0)
# cv2.destroyAllWindows()