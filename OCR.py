import cv2
import numpy as np
import pytesseract

def process_and_ocr(input_path, output_path='processed.jpg'):
    """
    Lee la imagen, la procesa para reducir ruido y aumentar nitidez,
    y finalmente realiza OCR con Tesseract.
    
    Parámetros:
    -----------
    input_path : str
        Ruta de la imagen de entrada (p. ej. 'boletoD.jpeg').
    output_path : str
        Ruta donde se guardará la imagen procesada (p. ej. 'processed.jpg').
    
    Retorna:
    --------
    text : str
        Texto extraído por Tesseract.
    """

    # 1) Cargar la imagen
    image = cv2.imread(input_path)

    # 2) Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3) Aumentar la resolución (factor de escalado)
    #    Prueba con INTER_LINEAR o INTER_CUBIC
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # 4) Reducir ruido (fastNlMeansDenoising) con h moderado para no borrar detalles
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 5) Aumentar nitidez con Unsharp Mask
    #    - Se crea una versión desenfocada con GaussianBlur
    #    - Se combinan (imagen - desenfocada) para resaltar bordes
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # 6) Binarización
    #    Puedes probar OTSU o adaptativeThreshold según la calidad de tu imagen
    #    Ejemplo con OTSU:
    _, binarized = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 7) (Opcional) Deskew (corrección de inclinación) si la imagen está girada
    # deskewed = deskew(binarized)
    # Si no necesitas deskew, omite este paso y usa 'binarized' directamente.

    # 8) Operaciones morfológicas ligeras para eliminar ruido residual
    #    - Cierre (closing) para rellenar pequeños huecos en los caracteres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

    # 9) Guardar la imagen final para inspeccionar cómo quedó
    cv2.imwrite(output_path, cleaned)

    # 10) Realizar OCR con Tesseract
    #     Prueba distintos modos psm (3, 6, 11) si no reconoce bien.
    config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(cleaned, lang='spa', config=config)

    return text


# --------------------------------------------------------------------------
# (Opcional) Función para deskew (corrección de inclinación)
# --------------------------------------------------------------------------
def deskew(thresh_img):
    """
    Corrige la inclinación de la imagen binaria (thresh_img) usando
    la técnica de minAreaRect. Devuelve la imagen rotada.
    """
    coords = np.column_stack(np.where(thresh_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # Ajustar el ángulo para rotar correctamente
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = thresh_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh_img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


if __name__ == "__main__":
    # Ejecutar el procesamiento y OCR
    ruta_imagen_entrada = "boletoDrec.jpg"
    ruta_imagen_salida = "processed.jpg"
    
    texto_resultado = process_and_ocr(ruta_imagen_entrada, ruta_imagen_salida)
    
    print("Texto extraído:")
    print(texto_resultado)
