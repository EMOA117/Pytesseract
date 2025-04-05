import cv2
import numpy as np
import pytesseract

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

def correct_orientation(image):
    """
    Detecta la orientación usando Tesseract y rota la imagen para que el texto quede horizontal.
    """
    # Ejecutamos OSD sobre la imagen (mejor si está en color o escala de grises, no binarizada).
    osd = pytesseract.image_to_osd(image)
    rotation_angle = 0

    # Buscamos la línea donde aparece 'Rotate:'
    for line in osd.split('\n'):
        if "Rotate:" in line:
            rotation_angle = int(line.split(':')[1].strip())
            break
    
    # Si Tesseract indica que la imagen está rotada, la corregimos
    if rotation_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # Rota en sentido contrario (negativo) al ángulo detectado
        M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), 
                               flags=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_REPLICATE)
    return image

def apply_roi(image):
    """
    Aplica una región de interés (ROI) a la imagen usando proporciones negativas.
    La ROI se define como la mitad inferior de la imagen, usando todo el ancho.
    
    Parámetros:
    -----------
    image : np.ndarray
        Imagen de entrada (después de corregir la orientación).
    
    Retorna:
    --------
    roi : np.ndarray
        Imagen recortada a la región de interés.
    """
    # Obtener dimensiones de la imagen
    h, w = image.shape[:2]
    
    # Definir proporciones para la ROI
    x_start = 0          # Desde el inicio del ancho
    x_end = w            # Hasta el final del ancho
    y_start = int(h * 0.5)  # Desde la mitad de la altura
    y_end = h            # Hasta el final de la altura
    
    # Recortar la imagen a la ROI
    roi = image[y_start:y_end, x_start:x_end]
    
    return roi

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

    # 1.1) Corregir la orientación
    image = correct_orientation(image)

    # 1.2) Aplicar la región de interés (ROI)
    image = apply_roi(image)

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
    #deskewed = deskew(binarized)
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



if __name__ == "__main__":
    # Ejecutar el procesamiento y OCR
    ruta_imagen_entrada = "imagenRotada.jpeg"
    ruta_imagen_salida = "processedrotateROI.jpg"
    
    texto_resultado = process_and_ocr(ruta_imagen_entrada, ruta_imagen_salida)
    
    print("Texto extraído:")
    print(texto_resultado)
