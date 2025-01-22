import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
import os

# ==========================================================
#                       CONFIGURACIÓN
# ==========================================================
# image_size: Tamaño total de las imágenes (218x178 píxeles).
# nz: Dimensión del vector de ruido, que será la entrada del generador.
image_size = 218 * 178  # Producto de las dimensiones de la imagen
nz = 100  # Tamaño del vector de ruido



# ==========================================================
#                       GENERADOR
# ==========================================================
# El generador transforma un vector de ruido aleatorio (nz) en una imagen.
# Utiliza capas completamente conectadas (nn.Linear) y funciones de activación no lineales
# para aprender cómo generar imágenes realistas.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 1024),         # Primera capa: de ruido (nz) a vector de tamaño 1024
            nn.ReLU(True),              # ReLU introduce no linealidad para aprender patrones complejos
            nn.Linear(1024, 4096),      # Segunda capa: vector intermedio más grande
            nn.ReLU(True),
            nn.Linear(4096, image_size),# Última capa: genera el vector plano de la imagen
            nn.Tanh()                   # Escala los valores de salida entre -1 y 1
        )

    def forward(self, input):
        # Pasa el vector de ruido a través de las capas y da una imagen de tamaño 218x178.
        return self.main(input).view(-1, 1, 218, 178)  # Formato de salida: (batch, canal, alto, ancho)




# ==========================================================
#                       DISCRIMINADOR
# ==========================================================
# El discriminador clasifica las imágenes como reales (1) o falsas (0).
# Su entrada son imágenes planas, que se pasan por capas completamente conectadas
# con activaciones no lineales.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, 4096),  # Entrada: imagen plana a vector intermedio
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU permite pequeñas pendientes para valores negativos
            nn.Linear(4096, 1024),       # Capa intermedia más pequeña
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),          # Salida: probabilidad de que sea real (1 valor)
            nn.Sigmoid()                 # Convierte la salida a rango [0, 1]
        )

    def forward(self, input):
        # Pasa la imagen por las capas para obtener la probabilidad de real/falso.
        input = input.view(-1, image_size)  # Asegura que la entrada esté en formato plano
        return self.main(input)



# ==========================================================
#             INSTANCIACIÓN DE GENERADOR Y DISCRIMINADOR
# ==========================================================
# Creamos las instancias del generador (netG) y el discriminador (netD).
netG = Generator()
netD = Discriminator()



# ==========================================================
#                   PARÁMETROS DE ENTRENAMIENTO
# ==========================================================
# Función de pérdida y optimizadores para ambos modelos.
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss, adecuada para clasificación binaria.

# Optimización con Adam para ambos modelos:
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))



# ==========================================================
#             EXPLICACIÓN GENERAL DEL ENTRENAMIENTO
# ==========================================================
# Durante el entrenamiento:
# 1. Se pasa un lote de imágenes reales al discriminador para calcular la pérdida con etiquetas reales (1).
# 2. Se genera un lote de imágenes falsas con el generador, y se pasa al discriminador para calcular la pérdida con etiquetas falsas (0).
# 3. Se actualizan los pesos del discriminador para mejorar su capacidad de clasificación.
# 4. Se genera otro lote de imágenes falsas y se calcula la pérdida del generador, que busca engañar al discriminador.
# 5. Se actualizan los pesos del generador para crear imágenes más realistas.

# Transformaciones:
# - Convierte las imágenes a tensores (torch.Tensor).
# - Normaliza los valores de los píxeles entre -1 y 1 (necesario por la activación Tanh del generador).
transform = transforms.Compose([
    transforms.ToTensor(),                     # Convertir a tensor
    transforms.Normalize((0.5,), (0.5,))       # Normalización: media 0.5, desviación estándar 0.5
])

# Cargar el dataset desde la carpeta `/dataset`:
# El dataset debe estar descomprimido y organizado como una carpeta de imágenes.
dataset = datasets.ImageFolder(root='./dataset', transform=transform)

subset_indices = range(0, 12000)
subset = Subset(dataset, subset_indices)


# Crear el DataLoader:
# - shuffle=True baraja las imágenes para asegurar variedad en los lotes.
# - batch_size define cuántas imágenes se procesan por iteración.
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================================
#               FUNCIÓN PARA GUARDAR IMÁGENES GENERADAS
# ==========================================================
# Directorio donde se guardarán las imágenes generadas:
os.makedirs('./generated_images', exist_ok=True)

# Función para guardar un lote de imágenes generadas.
def save_generated_images(epoch, fake_images):
    # vutils.save_image guarda las imágenes en formato cuadrícula.
    # Normalizamos las imágenes generadas al rango [0, 1] para que sean visibles.
    vutils.save_image(fake_images,
                      f"./generated_images/epoch_{epoch:03d}.png",
                      normalize=True)

             
                      
# ==========================================================
#                   BUCLE DE ENTRENAMIENTO
# ==========================================================
# El bucle recorre el dataset, entrena el discriminador y el generador, 
# y guarda las imágenes generadas después de cada época.
num_epochs = 50  # Número total de épocas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usar GPU si está disponible

# Mover modelos al dispositivo (CPU/GPU)
netG.to(device)
netD.to(device)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # ----------------------------------------
        # (1) Actualizar Discriminador (netD)
        # ----------------------------------------
        # Entrenamos el discriminador en imágenes reales y falsas.
        netD.zero_grad()

        # Entrenamiento con imágenes reales
        real_images = data[0].to(device)        # Cargamos imágenes reales al dispositivo
        batch_size = real_images.size(0)       # Tamaño del lote actual
        labels_real = torch.ones(batch_size, 1, device=device)  # Etiquetas reales (1)
        output_real = netD(real_images)        # Clasificar imágenes reales
        loss_real = criterion(output_real, labels_real)  # Calcular pérdida con etiquetas reales

        # Entrenamiento con imágenes generadas (falsas)
        noise = torch.randn(batch_size, nz, device=device)  # Generar ruido aleatorio
        fake_images = netG(noise)                          # Crear imágenes falsas con el generador
        labels_fake = torch.zeros(batch_size, 1, device=device)  # Etiquetas falsas (0)
        output_fake = netD(fake_images.detach())           # Clasificar imágenes falsas
        loss_fake = criterion(output_fake, labels_fake)    # Calcular pérdida con etiquetas falsas

        # Pérdida total del discriminador (reales + falsas)
        loss_D = loss_real + loss_fake
        loss_D.backward()                                  # Calcular gradientes
        optimizerD.step()                                  # Actualizar pesos del discriminador

        # ----------------------------------------
        # (2) Actualizar Generador (netG)
        # ----------------------------------------
        netG.zero_grad()

        # Etiquetas invertidas (el generador quiere engañar al discriminador)
        labels_g = torch.ones(batch_size, 1, device=device)  # Etiquetas "reales" para imágenes falsas
        output_fake = netD(fake_images)                     # Recalcular con imágenes falsas actuales
        loss_G = criterion(output_fake, labels_g)           # Pérdida del generador
        loss_G.backward()                                   # Calcular gradientes
        optimizerG.step()                                   # Actualizar pesos del generador

        # ----------------------------------------
        # (3) Mostrar progreso
        # ----------------------------------------
        if i % 100 == 0:  # Mostrar información cada 100 lotes
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[Loss_D: {loss_D.item():.4f}] [Loss_G: {loss_G.item():.4f}]")

    # Guardar imágenes generadas después de cada época
    save_generated_images(epoch, fake_images)

# ==========================================================
#                FIN DEL ENTRENAMIENTO
# ==========================================================
print("Entrenamiento completado. Las imágenes generadas se guardaron en './generated_images'.")