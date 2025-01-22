"""
Módulo: gan_training.py

Este script entrena una Generative Adversarial Network (GAN) básica usando PyTorch.
Consta de:
- Definición del Generador (Generator).
- Definición del Discriminador (Discriminator).
- Preparación del dataset y del DataLoader.
- Un bucle de entrenamiento que:
  - Actualiza el Discriminador con imágenes reales y falsas.
  - Actualiza el Generador para engañar al Discriminador.
  - Guarda checkpoints e imágenes generadas en cada época.
"""

import os
import time  # Para calcular tiempos restantes
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils

# ==========================================================
#                       CONFIGURACIÓN
# ==========================================================
image_size = 218 * 178  # Tamaño de la imagen aplanada (218x178)
nz = 100                # Dimensión del vector de ruido
batch_size = 64
num_epochs = 50         # Número de épocas totales

# ==========================================================
#                       GENERADOR
# ==========================================================
class Generator(nn.Module):
    """Red Generadora (Generator) para la GAN.

    Esta red toma como entrada un vector de ruido de dimensión `nz`
    y devuelve un mapa de características aplanado del tamaño `image_size`,
    que luego se reconfigura en formato (1, 218, 178).

    Atributos:
        main (nn.Sequential): Bloque principal de capas lineales y activaciones.
    """

    def __init__(self):
        """
        Constructor de la clase Generator.

        Inicializa las capas lineales y funciones de activación (ReLU, Tanh).
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Linear(4096, image_size),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Realiza la propagación hacia adelante del Generador.

        Args:
            input (torch.Tensor): Tensor de entrada (ruido) de tamaño (batch_size, nz).

        Returns:
            torch.Tensor: Tensor de salida con forma (batch_size, 1, 218, 178),
                con valores en el rango [-1, 1].
        """
        return self.main(input).view(-1, 1, 218, 178)

# ==========================================================
#                       DISCRIMINADOR
# ==========================================================
class Discriminator(nn.Module):
    """Red Discriminadora (Discriminator) para la GAN.

    Esta red toma como entrada una imagen (real o generada) y
    produce una probabilidad de que sea real (valor entre 0 y 1).

    Atributos:
        main (nn.Sequential): Bloque principal de capas lineales y funciones
            de activación (LeakyReLU, Sigmoid).
    """

    def __init__(self):
        """
        Constructor de la clase Discriminator.

        Inicializa las capas lineales y funciones de activación (LeakyReLU, Sigmoid).
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Realiza la propagación hacia adelante del Discriminador.

        Args:
            input (torch.Tensor): Tensor de entrada con dimensión (batch_size, 1, 218, 178).

        Returns:
            torch.Tensor: Probabilidad de que la imagen sea real (valor entre 0 y 1).
        """
        input = input.view(-1, image_size)
        return self.main(input)

# ==========================================================
#       INSTANCIAR MODELOS Y PARÁMETROS DE ENTRENAMIENTO
# ==========================================================
# Crear instancias del Generador y Discriminador
netG = Generator()
netD = Discriminator()

# Definir criterio de pérdida y optimizadores
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ==========================================================
#          PREPARAR DATASET Y DATALOADER (CELEBA, ETC.)
# ==========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar dataset desde la carpeta 'dataset' y usar solo subset de 12,000 imágenes
dataset = datasets.ImageFolder(root='./dataset', transform=transform)
subset_indices = range(0, 12000)  # Usar las primeras 12,000 imágenes
subset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

# ==========================================================
#               BUCLE DE ENTRENAMIENTO
# ==========================================================
# Seleccionar dispositivo (CPU o GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)

# Directorio para guardar imágenes generadas
os.makedirs('./generated_images_cronometro', exist_ok=True)

# Intentar cargar checkpoints para continuar entrenamiento
try:
    netG.load_state_dict(torch.load("generator_checkpoint.pth"))
    netD.load_state_dict(torch.load("discriminator_checkpoint.pth"))
    optimizerG.load_state_dict(torch.load("optimizerG_checkpoint.pth"))
    optimizerD.load_state_dict(torch.load("optimizerD_checkpoint.pth"))
    with open("epoch_checkpoint.txt", "r") as f:
        start_epoch = int(f.read()) + 1
    print(f"Checkpoint encontrado. Continuando desde la época {start_epoch}...")
except FileNotFoundError:
    print("No se encontraron checkpoints. Entrenamiento comenzará desde cero.")
    start_epoch = 0

# Variables para medición de tiempo
start_time = time.time()  # Tiempo inicial
total_batches = len(dataloader)

# ==========================================================
#                  EJECUCIÓN DE ÉPOCAS
# ==========================================================
for epoch in range(start_epoch, num_epochs):
    epoch_start_time = time.time()  # Tiempo de inicio de la época

    for i, data in enumerate(dataloader, 0):
        # ----------------------------------------
        # (1) Actualizar Discriminador (netD)
        # ----------------------------------------
        netD.zero_grad()

        # Entrenamiento con imágenes reales
        real_images = data[0].to(device)
        batch_size = real_images.size(0)  
        output_real = netD(real_images)
        labels_real = torch.ones_like(output_real, device=device)
        loss_real = criterion(output_real, labels_real)

        # Entrenamiento con imágenes falsas
        noise = torch.randn(batch_size, nz, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        labels_fake = torch.zeros_like(output_fake, device=device)
        loss_fake = criterion(output_fake, labels_fake)

        # Pérdida total del discriminador y actualización
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # ----------------------------------------
        # (2) Actualizar Generador (netG)
        # ----------------------------------------
        netG.zero_grad()

        # Generar imágenes falsas y calcular la pérdida del generador
        output_fake = netD(fake_images)
        labels_g = torch.ones_like(output_fake, device=device)
        loss_G = criterion(output_fake, labels_g)
        loss_G.backward()
        optimizerG.step()

        # ----------------------------------------
        # (3) Mostrar progreso y estimar tiempo
        # ----------------------------------------
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            batches_done = epoch * total_batches + i + 1
            total_batches_done = num_epochs * total_batches
            estimated_total_time = (elapsed_time / batches_done) * total_batches_done
            time_left = estimated_total_time - elapsed_time

            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{total_batches}] "
                  f"[Loss_D: {loss_D.item():.4f}] [Loss_G: {loss_G.item():.4f}] "
                  f"[Elapsed: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s] "
                  f"[Remaining: {time_left//60:.0f}m {time_left%60:.0f}s]")

    # Guardar imágenes generadas al final de cada época
    with torch.no_grad():
        noise = torch.randn(16, nz, device=device)  # Ruido fijo para comparar entre épocas
        fake_images = netG(noise)
        vutils.save_image(
            fake_images,
            f"./generated_images/epoch_{epoch+1:03d}.png",
            normalize=True
        )

    # Guardar el estado del modelo y los optimizadores
    torch.save(netG.state_dict(), "generator_checkpoint.pth")
    torch.save(netD.state_dict(), "discriminator_checkpoint.pth")
    torch.save(optimizerG.state_dict(), "optimizerG_checkpoint.pth")
    torch.save(optimizerD.state_dict(), "optimizerD_checkpoint.pth")

    # Guardar el número de la época actual
    with open("epoch_checkpoint.txt", "w") as f:
        f.write(str(epoch))

# ==========================================================
#                FIN DEL ENTRENAMIENTO
# ==========================================================
print("Entrenamiento completado. Las imágenes generadas se guardaron en './generated_images'.")
