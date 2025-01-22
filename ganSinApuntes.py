import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
import os
import time  # Para calcular tiempos restantes

# ==========================================================
#                       CONFIGURACIÓN
# ==========================================================
image_size = 218 * 178
nz = 100  # Dimensión del vector de ruido
batch_size = 64
num_epochs = 50  # Número de épocas totales

# ==========================================================
#                       GENERADOR
# ==========================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Linear(4096, image_size),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, 218, 178)

# ==========================================================
#                       DISCRIMINADOR
# ==========================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(-1, image_size)
        return self.main(input)

# Instanciar modelos
netG = Generator()
netD = Discriminator()

# ==========================================================
#                   PARÁMETROS DE ENTRENAMIENTO
# ==========================================================
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ==========================================================
#               PREPARAR DATASET Y DATALOADER
# ==========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root='./dataset', transform=transform)
subset_indices = range(0, 12000)  # Usar las primeras 12,000 imágenes
subset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

# ==========================================================
#               BUCLE DE ENTRENAMIENTO
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)

# Directorio para guardar imágenes generadas
os.makedirs('./generated_images', exist_ok=True)

start_time = time.time()  # Tiempo inicial
total_batches = len(dataloader)
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # ----------------------------------------
        # (1) Actualizar Discriminador (netD)
        # ----------------------------------------
        netD.zero_grad()

        # Entrenamiento con imágenes reales
        real_images = data[0].to(device)
        batch_size = real_images.size(0)  # Tamaño del lote dinámico
        output_real = netD(real_images)
        labels_real = torch.ones_like(output_real, device=device)  # Crear etiquetas dinámicas
        loss_real = criterion(output_real, labels_real)

        # Entrenamiento con imágenes falsas
        noise = torch.randn(batch_size, nz, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach())
        labels_fake = torch.zeros_like(output_fake, device=device)  # Crear etiquetas dinámicas
        loss_fake = criterion(output_fake, labels_fake)

        # Pérdida total del discriminador
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # ----------------------------------------
        # (2) Actualizar Generador (netG)
        # ----------------------------------------
        netG.zero_grad()
        output_fake = netD(fake_images)
        labels_g = torch.ones_like(output_fake, device=device)  # Etiquetas para engañar al discriminador
        loss_G = criterion(output_fake, labels_g)
        loss_G.backward()
        optimizerG.step()

        # ----------------------------------------
        # (3) Mostrar progreso con tamaños de tensores
        # ----------------------------------------
        if i % 100 == 0:  # Cada 100 lotes
            print(f"Batch size: {batch_size}")
            print(f"Output real size: {output_real.size()}, Labels real size: {labels_real.size()}")
            print(f"Output fake size: {output_fake.size()}, Labels fake size: {labels_fake.size()}")
            print(f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # Guardar imágenes generadas
    with torch.no_grad():
        noise = torch.randn(16, nz, device=device)  # Fijo para ver progreso consistente
        fake_images = netG(noise)
        vutils.save_image(fake_images,
                          f"./generated_images/epoch_{epoch+1:03d}.png",
                          normalize=True)

# ==========================================================
#                FIN DEL ENTRENAMIENTO
# ==========================================================
print("Entrenamiento completado. Las imágenes generadas se guardaron en './generated_images'.")