import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import grad
import streamlit as st
import torch.optim as optim

class WCGAN_GP:
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.G = Generator(input_dim).to(device)
        self.D = Discriminator(input_dim).to(device)
        self.device = device
        self.lambda_gp = 10

class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)


class WCGAN_GP:
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.G = Generator(input_dim).to(device)
        self.D = Discriminator(input_dim).to(device)
        self.device = device
        self.lambda_gp = 10

    def _gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1).to(self.device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        gradients = grad(outputs=d_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones_like(d_interpolates),
                         create_graph=True, retain_graph=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def train(self, X, epochs=100, batch_size=128):
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt_G = optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.9))
        opt_D = optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.9))

        st.session_state.loss_g = []
        st.session_state.loss_d = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()

        for epoch in range(epochs):
            for real in loader:
                real = real[0].to(self.device)

                # Train Discriminator
                opt_D.zero_grad()
                z = torch.randn(real.size(0), 100).to(self.device)
                fake = self.G(z)

                loss_D = -torch.mean(self.D(real)) + torch.mean(self.D(fake)) + self._gradient_penalty(real, fake)
                loss_D.backward()
                opt_D.step()

                # Train Generator
                opt_G.zero_grad()
                gen_loss = -torch.mean(self.D(self.G(z)))
                gen_loss.backward()
                opt_G.step()

            # Update session state
            st.session_state.loss_g.append(gen_loss.item())
            st.session_state.loss_d.append(loss_D.item())

            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs}\nG Loss: {gen_loss.item():.4f} | D Loss: {loss_D.item():.4f}")

            # Update loss chart
            if epoch % 5 == 0:
                fig, ax = plt.subplots()
                ax.plot(st.session_state.loss_g, label='Generator')
                ax.plot(st.session_state.loss_d, label='Discriminator')
                ax.set_title("Training Loss Progress")
                ax.legend()
                chart_placeholder.pyplot(fig)

            # Terminal logging
            print(f"Epoch {epoch + 1}/{epochs} | G Loss: {gen_loss.item():.4f} | D Loss: {loss_D.item():.4f}")

        return self.G, self.D
