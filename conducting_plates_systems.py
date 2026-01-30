import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cho_factor, cho_solve

# Physical Constants
epsilon_0 = 8.854e-12 # Permittivity of free space
k_coulomb = 1 / (4 * np.pi * epsilon_0)

N_grid = 80   # Number of cells per side of the square (which contains the disk)

# Defining the class structure of a Plate
@dataclass
class Plate:
    radius: float
    V0: float
    center: tuple = (0.0, 0.0, 0.0) #(X_c, Y_c, Z_c)

    def __post_init__(self):
        self.build_grid()
        self.sigma = None
        self.Q = None
        self.r_matrix = None

    def build_grid(self):
        R = self.radius
        r_c = self.center
        X_c = r_c[0]
        Y_c = r_c[1]
        Z_c = r_c[2]

        # To integrate numerically, dividing the disk into a grid of square cells
        L = 2*R
        dx = L / N_grid
        area = dx**2  # Area of each cell

        # Creating Squares grid
        x = np.linspace(-L/2, L/2, N_grid)
        y = np.linspace(-L/2, L/2, N_grid)
        X, Y = np.meshgrid(x, y)

        # Masking: select only points within the disk
        mask = (X**2 + Y**2) <= R**2
        X_plate = X[mask] + X_c
        Y_plate = Y[mask] + Y_c
        Z_plate = np.full_like(X_plate, Z_c)

        x_vec = X_plate[:, np.newaxis]
        y_vec = Y_plate[:, np.newaxis]
        z_vec = Z_plate[:, np.newaxis]

        N = X_plate.size

        V_vec = np.full(N, self.V0)

        self.dx = dx
        self.area = area
        self.mask = mask
        self.X_plate, self.Y_plate, self.Z_plate = X_plate, Y_plate, Z_plate
        self.x_vec, self.y_vec, self.z_vec = x_vec, y_vec, z_vec
        self.N = N  # Number of squares in the plate
        self.V_vec = V_vec

    def radial_distance(self, X_mesh, Y_mesh, Z_mesh):
        x_vec, y_vec, z_vec = self.x_vec, self.y_vec, self.z_vec
        dx = X_mesh.flatten() - x_vec
        dy = Y_mesh.flatten() - y_vec
        dz = Z_mesh.flatten() - z_vec
        eps = 1e-12  # To avoid division by zero
        
        r = np.sqrt((dx)**2 + (dy)**2 + (dz)**2) + eps

        self.r_matrix = r

        return r

    def computeV(self, X_mesh, Y_mesh, Z_mesh):
        x_vec, y_vec, z_vec = self.x_vec, self.y_vec, self.z_vec
        area = self.area
        if self.sigma is None:
            print("Warning: Charge distribution not calculated yet. Solve the system.")
            return
        else: sigma = self.sigma

        r = self.r_matrix if self.r_matrix is not None else self.radial_distance(X_mesh, Y_mesh, Z_mesh)
        V_matrix = k_coulomb * sigma[:, np.newaxis] * area / r

        V = np.sum(V_matrix, axis=0).reshape(X_mesh.shape)

        self.V = V

        return V
    
    def computeE(self, X_mesh, Y_mesh, Z_mesh):
        x_vec, y_vec, z_vec = self.x_vec, self.y_vec, self.z_vec
        area = self.area
        if self.sigma is None:
            print("Warning: Charge distribution not calculated yet. Solve the system.")
            return
        else: sigma = self.sigma

        r = self.r_matrix if self.r_matrix is not None else self.radial_distance(X_mesh, Y_mesh, Z_mesh)

        dx = X_mesh.flatten() - self.x_vec
        dy = Y_mesh.flatten() - self.y_vec
        dz = Z_mesh.flatten() - self.z_vec

        Ex_matrix = k_coulomb * sigma[:, np.newaxis] * area * (dx) / (r**3)
        Ey_matrix = k_coulomb * sigma[:, np.newaxis] * area * (dy) / (r**3)
        Ez_matrix = k_coulomb * sigma[:, np.newaxis] * area * (dz) / (r**3)
        Ex_total = np.sum(Ex_matrix, axis=0).reshape(X_mesh.shape)
        Ey_total = np.sum(Ey_matrix, axis=0).reshape(X_mesh.shape)
        Ez_total = np.sum(Ez_matrix, axis=0).reshape(X_mesh.shape)

        E = (Ex_total, Ey_total, Ez_total)
        self.E = E

        return E
    
    def plot_Sigma(self):
        if self.sigma is None:
            print("Warning: Charge distribution not calculated yet. Solve the system.")
            return
        sigma_grid = self.sigma_grid
        R = self.radius
        Q0 = self.Q

        plt.figure(figsize=(10, 8))
        plt.rc('text', usetex=True)
        plt.imshow(sigma_grid, origin='lower', extent=[-R, R, -R, R], cmap='inferno')
        plt.colorbar(label=r'Charge density $\sigma$ [C/m$^2$]')
        plt.title(f'Charge Distribution on Conducting Disk at Equilibrium\nQ_calc = {Q0:.2e} C')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()


@dataclass
class System:
    plates: list  # List of Plate objects
    X_space: np.ndarray
    Y_space: np.ndarray
    Z_space: np.ndarray

    def __post_init__(self):
        self.buildSystem()

        self.sigma_all = None

    def buildSystem(self):

        plates = self.plates

        X_all = np.concatenate([p.X_plate for p in plates], axis=0)
        Y_all = np.concatenate([p.Y_plate for p in plates], axis=0)
        Z_all = np.concatenate([p.Z_plate for p in plates], axis=0)
        x_vec = X_all[:, np.newaxis]
        y_vec = Y_all[:, np.newaxis]
        z_vec = Z_all[:, np.newaxis]

        self.X_all, self.Y_all, self.Z_all = X_all, Y_all, Z_all
        self.x_vec, self.y_vec, self.z_vec = x_vec, y_vec, z_vec

        area_all = np.concatenate([np.full(p.N, p.area) for p in plates], axis=0)
        V_all = np.concatenate([p.V_vec for p in plates], axis=0)
        dx_all = np.concatenate([np.full(p.N, p.dx) for p in plates], axis=0)

        self.area_all = area_all
        self.V_all = V_all
        self.dx_all = dx_all

        N = area_all.size
        self.N = N
    

    def Sigma(self):

        plates = self.plates

        X_All, Y_All, Z_All = self.X_all, self.Y_all, self.Z_all
        area = self.area_all
        dx = self.dx_all
        V_all = self.V_all
        N = self.N

        # Calculate distances between squares using the distance matrix
        coords = np.column_stack([X_All, Y_All, Z_All])

        # To improve performance, calculate only the superior triangular matrix
        pdist_vals = pdist(coords, metric='euclidean')
        dist_matrix = squareform(pdist_vals)

        # Constructing M, matrix of known terms
        # For off-diagonal terms of A (i != j) consider the charge as point-like,
        # concentrated at the center of the square
        M = k_coulomb * area / dist_matrix

        # For diagonal terms (i == j) the potential can be calculated as generated 
        # by a charge uniformly distributed at the center of the square
        # If the square has side a, the potential at the center is given by:
        # V(0, 0) = (sigma*/(4*pi*epsilon_0)) * (4*a*ln(1 + sqrt(2)))
        np.fill_diagonal(M, k_coulomb * (4 * np.log(1 + np.sqrt(2))) * dx)

        # To estimate the charge distribution, setting the potential of each square to V0 
        # (uniform because conductor is in equilibrium)
        # Solve for the total charges q on each pixel
        c, low = cho_factor(M)
        sigma_all = cho_solve((c, low), V_all)   

        self.sigma_all = sigma_all
        
        # Total Charge Calculation (global)
        Q_all = np.sum(sigma_all * area)
        self.Q_all = Q_all

        # Assign results to each Plate object
        idx = 0
        for p in plates:

            p.sigma = sigma_all[idx : idx + p.N]
            
            sigma_grid = np.full((N_grid, N_grid), np.nan)
            sigma_grid[p.mask] = p.sigma
            p.sigma_grid = sigma_grid
            
            p.Q = np.nansum(sigma_grid) * (p.dx**2)
            
            idx += p.N


    def solveSystem(self):

        X_space, Y_space, Z_space = self.X_space, self.Y_space, self.Z_space

        self.Sigma()

        for p in self.plates:

            p.computeV(self.X_space, self.Y_space, self.Z_space)
            p.computeE(self.X_space, self.Y_space, self.Z_space)

    def plotVSystem(self):

        if self.sigma_all is None:
            self.solveSystem()
        
        plt.figure(figsize=(10, 8))
        V_tot = np.sum([p.V for p in self.plates], axis=0)
        plt.imshow(V_tot, origin='lower', extent=[self.X_space.min(), self.X_space.max(), self.Z_space.min(), self.Z_space.max()], cmap='viridis', aspect='auto')
        plt.colorbar(label='Potential V [V]')
        plt.title('Electric Potential V(x, z) in the Observation Plane (y=0)')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.show()

    def plotESystem(self):

        if self.sigma_all is None:
            self.solveSystem()
        
        plt.figure(figsize=(12, 8))
        E_tot = np.sum([p.E for p in self.plates], axis=0)
        Ex_tot, Ey_tot, Ez_tot = E_tot
        quiv = plt.streamplot(self.X_space, self.Z_space, Ex_tot, Ez_tot, color='C0', density=1.5, linewidth=1, arrowsize=1.5)
        
        for p in self.plates:
            R = p.radius
            X_c = p.center[0]
            Z_c = p.center[2]
            plt.plot([X_c - R, X_c + R], [Z_c, Z_c], color='black', linewidth=4, label=f'Plate at z={Z_c:.4f} m')

        plt.title('Electrostatic Field Distribution (XZ plane)')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plotSystem(self):

        if self.sigma_all is None:
            self.solveSystem()

        self.plotVSystem()
        self.plotESystem()

    def plotSigmaSystem(self):

        if self.sigma_all is None:
            self.solveSystem()

        for p in self.plates:
            p.plot_Sigma()


# Test on a single conducting plate
R = 2.0 
V0 = 1.0 
plate0 = Plate(radius=R, V0=V0)

x_0 = np.linspace(-10, 10, 100)
z_0 = np.linspace(-10, 10, 100)
Xm_0, Zm_0 = np.meshgrid(x_0, z_0)
Ym_0 = np.zeros_like(Xm_0)

sys0 = System(plates=[plate0], X_space=Xm_0, Y_space=Ym_0, Z_space=Zm_0)
# sys0.plotSystem()


# Test on a two plates capacitor configuration
x_space_cap = np.linspace(-5, 5, N_grid)
z_space_cap = np.linspace(-10, 10, N_grid)
X_plane_cap, Z_plane_cap = np.meshgrid(x_space_cap, z_space_cap)
Y_plane_cap = np.zeros_like(X_plane_cap)

d = 8.0  # Distance between plates
plate1 = Plate(radius=R, V0=V0/2, center=(0.0, 0.0, d/2))
plate2 = Plate(radius=R, V0=-V0/2, center=(0.0, 0.0, -d/2))
sys1 = System(plates=[plate1, plate2], X_space=X_plane_cap, Y_space=Y_plane_cap, Z_space=Z_plane_cap)
# sys1.plotSystem()

# Test on a 5 plates system
num_plates = 3
plates = []
d_multi = 4.0  # Distance between plates
x_pos_multi = [0.0, -2.0, 1.0]
for i in range(num_plates):
    V_plate = V0 * (num_plates - 1 - 2*i) / (num_plates - 1)
    z_pos = d_multi * (i - (num_plates - 1) / 2)
    plate = Plate(radius=R, V0=V_plate, center=(x_pos_multi[i], 0.0, z_pos))
    plates.append(plate)
sys2 = System(plates=plates, X_space=X_plane_cap, Y_space=Y_plane_cap, Z_space=Z_plane_cap)
sys2.plotSystem()
# sys2.plotSigmaSystem()