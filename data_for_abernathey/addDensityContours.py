import numpy as np
import matplotlib.pyplot as plt
import seawater as sw

# function to add density lines on T-S plot
def addDensityContours(ax, temp_range=[-2,5], sal_range=[33.5, 35.5], clvl_range=[25, 28.5], dT=0.02, dS=0.02, drho=0.05):
    
    """
    Adds density lines to T-S plots
    """
    
    ymin = np.min(temp_range)
    ymax = np.max(temp_range)
    
    xmin = np.min(sal_range)
    xmax = np.max(sal_range)
    
    pdens_min = np.min(clvl_range)
    pdens_max = np.max(clvl_range)
    
    
    # add density lines
    temps = np.arange(ymin, ymax, dT)
    sals = np.arange(xmin, xmax, dS)
    sal_grid, temp_grid = np.meshgrid(sals, temps)
    pdens_grid= sw.pden(sal_grid, temp_grid, 0)-1000
    plvls = np.arange(pdens_min, pdens_max, drho)
    cs = ax.contour(sal_grid, temp_grid, pdens_grid, plvls, colors='0.5',linewidth=0.5, alpha=0.6)
    plt.clabel(cs, inline=1, fontsize=10,fmt='%1.2f',inline_spacing=10)
    