# For final assignment of the Engineering Mathematic lecture.
## author: ChenzJ
## Date: 30 Dec.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Lape(space, mesh_size=1):
    ''' caculate the approximate value for Laplace's equation. '''
    x, y = space.shape
    dif_xx = np.array(space, dtype=np.float)
    dif_yy = np.array(space, dtype=np.float)

    h = mesh_size # dx
    dif_xx[h:x-h,:] = h**(-2) * (space[2*h:x,:] - 2*space[h:x-h,:] + space[0:x-2*h,:])
    dif_yy[:,h:y-h] = h**(-2) * (space[:,2*h:y] - 2*space[:,h:y-h] + space[:,0:y-2*h])
    
    return np.add(dif_xx, dif_yy)

def show_heatmap(space, fig, *mode):
    ''' show the heat flow spread in space. '''
    x, y = space.shape

    if 'heatmap' in mode:
        plt.imshow(space, cmap='hot')
        plt.colorbar()
        plt.pause(0.5)
    
    elif 'surface3d' in mode:
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(range(x), range(y))
        surf = ax.plot_surface(X, Y, space, cmap='jet',
                       linewidth=0, antialiased=False)
        fig.colorbar(surf)


def heatflow(size, temperature, n_iter, time_step=0.01, initmode='pt', plotway='heatmap'):
    ''' simulation function for the procedure of 
        heat flow from a body in space. 2-dimension '''
    
    # TODO: initialize a 2-d space.
    u = np.zeros(size, dtype=np.float)

    # TODO: initialize the temperature.
    idx_x, idx_y = np.random.randint(size[0]//4, size[0]//4*3, 2)
    if initmode == 'pt':
        u[idx_x, idx_y] = temperature
        print('init point is ({}, {}), temperature is {}.'
            .format(idx_x, idx_y, u[idx_x, idx_y]))
    
    elif initmode == 'multipt':
        for _ in range(5):
            idx_x, idx_y = np.random.randint(size[0]//5, size[0]//4*4, 2)
            u[idx_x, idx_y] = temperature

    elif initmode == 'region':
        range_r = np.random.randint(2, min(idx_x, idx_y), 1)[0]
        u[idx_x-range_r:idx_x+range_r, idx_y-range_r:idx_y+range_r] = temperature
        print('init point is ({}, {}), dist is {}, temperature is {}.'
                .format(idx_x, idx_y, range_r, u[idx_x, idx_y]))

    # TODO: calculate the velocity v of the heat flow in the body.
    # Then update the heat flow in space u.
    for t in range(n_iter):
        Lap_u = Lape(u)
        u = u + time_step*Lap_u
        #print(u[idx_x, idx_y], Lap_u[idx_x, idx_y]) 
        if t in [0, n_iter//4, n_iter//2, n_iter//4*3, n_iter-1]:
            fig = plt.figure()
            show_heatmap(u, fig, plotway)
            plt.savefig('./res/{}_{}_{}.png'.format(initmode, plotway, t))
            print('Saving...{}%'.format(t/n_iter))

    # plt.show()

if __name__ == '__main__':
    way = ['heatmap', 'surface3d']
    for _, mode in enumerate(['pt', 'multipt', 'region']):
        heatflow((50, 50), 
            temperature=100, 
            n_iter=200, 
            time_step=0.05, 
            initmode=mode, plotway=way)
