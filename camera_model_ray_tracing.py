# This code implements the pinhole camera model and raytracing functions to render a sphere in 3D space.
# Part of this code is based on the UDEMY course on this topic.
# I'm just putting this as entry point scripts and patching the code a little bit to make it easier for me to understand.

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Camera parameters
    H = 400
    W = 400
    f = 1200 # Focal length
    
    # Rays parameters from the origin and direction
    # rays_o: origin of the rays
    # rays_d: direction of the rays
    # Standard here is to stored the rays in a 2D array with shape (H*W, 3)
    rays_o = np.zeros((H*W, 3))
    rays_d = np.zeros((H*W, 3))

    # Pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Seting the directions of the rays, in x, y and z
    dirs = np.stack((u - W / 2,
                 -(v - H / 2),
                 - np.ones_like(u) * f), axis=-1)
    rays_d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    rays_d = rays_d.reshape(-1, 3)

    print("Plotting rays directions in a 3D space")
    plot_rays(rays_o, rays_d, 1)
    
    # Now we will render a sphere in 3D space
    print("Rendering a sphere in 3D space")
    
    sphere = Sphere(np.array([0, 0, -1]), 0.1, np.array([1, 0, 0]))
    colors = sphere.intersect(rays_o, rays_d)
    
    img_sphere = colors.reshape(H, W, 3)
    
    # Plotting the sphere
    fig = plt.figure()
    plt.title("Rendering a sphere in 3D space")
    
    plt.imshow(img_sphere)
    
    plt.show()
    

def plot_rays(o, d, t):
    
    plt.figure(figsize=(12, 12))
    plt.title("Rays directions in a 3D space")
    _ = plt.axes(projection='3d')
    
    pt1 = o
    pt2 = o + t * d
    
    for p1, p2 in zip(pt1[::100], pt2[::100]):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    
    
    plt.show()
    
class Sphere():
    """
    Class the represents the rendering of a sphere in 3D space.
    
    The equation of a sphere is given by
    
    $'(x - p_x)^2 + (y - p_y)^2 + (z - p_z)^2 = r^2'$
    
    where p is the center of the sphere, and $'r'$ is the radius, and x,y and z
    are the points in the surface of the sphere, or where the ray r(t) intesects
    the surface of the sphere.
    
    The ray intersection $r(t)$ on the sphere is given by the set of equations below:
        - $'x = o_x + t * d_x'$, where o_x is the origin of the ray in the x-axis, and d_x is the direction of the ray in the x-axis
        - $'y = o_y + t * d_y'$, where o_y is the origin of the ray in the y-axis, and d_y is the direction of the ray in the y-axis
        - $'z = o_z + t * d_z'$, where o_z is the origin of the ray in the z-axis, and d_z is the direction of the ray in the z-axis
    $'t'$ is the parameter that defines the time.
    
    So, we can substitute the equations above in the equation of the sphere to find the intersection points.
    by solving the equation below:
    
    $'(o_x + t * d_x - p_x)^2 + (o_y + t * d_y - p_y)^2 + (o_z + t * d_z - p_z)^2 = r^2'$
    
    Since we know the origin of the ray and the direction of the ray, 
    we have a quadratic equation in the form $'at^2 + bt + c = 0'$,
    so we can solve this equation to find the intersection points for t.
    
    The method intersect will return the color of the sphere at the intersection points by
    solving the equation above.
    
    .. note::
      Of course this is a simple model, and in real world we have to consider the refraction of the light
      and other factors that can affect the color of the sphere.
      Also, for more complex shapes and objects, usually people build them as a mesh of triangles, and
      the intersection of the rays are computed using this mesh.
      
     Another way of rendering objects in the scene is via Volumetric Rendering,
     
    """
    
    def __init__(self, p, r, c):
        self.p = p
        self.r = r
        self.c = c
        
    def intersect(self, o, d):
        
        # Solve equation at^2 + bt + c = 0
        # (ox + t * dx  - xc)^2 + (oy + t * dy-yc)^2 + (oz + t * dz-zc)^2 = r^2 
        a = d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2
        b = 2 * ((d[:, 0] * (o[:, 0] - self.p[0])) + (d[:, 1] * (o[:, 1] - self.p[1])) + (d[:, 2] * (o[:, 2] - self.p[2])))
        c = (o[:, 0] - self.p[0])**2 + (o[:, 1] - self.p[1])**2 + (o[:, 2] - self.p[2])**2 - self.r**2
        
        pho = b**2 - 4 * a * c
        cond = pho >= 0
        
        num_rays = o.shape[0]
        colors = np.zeros((num_rays, 3))
        colors[cond] = self.c
        
        return colors    


if __name__ == "__main__":
    main()