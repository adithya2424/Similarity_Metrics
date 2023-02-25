import nrrd
import numpy as np
from skimage import measure
from mayavi import mlab
from collections import defaultdict
from queue import Queue
from collections import OrderedDict

class surface:
    def __init__(self):
        self.verts = None
        self.faces = None
        self.normals = None
        self.color = [1., 0., 0.]
        self.mlab_handle = None
        self.opacity = 1

    def createSurfaceFromVolume(self, img, voxsz, isolevel):
        self.verts, self.faces, self.normals, values = measure.marching_cubes(img, isolevel, spacing=voxsz)

    def display(self, axes=False):
        # mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        if self.mlab_handle is None:
            self.mlab_handle = mlab.triangular_mesh(self.verts[:, 0], self.verts[:, 1], self.verts[:, 2], self.faces,
                                                    color=(self.color[0], self.color[1], self.color[2]),
                                                    opacity=self.opacity)

        else:
            self.mlab_handle.mlab_source.set(x=self.verts[:, 0], y=self.verts[:, 1], z=self.verts[:, 2])

        if axes == True:
            mlab.axes(self.mlab_handle)

        # mlab.show()

    def connectedComponents(self):
        old_faces = self.faces
        edges = [set() for i in range(np.shape(self.verts)[0])]
        # the below loop computes the edge list
        for face in old_faces:
            for vert_idx in face:
                adjacent_nodes = [v for v in face if v != vert_idx]
                edges[vert_idx].update(adjacent_nodes)
        components = []
        marked = set()
        List = []
        for startnode, edg in enumerate(edges):
            if startnode not in marked:
                List.append(startnode)
                component = set()
                while List:
                    n = List.pop()
                    if n not in marked:
                        component.add(n)
                        marked.add(n)
                        List.extend(m for m in edges[n] if m not in marked)
                components.append(component)
            # print(startnode)
        max_value = max([max(component) for component in components])
        labels = [0] * (max_value + 1)
        for i, component in enumerate(components):
            for value in component:
                labels[value] = i
        labels = np.array(labels)
        H = [surface() for label in range(np.max(labels)+1)]
        key = np.zeros(np.shape(self.verts)[0])
        for k in range(0, np.size(components)):
            d = np.size(list(components[k]))
            a = np.array(list(components[k]))
            key[a] = range(d)
            H[k].verts = self.verts[a, :]
            msk = labels[self.faces[:, 0]] == k
            H[k].faces = key[self.faces[msk, :]]
            # print(k)
        return H

    def volume(self, numsurf, surfaces):
        final_vol = 0
        total_vol = 0
        all_vol = []
        for i in range(numsurf):
            spatial_value = surfaces[i].verts
            x = spatial_value[:, 0]
            y = spatial_value[:, 1]
            z = spatial_value[:, 2]
            for face in surfaces[i].faces:
                face = np.array(face, dtype="int")
                vol = (-x[face[2]] * y[face[1]] * z[face[0]]) + \
                      (x[face[1]] * y[face[2]] * z[face[0]]) + \
                      (x[face[2]] * y[face[0]] * z[face[1]]) - \
                      (x[face[0]] * y[face[2]] * z[face[1]]) - \
                      (x[face[1]] * y[face[0]] * z[face[2]]) + \
                      (x[face[0]] * y[face[1]] * z[face[2]])
                total_vol = (vol / 6) + total_vol
                final_vol = np.abs(total_vol)
            all_vol.append(final_vol)
        return all_vol




















