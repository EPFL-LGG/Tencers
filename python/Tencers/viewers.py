import numpy as np
import tencers
import elastic_rods
from tri_mesh_viewer import TriMeshViewer, RawMesh
from linkage_vis import CenterlineViewer
import vis
import mesh_operations
from Tencers.geomtools import cylindrical_trimesh_between
from elastic_rods import PeriodicRod, RodMaterial, ElasticRod
from tencers import *

class HybridViewer(TriMeshViewer):
    def __init__(self, trimesh, width=768, height=512, textureMap=None, scalarField=None, vectorField=None, superView=None, transparent=False, wireframe=False, use_subviews=False, **style_kwargs):
        self.geom_objects = []

        self.child_viewers = {}


        assert trimesh is not None 
        if use_subviews:
            if isinstance(trimesh, (list, tuple)):
                self.registerGeometry(trimesh[0], **style_kwargs)
            
            else:
                self.registerGeometry(trimesh, **style_kwargs)
        else:
            self.registerGeometry(trimesh, **style_kwargs)
            V, F, _ = self.getVisualizationGeometry()
            
        V, F, _ = self.getVisualizationGeometry()

        self.vector_field = np.zeros_like(V)
        self.scalar_field = np.zeros_like(V[:, 0])

        self.vertex_size = np.array([len(v) for v, _, _ in map (lambda o: o.visualizationGeometry(), self.geom_objects)])
        self.vertex_offset = np.cumsum(np.concatenate(([0], self.vertex_size[:-1])))

        assert V.shape[1] == 3
        assert F.shape[1] == 3

        super().__init__((V, F), width, height, textureMap, scalarField, vectorField, superView, transparent, wireframe)

        if use_subviews:
            if isinstance(trimesh, (list, tuple)):
                if len(trimesh) > 1:
                    self.createChildViewers(trimesh[1:])


    def createChildViewers(self, item, allow_nested_iterables=True):
        _create_and_add_viewer = lambda item: self.child_viewers[len(self.child_viewers)](HybridViewer(item, superView=self))
        
        if isinstance(item, elastic_rods.PeriodicRod):
            _create_and_add_viewer(item.rod)
        
        elif "Spring" in item.__class__.__name__:
            _create_and_add_viewer(VisualizationWrapperMaker(visualizeSpring).wrap(item))
            
        elif isinstance(item, Tencer):
            closedRods = item.getClosedRods()
            for rod in item.getOpenRods():
                material = rod.material(0)
                newrod = ElasticRod(rod.deformedPoints())
                newrod.setMaterial(material)
                _create_and_add_viewer(newrod)
            for rod in closedRods:
                material = rod.rod.material(0)
                newrod = ElasticRod(rod.rod.deformedPoints())
                newrod.setMaterial(material)
                _create_and_add_viewer(newrod)
            for spring in item.getSprings():
                _create_and_add_viewer(VisualizationWrapperMaker(visualizeSpring).wrap(spring))
        
        elif hasattr(item, "visualizationGeometry"):
            _create_and_add_viewer(item)
        
        elif isinstance(item, (list, tuple)):
            if not allow_nested_iterables:
                raise ValueError(f"Unsupported {item = }, {allow_nested_iterables = }")

            for it in item:
                _create_and_add_viewer(it)
        
        else:
            raise ValueError(f"Unsupported type {type(item)}, {item = }")

    
    def registerGeometry(self, item, allow_nested_iterables=True, **kwargs):
        
        if isinstance(item, elastic_rods.PeriodicRod):
            self.geom_objects.append(item.rod)
        
        elif "Spring" in item.__class__.__name__:
            cylinder_radius = kwargs.get("spring_cylinder_radius", 0.01)    
            self.geom_objects.append(VisualizationWrapperMaker(visualizeSpring, cylinder_radius=cylinder_radius).wrap(item))
            
        elif isinstance(item, Tencer):
            closedRods = item.getClosedRods()
            cylinder_radius = kwargs.get("spring_cylinder_radius", 0.01)  
            for rod in item.getOpenRods():
                material = rod.material(0)
                newrod = ElasticRod(rod.deformedPoints())
                newrod.setMaterial(material)
                self.geom_objects.append(newrod)
            for rod in closedRods:
                material = rod.rod.material(0)
                newrod = ElasticRod(rod.rod.deformedPoints())
                newrod.setMaterial(material)
                self.geom_objects.append(newrod)
            for spring in item.getSprings():
                self.geom_objects.append(VisualizationWrapperMaker(visualizeSpring, cylinder_radius=cylinder_radius).wrap(spring))
        
        elif hasattr(item, "visualizationGeometry"):
            self.geom_objects.append(item)
        
        elif isinstance(item, (list, tuple)):
            if not allow_nested_iterables:
                raise ValueError(f"Unsupported {item = }, {allow_nested_iterables = }")

            for it in item:
                self.registerGeometry(it, allow_nested_iterables=False, **kwargs)
        
        else:
            raise ValueError(f"Unsupported {item = }")
    

    def getVisualizationGeometry(self, index=None):
        """
        return the visualization geometry of the object at index, if index is None (default), return all geometry concatenated
        """
        assert isinstance(self.geom_objects, list)
        assert all([hasattr(obj, "visualizationGeometry") for obj in self.geom_objects])

        if index == None:
            return combineVisualizationGeometry(self.geom_objects)
        else:
            return self.geom_objects[index].visualizationGeometry()
        

    def setScalarField(self, index, data):
        offset = self.vertex_offset[index]
        size = self.vertex_size[index]

        assert data.shape == (size, ) or data.size == 1

        self.scalar_field[offset: offset + size] = data


    def setVectorField(self, index, data):
        offset = self.vertex_offset[index]
        size = self.vertex_size[index]

        assert data.shape == (size, 3) or data.shape == (3, )

        self.vector_field[offset: offset + size] = data
    

    def update(self, preserveExisting=False, mesh=None, updateModelMatrix=False, textureMap=None, scalarField=None, vectorField=None, transparent=False, displacementField=None):
        if scalarField is None and not np.allclose(self.scalar_field, 0.0):
            scalarField = self.scalar_field

        if vectorField is None and not np.allclose(self.vector_field, 0.0):
            vectorField = self.vector_field

        return super().update(preserveExisting, mesh, updateModelMatrix, textureMap, scalarField, vectorField, transparent, displacementField)
    
    def update_mesh(self,trimesh):
        self.geom_objects = []
        self.registerGeometry(trimesh)
        V, F, _ = self.getVisualizationGeometry()
        self.vector_field = np.zeros_like(V)
        self.scalar_field = np.zeros_like(V[:, 0])

        self.vertex_size = np.array([len(v) for v, _, _ in map (lambda o: o.visualizationGeometry(), self.geom_objects)])
        self.vertex_offset = np.cumsum(np.concatenate(([0], self.vertex_size[:-1])))

        assert V.shape[1] == 3
        assert F.shape[1] == 3

        # accept (V, F) tuples as meshes, wrapping in a RawMesh
        mesh = RawMesh(*(V,F))

        return super().update(preserveExisting=False, mesh=mesh, updateModelMatrix=False, textureMap=None, scalarField=self.scalar_field, vectorField=self.vector_field, transparent=False, displacementField=None)

    def show(self):
        self.update(scalarField=self.scalar_field, vectorField=self.vector_field)
        return super().show()


    def create_centerline_viewer(self, viewer_name, rod, vector_field=None, **kwargs):
        centerline_viewer = CenterlineViewer(rod, superView=self)
        self.child_viewers[viewer_name] = centerline_viewer
        
        if vector_field is not None:
            self.update_centerline_viewer(viewer_name, vector_field, **kwargs)

        return centerline_viewer


    def update_centerline_viewer(self, viewer_name, vector_field, vmin=None, vmax=None, **kwargs):
        import vis, matplotlib
        assert(vector_field.shape[1] == 3)
        if vmin is None: vmin = np.min(np.linalg.norm(vector_field, axis=1))
        if vmax is None: vmax = np.max(np.linalg.norm(vector_field, axis=1))
        if (vmax - vmin) < vmax / 1000:
            vmin *= 0.9999
            vmax *= 1.00005
        
        kwargs = {
            **kwargs,
            "colormap": matplotlib.cm.autumn_r,
            "vmin": vmin,
            "vmax": vmax
        }

        viewer = self.child_viewers[viewer_name]

        frame = vis.fields.VectorField(viewer.mesh, vector_field, **kwargs)
        # frame = vis.fields.VectorField(viewer.mesh, vector_field, colormap=matplotlib.cm.Greys, glyph=vis.fields.VectorGlyph.CYLINDER, vmin=vmin, vmax=vmax)
        viewer.update(vectorField=frame)


def combineVisualizationGeometry(objects):
    from functools import reduce
    return reduce(
        lambda a, b: (np.vstack((a[0], b[0])), np.vstack((a[1], b[1] + len(a[0]))), np.vstack((a[2], b[2]))),
        map(lambda o: o.visualizationGeometry(), objects))


class MeshWithNormals:
    def __init__(self, V, F, N):
        self.V = V
        self.F = F
        self.N = N
    
    def visualizationGeometry(self):
        return self.V, self.F.astype(np.uint64), self.N


class VisualizationWrapperMaker():
    def __init__(self, func, *func_args, **func_kwargs):
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def __call__(self, object):
        return self.wrap(object)
    
    def wrap(self, object):
        class _VisWrapper:
            def visualizationGeometry(vis_self):
                return self.func(object, *self.func_args, **self.func_kwargs)

        return _VisWrapper()

def visualizeSpring(spring, cylinder_radius=0.01):
    v = spring.get_coords()
    c_V, c_F = cylindrical_trimesh_between(v[:3], v[3:], cylinder_radius, cylinder_radius)
    c_N = mesh_operations.getVertexNormalsRaw(c_V, c_F)
    return c_V, c_F, c_N
    

def visualizeRestLengthSpring(spring, cylinder_radius=0.01):
    v1, v2 = spring.getTargetPositions()
    c_V, c_F = cylindrical_trimesh_between(v1, v2, cylinder_radius, cylinder_radius)
    c_N = mesh_operations.getVertexNormalsRaw(c_V, c_F)
    return c_V, c_F, c_N
    

class LoadVisualizationWrapper():
    """
    For comptability with the HybridViewer class
    """
    def __init__(self, eo, spring_data):
        self.eo = eo
        self.spring_data = spring_data

    def visualizationGeometry(self):
        meshes = []
        for data in self.spring_data:
            if isinstance(self.eo, elastic_rod_sheet.ElasticRodSheetSystem):
                r1, v1, r2, v2 = data
                vertex1 = self.eo.rodAssembly().rod(r1).rod.deformedPoints()[v1]
                vertex2 = self.eo.rodAssembly().rod(r2).rod.deformedPoints()[v2]

            elif isinstance(self.eo, elastic_rods.PeriodicRod):
                v1, v2 = data
                vertex1 = self.eo.rod.deformedPoints()[v1]
                vertex2 = self.eo.rod.deformedPoints()[v2]

            c_V, c_F = cylindrical_trimesh_between(vertex1, vertex2, 0.01, 0.01)
            c_N = mesh_operations.getVertexNormalsRaw(c_V, c_F)
            meshes.append(MeshWithNormals(c_V, c_F, c_N))
        
        combined = combineVisualizationGeometry(meshes)

        return combined


def render_to(viewer, filename, pos=None, target=None, up=None):
    # pos    = np.array([0, 1.0, 8.0]) if pos is None else pos
    # target = np.array([0.0, 1.0, 0.0]) if target is None else target
    # up     = np.array([0, 0.0, -2.5]) if up is None else up
    renderScale = 3
    outputScale = 1
    renderer = viewer.offscreenRenderer(scale=renderScale)
    for m in renderer.meshes:
        m.alpha = 0.6
        m.lineWidth = 0.0
        m.shininess = 100.0
    
    cam_pos, cam_target, cam_up = viewer.getCameraParams()
    pos    = cam_pos if pos is None else pos
    target = cam_target if target is None else target
    up     = cam_up if up is None else up
    print(f"using {pos = }, {target = }, {up = }")
    renderer.setCameraParams(np.vstack([pos, target, up]))
    renderer.render()

    renderer.scaledImage(outputScale / renderScale).save(filename)
    print(f"saved to '{filename}'")


def make_viewer_update_callback(viewer, every_n_iters=1):
    assert every_n_iters > 0
    def _update(prob, it):
        if it % every_n_iters == 0:
            viewer.update()

    return _update