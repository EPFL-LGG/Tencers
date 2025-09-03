import ElasticRods
import elastic_rods

import matplotlib.pyplot as plt

from py_newton_optimizer import NewtonOptimizerOptions
import py_newton_optimizer

from tencers import *
from elastic_rods import ElasticRod, PeriodicRod, RodMaterial

from Tencers.viewers import HybridViewer, make_viewer_update_callback, CenterlineViewer
from Tencers.springs import *
from Tencers.state_saver import *
from Tencers.rods_IO import *
from registration import register_points


def remove_compressed_springs(tencer:Tencer, fixed_vars, opt):
    """ During initialization, repeatedly compute equilibrium and remove compressed springs until all springs are tensionned.
    Args:
        tencer (Tencer): the tencer to sparsify
        fixed_vars (list[int]) : fixed variables for equilibrium computation
        opt (NewtonOptimizerOption) : Equilibrium optimization options
    Returns:
        tencer (Tencer): the tencer once all compressed springs have been removed
    """

    all_tensionned = False
    old_num_springs = len(tencer.getSprings())
    
    print("Initial number of springs : ", old_num_springs)

    while not all_tensionned:

        # Set gradient tolerance for equilibrium solve
        opt.gradTol = compute_grad_tol(tencer)
        if opt.verbose:
            print("Gradient tolerance : ", opt.gradTol)
        
        # Compute equilibrium
        computeEquilibrium(tencer,fixedVars=fixed_vars, opts=opt,hessianShift=1e-8)

        # Get tensionned and compressed springs
        s = tencer.getSprings()
        tensionned_springs = []
        compressed_springs = []
        for i,spring in enumerate(s):
            spring_length = np.linalg.norm(spring.get_coords()[:3] - spring.get_coords()[3:])
            spring_rest_length = spring.get_rest_length()
            if spring_length > spring_rest_length:
                tensionned_springs.append(i)
        tensionned_springs = np.array(tensionned_springs)

        new_num_springs = len(tensionned_springs)
        print("New number of springs : ", new_num_springs)

        if new_num_springs == old_num_springs:
            all_tensionned = True 
            print("Done.")
        else : 
            # Remove compressed springs
            tencer = Tencer(tencer.getOpenRods(),tencer.getClosedRods(),np.array(tencer.getSprings())[tensionned_springs],np.array(tencer.getAttachmentVertices())[tensionned_springs],tencer.getTargetRods())
            old_num_springs = new_num_springs 

    return tencer

def get_compressed_springs(tencer:Tencer):
    """ Get compressed springs in a tencer.
    Args:
        tencer (Tencer) : a tencer
    Returns : 
        (list[float]) : the list of compressed springs
    """
    s = np.array(tencer.getSprings())
    tensionned_springs = []
    compressed_springs = []
    for i,spring in enumerate(s):
        spring_length = np.linalg.norm(spring.get_coords()[:3] - spring.get_coords()[3:])
        spring_rest_length = spring.get_rest_length()
        if spring_length < spring_rest_length:
            compressed_springs.append(i)
        else : 
            tensionned_springs.append(i)
    return compressed_springs

def distance_to_target(tencer:Tencer,target_rods):
    """ Distance between the tencer's rods and the target rods.
    Args:
        tencer (Tencer) : a tencer
        target_rods (list[ElasticRod]) : the target rods
    Returns:
        (float) : the distance between the tencer's rods and the target rods.
    """
    d = 0
    total_length = 0
    total_num_vertices = 0
    for i,r in enumerate(tencer.getOpenRods()):
        d += np.linalg.norm(np.array(r.deformedPoints()) - np.array(target_rods[i].deformedPoints()))
        total_length += r.characteristicLength()
        total_num_vertices += r.numVertices()
    idx = len(tencer.getOpenRods())
    for i,r in enumerate(tencer.getClosedRods()):
        d += np.linalg.norm(np.array(r.deformedPoints()) - np.array(target_rods[i+idx].deformedPoints()))
        total_length += r.restLength()
        total_num_vertices += r.numVertices()
    return d / total_length / total_num_vertices
    

        
        
def compute_grad_tol(tencer:Tencer):
    """ Set gradient tolerance for forward equilibrium solve depending on the problem scale
    Args : 
        tencer (Tencer) : the tencer that will be simulated
    Returns : 
        (float) : gradient tolerance for equilibrium computation
    """
    num_springs = len(tencer.getSprings())
    rod_factor = 0
    for r in tencer.getOpenRods():
        n_vertices = r.numVertices()
        length = r.characteristicLength()
        m = r.material(0)
        y = m.youngModulus
        c = m.crossSectionHeight
        rod_factor += n_vertices*y*c*c*c/length
    for r in tencer.getClosedRods():
        n_vertices = r.numVertices()
        length = r.restLength()
        m = r.rod.material(0)
        c = m.crossSectionHeight
        y = m.youngModulus
        rod_factor += n_vertices*y*c*c*c/length
    return max(1e-9,1e-9*num_springs+5e-12*rod_factor)


def compute_initial_stiffness_pr(rod:PeriodicRod):
    """ Compute initial spring stiffness, according to formula k = 100 E c^2 / L
	Args:
		rod (PeriodicRod) : tencer’s longest rod
    """
    y = rod.rod.material(0).youngModulus
    c = rod.rod.material(0).crossSectionHeight
    l = rod.restLength()
    return 100 * y * c * c / l

def compute_initial_stiffness_er(rod:ElasticRod):
    """ Compute initial spring stiffness, according to formula k = 100 E c^2 / L
	Args:
		rod (ElasticRod) : tencer’s longest rod
    """
    y = rod.material(0).youngModulus
    c = rod.material(0).crossSectionHeight
    l = rod.restLength()
    return 100 * y * c * c / l


def greedy_decimation_step(tencer,aligned_rods, alignment_fct, optimizer_options, fixed_vars, distance_threshold=5e-6):
    """ Performs the greedy decimation step.
	Args:
		tencer (Tencer): current tencer.
		aligned rods (list of ElasticRod/PeriodicRod) : aligned target rods
		alignment_fct (Callable) : function that performs rigid alignment between the tencer and the target rods
		optimizer_options (NewtonOptimizerOption) : Equilibrium optimization options
		fixed_vars (list[int]) : fixed variables for equilibrium computation
		distance_threshold : distance threshold to stop the greedy decimation algorithm
	Returns:
		tencer (Tencer): tencer after greedy decimation
		aligned rods (list of ElasticRod/PeriodicRod) : aligned target rods
	"""
    
    distance = distance_to_target(tencer,aligned_rods)
    print("Distance : ", distance, ", number of springs", tencer.numRestVars())
    
    while distance < distance_threshold:

        # remove spring with lowest force
        forces = compute_spring_forces(tencer.getSprings())
        min_spring = np.argmin(forces)
        idx = np.array([i for i in range(tencer.numRestVars()) if i!=min_spring])
        new_springs = np.array(tencer.getSprings())[idx]
        new_a = np.array(tencer.getAttachmentVertices())[idx]
        tencer = Tencer(tencer.getOpenRods(),tencer.getClosedRods(),new_springs,new_a,tencer.getTargetRods())

        # Compute equilibrium
        c = computeEquilibrium(tencer, fixedVars=fixed_vars, opts=optimizer_options, hessianShift = 1e-8)

        # Check for compressed springs, remove them if necessary
        tencer = remove_compressed_springs(tencer,fixed_vars,optimizer_options)

        # Rigid registration of the target 
        aligned_rods = alignment_fct(tencer,aligned_rods)

        # Compute distance to target
        distance = distance_to_target(tencer,aligned_rods)
        print("Distance : ", distance, ", number of springs", tencer.numRestVars())
        
    return tencer, aligned_rods

def target_registration(tencer,aligned_rods):
    """ Rigid alignment: target rods to current tencer.
	Args:
		knot (Tencer) : current tencer.
		aligned rods (list of ElasticRod/PeriodicRod) : target rods to align
	Returns:
		new_aligned_rods (list of ElasticRod/PeriodicRod) : aligned target rods
    """
    target_points = []
    current_points = []
    num_open_rods = len(tencer.getOpenRods())
    for i in range(len(aligned_rods)):
        target_points = target_points + aligned_rods[i].deformedPoints()
        if i < num_open_rods:
            current_points = current_points + tencer.getOpenRods()[i].deformedPoints()
        else:
            current_points = current_points + tencer.getClosedRods()[i - num_open_rods].deformedPoints()
    target_points = np.array(target_points)
    current_points = np.array(current_points)
    R,t = register_points(current_points, target_points)
    new_target_points = np.einsum("ij,kj->ki",R,target_points) + t
    
    start_idx = 0
    new_aligned_rods = []
    for i in range(len(aligned_rods)):
        end_idx = start_idx + len(aligned_rods[i].deformedPoints())
        
        if i < num_open_rods:
            target_points = new_target_points[start_idx:end_idx]
            aligned_rod = ElasticRod(target_points)
            material = aligned_rods[i].material(0)
            aligned_rod.setMaterial(material)
        else:
            target_points = np.concatenate([new_target_points[start_idx:end_idx], [new_target_points[start_idx], new_target_points[start_idx+1]]])
            aligned_rod = PeriodicRod(target_points, zeroRestCurvature=True)
            material = aligned_rods[i].rod.material(0)
            aligned_rod.setMaterial(material)
        new_aligned_rods.append(aligned_rod)
        start_idx = end_idx
    
    return new_aligned_rods


def draw_initial_stiffness_curve(tencer:Tencer, fixed_vars, opt,stiffness_vals):
    num_springs = []
    for s in stiffness_vals:
        print("stiffness value : ", s)
        tencer1 = Tencer(tencer)
        tencer1.setRestVars([s]*tencer1.numRestVars())
        tencer1 = remove_compressed_springs(tencer1, fixed_vars, opt)
        num_springs.append(len(tencer1.getSprings()))
    plt.loglog(np.array(stiffness_vals),np.array(num_springs))
    plt.show()
    return np.array(stiffness_vals),np.array(num_springs)

def draw_initial_stiffness_and_distance_curve(tencer:Tencer, target_rods, fixed_vars, opt,stiffness_vals):
    num_springs = []
    distance = []
    for s in stiffness_vals:
        print("stiffness value : ", s)
        tencer1 = Tencer(tencer)
        tencer1.setRestVars([s]*tencer1.numRestVars())
        tencer1 = remove_compressed_springs(tencer1, fixed_vars, opt)
        num_springs.append(len(tencer1.getSprings()))
        distance.append(distance_to_target(tencer1,target_rods))
    plt.semilogx(np.array(stiffness_vals),np.array(num_springs))
    plt.semilogx(np.array(stiffness_vals),np.array(distance))
    plt.show()
    return np.array(stiffness_vals),np.array(num_springs), np.array(distance)