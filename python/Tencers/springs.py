import ElasticRods
from tencers import *

import numpy as np

#############################################
###       CREATE MULTI ROD SPRINGS        ###
#############################################


def create_multi_rod_springs(rods, vertices, stiffness, rest_lengths=None) : 
    """
    Create springs between multiple rods.
    Args : 
        rods (List[PeriodicRod]) : set of rods
        vertices (np.array[n,4]) : for each spring, array of attachment vertices in the format [rod1,vertex1,rod2,vertex2]
        stiffness (float, int, list, np.array) : stiffness of the springs
        rest_lengths (opt list/np.array) : spring rest lengths
    Returns : 
        springs (List[Spring]) : set of springs
        attachment_vertices (List[SpringAttachmentVertices]) : Multi-rod spring attachment vertices
    """
    attachment_vertices = []
    springs = []
    b = np.array([0,1,2])
    if rest_lengths is None:
        rest_lengths = np.zeros(len(vertices))
    for i in range(len(vertices)):
        v = vertices[i]
        attachment_vertices.append(SpringAttachmentVertices(v[0],v[1],v[2],v[3]))
        coordsA = rods[v[0]].getDoFs()[b + 3*v[1]]
        coordsB = rods[v[2]].getDoFs()[b + 3*v[3]]
        if type(stiffness) == float or type(stiffness) ==  int:
            springs.append(Spring(coordsA, coordsB, stiffness, rest_lengths[i]))
        elif type(stiffness) == list or type(stiffness) == np.ndarray:
            springs.append(Spring(coordsA, coordsB, stiffness[i], rest_lengths[i]))
        else :
            print("unknown stiffness type")
            return
    return springs, attachment_vertices




########################
### SPRING UTILITIES ###
########################

def spring_lengths(springs):
    """
    Lengths of a set of springs.
    Args : 
        springs (List[Spring]) : list of springs
    Returns :
        lengths (List[float]) : the springs' lengths
    """
    lengths = []
    for spring in springs:
        c = np.array(spring.get_coords())
        l = c[:3]-c[3:]
        lengths.append(np.linalg.norm(l))
    return lengths
        

def remove_zero_springs(knot):
    """
    Remove springs with zero stiffness during sparsification
    Args :
        knot (Tencer) : current knot
    Returns :
        (Tencer) : current knot without zero-stiffness springs
    """
    v = knot.getRestVars()
    idx = (v > 0)
    springs = np.array(knot.getSprings())[idx]
    attachment_vertices = np.array(knot.getAttachmentVertices())[idx]
    open_rods = knot.getOpenRods()
    closed_rods = knot.getClosedRods()
    target_rods = knot.getTargetRods()
    return Tencer(open_rods,closed_rods,springs,attachment_vertices,target_rods)


def compute_spring_forces(springs):
    """ Compute the forces exerted by each spring on a tencer.
	Args:
		springs (list[Spring]) : springs
	Returns:
		new_springs (list[Spring]) : springs with zero rest length
    """
    forces = []
    for spring in springs:
        spring_length = np.linalg.norm(spring.get_coords()[:3] - spring.get_coords()[3:])
        spring_rest_length = spring.get_rest_length()
        force = spring.stiffness * (spring_length - spring_rest_length) 
        forces.append(force)
    return np.array(forces)

def replace_springs(springs):
    """ Replace all springs with zero rest length springs that exert the same force
	Args:
		springs (list[Spring]) : springs to be replaced
	Returns:
		new_springs (list[Spring]) : springs with zero rest length
    """
    new_springs = []
    for spring in springs:
        spring_length = np.linalg.norm(spring.get_coords()[:3] - spring.get_coords()[3:])
        spring_rest_length = spring.get_rest_length()
        stiffness = spring.stiffness * (spring_length - spring_rest_length) / spring_length
        new_springs.append(Spring(spring.get_coords()[:3],spring.get_coords()[3:],stiffness,0))
    return new_springs

def replace_springs_with_rest_length(springs, rest_lengths):
    """ Replace all springs new springs with different rest lengths that exert the same force
	Args:
		springs (list[Spring]) : springs to be replaced
		rest_lengths (list) : new rest lengths
	Returns:
		new_springs (list[Spring]) : new springs with the given rest length
    """
    new_springs = []
    for i,spring in enumerate(springs):
        spring_length = np.linalg.norm(spring.get_coords()[:3] - spring.get_coords()[3:])
        spring_rest_length = spring.get_rest_length()
        stiffness = spring.stiffness * (spring_length - spring_rest_length) / (spring_length - rest_lengths[i])
        new_springs.append(Spring(spring.get_coords()[:3],spring.get_coords()[3:],stiffness,rest_lengths[i]))
    return new_springs
