import pickle
import numpy as np
import json

from tencers import *
from Tencers.springs import *
from elastic_rods import PeriodicRod, RodMaterial, ElasticRod

def save_state(knot, file_name):
    """
    Save a tencer to a pkl file.
    Args : 
        knot (Tencer) : knot to save
        file_name (string) : file name
    """
    to_pickle = dict()
    to_pickle["open_rods"] = knot.getOpenRods()
    to_pickle["closed_rods"] = knot.getClosedRods()
    to_pickle["target_rods"] = knot.getTargetRods()
    to_pickle["rest_vars"] = knot.getRestVars()
    to_pickle["rest_vars_type"] = knot.getRestVarsType()
    to_pickle["attachment_vertices"] = knot.getAttachmentVertices()
    to_pickle["spring_anchor_vars"] = knot.getSpringAnchorVars()
    
    # Workaround unimplemented pickling function for Spring object (TODO)
    springs = knot.getSprings()
    to_pickle["spring_stiffnesses"] = np.array([s.stiffness for s in springs])
    to_pickle["spring_rest_lengths"] = np.array([s.get_rest_length() for s in springs])

    pickle.dump(to_pickle,open( file_name, "wb" ))
    
    
def load_state(file_name):
    """
    Load a tencer from a pkl file
    Args : 
        file_name (string) : file name
    Returns : 
        knot (Tencer) : knot from pkl file
    """
    from tencers import RestVarsType
    
    d = pickle.load(open( file_name, "rb" ))
    
    open_rods = d["open_rods"]
    closed_rods = d["closed_rods"]
    target_rods = d["target_rods"]
    rest_vars = d["rest_vars"]
    rest_vars_type = d["rest_vars_type"]
    attachment_vertices = d["attachment_vertices"]
    spring_anchor_vars = d["spring_anchor_vars"]
    
    spring_stiffnesses = d["spring_stiffnesses"]
    spring_rest_lengths = d["spring_rest_lengths"]

    # Check if the rest vars are consistent with the rest vars type
    if rest_vars_type == RestVarsType.StiffnessOnVerticesFast or rest_vars_type == RestVarsType.Stiffness:
        assert((spring_stiffnesses == rest_vars).all())
        
    elif rest_vars_type == RestVarsType.SpringAnchors:
        sav_flattened = np.array([[sav.mat_coord_A, sav.mat_coord_B] for sav in spring_anchor_vars]).flatten()
        assert((sav_flattened == rest_vars).all())
        
    elif rest_vars_type == RestVarsType.StiffnessAndSpringAnchors:

        # TODO MICHELE: sketchy workaround to patch the fact that the arc-length position
        # of a sliding node is always computed modulo L *as a rest variable*,
        # but can effectively be out of [0, L] when stored in spring_anchor_vars.
        # Need to upgrade the setters (m_setRestVars and set_spring_anchors) in Tencer 
        # so that they take care of applying the modulo operator, 
        # instead of delegating to the SlidingNode class. 
        # This requires restructuring the SlidingNode class (TODO).
        
        def modulo_L(L, x):
            if L == np.inf or 0 <= x <= L:
                return x
            unew = x
            if -L <= x < 0:
                unew += L
            elif L < x <= 2*L:
                unew -= L
            else:
                raise ValueError("x = {} is not in the range [-L, 2L], with L = {}".format(x, L))
            return unew
    
        rod_lengths = [np.inf for er in open_rods] + [pr.restLength() for pr in closed_rods]
        sav_flattened_modulo_L = []
        for i in range(len(spring_anchor_vars)):
            sav = spring_anchor_vars[i]
            sav_flattened_modulo_L.append([
                modulo_L(rod_lengths[sav.rod_idx_A], sav.mat_coord_A),
                modulo_L(rod_lengths[sav.rod_idx_B], sav.mat_coord_B)
            ])
        sav_flattened_modulo_L = np.array(sav_flattened_modulo_L).flatten()

        assert((np.abs(np.concatenate([spring_stiffnesses, sav_flattened_modulo_L]) - rest_vars) < 1e-12).all())

    # Build the Tencer object
    if rest_vars_type == RestVarsType.StiffnessOnVerticesFast:
        attach_vertices_array = np.array([[av.rodIdxA, av.vertexA, av.rodIdxB, av.vertexB] for av in attachment_vertices])
        springs, attachment_vertices_recomputed = create_multi_rod_springs(open_rods + closed_rods, attach_vertices_array, spring_stiffnesses, spring_rest_lengths)
        assert(attachment_vertices_recomputed == attachment_vertices)
        
        knot = Tencer(open_rods, closed_rods, springs, attachment_vertices, target_rods)
        knot.setRestVarsType(RestVarsType.StiffnessOnVerticesFast)
        
    elif rest_vars_type == RestVarsType.Stiffness or rest_vars_type == RestVarsType.SpringAnchors or rest_vars_type == RestVarsType.StiffnessAndSpringAnchors:
        n_springs = len(spring_stiffnesses)
        # The only Tencer constructor currently available requires spacifying the attachment vertices.
        # Attachment vertices are not available if the pickeld state had rest vars type different from StiffnessOnVerticesFast.
        # We define dummy attachment vertices just for building a TK; setting the correct rest vars type will then delete any stored attachment vertex.
        dummy_vi, dummy_vj = 0, 1
        dummy_attachment_vertices_array = np.array([[spring_anchor_vars[si].rod_idx_A, dummy_vi, spring_anchor_vars[si].rod_idx_B, dummy_vj] for si in range(n_springs)])
        springs, dummy_attachment_vertices = create_multi_rod_springs(open_rods + closed_rods, dummy_attachment_vertices_array, spring_stiffnesses, spring_rest_lengths)
        
        knot = Tencer(open_rods, closed_rods, springs, dummy_attachment_vertices, target_rods)
        knot.setRestVarsType(rest_vars_type)
        
        knot.setSpringAnchorVars(spring_anchor_vars)  # ensure that spring anchor vars are set also for RestVarsType.Stiffness (they are not part of the rest variables)

        knot.setRestVars(rest_vars)
        
    return knot

#################################################
###   LOAD AND SAVE FILE FROM/TO JSON FILES   ###
#################################################
        

def load_from_json(filename):
    """ Load tencer to json file.
    Args:
        filename (string) : file name of the tencer's json file
    """
    
    with open(filename) as f:
        tencer_data = json.load(f)
    
    num_open_rods = tencer_data['numOpenRods']    
    num_closed_rods = tencer_data['numClosedRods']
    num_springs = tencer_data['numSprings']

    young_modulus = tencer_data['YoungModulus']
    c = tencer_data['crossSection']
    
    open_rods = []
    closed_rods = []
    target_rods = []
    springs = []
    attachment_vertices = []
    
    for i in range(num_open_rods):
        rod_points = tencer_data['restPoints'][i]
        open_rods.append(ElasticRod(rod_points))
        target_rods.append(ElasticRod(rod_points))
        rod_young_modulus = young_modulus[i]
        material = RodMaterial('ellipse', rod_young_modulus, 0.5, [c[i], c[i]])  # circular cross-section
        open_rods[-1].setMaterial(material)
        target_rods[-1].setMaterial(material)
        num_vertices = len(rod_points)
        open_rods[-1].setRestKappas(np.zeros((num_vertices,2)))
        open_rods[-1].setRestLengths(tencer_data['rodRestLengths'][i])
        open_rods[-1].setRestDirectors(buildRestDirectors(tencer_data['restDirectors'][i]))
        open_rods[-1].deformedConfiguration().initialize_from_data(rod_points,buildRestDirectors(tencer_data['referenceDirectors'][i]),tencer_data['sourceTangents'][i])
        target_rods[-1].setDoFs(tencer_data['targetDefoVars'][i])
        
        
    for i in range(num_open_rods, num_closed_rods+ num_open_rods):
        rod_points = tencer_data['restPoints'][i]
        closed_rods.append(PeriodicRod(rod_points, zeroRestCurvature=True))
        target_rods.append(ElasticRod(rod_points))
        rod_young_modulus = young_modulus[i]
        material = RodMaterial('ellipse', rod_young_modulus, 0.5, [c[i], c[i]])  # circular cross-section
        closed_rods[-1].setMaterial(material)
        target_rods[-1].setMaterial(material)
        closed_rods[-1].rod.setRestLengths(tencer_data['rodRestLengths'][i])
        closed_rods[-1].rod.setRestDirectors(buildRestDirectors(tencer_data['restDirectors'][i]))
        closed_rods[-1].rod.deformedConfiguration().initialize_from_data(rod_points,buildRestDirectors(tencer_data['referenceDirectors'][i]),tencer_data['sourceTangents'][i])
        target_rods[-1].setDoFs(tencer_data['targetDefoVars'][i])
        
      
    
    for i in range(num_springs):
        springs.append(Spring(np.zeros(3),np.zeros(3),0,tencer_data['springRestLengths'][i]))
        if len(tencer_data['attachmentVertices'])>0:
            attachment_vertices.append(SpringAttachmentVertices(*tencer_data['attachmentVertices'][i]))
        else:
            if 'spring_rod_anchor' not in tencer_data:
                attachment_vertices.append(SpringAttachmentVertices(0,0,0,1))
            else:
                attachment_vertices.append(SpringAttachmentVertices(tencer_data['spring_rod_anchor'][i][0],0,tencer_data['spring_rod_anchor'][i][1],1))
        
    tencer = Tencer(open_rods,closed_rods,springs,attachment_vertices,target_rods)
    
    
    tencer.setDefoVars(tencer_data['defoVars'])
    
    
    tencer.setRestVarsType(intToRestVarType(tencer_data['restVarType']))
    tencer.setRestVars(tencer_data['restVars'])
    
    return tencer

def save_to_json(tencer,filename):
    """ Save a tencer's state in a json file
    Args:
        tencer (Tencer) : a tencer
        filename (string) : the file's name
    """
    open_rods = tencer.getOpenRods()
    closed_rods = tencer.getClosedRods()
    target_rods = tencer.getTargetRods()
    springs = tencer.getSprings()

    dico = dict()
    dico['numOpenRods'] = len(open_rods)
    dico['numClosedRods'] = len(closed_rods)
    dico['numSprings'] = len(springs)

    # Rod material
    dico['YoungModulus'] = [r.material(0).youngModulus for r in open_rods] + [r.rod.material(0).youngModulus for r in closed_rods]
    dico['crossSection'] = [r.material(0).crossSectionHeight/2 for r in open_rods] + [r.rod.material(0).crossSectionHeight/2 for r in closed_rods]

    # Rod rest quantities
    dico['rodRestLengths'] = [r.restLengths() for r in open_rods] + [r.rod.restLengths() for r in closed_rods]
    dico['restPoints'] = [to_list(r.restPoints()) for r in open_rods] + [to_list(r.rod.restPoints()) for r in closed_rods]
    dico['restDirectors'] = [directors_to_list(r.restDirectors()) for r in open_rods] + [directors_to_list(r.rod.restDirectors()) for r in closed_rods]

    # Rod deformed configuration
    dico['defoVars'] = list(tencer.getDefoVars())
    dico['sourceTangents'] = [to_list(r.deformedConfiguration().sourceTangent) for r in open_rods] + [to_list(r.rod.deformedConfiguration().sourceTangent) for r in closed_rods]
    dico['referenceDirectors'] = [directors_to_list(r.deformedConfiguration().sourceReferenceDirectors) for r in open_rods] + [directors_to_list(r.rod.deformedConfiguration().sourceReferenceDirectors) for r in closed_rods]

    # Tencer rest variables
    dico['restVarType'] = restVarTypeToInt(tencer.getRestVarsType())
    dico['restVars'] = list(tencer.getRestVars())

    # Springs
    dico['attachmentVertices'] = attachment_vertices_to_list(tencer.getAttachmentVertices())
    dico['springRestLengths'] = [s.get_rest_length() for s in springs]
    if tencer.getRestVarsType() != RestVarsType.StiffnessOnVerticesFast:
        dico['spring_rod_anchor'] = [[tencer.getSpringAnchorVarsForSpring(i).rod_idx_A,tencer.getSpringAnchorVarsForSpring(i).rod_idx_B] for i in range(len(springs))]


    # Target rod
    dico['targetDefoVars'] = [list(t.getDoFs()) for t in target_rods]

    with open(filename, 'w') as f:
        json.dump(dico,f)
        

def intToRestVarType(x):
    if x == 0:
        return RestVarsType.StiffnessOnVerticesFast
    if x == 1:
        return RestVarsType.Stiffness
    if x == 2: 
        return RestVarsType.SpringAnchors
    if x == 3:
        return RestVarsType.StiffnessAndSpringAnchors
    print("Warning: unknown rest var type")

def restVarTypeToInt(restVarType):
    if restVarType == RestVarsType.StiffnessOnVerticesFast:
        return 0
    if restVarType == RestVarsType.Stiffness:
        return 1
    if restVarType == RestVarsType.SpringAnchors:
        return 2
    if restVarType == RestVarsType.StiffnessAndSpringAnchors:
        return 3
    print("Warning: unknown rest var type")
    
def buildRestDirectors(d):
    directors = []
    for i in range(len(d)):
        directors.append(ElasticRod.Directors(d[i][0],d[i][1]))
    return directors

def to_list(l):
    lst = []
    for i in range(len(l)):
        lst.append(list(l[i]))
    return lst

def directors_to_list(d):
    lst = []
    for i in range(len(d)):
        lst.append([list(d[i].d1),list(d[i].d2)])
    return lst

def attachment_vertices_to_list(attachment_vertices):
    lst = []
    for v in attachment_vertices: 
        lst.append([v.rodIdxA,v.vertexA,v.rodIdxB,v.vertexB])
    return lst