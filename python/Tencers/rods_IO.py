import elastic_rods
import numpy as np
import os
import json

def export_knot_to_obj(file_prefix, knot):
    """
    Export a tencer to an obj file.
    Args : 
        file_prefix (string) : file prefix
        knot (Tencer) : tencer to export
    """
    filename = file_prefix + ".obj"
    target_rods = knot.getTargetRods()
    closed_rods = knot.getClosedRods()
    open_rods = knot.getOpenRods()
    springs = knot.getSprings()
    start_edge = 0
    
    # If the file already exists, delete it and print a warning
    if os.path.exists(filename):
        print('Warning: file ' + filename + ' already exists. The content of the existing file has been overwritten.')
        os.remove(filename)

    n_or = len(open_rods)
    n_cr = len(closed_rods)
    for i in range(len(target_rods)):
        rod_type = 'open' if i < n_or else 'closed'
        with open(filename, 'a') as f:
            f.write('o ' + file_prefix + '_target_' + rod_type + '_' + str(i) + '\n')
        rod = target_rods[i]
        points_list = rod.deformedPoints()
        write_obj_data_to_file(filename, points_list, start_edge, rod.numEdges() + 1, False, True)
        start_edge += rod.numEdges() + 1
    for i in range(len(open_rods)):
        with open(filename, 'a') as f:
            f.write('o ' + file_prefix + '_open_rod_' + str(i) + '\n')
        rod = open_rods[i]
        points_list = rod.deformedPoints()
        write_obj_data_to_file(filename, points_list, start_edge, rod.numEdges() + 1, False,True)
        start_edge += rod.numEdges() + 1
    for i in range(len(closed_rods)):
        with open(filename, 'a') as f:
            f.write('o ' + file_prefix + '_closed_rod_' + str(i + len(open_rods)) + '\n')
        rod = closed_rods[i]
        points_list = rod.deformedPoints()
        write_obj_data_to_file(filename, points_list, start_edge, rod.numEdges(), False, True)
        start_edge += rod.numEdges()
    with open(filename, 'a') as f:
        f.write('o ' + file_prefix + '_springs \n')
        for s in springs:
            a = s.get_coords()[:3]
            b = s.get_coords()[3:]
            f.write('v ')
            for coord in a:
                f.write(str(coord) + ' ')
            f.write('\nv ')
            for coord in b:
                f.write(str(coord) + ' ')
            f.write('\n')
        for i in range(len(springs)):
            f.write('l ' + str(start_edge + 2*i+1) + ' ' + str(start_edge + 2*i+2) + '\n')


def import_knot_from_obj(file):
    """
    Read points from an obj file and return lists of target_rods, closed_rods, open_rods, and springs.
    Args:
        file (string): file name
    Returns:
        target_rods (List[List[np.array]]): list of target rods, each represented as a list of points
        closed_rods (List[List[np.array]]): list of closed rods, each represented as a list of points
        open_rods (List[List[np.array]]): list of open rods, each represented as a list of points
        springs (List[List[np.array]]): list of springs, each represented as a list of points
    """
    target_rods = []
    closed_rods = []
    open_rods = []
    springs = []

    current_object = None
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('o'):
                if '_target_' in line:
                    current_object = target_rods
                elif '_closed_' in line:
                    current_object = closed_rods
                elif '_open_' in line:
                    current_object = open_rods
                elif '_springs' in line:
                    current_object = springs
            elif line.startswith('v'):
                coords = line.split()[1:]
                point = [float(coord) for coord in coords]
                current_object.append(point)

    return target_rods, closed_rods, open_rods, springs

    
def write_obj_data_to_file(file, points_list, start_edge, num_edges, center, append = False):
    """
    Helper function : write a point list to an obj file
    """
    if center:  # for PeriodicRodList, globally center 
        points_list = list(np.array(points_list) - np.mean(np.array(points_list), axis=0))

    if append :
        c = 'a'
    else :
        c = 'w'
    with open(file, c) as f:
        for p in points_list:
            f.write('v ')
            for coord in p:
                f.write(str(coord) + ' ')
            f.write('\n')
        for i in range(start_edge+1, start_edge + num_edges):  # nodes' numbering starts from 1
            f.write('l ' + str(i) + ' ' + str(i+1) + '\n')
        f.write('l ' + str(start_edge + num_edges) + ' ' + str(start_edge+1) + '\n')
    
def export_periodic_rod_to_obj(file, rod, center=False):
    """
    Export a PeriodicRod to an obj file.
    Args : 
        file (string) : file name
        rod (PeriodicRod, np.array) : periodic rod
        center (bool) : center points around 0
    """

    path = os.path.dirname(file)
    os.makedirs(path, exist_ok=True)
    
    if isinstance(rod, elastic_rods.PeriodicRod):
        points_list = rod.deformedPoints()
        write_obj_data_to_file(file, points_list, 0, rod.numEdges(), center)
            
    elif isinstance(rod, np.ndarray):  # assume single rod
        points_list = list(rod)
        write_obj_data_to_file(file, points_list, 0, rod.shape[0], center)
        
    else:
        raise ValueError('Unknown input type')

        
def write_springs_to_obj(file,springs):
    """
    Export springs to an obj file.
    Args : 
        file (string) : file name
        springs (List[Springs]) : set of springs
    """
    path = os.path.dirname(file)
    os.makedirs(path, exist_ok=True)
    with open(file, 'w') as f:
        for s in springs:
            a = s.get_coords()[:3]
            b = s.get_coords()[3:]
            f.write('v ')
            for coord in a:
                f.write(str(coord) + ' ')
            f.write('\nv ')
            for coord in b:
                f.write(str(coord) + ' ')
            f.write('\n')
        for i in range(len(springs)):
            f.write('l ' + str(2*i+1) + ' ' + str(2*i+2) + '\n')
            
    
def replace_open_with_frame_in_obj(file):
    "Helper function to consistently deal with the hack of using an open rod to represent a frame when writing to obj"

    with open(file, 'r') as f:
        lines = f.readlines()

    n_target_open_rods = 0
    n_open_rods = 0
    # Read line by line, delete the object which has 'target_open' in the name,
    # replace 'open_rod' -> 'frame_rod', count the number of open rods to check there is only one
    for i, line in enumerate(lines):
        if 'target_open' in line:
            del lines[i]
            n_target_open_rods += 1
            while i < len(lines) and not lines[i].startswith('o'):
                del lines[i]
        if 'open_rod' in line:
            lines[i] = line.replace('open_rod', 'frame_rod')
            n_open_rods += 1
    
    # Check there was only one open rod
    assert n_open_rods == n_target_open_rods, 'Number of open rods and target open rods do not match'
    if n_open_rods != 1:
        raise ValueError('Expected one open rod, found ' + str(n_open_rods))
        
    # Write the modified lines back to the file
    with open(file, 'w') as f:
        f.writelines(lines)


def read_nodes_from_file(file, splitchar = ' '):
    """
    Supported extensions: obj, txt
    """
    nodes = []
    connectivity = []
    n_rods = 0
    if file.endswith('.obj'):
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('v'):
                    pt = []
                    for coord in line.split(splitchar)[1:4]:
                        pt.append(float(coord))
                    nodes.append(np.array(pt))
                if line.startswith('l'):
                    edge = []
                    s = line.split(splitchar)
                    for index in s[1:3]:
                        edge.append(int(index))
                        if len(edge) == 2 and abs(int(index) - edge[0]) != 1: # last edge of a rod 
                            n_rods += 1
                    connectivity.append(edge)
        if n_rods > 1:
            indices_connections = [i for i in range(len(connectivity)) if abs(connectivity[i][0] - connectivity[i][1]) != 1]
            ne_per_rod = np.append(indices_connections[0] + 1, np.diff(indices_connections))
            pts = np.array(nodes)
            pts_list = [pts[0:ne_per_rod[0], :]]
            for ri in range(0, n_rods-1):
                pts_list.append(pts[indices_connections[ri]:indices_connections[ri+1]])
            return pts_list
        else:
            return np.array(nodes)
    
    elif file.endswith('.txt'):
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                pt = []
                for coord in line.split(' ')[0:3]:
                    pt.append(float(coord))
                nodes.append(np.array(pt))
        return np.array(nodes)
    
    elif not '.' in file.split('/')[0]: # no extension, assum same formatting as .txt
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                pt = []
                for coord in line.split(' ')[0:3]:
                    pt.append(float(coord))
                nodes.append(np.array(pt))
        return np.array(nodes)
    
    
def export_vector_field(filename,vectors, positions, color_scheme=None, min_len=0.2, avg_len=0.5,max_len=0.5):
    
    # default color scheme
    if color_scheme is None:
        color_scheme = [[1,"num",[0.8,0.0,0.0]],[0.5,"num",[0.8,0.2,0.0]],[0.01,"num",[0.8,0.6,0.0]]]
        
    def create_dict_color_scheme(l):
        col = dict()
        col["value_to_map_to_this_color"] = l[0]
        col["color_space"] = "sRGB"
        col["format"] = l[1]
        col["color"] = l[2]
        return col
    
    def create_list_color_scheme(l):
        res = []
        for x in l:
            res.append(create_dict_color_scheme(x))
        return res
    
    def create_list_vectors(vectors, positions):
        res = []
        for position,vector in zip(positions,vectors):
            res.append(dict({"pos":list(position),"vec" : list(vector)}))
        return res
    
    # create dictionary
    params = dict()
    params["min_len"] = min_len
    params["avg_len"] = avg_len
    params["max_len"] = max_len
    params["frame_offset"] = 5
    params["colorscheme"] = create_list_color_scheme(color_scheme)
    params["vectors"] = create_list_vectors(vectors,positions)
    
    # dump into json file
    json_object = json.dumps(params, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_object)



    

