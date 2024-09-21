# ----------------------
# (c) 2024 Magimine, LLC
# ----------------------

import cv2
import numpy as np
import glob
import os

def get_bounding_box_edges(min_coords, max_coords):
    # Compute the 8 vertices of the bounding box
    vertices = np.array([[min_coords[0], min_coords[1], min_coords[2]],
                         [max_coords[0], min_coords[1], min_coords[2]],
                         [max_coords[0], max_coords[1], min_coords[2]],
                         [min_coords[0], max_coords[1], min_coords[2]],
                         [min_coords[0], min_coords[1], max_coords[2]],
                         [max_coords[0], min_coords[1], max_coords[2]],
                         [max_coords[0], max_coords[1], max_coords[2]],
                         [min_coords[0], max_coords[1], max_coords[2]]], dtype=np.float32)

    # Define the edges of the bounding box
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
             (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
             (0, 4), (1, 5), (2, 6), (3, 7)]  # Vertical edges

    return vertices, edges


def face_blocks_point(v0, v1, v2, x, camera_pos): # camera is at camera_pos
    edge1 = v1 - v0
    edge2 = v2 - v0
    dx = x - camera_pos

    A = np.hstack((dx.reshape(-1, 1), edge1.reshape(-1, 1), edge2.reshape(-1, 1)))
    
    s = np.linalg.solve(A, x - v0)
    
    if s[0] > 0 and s[0] < 1 and s[1] > 0 and s[1] < 1 and s[2] > 0 and s[2] < 1:
        return True # face blocks point x
    else:
        return False


def vertex_is_visible(vertices, faces, v, camera_pos):
    for v0, v1, v2, v3 in faces:
        if v not in (v0, v1, v2, v3):
            x = vertices[v]
            if face_blocks_point(vertices[v0], vertices[v1], vertices[v2], x, camera_pos):
                return False
    return True


# Project vertices onto camera view
def project_bbox_onto_image(vertices, rotation_matrix, tvec, camera_matrix, color, image):

    # Define the faces of the bounding box (defined by 3 vertices each)
    faces = [
        (0, 1, 3, 2), (0, 1, 4, 5), (1, 2, 5, 6), (2, 3, 6, 7), (3, 0, 7, 4), (4, 5, 7, 6)
    ]

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    bb_vertices, bb_edges = get_bounding_box_edges(min_coords, max_coords)
    
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    
    projected_points, _ = cv2.projectPoints(bb_vertices, rvec, tvec, camera_matrix, None)
    
    bb_vertices = np.dot(bb_vertices, rotation_matrix.T)

    tvec = tvec.reshape(1,3)
    
    # Draw edges of visible faces
    for start, end in bb_edges:
        # Check if edge is visible
        if vertex_is_visible(bb_vertices, faces, start, -tvec) and vertex_is_visible(bb_vertices, faces, end, -tvec):
            p1 = projected_points[start][0]
            p2 = projected_points[end][0]
            p1 = tuple(map(int, p1))
            p2 = tuple(map(int, p2))
            cv2.line(image, p1, p2, color, 2)

    return image


inch_to_m = 0.0254

# Box dimensions
box_width = 4 * inch_to_m
box_height = 2 * inch_to_m
box_depth = 1 * inch_to_m

# Define vertices of the box    
box_vertices = np.array([
    # Bottom vertices
    [-box_depth/2, -box_width/2, -box_height/2],
    [box_depth/2, -box_width/2, -box_height/2],
    [box_depth/2, -box_width/2, box_height/2],
    [-box_depth/2, -box_width/2, box_height/2],
    # Top vertices
    [-box_depth/2, box_width/2, -box_height/2],
    [box_depth/2, box_width/2, -box_height/2],
    [box_depth/2, box_width/2, box_height/2],
    [-box_depth/2, box_width/2, box_height/2]  
])

camera_matrix = np.array([[1762, 0, 946], [0, 1776, 523], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32) 

# Parameters used to create the ArUco GridBoard
markersX = 5              # Number of markers in the x-direction (columns)
markersY = 7              # Number of markers in the y-direction (rows)
markerLength = 0.0175     # Length of one marker's side (in meters)
markerSeparation = 0.0016 # Separation between markers (in meters)

# Calculate total width and height of the ArUco board
total_width = (markersX * markerLength) + ((markersX - 1) * markerSeparation)
total_height = (markersY * markerLength) + ((markersY - 1) * markerSeparation)

# Compute the center of the board in 3D space (assuming Z=0 for the board plane)
board_center = np.array([total_width / 2, total_height / 2, 0], dtype=np.float32).reshape(3,1)

box_center = board_center - np.array([0, 0, box_height/2], dtype=np.float32).reshape(3,1)

# Initialize Aruco board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.GridBoard_create(markersX, markersY, markerLength, markerSeparation, aruco_dict) 
parameters =  cv2.aruco.DetectorParameters_create()
parameters.cornerRefinementWinSize=32
parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_CONTOUR   

rvec = np.zeros((1, 3), dtype=np.float32)
tvec = np.zeros((1, 3), dtype=np.float32)

# Directory containing the images
image_dir = "data/image_frames2" 

# Get a list of all images in the directory that match the pattern "img_*.png"
image_paths = sorted(glob.glob(os.path.join(image_dir, "img_*.png")))

# Loop through each image
for image_path in image_paths:
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)

    if retval > 0:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        box_tvec = tvec + rotation_matrix.dot(box_center)
        print("Rotation: ",rotation_matrix)
        print("Translation: ", box_tvec)
        # Draw axis on the board (length of axis in meters)
        axis_length = 0.03  # Adjust based on your scene, 10 cm for the axes length
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

        # Draw bounding box
        bbox_color = (0, 240, 0)
        project_bbox_onto_image(box_vertices, rotation_matrix, box_tvec, camera_matrix, bbox_color, image)
        
        # Display the image with the pose axis drawn
        cv2.imshow('Pose Estimation', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Pose estimation failed.")











