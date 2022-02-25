import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
import copy
import pyglet

point_list = list()
polygon_list = list()

def read_file():
    global point_list , polygon_list  # use global if other functions need

    with open("./D files/queen.d.txt") as f:  #read .d   D files in the same folder as main.py
        original_data = f.read()  #str
    #then clean the data
    data = original_data.replace("\t", " ")  # short the white space
    data = data.replace("data", '')  # remove the first row "data"
    data = data.split("\n")  # use split() to list for each row
    lineone = data[0].split(" ")  # list ['','piont_num','polygon_num']
    while '' in lineone:
        lineone.remove("")
    point_num = lineone[0]
    polygon_num = lineone[1]
    data = data[1:]

    for i in data[:int(point_num)]:  # get the point part
        point_list.append(i.split(" "))

    for i in data[int(point_num):]:  # get the polygon part
        if i:
            polygon_list.append(i.split(" "))

    # however there are many '' in list
    for i, point in enumerate(point_list):  # turn float
        point_list[i] = [float(j) for j in point if j]  # '' = False
    for i, polygon in enumerate(polygon_list):
        polygon_list[i] = [int(j) for j in polygon if j]
    #data = original_data.replace("\t", " ").replace("data", "").split("\n")

def world_matrix(l: list):  # from local coord to world coord
    martix = np.identity(4)
    #set vector from local zero piont to world  l
    martix[0][3] = l[0]
    martix[1][3] = l[1]
    martix[2][3] = l[2]
    world_matrix = martix
    return world_matrix

def view_matrix(uvn, c): #view_matrix = R.T
    r = np.identity(4)  # r matrix = Ux, Uy, Uz, 0
    r[0][0] = uvn[0][0]  #           Vx, Vy, Vz, 0
    r[0][1] = uvn[0][1]  #           Nx, Ny, Nz, 0
    r[0][2] = uvn[0][2]  #           0,  0,  0,  1
    r[1][0] = uvn[1][0]
    r[1][1] = uvn[1][1]
    r[1][2] = uvn[1][2]
    r[2][0] = uvn[2][0]
    r[2][1] = uvn[2][1]
    r[2][2] = uvn[2][2]

    t = np.identity(4)   #  t matrix = 1, 0, 0, -cx
    for i in range(0, 3):  #           0, 1, 0, -cy
        t[i][3] = -c[i]   #            0, 0, 1, -cz
    #print("T:", t)       #            0, 0, 0, 1

    view_matrix = r @ t  # this is view_matrix = R.T

    return view_matrix

def cal_uvn(c, upvector):  # C is c V' is upvector
    n = np.zeros(3)  # N= P-C/abs(P-C)
    n = np.subtract(n, c)
    n = n / np.linalg.norm(n)  # default norm = 2 this is N= P-C/abs(P-C)

    u = np.zeros(3)  # U=N X V' / abs(N X V')
    u1 = np.cross(n, upvector)
    u1 = u1 / np.linalg.norm(u1)
    u = u1  # this is U=N X V' / abs(N X V')

    v = np.cross(u, n)  # this is V= U X N

    uvn = [0]*3  # return three vector at one time
    uvn[0] = u
    uvn[1] = v
    uvn[2] = n
    return uvn

def perspective_matrix(far, near, h):  # according to slide 21  far = f  h = half of window side
    # assume the projection window(plane) is same as near clipping plane, so d = near
    per_matrix = np.zeros((4,4))  #     per_matrix =      d/h  0   0     0
    per_matrix[0][0] = near / h    #                      0   d/h  0     0
    per_matrix[1][1] = near / h    #                      0    0 f/(f-d) -df/(f-d)
    per_matrix[2][2] = far / (far - near)   #             0    0   1     0
    per_matrix[2][3] = -(far * near / (far - near))
    per_matrix[3][2] = 1

    return per_matrix

def view_transform(world_matrix, view_matrix, perspective_matrix):
    # use view_transform() to transform local points to screen coordinate
    global point_list
    point_list_copy = copy.deepcopy(point_list)
    point_list1 = []   #add 1 in the four raw to do matrix calculate
    for j in point_list_copy:
        j.append(1)
        point_list1.append(j)

    screen_co = list()  # screen coordinate
    screen_co_x = list() # x value of all points
    screen_co_y = list()  # y value of all points
    for point in point_list1:
        point = perspective_matrix @ view_matrix @ world_matrix @ point

        #we only need x,y location of screen for this project, create a coordinate
        screen_co_x.append(point[0])
        screen_co_y.append(point[1])
        screen_co.append([point[0], point[1]])
    # we need to put x, y to -1 to 1 use for device transformation
    screen_co_norm = list()
    max_x = max(screen_co_x)
    max_y = max(screen_co_y)
    min_x = min(screen_co_x)
    min_y = min(screen_co_y)
    for point in screen_co:
        point[0] = (point[0] - min_x) / (max_x - min_x)
        point[1] = (point[1] - min_y) / (max_y - min_y)
    return screen_co

def back_face_culling(world_matrix, view_matrix, perspective_matrix):  # simpler in screen space:) yeah!
    # to get the point in screen space, do the same thing in transform
    global point_list
    point_list_copy = copy.deepcopy(point_list)
    screen_co = list()
    for j in point_list_copy:
        j.append(1)
    for point in point_list_copy:
        point = perspective_matrix @ view_matrix @ world_matrix @ point
        screen_co.append(point)
    # for each polygon find two vector, use point num1 to num2 & 3
    # point num  from polygon is 1 in list is 0 make -1 to get the num in list
    culling_polygon = list()
    for polygon in polygon_list:
        #print("polygon:",polygon)
        #print(point_list_copy[polygon[2] - 1])
        vector1 = np.subtract(screen_co[polygon[2] - 1], screen_co[polygon[1] - 1])
        vector2 = np.subtract(screen_co[polygon[3] - 1], screen_co[polygon[1] - 1])
        # we have 4 dimension in matrix calculate, delete the 4
        vector1 = vector1[:-1]
        vector2 = vector2[:-1]

        #print(vector2, vector1)
        vector_n = np.cross(vector1, vector2)  # two vector . for n
        #  if NP . N >0 the  polygon is visible  N is positive Z-axis so if np.z>0, NP . N >0
        if vector_n[2] > 0:
            culling_polygon.append(polygon)

    return culling_polygon

def draw():
    global screen_co, culling_polygon, polygon_list
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    #for not culling test, use polygon_List not culling_polygon
    for polygon in culling_polygon:
        num_point = polygon[0]
        polygon = polygon[1:]
        xshift = -0.5
        yshift = -0.5  #put graph in center
        for i in range(num_point):
        # points number start by 1, in screen_co, it should be +1, so we use [num - 1]
            if i == num_point-1:  #last point line with first point
                x1 = screen_co[polygon[i] - 1][0] + xshift
                y1 = screen_co[polygon[i] - 1][1] + yshift
                x2 = screen_co[polygon[0] - 1][0] + xshift
                y2 = screen_co[polygon[0] - 1][1] + yshift
            else: #other points
                x1 = screen_co[polygon[i] - 1][0] + xshift
                y1 = screen_co[polygon[i] - 1][1] + yshift
                x2 = screen_co[polygon[i + 1] - 1][0] + xshift
                y2 = screen_co[polygon[i + 1] - 1][1] + yshift

            glBegin(GL_LINES)
            glVertex2f(x1, y1)
            glVertex2f(x2, y2)
            glEnd()
    glFlush()


if __name__ == '__main__':
    far = 100  # far clipping panel
    near = 20  # near clipping panel
    h = 10  # half of view windows' side
    c = [-30, -40, 40]  # location of camera

    read_file()
    world_matrix = world_matrix([0, 0, 0])  # assume the world coord is as same as the local coord
    uvn = cal_uvn(c, [0, 5, 0])  # location and up vector of camera
    view_matrix = view_matrix(uvn, c)
    perspective_matrix = perspective_matrix(far, near, h)
    screen_co = view_transform(world_matrix, view_matrix, perspective_matrix)
    culling_polygon = back_face_culling(world_matrix, view_matrix, perspective_matrix)
    glutInit()
    glutCreateWindow("YHproject1")
    glutDisplayFunc(draw)
    glutMainLoop()




