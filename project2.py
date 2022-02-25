import numpy as np
import random
from OpenGL.GL import *
from OpenGL.GLUT import *
import copy

'''
Observer is class of cameras, usually one object
Model is class of data in txt. every object consider as one object in the image
Window is class of drawn window, one object
'''
class Vertex:  # use in view transform and z buffer
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Edge:  # to create edge table, create class of polygon edge
    def __init__(self):
            self.ymax = 0  # first part in edge table
            self.ymin = 0  # use to sort from buttom
            self.xmin = 0   # second
            self.slope = 0  # dx / dy use in edge table
            # z use in z buffer
            self.z_v1 = 0  # z of vertex 1
            self.z_v2 = 0  # z2
            self.zmin = 0
            self.zyslope = 0  # dz/dy use in z buffer
            #self.zxslope = 0  # dz/dx

class Pixel:  # pixels, used in zbuffer and scanline
    def __init__(self, x, y, z, color):
        self.x = x
        self.y = y
        self.z = z
        self.color = color

class Polygon:
    # Polygon class create in back face culling()
    # read file get a list because it copy in project 1
    # and list easy to use in other project
    def __init__(self):
        self.vertex_num = 0
        self.vertex_list = []
        self.edge_list = []
        self.color = (255, 255, 255) # rgb defalut is white
        self.edge_table = []
        self.pixel_list = []  # all pixels inside this polygon(including vertex

class Model:
    # a model read from data, create the pixel list,and use it on observer
    # read file functions based on the project one
    def __init__(self):
        self.point_list = []
        self.polygon_list = []
        self.screen_vertex = []  # in screen coordinate
        self.culling_polygon = []  # in current observer after backface-culling

    def read_file(self, s:str): # base on project 1
        point_list = list()
        polygon_list = list()
        with open(s) as f:  # read .d   D files in the same folder as main.py
            original_data = f.read()  # str
        # then clean the data
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

        self.point_list = point_list
        self.polygon_list = polygon_list

    def create_edge_table(self): # the edge table by polygon
        for p in self.culling_polygon:
            num = p.vertex_num
            horizon = list()
            horizon_y = list()
            horizon_signal = False
            for i in range(0, p.vertex_num):
                #for each polygon create a edge table
                # ymax, xmin, slope(dx/dy)
                # vertex in polygon number - 1 is the num in list
                v1 = self.screen_vertex[p.vertex_list[i]]
                if i == num - 1:
                    v2 = self.screen_vertex[p.vertex_list[0]]
                else:
                    v2 = self.screen_vertex[p.vertex_list[i + 1]]
                if int(v1.y) == int(v2.y):
                    continue
                    #horizon_signal = True
                    # add horizon_signal will take program lots of time now
                    # draw bottom horizon do not draw top horizon
                e = Edge() # save as edge and calculate the attributes
                e.ymax = int(max(v1.y, v2.y))
                e.ymin = int(min(v1.y, v2.y))
                e.xmin = v1.x if v1.y < v2.y else v2.x  # start with xmin and add slope each time
                e.slope = (v1.x - v2.x) / (v1.y - v2.y)  # dx-dystore the edge slope for coherence
                e.z_v1 = v1.z
                e.z_v2 = v2.z
                e.zmin = v1.z if v1.y < v2.y else v2.z
                e.zyslope = (v1.z - v2.z) / (v1.y - v2.y)
                # to ensure the odd-even, do not draw y max pixel
                e.ymax -= 1
                # add one edge in polygon
                if horizon_signal:
                    horizon.append(e)
                    horizon_y.append(e.ymax)
                else:
                    p.edge_table.append(e)
            '''
            for i in horizon:
                if i.ymax == min(horizon_y):  # choose the bottom horizon edge
                    p.edge_table.append(e)
            '''
            # list.sort() need a function to sort
            def get_ymin(edge):
                return edge.ymin
            p.edge_table.sort(key=get_ymin)

    def scan_conversion(self):  # scan with AET for each polygon, save in Polygon.pixel_list
        for p in self.culling_polygon:
            aet = []  # Active edge table, update every scanline
            p.color =  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if not p.edge_table:
                continue
            # get the range of scanline
            down, top = 0, 0
            for i in p.edge_table:
                if int(i.ymin) < down: # use int() consistent in x,y
                    down = int(i.ymin)
                if int(i.ymax) > top:
                    top = int(i.ymax)

            for scanY in range(down, top + 1):  # scanline from down to top
                for e in p.edge_table:
                    if e.ymin == scanY:  # move edge to AET
                        aet.append(e)

                for i in range(len(aet) // 2): # odd in ever step
                    za = aet[i].zmin
                    zb = aet[i + 1].zmin
                    xb = int(aet[i + 1].xmin)
                    xa = int(aet[i].xmin)
                    for j in range(xa, xb): # j = x
                        # z buffer first cal vertex's z for the scanline
                        z = zb - (zb - za) * (xb - j) / (xb - xa)

                        # for each intersections between scanline and edge
                        # store all pixels coordinate between them into a pixel list
                        pixel = Pixel(j, scanY, z, p.color)
                        p.pixel_list.append(pixel)

                for e in aet:
                    if e.ymax == scanY:  # remove edges that no longer intersect with the next scanline
                        aet.remove(e)
                for e in aet:
                    e.xmin += e.slope  # update x and zmin
                    e.zmin += e.zyslope

                def x_cmp(edge): # use for sort
                    return edge.xmin
                aet.sort(key=x_cmp)  # re-sort aet by X value



class Observer:
    # Observer class include the camera and its
    # Coordinate system transformation
    # based on project one, set every input as a self.attribute
    # set every return as a self.attribute
    # read data in class model
    def __init__(self):
        self.local = []  # from local coord to world coord
        self.camera = []  # the camera look direction
        self.up_vector = []  # up vector
        #self.p_ref = []
        self.near = 0  # near clipping panel
        self.far = 0  # far clipping panel
        self.h = 0  # half of view windows' side
        # U V N in view matrix
        self.u = []
        self.v = []
        self.n = []

        # three matrix and their matrix multiple result
        self.world_matrix = np.identity(4)
        self.view_matrix = np.zeros(4)
        self.per_matrix = np.zeros(4)
        #self.final_matrix = np.zeros(4)

    def get_world_matrix(self):  # from local coord to world coord
        l = self.local
        matrix = np.identity(4)
        # set vector from local zero piont to world  l
        matrix[0][3] = l[0]
        matrix[1][3] = l[1]
        matrix[2][3] = l[2]
        self.world_matrix = matrix

    def cal_uvn(self):  # C is c V' is upvector
        c, upvector = self.up_vector, self.camera
        n = np.zeros(3)  # N= P-C/abs(P-C)
        n = np.subtract(n, c)
        n = n / np.linalg.norm(n)  # default norm = 2 this is N= P-C/abs(P-C)

        u = np.zeros(3)  # U=N X V' / abs(N X V')
        u1 = np.cross(n, upvector)
        u1 = u1 / np.linalg.norm(u1)
        u = u1  # this is U=N X V' / abs(N X V')

        v = np.cross(u, n)  # this is V= U X N

        self.u = u
        self.v = v
        self.n = n

    def get_perspective_matrix(self):  # according to slide 21  far = f  h = half of window side
        # assume the projection window(plane) is same as near clipping plane, so d = near
        far, near, h = self.far, self.near, self.h
        per_matrix = np.zeros((4, 4))  # per_matrix =      d/h  0   0     0
        per_matrix[0][0] = near / h  # 0   d/h  0     0
        per_matrix[1][1] = near / h  # 0    0 f/(f-d) -df/(f-d)
        per_matrix[2][2] = far / (far - near)  # 0    0   1     0
        per_matrix[2][3] = -(far * near / (far - near))
        per_matrix[3][2] = 1

        self.per_matrix = per_matrix

    def get_view_matrix(self):  # view_matrix = R.T
        u, v, n = self.u, self.v, self.n
        c = self.camera
        r = np.identity(4)  # r matrix = Ux, Uy, Uz, 0
        r[0][0] = u[0]  #                Vx, Vy, Vz, 0
        r[0][1] = u[1]  #                Nx, Ny, Nz, 0
        r[0][2] = u[2]  #                0,  0,  0,  1
        r[1][0] = v[0]
        r[1][1] = v[1]
        r[1][2] = v[2]
        r[2][0] = n[0]
        r[2][1] = n[1]
        r[2][2] = n[2]

        t = np.identity(4)  # t matrix = 1, 0, 0, -cx
        for i in range(0, 3):  #         0, 1, 0, -cy
            t[i][3] = -c[i]  #           0, 0, 1, -cz
        # print("T:", t)       #         0, 0, 0, 1

        view_matrix = r @ t  # this is view_matrix = R.T
        self.view_matrix = view_matrix

class Window:
    def __init__(self):
        self.zbuffer = dict()  #  pixel object dict {(x,y) : z}
        self.ibuffer = list()  #  pixel object list [Pixel, Pixel]

    def add_model(self, model: Model):  # add all pixel in polygons in model
        for p in model.culling_polygon:
            for pixel in p.pixel_list:
                if f"{pixel.x}, {pixel.y}" not in self.zbuffer:  # first time the x,y
                    self.ibuffer.append(pixel)
                    self.zbuffer[f"{pixel.x}, {pixel.y}"] = pixel.z
                else:
                    if self.zbuffer[f"{pixel.x}, {pixel.y}"] > pixel.z:  # front of current pixel
                        self.zbuffer[f"{pixel.x}, {pixel.y}"] = pixel.z
                        for i in self.ibuffer:
                            if i.x == pixel.x and i.y == pixel.y:
                                i.z = pixel.z
                                i.color = pixel.color
                                break
    def draw(self):  # based on project one use opengl
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        # because the opengl need put x,y into -1 to 1
        # normalize data first
        x_list = list()  # list of x used for normalize
        y_list = list()  # list of y used for normalize
        for i in self.ibuffer:
            x_list.append(i.x)
            y_list.append(i.y)
        max_x = max(x_list)
        max_y = max(y_list)
        min_x = min(x_list)
        min_y = min(y_list)
        xshift = -0.5  #put graph in center
        yshift = -0.5  #put graph in center

        for i in self.ibuffer:  # norm x,y
            i.x = (i.x - min_x) / (max_x - min_x) + xshift
            i.y = (i.y - min_y) / (max_y - min_y) + yshift

        glBegin(GL_POINTS)
        for i in self.ibuffer:  # draw pixel and color
            glColor3ubv(i.color)
            glVertex2f(i.x, i.y)
        glEnd()
        glFlush()


    def drawwindow(self):  # OpenGL window function

        glutInit()
        glutCreateWindow("YHproject2")
        glutDisplayFunc(self.draw)
        glutMainLoop()



def view_transform(model: Model, observer: Observer, times: int):
    # use view_transform() to transform local points to screen coordinate
    # because we use int() to round x,y in scan convertion
    # to avoid there are only a few of pixels
    # use times(int) to mutiple x,y coordinate, make it bigger range, it won't influence the shape
    world_matrix, view_matrix = observer.world_matrix, observer.view_matrix
    perspective_matrix = observer.per_matrix
    point_list = model.point_list
    point_list_copy = copy.deepcopy(point_list)
    point_list1 = []   #add 1 in the four raw to do matrix calculate
    for j in point_list_copy:
        j.append(1)
        point_list1.append(j)

    screen_co = list()  # screen coordinate contain Vertex object
    # the Vertex num in data start 1 at list start at 0
    screen_co.append(Vertex(0, 0, 0))  # add a null Vertex
    #screen_co_x = list() # x value of all points
    #screen_co_y = list()  # y value of all points
    #screen_co_z = list()
    for point in point_list1:
        point = perspective_matrix @ view_matrix @ world_matrix @ point
        screen_co.append(Vertex(point[0]*times, point[1]*times, point[2]))
    model.screen_vertex = screen_co

def back_face_culling(model: Model, observer: Observer):  # return list of Polygon object
    # we do back face culling then do the scan conversion
    # to get the point in screen space, do the same thing in transform
    point_list = model.point_list
    world_matrix = observer.world_matrix
    view_matrix = observer.view_matrix
    perspective_matrix = observer.per_matrix
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
    for polygon in model.polygon_list:
        vector1 = np.subtract(screen_co[polygon[2] - 1], screen_co[polygon[1] - 1])
        vector2 = np.subtract(screen_co[polygon[3] - 1], screen_co[polygon[1] - 1])
        # we have 4 dimension in matrix calculate, delete the 4
        vector1 = vector1[:-1]
        vector2 = vector2[:-1]

        #print(vector2, vector1)
        vector_n = np.cross(vector1, vector2)  # two vector . for n
        #  if NP . N >0 the  polygon is visible  N is positive Z-axis so if np.z>0, NP . N >0
        if vector_n[2] > 0:
            p = Polygon()
            p.vertex_num = polygon[0]
            for i in range(1,p.vertex_num+1):
                p.vertex_list.append(polygon[i])
            model.culling_polygon.append(p)


if __name__ == "__main__":
    # create models from file
    m1 = Model()
    m1.read_file("./D files/knight.d.txt")
    m2 = Model()
    m2.read_file("./D files/queen.d.txt")
    m3 = Model()
    m3.read_file("./D files/pawn.d.txt")
    # we only have one observer as camera, set it as project1
    observer = Observer()
    observer.far = 100  # far clipping panel
    observer.near = 20  # near clipping panel
    observer.h = 10  # half of view windows' side
    observer.camera = [30, -40, 40]  # location of camera
    observer.up_vector = [0, 5, 0] # up vector
    observer.local = [0, 0, 0]  # local coordinate
    observer.get_world_matrix()
    observer.cal_uvn()
    observer.get_view_matrix()
    observer.get_perspective_matrix()

    view_transform(m1, observer, 40)
    back_face_culling(m1, observer)
    m1.create_edge_table()
    m1.scan_conversion()
    view_transform(m2, observer, 40)
    back_face_culling(m2, observer)
    m2.create_edge_table()
    m2.scan_conversion()
    '''
    view_transform(m3, observer)
    back_face_culling(m3, observer)
    m3.create_edge_table()
    m3.scan_conversion()
    '''
    w = Window()
    w.add_model(m2)
    w.add_model(m1)
    #w.add_model(m3)
    w.drawwindow()
