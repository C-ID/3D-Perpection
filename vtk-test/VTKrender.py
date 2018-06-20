import json
import argparse
import vtk
import numpy as np
import pcl
import visualizercontrol
import struct

color={
    "car":(0, 255, 0),          #green
    "pedestrian":(255, 128, 128),   #pink
    "bicycle":(0, 0, 255),     #bule
    "unknown":(0, 255, 255)    #yellow
}

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)

        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(-255, 255)
        mapper.SetScalarVisibility(1)


        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)


    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = np.random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


def load_josn(json_file):
    with open(json_file, "rb") as f:
        file = json.load(f)
    pos, scores, points, names = [], [], [], []
    for obj in file['object']:
        names.append(obj['name'])
        scores.append(obj["score"])
        for point in obj['points']:
            pos.append([point['x'], point['y'], point['z']])
        points.append(pos)
        pos = []
    return points, scores, names

def read_bin(bin_file):
    points = np.fromfile(bin_file,dtype=np.float32).reshape([-1,4])
    return points[:,:3]

def prepare_data(json_path, pcd_file):
     points, scores, names = load_josn(json_file=json_path)

     oripoints = []
     cloud = pcl.io.loadpcd(pcd_file)
     for data in cloud:
         oripoints.append(list(data)[:3])
     # oripoints = np.fromfile(pcd_file, np.float32)
     return np.array(points), np.array(scores), np.array(oripoints), np.array(names)

def generator_color(length, aim):
    color_length = np.array(aim).shape[-1]
    color = np.full((length, color_length), np.asarray(aim))
    return color

import codecs
def read_txt(binary):
    name, score, tot_obj, cloud, coord = [], [], [], [], []
    with open(binary, 'r') as f:
        data = f.readlines()
        for line in data:
            # print(data[0])
            lindata = line.strip()
            lindata.split((': '))
            if "type-name: " in lindata:
                lindata = lindata[11:]
                name.append(lindata)
            if "type-score: " in lindata:
                lindata = float(lindata[12:])
                score.append(lindata)
            elif "Total-objects-size: " in lindata:
                lindata = int(lindata[20:])
                tot_obj.append(lindata)
            elif "cloud-points size: " in lindata:
                lindata = int(lindata[19:])
                cloud.append(lindata)
            else:
                # print(lindata.split())
                # print(lindata[0], lindata[1], lindata[2])
                coord.append([float(lindata.split()[0]), float(lindata.split()[1]), float(lindata.split()[2])])
    return name, score, tot_obj, cloud, coord




def render(args):
    #names, scores, tot_objs, clouds, coords = read_txt(args.binary_path)
    #points, scores, oripoints, names = prepare_data(args.json_path, args.pcd_path)

    oripoints = read_bin(args.pcd_path)
    #points, scores = load_josn(json_file=args.json_path)

    # pointCloud = VtkPointCloud()
    # for orip in oripoints:
    #     pointCloud.addPoint(orip)
    #
    # for ps in points:
    #     for p in ps:
    #         pointCloud.addPoint(p)
    #
    #
    # renderer = vtk.vtkRenderer()
    # renderer.AddActor(pointCloud.vtkActor)
    # renderer.SetBackground(.0, .0, .0)
    # renderer.ResetCamera()
    #
    # #Render Window\
    # renderWindow = vtk.vtkRenderWindow()
    # renderWindow.AddRenderer(renderer)
    #
    # # Interactor
    # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    # renderWindowInteractor.SetRenderWindow(renderWindow)
    #
    # # Begin Interaction
    # renderWindow.Render()
    # renderWindowInteractor.Start()
    # print(oripoints.shape[0])
    # colors1 = generator_color(oripoints.shape[0], [255, 255, 255])
    # obj = pointobject.VTKObject()
    # obj.CreateFromArray(np.array(oripoints))
    # obj.AddColors(colors1)
    #
    # for po in points:
    #     colors2 = generator_color(np.asarray(po).shape[0], [0, 255, 255])
    #     obj.CreateFromArray(np.array(po))
    #     obj.AddColors(colors2)
    #
    # ren = vtk.vtkRenderer()
    # ren.AddActor(obj.GetActor())
    #
    # renWin = vtk.vtkRenderWindow()
    # renWin.AddRenderer(ren)
    #
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)
    #
    # style = vtk.vtkInteractorStyleTrackballCamera()
    # iren.SetInteractorStyle(style)
    #
    # iren.Initialize()
    # iren.Start()

    #start from this line
    vtkControl = visualizercontrol.VTKVisualizerControl()
    vtkControl.AddPointCloudActor(oripoints)
    nID = vtkControl.GetLastActorID()
    vtkControl.SetActorColor(nID, (255, 255, 255))
    # for i in range(len(points)):
    #     vtkControl.AddPointCloudActor(np.asarray(points[i]))
    #     nID = vtkControl.GetLastActorID()
    #     vtkControl.SetActorColor(nID, names[i])

    vtkControl.ResetCamera()

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(vtkControl.renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renWin.Render()
    iren.Start()

if __name__ == "__main__":
    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="Python", description="Aim at testing cnn_seg results through VTK rendering")
    parser.add_argument("--json-path", type=str, required=True, help="path for json file")
    parser.add_argument("--pcd-path", type=str, required=True, help="path for pcd file")
    # parser.add_argument("--binary-path", type=str, required=True, help="path for binary file")
    args = parser.parse_args()

    render(args)