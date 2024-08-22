from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file, jsonify
import os
import numpy as np
import plotly.graph_objects as go
import math
import zipfile
import io
from datetime import datetime
from werkzeug.utils import secure_filename
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomLib import GeomLib_Tool
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_ProjectPointOnSurf
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Face
from OCC.Core.Geom import Geom_Curve, Geom_Plane
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.gp import gp_Pnt, gp_Pln, gp_Ax3, gp_Dir, gp_Vec, gp_XYZ
from OCC.Core.TopExp import topexp
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge, shapeanalysis
from OCC.Core.StlAPI import StlAPI_Writer
# from OCC.Core.TopExp import TopExp
# from OCC.Core.GCPnts import GCPnts_AbscissaPoint

app = Flask(__name__)
#initializing all the directories
UPLOAD_FOLDER='uploads'
ALLOWED_EXTENSIONS = {'step', 'stp'}
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

#specifying the desired file name
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
date_time=str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
#trigering index.html file to open the landing page
@app.route('/')
def index():
    return render_template('landing.html')

#initializing user email for creating folder
email=str()
name=str()
spir_folder='Spiral_'+date_time
cont_folder='Contour_'+date_time
@app.route('/return')
def back():
    message='Welcome '+name
    return render_template('index.html', message=message)

@app.route('/signin')
def log():
    return render_template('login.html')

sspiral=np.empty((2,3))
scontour=np.empty((2,3))
#trigering the view.html file to show the results
@app.route('/upload', methods=['POST'])
def upload_file():
    global email
    #checking if the chosen file is desired format
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    #if chosen file is in desired format then upload it in the designated folder
    if file and allowed_file(file.filename):
        filename = email[:email.index('@')]
        step_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(step_path)
        
        #taking inputs from user
        TD1=request.form['tool_dia']
        Feed=request.form['feedrate']
        cnc=request.form['cnc']
        
        #extracting the shape from the uploaded .step file
        shape = load_step(step_path)
        if shape is None:
            return "Error: Failed to load or read STEP file", 400
        #converting it to .stl
        stl_filename = convert_step_to_stl(step_path, shape)
        #calling all the functions
        process_step_file(step_path)
        gen_toolpath(f_N1, f_N2, TD1, Feed, cnc, 'contourSPIF_', cont_folder)
        gen_toolpath(f_N4, f_N5, TD1, Feed, cnc, 'spiralSPIF_', spir_folder)
        
        global scontour
        global sspiral
        
        #creating html files to store the slice and spiral plots to display
        h1="C:/Rnd_ISF_Slicer/IncrementalForming/static/pnt.html"
        h2="C:/Rnd_ISF_Slicer/IncrementalForming/static/spnt.html"
        #ploting slice points
        scontour=plot(f_N1, h1, 'Contour Trajectory')
        #ploting spiral points
        sspiral=plot(f_N4, h2, 'Spiral Trajectory')
        
        return redirect(url_for('view_file', filename=stl_filename))
    else:
        message2='Welcome '+name
        return render_template('index.html', message1='Please choose a .STEP or .STP file', message=message2)

@app.route('/create-folder/<folder_name>')
def create_subfolder(folder_name):
    base_dir = app.config['USER_FOLDER']
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    return folder_name

@app.route('/loginpage')
def login_page():
    return render_template('login.html')

@app.route('/createaccount')
def signin_page():
    return render_template('signup.html')
USER_FOLDER=str()
with open('database.txt', 'r') as file:
    for line in file:
        arr1=line.replace("'","").replace(",","").strip().split()
        email=arr1[1]


@app.route('/signup', methods=['POST'])
def signup():
    global USER_FOLDER, name, email
    name=request.form['Name']
    email=request.form['email']
    password=request.form['password1']
    cnfpassword=request.form['password2']
    if password==cnfpassword and len(password)>8:
        l=[]
        l.append(name)
        l.append(email)
        l.append(password)
        with open('database.txt', 'r') as file:
            flag=True
            for line in file:
                arr=line.replace("'","").replace(",","").strip().split()
                if arr[1]==email:
                    flag=False
        if flag==True:
            d=open('database.txt', 'a')
            d.write(str(l).replace('[','').replace(']','')+'\n')
            USER_FOLDER=email
            app.config['USER_FOLDER']=USER_FOLDER
            if not os.path.exists(USER_FOLDER):
                os.makedirs(USER_FOLDER)
            create_subfolder(spir_folder)
            create_subfolder(cont_folder)
            message='Welcome '+name
        else:
            return render_template('signup.html', message="Email Already registered")
    else:
        return render_template('signup.html', message="Recheck your password, length should be more than 8 and confirm password should be same as password")
    return render_template('index.html', message=message)

@app.route('/signin', methods=['POST'])
def signin():
    global name, email, USER_FOLDER
    email=request.form['email']
    password=request.form['password']
    
    with open('database.txt', 'r') as file:
        flag=False
        for line in file:
            arr=line.replace("'","").replace(",","").strip().split()
            if arr[1]==email:
                if arr[2]==password:
                    flag=True
                    email=arr[1]
                    name=arr[0]
                    USER_FOLDER=email
                    app.config['USER_FOLDER']=USER_FOLDER
                    if not os.path.exists(USER_FOLDER):
                        os.makedirs(USER_FOLDER)
                    create_subfolder(cont_folder)
                    create_subfolder(spir_folder)
                    message='Welcome '+name
                    return render_template('index.html', message=message)
                else:
                    message='Wrong Password, try again'
                    return render_template('login.html', message=message)
        if flag==False:
            message='User not found' 
            return render_template('login.html', message=message)
        
#loading the .step file uploded by user
def load_step(file_path):
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        if status == 1:
            reader.TransferRoots()
            shape = reader.Shape()
            return shape
        else:
            return None
    except Exception as e:
        print(f"Error reading .step file: {e}")
        return None

#converting to stl file(required for display only)
def convert_step_to_stl(step_path, shape):
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()
    stl_writer = StlAPI_Writer()
    stl_filename = os.path.splitext(step_path)[0] + '.stl'
    stl_writer.Write(shape, stl_filename)
    return os.path.basename(stl_filename)

#obtaining first edge
def stPnt(st_pnt, edges):
    #specifying the minimum distance between the points
    min_dist=100000
    p=0
    for j in range(len(edges)):
        #first point of an edge
        v1=TopoDS_Vertex()
        #last point of an edge
        v2=TopoDS_Vertex()
        b=bool()
        topexp.Vertices(edges[j], v1, v2, b)
        pnt_v1=BRep_Tool.Pnt(v1)
        #checking if the distance between first point of an edge in less than minimum distance
        dist=pnt_v1.Distance(st_pnt)
        if dist<min_dist:
            min_dist=dist
            p=j
    #swaping the edge to get the first edge having minimum distance from reference point
    temp1=edges[p]
    edges[p]=edges[0]
    edges[0]=temp1
    slice_edges=[]
    #iterating over a list of edges to store them in order
    for k in range(len(edges)):
        slice_edges.append(edges[k])
    #returning the list of edges with correct first edge
    return slice_edges

#ordering the edges in correct orientation
def orderedEdges(slice_edges):
    #storing ordered edges
    ord_edges=[]
    ord_edges.append(slice_edges[0])
    edg_rev=False
    #iterating over the list of unordered edges
    for l in range(len(slice_edges)):
        #first vertex
        v1=TopoDS_Vertex()
        #last vertex
        v2=TopoDS_Vertex()
        b=bool()
        topexp.Vertices(slice_edges[l], v1, v2, b)
        if edg_rev==True:
            lpnt=BRep_Tool.Pnt(v1)
        else:
            lpnt=BRep_Tool.Pnt(v2)
        for m in range(l+1,len(slice_edges)):
            mv1=TopoDS_Vertex()
            mv2=TopoDS_Vertex()
            mb=bool()
            topexp.Vertices(slice_edges[m], mv1, mv2, mb)
            pnt_mv1=BRep_Tool.Pnt(mv1)
            pnt_mv2=BRep_Tool.Pnt(mv2)
            #checking if first point of current edge is equal to the last point of the last edges
            if lpnt.IsEqual(pnt_mv1, 1e-3):
                #swaping edges to get the connected edges together
                temp=slice_edges[l+1]
                slice_edges[l+1]=slice_edges[m]
                slice_edges[m]=temp
                ord_edges.append(slice_edges[l+1])
                edg_rev=False
                break
            #checking if last point of current edge is equal to last point of last edge
            elif lpnt.IsEqual(pnt_mv2, 1e-3):
                #swaping the points to get the connected edge together
                temp=slice_edges[l+1]
                slice_edges[l+1]=slice_edges[m]
                slice_edges[m]=temp
                ord_edges.append(slice_edges[l+1])
                edg_rev=True
                break          
    #retuing the list of ordered edges on the single slice
    return ord_edges

#ordering the loops of edges obtained from different slicings
def orientedEdges(ord_edges):
    global z_dir
    #storing first vertex of every edge
    vFirstVertices = []
    #iterating over the loop of ordered edges
    for iEdge in ord_edges:
        #first vertex
        v1=TopoDS_Vertex()
        #last vertex
        v2=TopoDS_Vertex()
        b=bool()
        topexp.Vertices(iEdge, v1, v2, b)
        ori_v1=BRep_Tool.Pnt(v1)
        vFirstVertices.append(ori_v1)
    vFirstVertices.append(vFirstVertices[0])
    #initializing loop area
    Area = 0
    pnt0 = gp_Pnt(0, 0, 0)
    #iterating over the list of first vertices
    for o in range(1, len(vFirstVertices)):
        pnt1 = (vFirstVertices[o - 1])
        pnt2 = (vFirstVertices[o])
        vec1 = gp_Vec(pnt1, pnt0)
        vec2 = gp_Vec(pnt2, pnt0)
        vec_Area = vec1.Crossed(vec2)
        Area += vec_Area.Z()
    #storing oriended edges based on loop orientation
    oriented_edges=[]
    if Area * z_dir.Z() < 0:
        for ritrEdge in reversed(ord_edges):
            rev=topods.Edge(TopoDS_Shape.Reversed(ritrEdge))
            oriented_edges.append(rev)           
    else:
        for itrEdge in (ord_edges):
            oriented_edges.append(itrEdge)
    #reurning list of oriented edges on a single slice
    return oriented_edges

#generateds orderd points
def pointGen(edge, ref_pnt):
    #storing ordered points on latest sliced edge
    pnts=[]
    #storing ordered points on latest sliced edge as gp_Pnt
    pnts_gp=[]
    edge_adaptor = BRepAdaptor_Curve(edge)
    abscissa = GCPnts_UniformAbscissa(edge_adaptor, 20)

    #stores unordered points on latest sliced edge
    points=[]
    #stores unordered points on latest sliced edge as gp_Pnt
    points_gp=[]
    num_points = 20
    #iterating over the parameter to get the desired number of points 
    for i in range(1, num_points + 1):
        param = abscissa.Parameter(i)
        point = edge_adaptor.Value(param)
        edge_points = (round(point.X(),1), round(point.Y(),1), round(point.Z(),1))
        points.append(edge_points)
        points_gp.append(point)
    #first point on edge
    v1=(points[0])
    #last point on edge
    v2=(points[len(points)-1])
    d1=math.dist(ref_pnt, v1)
    d2=math.dist(ref_pnt, v2)
    if d1>d2:
        #iterating over the reversed loop of unordered points
        for pnt in reversed(points):
            pnts.append(pnt)
        #iterating over the reversed loop of unordered gp_Pnt        
        for gp_pnts in reversed(points_gp):
            pnts_gp.append(gp_pnts)
    else:
        for pnt in points:
            pnts.append(pnt)
                
        for gp_pnts in points_gp:
            pnts_gp.append(gp_pnts)           
    #returning list of orderded points and gp_Pnt   
    return pnts, pnts_gp

# def generate_points(edge, ref_point):
#     # Get the first and last vertices of the edge
#     first_vertex = TopExp.FirstVertex(edge)
#     last_vertex = TopExp.LastVertex(edge)
    
#     # Convert vertices to points
#     pntFVertex = BRep_Tool.Pnt(first_vertex)
#     pntLVertex = BRep_Tool.Pnt(last_vertex)
    
#     # Create an adaptor curve from the edge
#     curve = BRepAdaptor_Curve(edge)
    
#     # Get the first and last parameters of the curve
#     first_param = curve.FirstParameter()
#     last_param = curve.LastParameter()
    
#     # Calculate the length of the curve
#     curve_length = GCPnts_AbscissaPoint.Length(curve)
    
#     # Set the number of points to generate
#     n_pnts = 300
#     dp = (last_param - first_param) / n_pnts
    
#     # Initialize a list to store the contour points
#     contour_points = []
    
#     # Determine the direction to generate points based on the reference point
#     if ref_point.Distance(pntFVertex) < ref_point.Distance(pntLVertex):
#         p = first_param + dp
#         while p <= last_param - dp:
#             point = curve.Value(p)
#             contour_points.append(point)
#             p += dp
#     else:
#         p = last_param - dp
#         while p >= first_param + dp:
#             point = curve.Value(p)
#             contour_points.append(point)
#             p -= dp
    
#     # Return the length of the curve and the list of contour points
#     return curve_length, contour_points

#generating normals on ordered points
def normalGen(gSurface, pnts_gp):
    #storing normals of generated points
    normals=[]
    nor_gp=[]
    #iterating over the points on current edge
    for npnt in pnts_gp:
            b, u, v=GeomLib_Tool.Parameters(gSurface, gp_Pnt(npnt.X(), npnt.Y(), npnt.Z()), 1)
            if b==True:
                evaluator=GeomLProp_SLProps(gSurface, u, v, 2, 1e-6)
                normal=evaluator.Normal()
                if normal.XYZ()*(z_dir.XYZ())<0:
                    #reverses the direction of normal
                    normal=-normal
                normal_vec=gp_Vec(normal)
                nor_gp.append(normal_vec)
                normal_points=(normal_vec.X(), normal_vec.Y(), normal_vec.Z())
                normals.append(normal_points)
    #returning list of normals for ordered points       
    return normals, nor_gp

def spiralGen(pnt_slice, prevpnt_slice):
    #storing points on current slice 
    triplepnt=[]
    #storing points on previous slice
    prevtriplepnt=[]
    #storing points on current spiral
    spiralPnts=[]
    #iterating over the list of points on current sliced edge
    for i in range(len(pnt_slice)):
        X=pnt_slice[i].X()
        Y=pnt_slice[i].Y()
        Z=pnt_slice[i].Z()
        triplepnt.append(gp_XYZ(X, Y, Z))
    #iterating over the list of points on previous sliced edge
    for i in range(len(prevpnt_slice)):
        X=prevpnt_slice[i].X()
        Y=prevpnt_slice[i].Y()
        Z=prevpnt_slice[i].Z()
        prevtriplepnt.append(gp_XYZ(X, Y, Z))
    #generating denominator for slice
    deno=0
    for cnt in range(len(pnt_slice)-1):
        pntcurrent=(triplepnt[cnt])
        pntnext=(triplepnt[cnt+1])
        subd=pntcurrent.Subtracted(pntnext)
        deno=deno+subd.Modulus()
    #generating numerator for slice
    num=0
    for cnt in range(len(pnt_slice)):
        if cnt<len(pnt_slice)-1:
            pntcurrent=(triplepnt[cnt])
            pntnext=(triplepnt[cnt+1])
            subd=pntcurrent.Subtracted(pntnext)
            num=num+subd.Modulus()
        s=num/deno
        if len(prevpnt_slice)==len(pnt_slice):
            pntfirst=(triplepnt[cnt]).Multiplied(s)
            pntsecond=(prevtriplepnt[cnt]).Multiplied(1-s)
            sp_pnt=pntfirst.Added(pntsecond)
            spiralPnts.append(sp_pnt)
    #if len(spiralPnts)!=0:
           #spiralPnts.pop()
        
    return spiralPnts

def spnorm(slice_norm, prevslice_norm):
    #storing points on current slice 
    triplepnt=[]
    #storing points on previous slice
    prevtriplepnt=[]
    #storing points on current spiral
    spiralPnts=[]
    #iterating over the list of points on current sliced edge
    for i in range(len(slice_norm)):
        X=slice_norm[i].X()
        Y=slice_norm[i].Y()
        Z=slice_norm[i].Z()
        triplepnt.append(gp_XYZ(X, Y, Z))
    #iterating over the list of points on previous sliced edge
    for i in range(len(prevslice_norm)):
        X=prevslice_norm[i].X()
        Y=prevslice_norm[i].Y()
        Z=prevslice_norm[i].Z()
        prevtriplepnt.append(gp_XYZ(X, Y, Z))
    #generating denominator for slice
    deno=0
    for cnt in range(len(slice_norm)-1):
        pntcurrent=(triplepnt[cnt])
        pntnext=(triplepnt[cnt+1])
        subd=pntcurrent.Subtracted(pntnext)
        deno=deno+subd.Modulus()
    #generating numerator for slice
    num=0
    for cnt in range(len(slice_norm)):
        if cnt<len(slice_norm)-1:
            pntcurrent=(triplepnt[cnt])
            pntnext=(triplepnt[cnt+1])
            subd=pntcurrent.Subtracted(pntnext)
            num=num+subd.Modulus()
        s=num/deno
        if len(prevslice_norm)==len(slice_norm):
            pntfirst=(triplepnt[cnt]).Multiplied(s)
            pntsecond=(prevtriplepnt[cnt]).Multiplied(1-s)
            sp_pnt=pntfirst.Added(pntsecond)
            spiralPnts.append(sp_pnt)
    #if len(spiralPnts)!=0:
           #spiralPnts.pop()
        
    return spiralPnts

#address of file storing toolpath
f_N3=''
def gen_toolpath(f_N1, f_N2, TD1, Feed, cnc, gen_type, folder):
    global spir_folder, cont_folder
    #tool diameter (taken as a input from user)
    TD1=float(TD1)
    #feedrate (taken as a input from user)
    Feed=int(Feed)
    #tool radius
    R1=TD1/2
    #removing the unnecessary characters from the the files storing points and normal
    #inorder to perform required calcultions 
    def clean_and_loadtxt(file_path):
        cleaned_data = []
        with open(file_path, 'r') as file:
            for line in file:
                #Replacing comas with blank
                cleaned_line = line.replace(',', ' ').strip().split()
                cleaned_row = []
                for value in cleaned_line:
                    try:
                        cleaned_row.append(float(value))
                    except ValueError:
                        print(f"Warning: Skipping invalid value '{value}'.")
                if cleaned_row:
                    cleaned_data.append(cleaned_row)
        #returing a numpy array
        return np.array(cleaned_data)
    #loading points to perform calculations
    C=clean_and_loadtxt(f_N1)
    #loading normal to perform calculations
    nC=clean_and_loadtxt(f_N2)
    #checking if they have same number of data
    if C.shape != nC.shape:
        raise ValueError("S and nS must have the same shape")
    
    TCS = C+nC*R1

    TTS = TCS.copy()

    TTS[:, 2] = TCS[:, 2]-R1

    L = TTS.shape[0]

    LNO = 4
    
    file_name=gen_type+".mpf"
    #file path for stotring toolpath(change the address to the address of your pc while locally hosting)
    file_path=f"C:/Rnd_ISF_Slicer/IncrementalForming/{email}/{folder}/"+file_name
    global f_N3
    f_N3=file_path

    with open(file_path, 'w') as fid:
        #storing gcodes in the file in the format accepted by fanuc type controllers
        if cnc=='Fanuc':
            fid.write('N1 G54 F2500;\n')
            fid.write('N2 G00 Z50;\n')
            fid.write('N3 G64;\n')
            fid.write(f'N4  G01   X{TTS[0, 0]:5.5f}   Y{TTS[0, 1]:5.5f}   F{Feed:5.5f};\n')

            for i in range(L):
                fid.write(f'N{LNO + i + 1}  G01   X{TTS[i, 0]:5.5f}   Y{TTS[i, 1]:5.5f}   Z{TTS[i, 2]:5.5f};\n')

            fid.write(f'N{LNO + L + 1}  G01  Z50.00000;\n')
            fid.write(f'N{LNO + L + 2}  M30;\n')
        #storing gcodes in the file in the format accepted by siemens controllers
        elif cnc=='Siemens':
            fid.write('N1 G54 F2500\n')
            fid.write('N2 G00 Z=50\n')
            fid.write('N3 G64\n')
            fid.write(f'N4  G01   X={TTS[0, 0]:5.5f}   Y={TTS[0, 1]:5.5f}   F{Feed:5.5f}\n')

            for i in range(L):
                fid.write(f'N{LNO + i + 1}  G01   X={TTS[i, 0]:5.5f}   Y={TTS[i, 1]:5.5f}   Z={TTS[i, 2]:5.5f}  F{Feed:5.5f}\n')

            fid.write(f'N{LNO + L + 1}  G01  Z=50.00000  F{Feed:5.5f}\n')
            fid.write(f'N{LNO + L + 2}  M30\n')

#generating plots of sliced and spiral geomerty
def plot(txt, html, plotTitle):
    def clean_line(line):
        #Removing any unwanted characters, like commas
        clean_line = line.replace(',', '').strip()
        #Spliting the cleaned line into individual string numbers
        string_numbers = clean_line.split()
        #Converting the string numbers to floats
        float_numbers = [float(num) for num in string_numbers]
        return float_numbers
    
    
    #Loading the data from the file
    file_path = txt
    data1 = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                #Cleaning and converting each line to a list of floats
                data1.append(clean_line(line))
            except ValueError as e:
                print(f"Error converting line to floats: {line}")
                print(f"Error message: {e}")

    #Converting the list of lists to a NumPy array if it's not empty
    if data1:
        s = np.array(data1)
        #Extracting x, y, z coordinates
        x = s[:, 0]
        y = s[:, 1]
        z = s[:, 2]
        #Createing a 3D plot using Plotly
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', marker=dict(size=2))])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ), title= plotTitle
        )
        #Saving the plot as an HTML file
        output_file = html
        fig.write_html(output_file)
        print(f"3D plot saved to {output_file}")
    else:
        print("No data to plot.")

    return s

#simulating the toolpath
def simulate(s, html1, plotTitle):
    frames = []
    for i in range(1, len(s) + 1):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=s[:i, 0],
                    y=s[:i, 1],
                    z=s[:i, 2],
                    mode='lines',
                    line=dict(color='blue')
                )
            ],
            name=f'frame{i}'
        )
        frames.append(frame)

    #Creating Plotly figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[s[0, 0]],
                y=[s[0, 1]],
                z=[s[0, 2]],
                mode='lines',
                #marker=dict(size=5, color='blue'),
                line=dict(color='blue')
            )
        ],
        layout=go.Layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title=plotTitle,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 1, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}, 'fromcurrent': True}]
                }, {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                }]
            }]
        ),
        frames=frames
    )
    #storing plot as html file
    output_file = html1
    fig.write_html(output_file)


#address of files storing points
f_N1=''
#address of files storing normals
f_N2=''
#address of files storing spiral points
f_N4=''
#address of files storing spiral normals
f_N5=''
#reference point for generating points to check whether any repetative points are there
ref_pnt1=[-1000,0,0]
#referance points for generating point to be stored in a file
ref_pnt2=[-1000,0,0]
#reference point to obtain first edge
st_pnt=gp_Pnt(-1000, 0, 0)
#direction vector
z_dir=gp_Dir(0,0,1)
#main function (calling all the function here)
def process_step_file(step_path):
    #calling global variables
    global spir_folder, cont_folder
    global z_dir
    global st_pnt
    global ref_pnt1
    global ref_pnt2
    
    
    file_name1="pntContour.txt"
    #file path for stotring points(change the address to the address of your pc while locally hosting)
    file_path1=f"C:/Rnd_ISF_Slicer/IncrementalForming/{email}/{cont_folder}/"+file_name1
    global f_N1
    f_N1=file_path1
    #Opens a file to store points  
    f1=open(file_path1,"w")

    file_name2="nContour.txt"
    #file path for stotring normal(change the address to the address of your pc while locally hosting)
    file_path2=f"C:/Rnd_ISF_Slicer/IncrementalForming/{email}/{cont_folder}/"+file_name2
    global f_N2
    f_N2=file_path2
    #Opens a file to store normals
    f2=open(file_path2,"w")

    file_name4="pntspiral.txt"
    #file path for stotring spiral points(change the address to the address of your pc while locally hosting)
    file_path4=f"C:/Rnd_ISF_Slicer/IncrementalForming/{email}/{spir_folder}/"+file_name4
    global f_N4
    f_N4=file_path4
    #opens a file to store spiral points
    f4=open(file_path4,'w')

    file_name5="nspiral.txt"
    #file path for stotring spiral normal(change the address to the address of your pc while locally hosting)
    file_path5=f"C:/Rnd_ISF_Slicer/IncrementalForming/{email}/{spir_folder}/"+file_name5
    global f_N5
    f_N5=file_path5
    #opens a file to store spiral normals
    f5=open(file_path5,'w')
    
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(step_path)
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    #creating a box around the geometry to get the height
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    z_min, z_max = box.CornerMin().Z(), box.CornerMax().Z()
    l=z_max-z_min
    dz=1
    n=int(l/dz)
    
    #storing all points on slice, normals and points on spiral
    all_pnt=[]
    all_normal=[]
    all_spiralpnt=[]
    all_spnormal=[]

    #storing points on current slice
    pnt_slice=[]
    #storing points on previous slice
    prevpnt_slice=[]
    norm_slice=[]
    prevnorm_slice=[]
    #slicing with the incremental depth of dz
    for i in range(1,n):
        if i==1:
            z=z_max-0.1
        else:
            z=z-dz
        #defining slicing plane
        plane = gp_Pln(gp_Ax3(gp_Pnt(0, 0, z), z_dir))
        section = BRepAlgoAPI_Section(shape, plane, True)
        section.ComputePCurveOn1(True)
        section.Approximation(True)
        section.Build()
        if not section.IsDone():
            return 'Slicing failed'
    
        exp = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
        #stors edges as list
        edges=[]
        while exp.More():
            edge = topods.Edge(exp.Current())
            edges.append(edge)
            exp.Next()

        #getting first edge
        slice_edges=stPnt(st_pnt, edges)
        #getting edges ordered
        ordered_edges=orderedEdges(slice_edges)
        #getting edges oriented to have same direction of loop area
        oriented_edges=orientedEdges(ordered_edges)
        norm_slice.clear()
        #storing list of points on edges on a single slice
        currpnt=[]
        for edge in oriented_edges:
            pnts, pnts_gp = pointGen(edge, ref_pnt1)
            ref_pnt1=pnts[len(pnts)-1]
            currpnt.append(pnts)
        #storing the repeated edges
        pedge=[]
        #iterating over the list of lists of points on single edge
        for l in range(len(currpnt)):
            for m in range(l+1, len(currpnt)):
                #iterating over the list of points on single edge
                for lp in range(1,len(currpnt[0])-1):
                    #checking if any two lists has same points
                    if currpnt[l][lp]==currpnt[m][len(currpnt[0])-1-lp] or currpnt[l][lp]==currpnt[m][lp]:
                        f=0
                        for ie in pedge:
                            if ie==m:
                                f=1
                        if f==0:
                            #appending the repeated edge 
                            pedge.append(m)              
        #iterating the repeated edges to remove the from the list
        redge=[]
        for ed in pedge:
            redge.append(oriented_edges[ed])
        for redg in redge:
            oriented_edges.remove(redg)
            
        #storing current slice points
        vconstpnt=[]
        #iterating over the modified list of edges to genrate points
        for e in range(len(oriented_edges)): 
            #getting ordered points
            pnts1, pnts_gp1 = pointGen(oriented_edges[e], ref_pnt2)
            ref_pnt2=pnts1[len(pnts1)-1]
            for p in pnts_gp1:
                vconstpnt.append(p)

            #getting face containing that point
            face=TopoDS_Shape()
            bAncestor1 = BRepAlgoAPI_Section.HasAncestorFaceOn1(section, oriented_edges[e], face)
            #checking if edge lies on this face or not
            if bAncestor1==True:
                gSurface = BRep_Tool.Surface(topods.Face(face))
                #getting normal vectors on generated points
                normals, nor_gp=normalGen(gSurface, pnts_gp1)
            for normal in normals:
                all_normal.append(normal)
            for norm in nor_gp:
                norm_slice.append(norm)

        #appending ordered points on current slice to list of all points
        for v in vconstpnt:
            all_pnt.append(v)
        
        #deleting all previous elements
        pnt_slice.clear()
        #appending points on current slice
        pnt_slice=vconstpnt
        #generating spiral points
        spiralPnts=spiralGen(pnt_slice, prevpnt_slice)
        for sp in (spiralPnts):
            all_spiralpnt.append(sp)
            
        prevpnt_slice.clear()
        #appending current slice points to previous slice points
        for spnt in (pnt_slice):
            prevpnt_slice.append(spnt)

        spnor=spnorm(norm_slice, prevnorm_slice)
        for no in spnor:
            all_spnormal.append(no)
        prevnorm_slice.clear()
        for np in norm_slice:
            prevnorm_slice.append(np)
          
    #appending all the complete list to  respective files
    #slice points
    for pcontour in all_pnt:
        pnt=(pcontour.X(), pcontour.Y(), pcontour.Z())
        point_to_append1=str(pnt).replace("(","").replace(")","")
        f1.write(point_to_append1+"\n")
    #normals
    for ncontour in all_normal:
        normal_str=str(ncontour).replace("(","").replace(")","")
        f2.write(normal_str+"\n")
    #spiral points
    for sp_pnt in all_spiralpnt:
        spiral=(sp_pnt.X(), sp_pnt.Y(), sp_pnt.Z())
        str_sp=str(spiral).replace("(","").replace(")","")
        f4.write(str_sp+'\n')
    #append spiral noirmals to file
    for sp_nor in all_spnormal:
        gp_nor=(sp_nor.X(), sp_nor.Y(), sp_nor.Z())
        str_nsp=str(gp_nor).replace("(","").replace(")","")
        f5.write(str_nsp+"\n")

    
    f1.close()
    f2.close()
    f4.close()
    f5.close()
    return "Points and normals are successfully generated!"

# Function to zip a folder
def zip_folder(folder_path):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zf.write(os.path.join(root, file),
                         os.path.relpath(os.path.join(root, file), 
                                         os.path.join(folder_path, '..')))
    memory_file.seek(0)
    return memory_file

@app.route('/comment', methods=['post'])
def feedback():
    global name
    comment=request.form['comment']
    f_comment=open('feedback.txt', 'a')
    f_comment.write(name+":"+comment+'\n')

    return render_template('index.html',message2="feedback submitted successfully", message='Welcome '+name)

#downloads file containing points
@app.route('/download1')
def download_file1():
    file_location = zip_folder(f"C:/Rnd_ISF_Slicer/IncrementalForming/users/{name}/{cont_folder}") 
    return send_file(file_location, as_attachment=True, download_name='Contour_'+str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))+'.zip')

#downloads flie containing normals
@app.route('/download2')
def download_file2():
    file_location = zip_folder(f"C:/Rnd_ISF_Slicer/IncrementalForming/users/{name}/{spir_folder}") 
    return send_file(file_location, as_attachment=True, download_name='Spiral_'+str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))+'.zip')

@app.route('/simul1')
def simulate_contour():
    ht1="C:/Rnd_ISF_Slicer/IncrementalForming/static/simulContour.html"
    simulate(scontour, ht1, 'Contour Trajectory')
    return render_template('simulate1.html')

@app.route('/simul2')
def simulate_spiral():
    ht2="C:/Rnd_ISF_Slicer/IncrementalForming/static/simulSpiral.html"
    simulate(sspiral, ht2, 'Spiral Trajectory')
    return render_template('simulate2.html')

@app.route('/view')
def exit_simul():
    return render_template('view.html')

@app.route('/view/<filename>')
def view_file(filename):
    return render_template('view.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    #creating the directories if already not created
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000)
