import logging
import os
from typing import Annotated, Optional
import numpy as np

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

# scipy.optimize
try:
    from scipy.optimize import minimize, Bounds
except ModuleNotFoundError:
    slicer.util.pip_install("scipy")
    from scipy.optimize import minimize, Bounds

# spatialmath
try:
    from spatialmath import SE3
except ModuleNotFoundError:
    slicer.util.pip_install("spatialmath-python")
    from spatialmath import SE3

# NetworkX
try:
    import networkx as nx
except ModuleNotFoundError:
    slicer.util.pip_install("networkx")
    import networkx as nx
    
# ikpy
try:
    from ikpy.chain import Chain
    from ikpy.link import URDFLink
except ModuleNotFoundError:
    slicer.util.pip_install("ikpy")
    from ikpy.chain import Chain
    from ikpy.link import URDFLink
    
# json
try:
    import json
except ModuleNotFoundError:
    slicer.util.pip_install("json")
    import json
    
# time
try:
    import time
except ModuleNotFoundError:
    slicer.util.pip_install("time")
    import time
    


from slicer import vtkMRMLScalarVolumeNode


#
# RobotCavityScan
#


class RobotCavityScan(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("RobotCavityScan")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#RobotCavityScan">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # RobotCavityScan1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RobotCavityScan",
        sampleName="RobotCavityScan1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "RobotCavityScan1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="RobotCavityScan1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="RobotCavityScan1",
    )

    # RobotCavityScan2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="RobotCavityScan",
        sampleName="RobotCavityScan2",
        thumbnailFileName=os.path.join(iconsPath, "RobotCavityScan2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="RobotCavityScan2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="RobotCavityScan2",
    )


#
# RobotCavityScanParameterNode
#


@parameterNodeWrapper
class RobotCavityScanParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode
    
    

#
# RobotCavityScanWidget
#


class RobotCavityScanWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.downsampled_points = None
        self.optimalidx = None
        self.optialdistance = None
        self.bestpath = None
        self.bestcost = None
        
        # Create ROS2 publisher to send optimal joint angles to /optangles topic
        rosLogic = slicer.util.getModuleLogic('ROS2')
        rosNode = rosLogic.GetDefaultROS2Node()
        pub = rosNode.CreateAndAddPublisherNode('DoubleArray', '/optangles')
        self.pub = pub

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/RobotCavityScan.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)


        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RobotCavityScanLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.applytspbutton.connect("clicked(bool)", self.onapplytspbutton)
        self.ui.LoadURDFbutton.connect("clicked(bool)", self.LoadURDFbutton)
        self.ui.Planbutton.connect("clicked(bool)", self.onPlanButton)
        self.ui.executebutton.connect("clicked(bool)", self.onExecuteButton)  
        self.ui.loadpc.connect("clicked(bool)", self.onloadpcbutton)
        self.ui.ikbutton.connect("clicked(bool)", self.onikbutton)  

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[RobotCavityScanParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)
                                   
    def LoadURDFbutton(self) -> None:
        urdf_filepath = self.ui.urdfPathLineEdit.currentPath
        
        if not urdf_filepath:
            slicer.util.errorDisplay(_("Please select a URDF file to load."))
        else:
            self.logic.load_urdf_file(urdf_filepath)
            
    def onloadpcbutton(self) -> None:
        # Get downsample value
        value = self.ui.downsamplevalue.value
        # Get points from JSON file
        json_filepath = self.ui.jsonpath.currentPath
        # Check if JSON file is selected
        if not json_filepath:
            slicer.util.errorDisplay(_("Please select a JSON file with points."))
            return
        # Read in points from JSON file
        points, markupsNode = self.logic.read_json(json_filepath)
        # Downsample pointcloud by value ratio
        downsample, indices = self.logic.downsample_points(points,value/100)
        
        # Step 4: Create a new markups node to show the downsampled points
        downsampledNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "DownsampledPoints")
        slicer.util.updateMarkupsControlPointsFromArray(downsampledNode, downsample)

        # After creating the new node and adding downsampled points
        original_display = markupsNode.GetDisplayNode()
        new_display = downsampledNode.GetDisplayNode()
        new_display.CopyContent(original_display)
        new_display.PointLabelsVisibilityOff()
        original_display.SetVisibility(False)  # Hide original markups

        self.downsampled_points = downsample
    
    def onapplytspbutton(self) -> None:
        if self.downsampled_points is None:
            slicer.util.errorDisplay("❌ Please load a point cloud first.")
            return

        initial_path = list(range(len(self.downsampled_points)))
        initial_distance = self.logic.calculate_total_distance(self.downsampled_points, initial_path)
        optimized_path_indices, optimized_distance = self.logic.two_opt(self.downsampled_points, initial_path)

        slicer.util.infoDisplay(
            f"Path optimized from {initial_distance:.2f} to {optimized_distance:.2f} (no return)."
        )

        self.optimalidx = optimized_path_indices
        self.optialdistance = optimized_distance

        # Remove existing path node if it exists
        existing_path = slicer.mrmlScene.GetFirstNodeByName("TSP_CurvePath")
        if existing_path:
            slicer.mrmlScene.RemoveNode(existing_path)

        # Create a curve node for the path
        curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "TSP_CurvePath")
        displayNode = curveNode.GetDisplayNode()
        displayNode.SetVisibility3D(True)
        displayNode.SetVisibility2D(False)
        displayNode.SetLineThickness(2.0)
        displayNode.SetSelectedColor(1.0, 0.0, 0.0)  # Optional: green path

        # Add the optimized points to the curve node
        for idx in optimized_path_indices:
            pt = self.downsampled_points[idx]
            curveNode.AddControlPointWorld(pt)


    def onikbutton(self) -> None:
        # Get perturbation value
        eps = self.ui.perturbval.value
        # Grab all movable joints
        idx = self.logic.get_movable_indices()
        # Build a full 4×4 target frame for first target point
        pos = self.downsampled_points[self.optimalidx[0]]
        T = self.logic.build_target_frame(pos)
        # Set home position as intial guess
        initialpos = [0.0] * (len(self.logic.chain.links))
        # Calculate IK solution for first target
        res,p_target,R_target = self.logic.solve_ik_bfgs(T,initialpos)
        if not res.success:
            slicer.util.errorDisplay(_("First IK solution did not converge."))
        else:
            # Save initial solution
            q_solinit = res.x
            
            # Set the weights jused for edge cost calculation, THIS SHOULD BE USER DEFINED
            weights = np.ones(len(self.logic.chain.links))

            # Build graph
            G = self.logic.create_graph(q_solinit, initialpos, self.downsampled_points, weights, idx, eps)

            # Find shortest path
            best_path, best_cost = self.logic.get_shortest_path(G)
    
            # Convert list to numpy array
            best_path = np.array(best_path)
            
            # Save values to logic attributes for later use
            self.bestpath = best_path
            self.bestcost = best_cost
            slicer.util.infoDisplay("Best path cost: {:.2f}".format(best_cost))

        
    def onPlanButton(self) -> None:
        # Flatten the best path to a 1D array
        flat_data = self.bestpath.flatten()

        # Create blank message and set values
        msg = self.pub.GetBlankMessage()
        msg.SetNumberOfTuples(len(flat_data))
        for i, val in enumerate(flat_data):
            msg.SetValue(i, val)

        self.pub.Publish(msg)
        slicer.util.infoDisplay("Plan sent to MoveIt")

    def onExecuteButton(self) -> None:
        slicer.util.infoDisplay("Robot execution logic is not implemented yet.")
    	
    	

#
# RobotCavityScanLogic
#

class RobotCavityScanLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.chain = None
        
    def load_urdf_file(self, urdf_path: str):

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        try:
            self.chain = Chain.from_urdf_file(urdf_path, base_elements=['joint1'])
            slicer.util.infoDisplay("✅ URDF file loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}")
        
    def load_markups_node(self, filepath: str):
        [success, node] = slicer.util.loadMarkups(filepath, returnNode=True)
        if not success:
            raise RuntimeError(f"❌ Failed to load markups from: {filepath}")
        return node

    def get_control_points_as_array(self, markups_node):
        return slicer.util.arrayFromMarkupsControlPoints(markups_node)


    def read_json(self, json_path, tf=None):
        if tf is None:
            tf = np.eye(4)

        # Load markups node
        markupsNode = slicer.util.loadMarkups(json_path)

        # Extract control points as a NumPy array
        positions = slicer.util.arrayFromMarkupsControlPoints(markupsNode)

        positions_tf = np.zeros_like(positions)

        for i in range(len(positions)):
            pos_h = np.append(positions[i], 1)  
            pos_h = tf @ pos_h
            positions_tf[i] = pos_h[:3]

        return positions_tf, markupsNode
    
    def get_movable_indices(self):
        if self.chain is None:
            slicer.util.errorDisplay("❌ URDF not loaded. Please load the URDF file first.")
            return []
        return [i for i, link in enumerate(self.chain.links) if link.joint_type in ['revolute', 'prismatic']]

    
    def calculate_total_distance(self, points, path):
        """Compute total Euclidean distance along a given index path."""
        distance = 0.0
        for i in range(len(path) - 1):
            distance += np.linalg.norm(points[path[i]] - points[path[i+1]])
        return distance

    def two_opt(self, points, initial_path, max_iterations=100):
        """Improve a path via 2-opt heuristic."""
        best_path = initial_path.copy()
        best_distance = self.calculate_total_distance(points, best_path)
        iteration = 0
        improved = True

        while improved and iteration < max_iterations:
            improved = False
            for i in range(1, len(points) - 2):
                for j in range(i + 1, len(points)):
                    if j - i == 1:
                        continue
                    new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                    new_distance = self.calculate_total_distance(points, new_path)
                    if new_distance < best_distance:
                        best_path, best_distance = new_path, new_distance
                        improved = True
            iteration += 1

        return best_path, best_distance

    def downsample_points(self, points, downsample_ratio):
        """Randomly downsample and scale a point cloud."""
        n_keep = int(len(points) * downsample_ratio)
        indices = np.random.choice(len(points), n_keep, replace=False)
        downsampled = points[indices]
        print(f"Downsampled from {len(points)} to {len(downsampled)} points")
        return downsampled, indices

    def build_target_frame(self, pos):
        """
        Build an SE3 pose at pos = [x,y,z] with the Z-axis pointing down.
        Returns an SE3 object, so you can do T.t and T.R.
        """
        # start with a pure translation
        T = SE3(pos) * SE3.Rx(np.pi)
        return T
    
    def solve_ik_bfgs(self, T, initialpos, method="L-BFGS-B",
                      ftol=1e-6, maxiter=200):
        """
        Solve IK via quasi-Newton (BFGS or L-BFGS-B) for a target position `pos`
        with optional Z-down flipping. Respects joint limits.
        Returns the optimization result `res`.
        """
        # 1) Extyract target position and rotation
        p_target = T.t
        R_target = T.R

        # 2) initial guess
        q0 = initialpos

        full_bounds = [
            link.bounds if link.bounds is not None else (-np.pi, np.pi)
            for link in self.chain.links
        ]

        # 2) Split into lower/upper arrays
        lower_full, upper_full = zip(*full_bounds)

        # 4) Build your Bounds object
        bounds = Bounds(lower_full, upper_full)

        # 4) cost function: pos‖dp‖² + rot‖dR‖²
        def cost(q):
            Tsol = self.chain.forward_kinematics(q)
            dp   = Tsol[:3, 3] - p_target
            dR   = Tsol[:3, :3] - R_target
            return dp.dot(dp) + np.linalg.norm(dR, ord='fro')**2

        # 5) call SciPy minimize
        res = minimize(
            cost,
            q0,
            method=method,
            bounds=bounds,
            options={"ftol": ftol, "maxiter": maxiter}
        )
        return res, p_target, R_target
    
    def extract_active_joints(self, full_solution):
        """From IKPy’s full-chain solution (one value per link), drop the 0th element (the origin dummy) and then keep only the revolute joint angles."""
        # drop index 0
        sol = full_solution[1:]
        links = self.chain.links[1:]
        # filter on actual joint axes
        return [
            angle
            for angle, link in zip(sol, links)
            if getattr(link, "axis", None) is not None
        ]
    
    def perturb(self, q, idx, eps, n=3):
        """Generate n random perturbations around q, only at indices in idx, within ±eps."""
        q = np.asarray(q)
        perturbations = []
        for _ in range(n):
            delta = np.zeros_like(q)
            delta[idx] = np.random.uniform(-eps, eps, size=len(idx))
            perturbations.append(q + delta)
        return perturbations
    
    def weighted_joint_distance(self, q1, q2, weights):
        diff = q2 - q1
        return np.sqrt(np.sum(weights * diff**2))
    
    def create_graph(self, q_initsol, homepose, targetpoints, weights,idx,eps=0.05,n=3):
        # Initialize the graph
        G = nx.Graph()

        # Build first layer with home configuration
        G.add_node("L0_0", q=homepose, layer=0)

        # Create Layer nodes: original + 3 perturbations for first layer
        current_layer_nodes = []
        current_layer_nodes.append(("L1_0", q_initsol))
        perturbed_qs = self.perturb(q_initsol,idx,eps)

        # Add perturbed nodes to the first layer
        for i, q in enumerate(perturbed_qs, start=1):
            current_layer_nodes.append((f"L1_{i}", q))

        # Add layer 1 nodes to graph
        for node_name, q in current_layer_nodes:
            G.add_node(node_name, q=q, layer=1)

        # Connect home node to all Layer 1 nodes
        for node_name, q in current_layer_nodes:
            G.add_edge("L0_0", node_name)
            q1 = G.nodes["L0_0"]["q"]
            q2 = G.nodes[node_name]["q"]
            cost = self.weighted_joint_distance(q1, q2, weights)
            G["L0_0"][node_name]["weight"] = cost

        print("Layer 1 nodes added")
            
        # start building the rest of the layers starting from layer 2, outer loop over target points
        for layer_idx, target_point in enumerate(targetpoints[1:], start=2):
            print(f"Building layer {layer_idx}")
            # For each layer, we will create a new set of nodes based on the previous layer's nodes    
            next_layer_nodes = []
            
            # Iterate through the current layer nodes
            for parent_i, (parent_name, parent_q) in enumerate(current_layer_nodes):
                # Build target tf
                T = self.build_target_frame(target_point)
                # Solve IK for target pose with BFGS
                q_sol,p_target,R_target = self.solve_ik_bfgs(T,parent_q)
                
                # Check if the solution is valid
                if not q_sol.success:
                    print(f"⚠️ BFGS did not converge for {parent_name}: {q_sol.message}")
                    continue
                q_sol = q_sol.x
                print(f"  ✅ IK solution found for initial guess node {parent_name}")
                
                # Add the new node to the layer
                node_name = f"L{layer_idx}_{parent_i}"
                G.add_node(node_name, q=q_sol, layer=layer_idx)
                next_layer_nodes.append((node_name, q_sol))

            # Fully connect previous layer to this new layer, and calculate the edge cost
            for parent_name, _ in current_layer_nodes:
                q1 = G.nodes[parent_name]["q"]
                for child_name, _ in next_layer_nodes:
                    q2 = G.nodes[child_name]["q"]
                    G.add_edge(parent_name, child_name)
                    cost = self.weighted_joint_distance(q1, q2, weights)
                    G[parent_name][child_name]["weight"] = cost

            print(f"Layer {layer_idx} nodes added")
            current_layer_nodes = next_layer_nodes
            
        return G

    def layered_layout_truncated(self, G, max_layers_shown=8):
        pos = {}
        layers = {}

        # Group nodes by layer
        for node, data in G.nodes(data=True):
            layer = data.get("layer", 0)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        sorted_layers = sorted(layers.items())
        total_layers = len(sorted_layers)

        # Decide which layers to keep
        if total_layers > max_layers_shown:
            keep_first = 4
            keep_last = 4
            visible_layers = (
                sorted_layers[:keep_first] +
                [("...", ["..."])] +  # placeholder node for skipped layers
                sorted_layers[-keep_last:]
            )
        else:
            visible_layers = sorted_layers

        # Assign positions
        for x, (layer_idx, nodes) in enumerate(visible_layers):
            for y, node in enumerate(nodes):
                pos[node] = (x, -y)

        return pos
    
    
    def get_shortest_path(self, G, source="L0_0"):
        last_layer_idx = max(data["layer"] for _, data in G.nodes(data=True))
        last_layer_nodes = [n for n, d in G.nodes(data=True) if d["layer"] == last_layer_idx]

        best_path = None
        best_cost = float("inf")

        for target in last_layer_nodes:
            try:
                cost = nx.dijkstra_path_length(G, source, target=target, weight="weight")
                if cost < best_cost:
                    best_cost = cost
                    best_path = nx.dijkstra_path(G, source, target=target, weight="weight")
            except nx.NetworkXNoPath:
                continue

        print("Best path:", best_path)
        print("Best cost:", best_cost)

        if best_path is None:
            return [], float("inf")

        # Extract joint state vectors from the best path
        joint_path = [G.nodes[node]["q"][1:7] for node in best_path]

        return joint_path, best_cost

    def getParameterNode(self):
        return RobotCavityScanParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# RobotCavityScanTest
#


class RobotCavityScanTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_RobotCavityScan1()

    def test_RobotCavityScan1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("RobotCavityScan1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = RobotCavityScanLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
