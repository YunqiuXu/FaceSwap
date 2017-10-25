import numpy as np
import cv2
import models
from dlib import rectangle
import NonLinearLeastSquares

def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ

def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]

def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)

    return mesh

def getShape3D(mean3DShape, blendshapes, params):
    s = params[0]
    r = params[1:4]
    t = params[4:6]
    w = params[6:]

    R = cv2.Rodrigues(r)[0]
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D

def getMask(renderedImg):
    mask = np.zeros(renderedImg.shape[:2], dtype=np.uint8)

def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D

# YUNQIU XU : current useless!!!
# idxs3D : [26,59,5,6,94,111,112,53,98,104,56,110,100,23,103,97,20,99,109,48,49,50,17,16,15,64,7,31,8,79,80,85,86,89,87,88,40,65,10,32,62,61,63,29,28,30]
# idxs2D : [35,31,30,33,28,34,32,36,37,38,39,40,41,42,43,44,45,46,47,17,19,21,22,24,26,48,51,54,57,53,49,55,59,60,62,64,66,7,8,9,0,2,4,16,14,12]
def refine_idxs(idxs3D, idxs2D):
    FACE_POINTS = range(17, 68)
    MOUTH_POINTS = range(48, 61)
    RIGHT_BROW_POINTS = range(17, 22)
    LEFT_BROW_POINTS = range(22, 27)
    RIGHT_EYE_POINTS = range(36, 42)
    LEFT_EYE_POINTS = range(42, 48)
    NOSE_POINTS = range(27, 35)
    JAW_POINTS = range(0, 17)
    # Points used to line up the images.
    # ALIGN_POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
    ALIGN_POINTS = range(17,68)
    refined_3D = []
    refined_2D = []
    for i in range(len(idxs2D)):
        if idxs2D[i] in ALIGN_POINTS:
            refined_3D.append(idxs3D[i])
            refined_2D.append(idxs2D[i])
    return np.array(refined_3D), np.array(refined_2D)


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    dets = detector(scaledImg, 1)
    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale), int(det.bottom() / imgScale))
        dlibShape = predictor(img, faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()]) # 68 * 2
        shape2D = shape2D.T
        shapes2D.append(shape2D)

    return shapes2D


def getFaceTextureCoords(img, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor):
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    keypoints = getFaceKeypoints(img, detector, predictor)[0]

    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)

    return textureCoords
