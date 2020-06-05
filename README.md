# VisionToolKit
A toolkit for computer vision.

## installlation
`pip install git+https://github.com/hktxt/VisionToolKit`

## how to use
```
import VisionToolKit as vtk  
vtk.draw_bbox(img, bbox)
vtk.draw_mask(img, mask)
vtk.imshow(img)
vtk.imsave('a.jpg', img)
vtk.readJson(pth)
vtk.xyxy2xywh(x)
vtk.xywh2xyxy(x)
vtk.rle2mask(rle, imgshape)
...
```
