# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 21:05:11 2022

@author: saif

"""
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""




from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
class yolovs (object):
    def __init__(self, weights):
        FILE = Path(__file__).resolve()
        self.ROOT = FILE.parents[0]  # YOLOv5 root directory
        if str(self.ROOT) not in sys.path:
            sys.path.append(str(self.ROOT))  # add ROOT to PATH
        self.ROOT = Path(os.path.relpath(self.ROOT, Path.cwd()))  # relative

        dnn = False
        # Load model
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        self.model.model.float()

    def detect(self, image,  # file/dir/URL/glob, 0 for webcam

               imgsz=(640, 640),  # inference size (height, width)
               conf_thres=0.8,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=10,  # maximum detections per image
               device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               # do not save images/videos
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               visualize=False,
               update=False,  # update all models

               name='exp',  # save results to project/name
               exist_ok=False,  # existing project/name ok, do not increment
               line_thickness=3,  # bounding box thickness (pixels)
               hide_labels=False,  # hide labels
               hide_conf=False,  # hide confidences
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               ):
        data = self.ROOT / 'data/coco128.yaml',  # dataset.yaml path
        project = self.ROOT / 'runs/detect',  # save results to project/name
        stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine

        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # 00000000000 check
        # dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        # for path, im, im0s, vid_cap, s in dataset:
        # Padded resize

        img = letterbox(image, imgsz, stride=stride)[0]
        print("img", img.shape, type(img))
        print("il image", image.shape, type(image))
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        # print("ba3ed",im)
        im = im.float()  # uint8 to fp16/32
        # print(im)
        im = im / 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=augment)

        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes,
            agnostic_nms,
            max_det=max_det)
        # dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        res = []
        for i, det in enumerate(pred):  # per image
            # seen += 1
            # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], image.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    res.append([int(cls), int(xyxy[0]), int(
                        xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    image = cv2.rectangle(
                        image, (int(
                            xyxy[0]), int(
                            xyxy[1])), (int(
                                xyxy[2]), int(
                                xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(
                        image, str(
                            int(cls)), (int(
                                xyxy[0]), int(
                                xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imwrite('saif2.png', image)
        return (res)
