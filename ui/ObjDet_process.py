import torch
import sys, cv2, time, os
from ui.object_detection_platform_rc import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
import os
import prettytable as pt
from mmdet.apis import (inference_detector, init_detector)
from mmcv.utils import print_log
import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import multi_gpu_test, single_gpu_test
from PyQt5.Qt import QThread


class mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.listWidget_init()
        self.pushButton.clicked.connect(self.open_directory)
        self.listWidget_7.currentItemChanged.connect(self.image)  # open image from directory
        self.pushButton_2.clicked.connect(self.detect_single_image)
        self.pushButton_3.clicked.connect(self.detection_compare)


    def listWidget_init(self):
        self.listWidget.setSortingEnabled(True)
        self.listWidget_7.setSortingEnabled(True)
        self.listWidget_8.setSortingEnabled(True)


    def open_directory(self):
        self.listWidget_7.clear()  # refresh
        global filename1
        filename1=QFileDialog.getExistingDirectory(self,
                                    "选取文件夹",
                                    "./")
        if len(filename1):
            for i in os.listdir(filename1):
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(filename1 +"/" +i).scaled(60, 30), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.listWidget_7.setIconSize(QtCore.QSize(60, 30))
                item = QtWidgets.QListWidgetItem(icon, filename1+"/"+i)
                self.listWidget_7.addItem(item)
        try:
            self.listWidget_7.setCurrentRow(0)
        except Exception as inst:
            print(inst)


    def image(self):  # load current image
        if self.listWidget_7.currentItem() is not None:
            imagefile=self.listWidget_7.currentItem().text()
            jpg = QtGui.QPixmap(imagefile).scaled(self.origin_img.width(), self.origin_img.height())
            self.origin_img.setPixmap(jpg)


    def detect_single_image(self):
        self.listWidget.clear()  # refresh
        root_path = '/home/mst10512/mmdetection_219/'
        img = self.listWidget_7.currentItem().text()  # current image
        if self.comboBox.currentText() == "Faster R-CNN + FPN":
            if self.comboBox_2.currentText() == "COCO":  # faster_rcnn_fpn_coco
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_coco.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "PASCAL VOC":  # faster_rcnn_fpn_voc
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_voc.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "DIOR":  # faster_rcnn_fpn_dior
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_dior.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/dior/faster_rcnn_r50_fpn_dior.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

        if self.comboBox.currentText() == "Faster R-CNN + CSA-FPN":
            if self.comboBox_2.currentText() == "COCO":  # faster_rcnn_fpn_csa_coco:
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_coco.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_csa_coco.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "PASCAL VOC":  # faster_rcnn_fpn_csa_voc
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_voc.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_csa_voc.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "DIOR":  # faster_rcnn_fpn_csa_dior
                config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_dior.py'
                checkpoint = root_path + 'checkpoints/faster_rcnn/dior/faster_rcnn_r50_fpn_csa_dior.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

        if self.comboBox.currentText() == "RetinaNet + FPN":
            if self.comboBox_2.currentText() == "COCO":  # retinanet_fpn_coco:
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
                checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_coco.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "PASCAL VOC":  # retinanet_fpn_voc
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_voc.py'
                checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_voc.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "DIOR":  # retinanet_fpn_dior
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_dior.py'
                checkpoint = root_path + 'checkpoints/retinanet/dior/retinanet_r50_fpn_dior.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

        if self.comboBox.currentText() == "RetinaNet + CSA-FPN":
            if self.comboBox_2.currentText() == "COCO":  # retinanet_fpn_csa_coco:
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_coco.py'
                checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_csa_coco.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "PASCAL VOC":  # retinanet_fpn_csa_voc
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_voc.py'
                checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_csa_voc.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

            if self.comboBox_2.currentText() == "DIOR":  # retinanet_fpn_csa_dior
                config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_dior.py'
                checkpoint = root_path + 'checkpoints/retinanet/dior/retinanet_r50_fpn_csa_dior.pth'
                #self.object_detect_api(config, checkpoint, img)  # detect image

        self.object_detect_api(config, checkpoint, img)  # detect image


    def object_detect_api(self, config, checkpoint, img):  # config , checkpoint and img are both path (string)
        # build the model from a config file and a checkpoint file
        model = init_detector(config, checkpoint, device='cuda:0')
        begin_time = time.time()  # inference time start
        # test a single image
        result = inference_detector(model, img)
        end_time = time.time()
        consume_time = round((end_time - begin_time) * 1000, 5)  # en
        self.lineEdit.setText(str(consume_time) + ' ms')
        # save the results
        of = '/home/mst10512/mmdetection_219/detect_results/' + img.split('/')[-1]
        ui_result = model.show_result(img, result, score_thr=0.5, show=False, out_file=of, ui_show=True)  # ui_show=True, return [img, ui_result]
        # show the detected image
        imagefile = of
        jpg = QtGui.QPixmap(imagefile).scaled(self.deteted_img.width(), self.deteted_img.height())
        self.deteted_img.setPixmap(jpg)

        # show detection details
        tb = pt.PrettyTable()
        tb.field_names = ['class', 'confidence', 'bbox left-top', 'bbox right-bottom']
        tb.align['class'] = 'l'
        tb.align['confidence'] = 'c'
        tb.align['bbox left-top'] = 'c'
        tb.align['bbox right-bottom'] = 'c'
        tb.border = False
        tb.padding_width = 5
        info = ui_result[1]
        for i in range(len(info)):
            row_data = [
                info[i][0], f'{info[i][1]:.3f}', ('{:.3f}'.format(info[i][2]),'{:.3f}'.format(info[i][3]))
                , ('{:.3f}'.format(info[i][4]), '{:.3f}'.format(info[i][5]))]
            tb.add_row(row_data)
        print_log(tb)
        text_data = str(tb)
        item = QtWidgets.QListWidgetItem(text_data)
        self.listWidget.addItem(item)


    def detection_compare(self):
        self.listWidget_8.clear()  # refresh
        root_path = '/home/mst10512/mmdetection_219/'
        if self.comboBox_3.currentText() == 'Faster R-CNN + FPN':
            if self.comboBox_5.currentText() == 'COCO':
                bl_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
                bl_checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_coco.pth'
                #self.detect_dataset_api(bl_config, bl_checkpoint)
            if self.comboBox_5.currentText() == 'PASCAL VOC':
                bl_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py'
                bl_checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_voc.pth'
                #self.detect_dataset_api(bl_config, bl_checkpoint)
            if self.comboBox_5.currentText() == 'DIOR':
                bl_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_dior.py'
                bl_checkpoint = root_path + 'checkpoints/faster_rcnn/dior/faster_rcnn_r50_fpn_dior.pth'
                #self.detect_dataset_api(bl_config, bl_checkpoint)

        if self.comboBox_3.currentText() == 'RetinaNet + FPN':
            if self.comboBox_5.currentText() == 'COCO':
                bl_config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
                bl_checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_coco.pth'
                #self.detect_dataset_api(bl_config, bl_checkpoint)
            if self.comboBox_5.currentText() == 'PASCAL VOC':
                bl_config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_voc.py'
                bl_checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_voc.pth'
                #self.detect_dataset_api(bl_config, bl_checkpoint)
            if self.comboBox_5.currentText() == 'DIOR':
                bl_config = root_path + 'configs/retinanet/retinanet_r50_fpn_1x_dior.py'
                bl_checkpoint = root_path + 'checkpoints/retinanet/dior/retinanet_r50_fpn_dior.pth'
                #self.detect_dataset_api(bl_config, bl_checkpoint)

        if self.comboBox_4.currentText() == 'Faster R-CNN + CSA-FPN':
            if self.comboBox_5.currentText() == 'COCO':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/coco/faster_rcnn_r50_fpn_csa_coco.pth'
                #self.detect_dataset_api(new_config, new_checkpoint)
            if self.comboBox_5.currentText() == 'PASCAL VOC':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/voc/faster_rcnn_r50_fpn_csa_voc.pth'
                #self.detect_dataset_api(new_config, new_checkpoint)
            if self.comboBox_5.currentText() == 'DIOR':
                new_config = root_path + 'configs/faster_rcnn/faster_rcnn_r50_fpn_csa_1x_dior.py'
                new_checkpoint = root_path + 'checkpoints/faster_rcnn/dior/faster_rcnn_r50_fpn_csa_dior.pth'
                #self.detect_dataset_api(new_config, new_checkpoint)

        if self.comboBox_4.currentText() == 'RetinaNet + CSA-FPN':
            if self.comboBox_5.currentText() == 'COCO':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_coco.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/coco/retinanet_r50_fpn_csa_coco.pth'
                #self.detect_dataset_api(new_config, new_checkpoint)
            if self.comboBox_5.currentText() == 'PASCAL VOC':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_voc.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/voc/retinanet_r50_fpn_csa_voc.pth'
                #self.detect_dataset_api(new_config, new_checkpoint)
            if self.comboBox_5.currentText() == 'DIOR':
                new_config = root_path + 'configs/retinanet/retinanet_r50_fpn_csa_1x_dior.py'
                new_checkpoint = root_path + 'checkpoints/retinanet/dior/retinanet_r50_fpn_csa_dior.pth'
                #self.detect_dataset_api(new_config, new_checkpoint)

        self.detect_dataset_api(bl_config, bl_checkpoint)
        self.detect_dataset_api(new_config, new_checkpoint)

    def detect_dataset_api(self, config, checkpoint1, launcher='none', eval=True):  # test
        info = str("method:  " + self.comboBox_3.currentText() + "   dataset:  " + self.comboBox_5.currentText())
        method1 = QtWidgets.QListWidgetItem(info)
        print(info)
        self.listWidget_8.addItem(method1)

        cfg = Config.fromfile(config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        # init distributed env first, since logger depends on the dist info.
        if launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(launcher, **cfg.dist_params)

        rank, _ = get_dist_info()

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint1, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model=model, data_loader=data_loader, show=False, out_dir=None)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model=model, data_loader=data_loader)

        rank, _ = get_dist_info()
        if rank == 0:
            if eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
                ]:
                    eval_kwargs.pop(key, None)
                #eval_kwargs.update(dict(metric=args.eval, **kwargs))
                metric = dataset.evaluate(outputs, **eval_kwargs)
                print(metric)
                metric_dict = dict(config=config, metric=metric)

                #if args.work_dir is not None and rank == 0:
                #    mmcv.dump(metric_dict, json_file)


















if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = mywindow()
    myWin.show()
    sys.exit(app.exec_())

